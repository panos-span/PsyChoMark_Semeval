# starter/prompt_sweep_joint.py
import argparse
import json
import pathlib
import random
import re
import math  # <-- add this
import sys
import os
from collections import defaultdict
from typing import Any, Dict, List
import logging

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

# repo root on path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pathlib as _pathlib
from src.psycomark.llm.bedrock_chat import Chat
from src.psycomark.llm.eda_support import (
    build_s1_policy,
    build_s2_policy,
    load_fewshots,
)

ALL_TECHNIQUES = "fs_policy_boundary_sc5,fs_policy_boundary_cot_sc5,fs_policy_boundary_neg_cot_sc5,fs_cot_sc10,zs_policy_cot_sc5,zs_cot_sc5,fs_neg_cot_sc10"

"""
Each of these covers a distinct behavior:

Technique	Purpose
fs_policy_boundary_sc5	Classic few-shot + rubric + boundary + ensemble
fs_policy_boundary_cot_sc5	Same but with short reasoning
fs_policy_boundary_neg_cot_sc5	Adds negative shot to boost recall calibration
fs_cot_sc10	Heavy reasoning ensemble without rubric cues
zs_policy_cot_sc5	Zero-shot, rubric-driven, with CoT + ensemble
zs_cot_sc5	Pure zero-shot CoT with self-consistency
fs_neg_cot_sc10	Few-shot with neg. CoT examples + high variance
"""


# ---------- .env loader (no deps) ----------
def _load_dotenv_into_environ():
    root = _pathlib.Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    # Map non-standard names to AWS_* so boto3 sees them
    if "ACCESS_KEY_ID" in os.environ and "AWS_ACCESS_KEY_ID" not in os.environ:
        os.environ["AWS_ACCESS_KEY_ID"] = os.environ["ACCESS_KEY_ID"]
    if "SECRET_ACCESS_KEY" in os.environ and "AWS_SECRET_ACCESS_KEY" not in os.environ:
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["SECRET_ACCESS_KEY"]


_load_dotenv_into_environ()
# ------------------------------------------

# ---------- prompt artifacts (load + render) ----------
ARTIFACT_FILES = {
    "lexicons": "lexicons.json",
    "conflicts": "conflicts.json",
    "boundary_prompts": "boundary_prompts.json",
    "priors_prompt": "priors_prompt.json",
    "fewshot_bank": "fewshot_bank.json",
}


def _load_prompt_artifacts(eda_root: pathlib.Path) -> Dict[str, Any]:
    """Load prompt artifacts with ultra-robust defaults."""
    arts, paths = {}, {}
    for k, fname in ARTIFACT_FILES.items():
        p = eda_root / fname
        try:
            if p.exists():
                arts[k] = json.loads(p.read_text(encoding="utf-8"))
                paths[k] = str(p)
            else:
                arts[k] = {} if k != "fewshot_bank" else {"s1": [], "s2": []}
                paths[k] = None
        except Exception:
            arts[k] = {} if k != "fewshot_bank" else {"s1": [], "s2": []}
            paths[k] = None
    arts["_paths"] = paths
    return arts


def _render_boundary_block(arts: Dict[str, Any]) -> str:
    """Render <=2 short boundary cues per label."""
    bp = arts.get("boundary_prompts") or {}
    if not isinstance(bp, dict) or not bp:
        return ""
    lines = []
    for lab in ["Actor", "Action", "Effect", "Victim", "Evidence"]:
        v = bp.get(lab) or {}
        befo = (v.get("before") or [])[:2]
        aftr = (v.get("after") or [])[:2]
        parts = []
        if befo:
            parts.append("before=" + "; ".join(befo))
        if aftr:
            parts.append("after=" + "; ".join(aftr))
        if parts:
            lines.append(f"- {lab}: " + " | ".join(parts))
    return "Boundary cues (keep spans tight):\n" + "\n".join(lines) if lines else ""


def _render_conflicts_block(arts: Dict[str, Any]) -> str:
    cf = arts.get("conflicts") or {}
    pairs = cf.get("pairs") or []
    if not pairs:
        return ""
    pairs_txt = ", ".join([f"{a}–{b}" for a, b in pairs[:4]])
    return (
        "Conflict reminders (overlap resolution): "
        f"{pairs_txt}. Prefer prior-closer span; keep both only if semantically distinct."
    )


def _render_priors_block(arts: Dict[str, Any]) -> str:
    pr = arts.get("priors_prompt") or {}
    if not isinstance(pr, dict) or not pr:
        return ""
    lines = []
    for lab in ["Actor", "Action", "Effect", "Victim", "Evidence"]:
        d = pr.get(lab) or {}
        q50 = d.get("q50_len")
        q90 = d.get("q90_len")
        mode = d.get("start_mode")
        bits = []
        if isinstance(q50, (int, float)):
            bits.append(f"q50≈{int(q50)}")
        if isinstance(q90, (int, float)):
            bits.append(f"q90≈{int(q90)}")
        if isinstance(mode, (int, float)):
            bits.append(f"start≈{mode:.2f}")
        if bits:
            lines.append(f"- {lab}: " + ", ".join(bits))
    return "Span priors (length & position):\n" + "\n".join(lines) if lines else ""


def _snapshot_artifacts(arts: Dict[str, Any], out_dir: pathlib.Path):
    d = out_dir / "prompts" / "_artifacts"
    d.mkdir(parents=True, exist_ok=True)
    for k in ARTIFACT_FILES.keys():
        # write the loaded dicts (even if repaired) to snapshot
        with open(d / f"{k}.json", "w", encoding="utf-8") as f:
            json.dump(arts.get(k, {}), f, ensure_ascii=False, indent=2)


# ---------- utils ----------
def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def strict_json_extract(s: str):
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def list_json_extract(s: str):
    m = re.search(r"\[.*\]", s, flags=re.S)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, list) else []
    except Exception:
        return []


def shorten(txt: str, n=1600):
    return txt if len(txt) <= n else txt[:n] + "..."


# ---------- S1 post-processing (token-snap + cleanup + conflict NMS) ----------

STOPWORDS = set("to of in on at for with by a an the and or".split())


def _valid_span(text, s, e):
    if e - s < 3:
        return False
    span = text[s:e].strip()
    if not span:
        return False
    toks = re.findall(r"\w+|[^\w\s]", span)
    if toks and toks[0].lower() in STOPWORDS:
        return False
    if span in (".", ",", ";", ":", "’", "”"):
        return False
    return True


def _salience(m, txt, lexicons):
    s = m.get("startIndex", m.get("start", 0))
    e = m.get("endIndex", m.get("end", s))
    s, e = int(s), int(e)
    span = txt[s:e].lower()
    cues = [
        "cover up",
        "secret",
        "rigged",
        "agenda",
        "plot",
        "blame",
        "evidence",
        "collude",
        "fraud",
        "hoax",
    ]
    score = 0.0
    score += 1.0 if len(span) >= 8 else 0.0
    score += sum(0.3 for c in cues if c in span)
    for w in lexicons.get("absolutist", []) or []:
        if w in span:
            score += 0.1
    return -score  # lower is better in sort


_EVAL_TOKEN_RE = re.compile(r"(\w+|[^\w\s])")

CONFLICT_PAIRS = {("Action", "Effect"), ("Actor", "Victim")}


def _tokenize_eval(text: str):
    return [(m.start(), m.end()) for m in _EVAL_TOKEN_RE.finditer(text or "")]


def _char_to_token_set(s: int, e: int, toks):
    covered = set()
    for i, (ts, te) in enumerate(toks):
        if s < te and e > ts:
            covered.add(i)
    return covered


def _snap_to_tokens(span, toks):
    """Snap [start,end) to min/max boundaries of covered tokens. If none, snap to nearest token."""
    s, e = int(span["start"]), int(span["end"])
    if e <= s or not toks:
        return None
    covered = _char_to_token_set(s, e, toks)
    if covered:
        ns = min(toks[i][0] for i in covered)
        ne = max(toks[i][1] for i in covered)
        return {**span, "start": ns, "end": ne}
    # no overlap -> snap to nearest token
    # pick token whose center is closest to span center
    c = (s + e) / 2.0
    best = min(range(len(toks)), key=lambda i: abs((toks[i][0] + toks[i][1]) / 2.0 - c))
    ts, te = toks[best]
    return {**span, "start": ts, "end": te}


def _iou(a, b):
    inter = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    if inter <= 0:
        return 0.0
    union = (a["end"] - a["start"]) + (b["end"] - b["start"]) - inter
    return inter / union if union > 0 else 0.0


def _prior_dist(priors: dict, label: str, s: int, e: int, text_len: int):
    """Lower is better: combine |z_len| + start_pos distance to Beta mode (robust to missing)."""
    span_len = max(1, e - s)
    start_pos = s / max(1, text_len)
    p = priors.get(label, {})
    # length z ~ log-normal
    mu = p.get("length_lognorm", {}).get("mu", 0.0)
    sig = max(1e-6, p.get("length_lognorm", {}).get("sigma", 1.0))
    z_len = abs((math.log(max(1, span_len)) - mu) / sig)
    # position distance to Beta mode
    if "start_beta" in p:
        a = p["start_beta"].get("alpha", 1.0)
        b = p["start_beta"].get("beta", 1.0)
        mode = (
            ((a - 1) / (a + b - 2))
            if (a > 1 and b > 1)
            else (a / (a + b) if (a > 0 and b > 0) else 0.5)
        )
    else:
        mode = 0.5
    pos_dist = abs(start_pos - mode)
    return 1.5 * z_len + 1.0 * pos_dist


def _merge_tiny_gaps(spans, gap=1):
    """Merge same-label neighbors if next.start - prev.end <= gap."""
    if not spans:
        return spans
    out = []
    spans = sorted(spans, key=lambda x: (x["label"], x["start"], x["end"]))
    i = 0
    while i < len(spans):
        cur = dict(spans[i])
        j = i + 1
        while (
            j < len(spans)
            and spans[j]["label"] == cur["label"]
            and spans[j]["start"] - cur["end"] <= gap
        ):
            cur["end"] = max(cur["end"], spans[j]["end"])
            j += 1
        out.append(cur)
        i = j
    return out


def _dedup_same_label(spans, iou_thr=0.90, text_len=1, priors=None):
    """Remove near-duplicates per label using IoU; keep span closer to priors, else longer."""
    priors = priors or {}
    out = []
    for lab in {"Actor", "Action", "Effect", "Victim", "Evidence"}:
        S = [s for s in spans if s["label"] == lab]
        S = sorted(S, key=lambda x: (x["start"], x["end"]))
        keep = []
        used = [False] * len(S)
        for i, a in enumerate(S):
            if used[i]:
                continue
            best = a
            used[i] = True
            for j in range(i + 1, len(S)):
                if used[j]:
                    continue
                b = S[j]
                if _iou(a, b) >= iou_thr:
                    # choose prior-closer
                    da = _prior_dist(priors, lab, a["start"], a["end"], text_len)
                    db = _prior_dist(priors, lab, b["start"], b["end"], text_len)
                    cand = a if da <= db else b
                    best = (
                        cand
                        if (cand["end"] - cand["start"])
                        >= (best["end"] - best["start"])
                        else best
                    )
                    used[j] = True
            keep.append(best)
        out.extend(keep)
    return sorted(out, key=lambda x: (x["start"], x["end"]))


def _conflict_nms(spans, iou_thr=0.50, text_len=1, priors=None):
    """Suppress overlaps across conflict pairs using priors (lower prior_dist wins)."""
    priors = priors or {}
    if not spans:
        return spans
    spans = sorted(spans, key=lambda x: (x["start"], x["end"]))
    keep = [True] * len(spans)
    for i in range(len(spans)):
        if not keep[i]:
            continue
        a = spans[i]
        for j in range(i + 1, len(spans)):
            if not keep[j]:
                continue
            b = spans[j]
            pair = tuple(sorted((a["label"], b["label"])))
            if pair not in CONFLICT_PAIRS:
                continue
            if _iou(a, b) >= iou_thr:
                da = _prior_dist(priors, a["label"], a["start"], a["end"], text_len)
                db = _prior_dist(priors, b["label"], b["start"], b["end"], text_len)
                # keep prior-closer; if tie, keep longer; if still tie, keep earlier
                score_a = (da, -(a["end"] - a["start"]), a["start"])
                score_b = (db, -(b["end"] - b["start"]), b["start"])
                if score_a <= score_b:
                    keep[j] = False
                else:
                    keep[i] = False
                    break
    return [s for k, s in zip(keep, spans) if k]


def postprocess_s1_spans(
    text: str,
    spans: List[dict],
    priors: dict = None,
    merge_gap: int = 1,
    dedup_iou: float = 0.90,
    conflict_iou: float = 0.50,
):
    """Main post-proc: snap -> drop zero -> merge tiny gaps -> dedup -> conflict NMS."""
    if not text or not spans:
        return []
    toks = _tokenize_eval(text)
    L = len(text)
    # 1) normalize schema + snap
    canon = []
    for m in spans:
        lab = (m.get("label") or m.get("type") or "").strip()
        s = m.get("start", m.get("startIndex"))
        e = m.get("end", m.get("endIndex"))
        try:
            s, e = int(s), int(e)
        except Exception:
            continue
        s = max(0, min(L, s))
        e = max(0, min(L, e))
        if e <= s:
            continue
        snapped = _snap_to_tokens({"label": lab, "start": s, "end": e}, toks)
        if snapped and snapped["end"] > snapped["start"]:
            canon.append(snapped)
    if not canon:
        return []
    # 2) merge tiny gaps (same label)
    canon = _merge_tiny_gaps(canon, gap=merge_gap)
    # 3) de-dup near duplicates (same label)
    canon = _dedup_same_label(canon, iou_thr=dedup_iou, text_len=L, priors=priors)
    # 4) conflict-aware NMS (Action/Effect, Actor/Victim)
    canon = _conflict_nms(canon, iou_thr=conflict_iou, text_len=L, priors=priors)
    # back to submission schema + attach text slice
    out = []
    for m in canon:
        out.append(
            {
                "type": m["label"],
                "startIndex": m["start"],
                "endIndex": m["end"],
                "text": text[m["start"] : m["end"]],
            }
        )
    return out


# ---------- default prompts ----------
S1_BASE = """You are a careful annotator for PsyCoMark (SemEval-2026 Task 10, Subtask 1).
Task: extract character spans that best reflect the following labels: Actor, Action, Effect, Victim, Evidence.

**These markers reflect evolutionary psychology principles: agency detection, in-group/out-group threats, coalitional behavior, moral violations, and threat sensitivity.**

Output format (strict JSON list, no extra text):
[{"label":"Actor|Action|Effect|Victim|Evidence","start":int,"end":int}]

Rules and rubric:
- Offsets are 0-indexed; end is exclusive. Spans must lie within the provided TEXT.
- Keep spans tight and semantically meaningful. Exclude leading/trailing whitespace and punctuation.
- Prefer minimal spans that uniquely identify the entity/event (e.g., "agency" not "the agency" unless needed to disambiguate).
- Do not invent text not present in the input. Omit a label if not evidenced in the text.
- Overlaps are allowed when justified (e.g., an Action containing a sub-phrase), but avoid redundant duplicates.
- Boundary hints: include core content words; exclude trailing stopwords unless essential (e.g., prepositions integral to meaning).
"""

S1_USER = """TASK: Extract spans for labels: Actor, Action, Effect, Victim, Evidence.
Return ONLY a strict JSON list (no prose, no keys other than label/start/end):
[{{"label":"Actor|Action|Effect|Victim|Evidence","start":int,"end":int}}]

TEXT:
{doc_text}
"""

S2_SYS = """You are a careful annotator for PsyCoMark (SemEval-2026 Task 10, Subtask 2).
Goal: decide whether the REDDIT COMMENT promotes a conspiracy narrative.
Use brief private reasoning, but return ONLY the final JSON below.

Label decision rubric:
- "conspiracy": The author asserts or clearly endorses a conspiratorial claim.
- "non": Neutral reporting, critique/debunking, jokes/irony, or unrelated content.
Avoid using subreddit as a proxy; rely on text content and framing.

Probability rubric (p_conspiracy, p_non should sum to 1.0):
- 0.90-1.00: Explicit assertion/endorsement of a conspiracy.
- 0.60-0.80: Strong implication or supportive framing without explicit claim.
- ~0.50: Ambiguous/uncertain.
- 0.00-0.20: Clearly non-conspiratorial (neutral/debunking/irrelevant).

Return strict JSON ONLY:
{"label":"conspiracy|non","p_conspiracy":0.xx,"p_non":0.xx,"rationale":"<=2 sentences"}
Constraints:
- Ensure p_conspiracy + p_non = 1.0 (within rounding). Set label to the higher of the two.
- Keep rationale concise and non-revealing of chain-of-thought.
"""

S2_USER = """TASK: Document-level classification (conspiracy vs non). Return JSON only.

TEXT:
{doc_text}

DETECTED_MARKERS_JSON:
{markers_json}

HOW_TO_USE_MARKERS:
- Treat Actor/Action/Effect/Victim/Evidence as signals of conspiratorial framing.
- Absence of markers ≠ non-conspiracy; decide based on content.
- If uncertain, choose "non".

RETURN JSON:
{{"label":"conspiracy|non","p_conspiracy":0.xx,"p_non":0.xx,"rationale":"..."}}"""


def _save_text(path, text):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text if isinstance(text, str) else json.dumps(text, ensure_ascii=False))


def _save_prompt_bundle(
    out_dir: pathlib.Path,
    task: str,
    doc_id: str,
    sys_blocks: List[str],
    user_block: str,
):
    base = out_dir / "prompts" / task
    _save_text(base / f"{doc_id}.system.txt", "\n\n---\n\n".join(sys_blocks))
    _save_text(base / f"{doc_id}.user.txt", user_block)


def _save_prompt_meta(out_dir: pathlib.Path, meta: dict):
    _save_text(
        out_dir / "prompts" / "metadata.json",
        json.dumps(meta, ensure_ascii=False, indent=2),
    )


# --- NEW: helpers to emit Codabench-style files ---
def _to_codabench_s1(markers):
    """Drop 'text' field; keep keys: startIndex, endIndex, type."""
    out = []
    for m in markers or []:
        if not all(k in m for k in ("startIndex", "endIndex", "type")):
            # tolerate your internal schema: {type,startIndex,endIndex,text}
            t = m.get("type") or m.get("label")
            s = m.get("startIndex", m.get("start"))
            e = m.get("endIndex", m.get("end"))
        else:
            t, s, e = m["type"], m["startIndex"], m["endIndex"]
        try:
            s, e = int(s), int(e)
        except Exception:
            continue
        if e <= s:
            continue
        out.append({"startIndex": s, "endIndex": e, "type": str(t)})
    return out


# ---------- fewshots rendering ----------
def render_fewshots_block(examples: List[Dict[str, Any]], is_s2: bool) -> str:
    if not examples:
        return ""
    blocks = []
    for ex in examples:
        txt = ex.get("text") or ex.get("doc_text") or ""

        if is_s2:
            lbl = (
                ex.get("label") or (ex.get("gold") or {}).get("label") or "non"
            ).lower()
            if lbl not in ("conspiracy", "non"):
                lbl = "non"
            # assign sensible example probabilities
            if lbl == "conspiracy":
                gold = {
                    "label": "conspiracy",
                    "p_conspiracy": 0.8,
                    "p_non": 0.2,
                    "rationale": "asserts conspiracy.",
                }
            else:
                gold = {
                    "label": "non",
                    "p_conspiracy": 0.2,
                    "p_non": 0.8,
                    "rationale": "neutral/debunking.",
                }

            mk = ex.get("markers") or []
            mk_norm = []
            for m in mk:
                lab = (m.get("type") or m.get("label") or "").strip()
                s = m.get("startIndex", m.get("start"))
                e = m.get("endIndex", m.get("end"))
                try:
                    s, e = int(s), int(e)
                except:
                    continue
                if lab and e > s:
                    mk_norm.append({"type": lab, "startIndex": s, "endIndex": e})

            block = f"EXAMPLE:\nTEXT:\n{shorten(txt, 800)}\n"
            if mk_norm:
                block += (
                    "DETECTED_MARKERS_JSON:\n"
                    + json.dumps(mk_norm, ensure_ascii=False)
                    + "\n"
                )
            block += "JSON:\n" + json.dumps(gold, ensure_ascii=False)

        else:
            # NEW: coerce S1 spans from ex["spans"] or ex["markers"]
            spans = (
                ex.get("spans")
                or ex.get("markers")
                or ex.get("gold")
                or ex.get("json")
                or []
            )
            norm = []
            for m in spans:
                lab = (m.get("label") or m.get("type") or "").strip()
                s = m.get("start", m.get("startIndex"))
                e = m.get("end", m.get("endIndex"))
                try:
                    s, e = int(s), int(e)
                except Exception:
                    continue
                if lab and e is not None and s is not None and e > s:
                    norm.append({"label": lab, "start": s, "end": e})

            MAX_EX_SPAN = 90
            norm = [m for m in norm if (m["end"] - m["start"]) <= MAX_EX_SPAN]
            block = f"EXAMPLE:\nTEXT:\n{shorten(txt, 800)}\nJSON:\n" + json.dumps(
                norm, ensure_ascii=False
            )

        blocks.append(block)
    return "\n\n".join(blocks)


# ---------- quick GT validation ----------
def _file_has_markers(jsonl_path: str, sample: int = 200) -> bool:
    try:
        n, y = 0, 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                r = json.loads(ln)
                mks = r.get("markers") or []
                if isinstance(mks, list) and len(mks) > 0:
                    y += 1
                n += 1
                if n >= sample:
                    break
        return y > 0
    except Exception:
        return False


def tune_threshold_dev(rows_s2, prob_rows):
    """Return (best_thr, best_f1, stats)."""
    # build gold
    y_true = []
    id_order = []
    for r in rows_s2:
        lab = (r.get("doc_label") or "").strip().lower()
        if lab not in ("conspiracy", "non"):  # skip cant_tell
            continue
        y_true.append(1 if lab == "conspiracy" else 0)
        id_order.append(r.get("_id") or r.get("doc_id"))
    # align probs
    p_map = {r["_id"]: float(r["p_conspiracy"]) for r in prob_rows}
    y_prob = [p_map.get(i, 0.5) for i in id_order]
    # sweep thresholds
    import numpy as np
    from sklearn.metrics import f1_score

    best_f1, best_t = -1.0, 0.50
    for t in np.linspace(0.10, 0.90, 33):
        y_pred = [1 if p >= t else 0 for p in y_prob]
        f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    mean_p = float(sum(y_prob) / len(y_prob)) if y_prob else 0.5
    if mean_p < 0.15:
        logging.warning(
            "[S2] mean_p extremely low; likely 'all-No' drift. Check few-shots and markers."
        )

    return best_t, best_f1, {"mean_p": mean_p, "n": len(y_prob)}


# ---------- S1 helpers ----------
LABELS = {"Actor", "Action", "Effect", "Victim", "Evidence"}
_CANON_LABEL = {lab.lower(): lab for lab in LABELS}


def clip_and_validate(spans, text):
    L = len(text)
    out = []
    for m in spans:
        lab_raw = (m.get("label") or "").strip()
        lab = _CANON_LABEL.get(lab_raw.lower())
        if not lab:
            continue
        try:
            s_val = m.get("start")
            e_val = m.get("end")
            if s_val is None or e_val is None:
                # Fallback to startIndex/endIndex if model used that schema
                s_val = m.get("startIndex")
                e_val = m.get("endIndex")
            s = int(s_val)
            e = int(e_val)
        except Exception:
            continue
        s = max(0, min(L, s))
        e = max(0, min(L, e))
        if e <= s:
            continue
        out.append({"label": lab, "start": s, "end": e})
    return out


def iou(a, b):
    inter = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = (a["end"] - a["start"]) + (b["end"] - b["start"]) - inter
    return (inter / union) if union > 0 else 0.0


def merge_by_freq(all_runs: List[List[Dict[str, int]]], min_votes=2, iou_thr=0.8):
    kept = []
    for lab in LABELS:
        spans = [s for run in all_runs for s in run if s["label"] == lab]
        spans = sorted(spans, key=lambda x: (x["start"], x["end"]))
        clusters = []
        for s in spans:
            placed = False
            for cl in clusters:
                if iou(s, cl["rep"]) >= iou_thr:
                    cl["members"].append(s)
                    cl["rep"] = cl["members"][len(cl["members"]) // 2]
                    placed = True
                    break
            if not placed:
                clusters.append({"rep": s, "members": [s]})
        for cl in clusters:
            if len(cl["members"]) >= min_votes:
                kept.append(cl["rep"])
    return kept


# ---------- prompt builders (tech-agnostic) ----------
def s1_prompt(
    doc_text,
    policy,
    fewshots,
    boundary_note: str,
    tech: str,
    prompt_arts: Dict[str, Any] = None,
):
    fs_block = render_fewshots_block(fewshots, is_s2=False)
    system = [S1_BASE]
    if policy:
        system.insert(0, policy)
    if "boundary" in tech and boundary_note:
        system.append("Boundary guidance:\n" + boundary_note)

    # NEW: CoT checklist
    if "cot" in tech:
        system.append(
            "Checklist before labeling:\n"
            "- Identify agents (Actor), what they do (Action), consequences (Effect), victims, and any evidence.\n"
            "- Keep spans token-tight; exclude trailing punctuation/stopwords.\n"
            "- If Action and Effect overlap, choose the minimal span that best fits each role.\n"
            "- For Actor vs Victim overlaps, prefer the smaller specific mention for each role."
        )
    n_samples = 1
    temp = 0.0
    # --- NEW: append artifact-driven blocks (compact) ---
    if prompt_arts:
        if "boundary" in tech:
            b = _render_boundary_block(prompt_arts)
            if b:
                system.append(b)
        c = _render_conflicts_block(prompt_arts)
        if c:
            system.append(c)
        p = _render_priors_block(prompt_arts)
        if p:
            system.append(p)
    if tech.startswith("sc"):  # self-consistency
        n = re.search(r"sc(\d+)", tech)
        n_samples = int(n.group(1)) if n else 5
        temp = 0.7
    user = ""
    if "fs" in tech or tech.startswith("sc"):
        user += (fs_block + "\n\n") if fs_block else ""
    user += S1_USER.format(doc_text=doc_text)
    return system, user, n_samples, temp


def _pick_balanced_s1_fewshots(pool, k=6, seed=42):
    labs = ["Actor", "Action", "Effect", "Victim", "Evidence"]
    rng = random.Random(seed)

    def ex_has_lab(ex, lab):
        spans = ex.get("spans") or ex.get("markers") or []
        for m in spans:
            lab_m = (m.get("label") or m.get("type") or "").strip()
            if lab_m == lab:
                return True
        return False

    # 1) ensure one per label if available
    chosen, used = [], set()
    for lab in labs:
        cand = [e for e in (pool or []) if ex_has_lab(e, lab)]
        if cand:
            e = rng.choice(cand)
            if id(e) not in used:
                chosen.append(e)
                used.add(id(e))

    # 2) fill by salience
    def salience(ex):
        txt = (ex.get("text") or "").lower()
        cues = [
            "control",
            "cover up",
            "agenda",
            "rigged",
            "evidence",
            "plot",
            "blame",
            "secret",
        ]
        score = sum(1 for c in cues if c in txt)
        score += 1 if len(txt) >= 160 else 0
        return -score

    rest = [e for e in (pool or []) if id(e) not in used]
    rest.sort(key=salience)
    for e in rest:
        if len(chosen) >= k:
            break
        chosen.append(e)
        used.add(id(e))
    return chosen[:k]


def _pick_balanced_s2_fewshots(pool, k=8):
    ys = [e for e in (pool or []) if (e.get("label") or "").lower() == "conspiracy"]
    ns = [e for e in (pool or []) if (e.get("label") or "").lower() == "non"]
    t = min(k // 2, len(ys), len(ns))
    out = ys[:t] + ns[:t]
    out += [e for e in (pool or []) if e not in out][: max(0, k - len(out))]
    return out[:k]


def s2_prompt_with_markers(
    doc_text,
    policy,
    fewshots,
    tech: str,
    markers_json: str,
    prompt_arts: Dict[str, Any] = None,
):
    fs_block = render_fewshots_block(fewshots, is_s2=True)
    system = [S2_SYS]
    if "policy" in tech and policy:
        system.insert(0, policy)

    # NEW: CoT checklist
    if "cot" in tech:
        system.append(
            "Checklist before labeling:\n"
            "- Read the comment carefully. Are there psycholinguistic markers suggesting conspiratorial framing?\n"
            "- Are Actor/Action/Effect/Victim/Evidence markers present or strongly implied?\n"
            "- Is the framing endorsing a conspiracy, or is it neutral/joking/critical?\n"
            "- If uncertain or neutral, choose 'non'."
        )

    if prompt_arts:
        if "boundary" in tech:
            b = _render_boundary_block(prompt_arts)
            if b:
                system.append(b)
        c = _render_conflicts_block(prompt_arts)
        if c:
            system.append(c)
        p = _render_priors_block(prompt_arts)
        if p:
            system.append(p)

    # SC detection
    sc_match = re.search(r"sc(\d+)", tech)
    n_samples = int(sc_match.group(1)) if sc_match else 1
    temp = 0.7 if n_samples > 1 else 0.0

    user_prefix = ""
    if "fs" in tech or n_samples > 1:
        user_prefix = (fs_block + "\n\n") if fs_block else ""

    user = user_prefix + S2_USER.format(doc_text=doc_text, markers_json=markers_json)

    return system, user, n_samples, temp


# ---------- run S1 (then S2) for one doc ----------
def run_s1(
    doc, sys_blocks, user_block, model_id, region, max_tokens, temperature, n_samples
):
    text = doc.get("text", "")

    if n_samples == 1:
        chat = Chat(
            model_id=model_id,
            region=region,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        for s in sys_blocks:
            chat.add_system(s)
        chat.add_user(user_block)
        try:
            out = chat.generate()
            spans = clip_and_validate(list_json_extract(out), text)
            return spans
        except Exception as e:
            logging.warning(
                f"[S1] generate() failed for doc -> fallback empty spans: {e}"
            )
            return []
    else:
        runs = []
        for _ in range(n_samples):
            chat = Chat(
                model_id=model_id,
                region=region,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            for s in sys_blocks:
                chat.add_system(s)
            chat.add_user(user_block)
            try:
                out = chat.generate()
                runs.append(clip_and_validate(list_json_extract(out), text))
            except Exception as e:
                logging.warning(
                    f"[S1] generate() failed (SC) -> counting as empty: {e}"
                )
                runs.append([])
        return merge_by_freq(runs, min_votes=max(2, n_samples // 3), iou_thr=0.8)


def _coerce_probabilities(js: Dict[str, Any]) -> Dict[str, float]:
    """Return a dict with label, p_conspiracy, p_non parsed from model output."""
    label = (js.get("label") or "non").strip().lower()
    p_con = js.get("p_conspiracy")
    p_non = js.get("p_non")

    def _float_or_none(val):
        try:
            return float(val)
        except Exception:
            return None

    p_con = _float_or_none(p_con)
    p_non = _float_or_none(p_non)

    if p_con is None and p_non is None:
        conf = _float_or_none(js.get("confidence"))
        if conf is None:
            conf = 0.5
        if label == "conspiracy":
            p_con = conf
            p_non = 1.0 - conf
        else:
            p_con = 1.0 - conf
            p_non = conf

    if p_con is None:
        if p_non is None:
            p_con = 0.5
            p_non = 0.5
        else:
            p_con = max(0.0, min(1.0, 1.0 - p_non))
    if p_non is None:
        p_non = max(0.0, min(1.0, 1.0 - p_con))

    total = p_con + p_non
    if total > 0:
        p_con /= total
        p_non /= total
    else:
        p_con = p_non = 0.5

    if label not in ("conspiracy", "non"):
        label = "conspiracy" if p_con >= p_non else "non"

    return {"label": label, "p_conspiracy": p_con, "p_non": p_non}


def run_s2(
    doc, sys_blocks, user_block, model_id, region, max_tokens, temperature, n_samples
):
    preds = []
    for _ in range(n_samples):
        chat = Chat(
            model_id=model_id,
            region=region,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        for s in sys_blocks:
            chat.add_system(s)
        chat.add_user(user_block)
        try:
            out = chat.generate()
            js = strict_json_extract(out) or {
                "label": "non",
                "rationale": "repair",
                "confidence": 0.51,
            }
        except Exception as e:
            logging.warning(f"[S2] generate() failed -> defaulting to non: {e}")
            js = {"label": "non", "rationale": "fallback", "confidence": 0.55}
        preds.append(_coerce_probabilities(js))

    avg_p_con = sum(p["p_conspiracy"] for p in preds) / len(preds)
    avg_p_non = sum(p["p_non"] for p in preds) / len(preds)
    total = avg_p_con + avg_p_non
    if total > 0:
        avg_p_con /= total
        avg_p_non /= total
    final_label = "conspiracy" if avg_p_con >= avg_p_non else "non"
    return final_label, avg_p_con, avg_p_non


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    # Use dev_rehydrated.jsonl for both S1 and S2 by default
    ap.add_argument("--test-file-s1", required=False, default="dev_rehydrated.jsonl")
    ap.add_argument("--test-file-s2", required=False, default="dev_rehydrated.jsonl")
    ap.add_argument("--eda-root", required=False, default=None)
    ap.add_argument(
        "--techniques",
        default=ALL_TECHNIQUES,
        help="Comma list applied jointly. Examples: zs,fs_boundary_policy,sc5,sc10",
    )
    ap.add_argument(
        "--save-prompts",
        choices=["none", "sample", "all"],
        default="sample",
        help="Save prompts to runs/<out>/prompts. 'sample' saves first 3 docs per task+tech.",
    )
    ap.add_argument(
        "--print-prompts-preview",
        action="store_true",
        default=True,
        help="Print a short preview (first 500 chars) of the final system+user prompts once per technique.",
    )
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--region", default=None)
    ap.add_argument("--max-tokens-s1", type=int, default=1200)
    ap.add_argument("--max-tokens-s2", type=int, default=900)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--sc-temperature", type=float, default=0.7)
    ap.add_argument("--s1-iou", type=float, default=0.5)
    ap.add_argument("--out-root", default="runs/joint_llm")
    ap.add_argument(
        "--pp-merge-gap",
        type=int,
        default=1,
        help="Merge same-label spans separated by <= gap chars.",
    )
    ap.add_argument(
        "--pp-dedup-iou",
        type=float,
        default=0.90,
        help="De-dup same-label spans with IoU>=thr (keep prior-closer).",
    )
    ap.add_argument(
        "--pp-conflict-iou",
        type=float,
        default=0.50,
        help="NMS IoU for conflict pairs (Actor/Victim, Action/Effect).",
    )
    ap.add_argument(
        "--max-markers-per-label",
        type=int,
        default=3,
        help="Limit markers passed to S2 to control prompt length.",
    )
    ap.add_argument(
        "--limit-docs",
        type=int,
        default=None,
        help="Process only the first N documents for each of S1/S2 (for quicker prompt sweeps).",
    )
    ap.add_argument(
        "--s2-thresh",
        default="auto",
        help='Decision threshold for S2. Use "auto" to tune on dev probs, or a float like 0.45.',
    )
    args = ap.parse_args()

    print("=== JOINT PROMPT SWEEP START ===")
    # Print model info
    print(f"Model ID: {args.model_id}, Region: {args.region}")

    random.seed(42)

    # EDA
    prompt_arts = {}
    s1_policy = s2_policy = ""
    s1_shots = []
    s2_shots = []
    boundary = ""
    if args.eda_root:
        eda = pathlib.Path(args.eda_root)
        print(f"Loading EDA artifacts from: {eda}")
        s1_policy = build_s1_policy(eda) or ""
        s2_policy = build_s2_policy(eda) or ""
        s1_shots = load_fewshots(eda, "s1", max_n=6) or []
        s2_shots = load_fewshots(eda, "s2", max_n=8) or []
        bctx = eda / "boundary_context.json"
        if bctx.exists():
            try:
                boundary = json.loads(bctx.read_text(encoding="utf-8")).get("note", "")
            except Exception:
                boundary = ""
        # --- NEW: load prompt artifacts & provide few-shot fallback ---
        prompt_arts = _load_prompt_artifacts(eda)
        if not s1_shots:
            s1_shots = (prompt_arts.get("fewshot_bank") or {}).get("s1", [])[:6]
        if not s2_shots:
            s2_shots = (prompt_arts.get("fewshot_bank") or {}).get("s2", [])[:8]

    # Data
    rows_s1 = list(read_jsonl(args.test_file_s1))
    rows_s2 = list(read_jsonl(args.test_file_s2))
    if args.limit_docs is not None:
        rows_s1 = rows_s1[: args.limit_docs]
        rows_s2 = rows_s2[: args.limit_docs]
        logging.info(
            f"Limiting to first {args.limit_docs} docs for S1 ({len(rows_s1)}) and S2 ({len(rows_s2)})"
        )
    id2doc_s2 = {(r.get("_id") or r.get("doc_id")): r for r in rows_s2}

    techniques = [t.strip() for t in args.techniques.split(",") if t.strip()]
    for tech in techniques:
        has_fs = "fs" in tech
        has_neg = "neg" in tech
        use_cot = "cot" in tech
        use_boundary = "boundary" in tech
        use_policy = "policy" in tech
        sc_match = re.search(r"sc(\d+)", tech)
        sc_n = int(sc_match.group(1)) if sc_match else 1

    if has_fs:
        s1_shots = load_fewshots(eda, "s1", max_n=6)
        s2_shots = load_fewshots(eda, "s2", max_n=8)
        if has_neg:
            s1_neg = [
                ex
                for ex in prompt_arts.get("fewshot_bank", {}).get("s1", [])
                if ex.get("markers") == []
            ][:1]
            s2_neg = [
                ex
                for ex in prompt_arts.get("fewshot_bank", {}).get("s2", [])
                if ex.get("gold", {}).get("label") == "non"
            ][:1]
            s1_shots += s1_neg
            s2_shots += s2_neg

    for tech in techniques:
        print(f"\n=== JOINT S1→S2 :: {tech} ===")

        tech_dir = pathlib.Path(args.out_root) / tech
        (tech_dir / "s1").mkdir(parents=True, exist_ok=True)
        (tech_dir / "s2").mkdir(parents=True, exist_ok=True)
        (tech_dir / "prompts").mkdir(parents=True, exist_ok=True)
        # --- NEW: snapshot artifacts into the run folder ---
        if prompt_arts:
            _snapshot_artifacts(prompt_arts, tech_dir)
        s1_sub = tech_dir / "s1" / "submission.jsonl"
        s1_pruned_sub = tech_dir / "s1" / "submission_pruned.jsonl"
        s2_sub = tech_dir / "s2" / "submission.jsonl"
        # If limiting docs, create a ground-truth subset for fair S1 eval
        gt_subset_path = None
        if args.limit_docs is not None:
            gt_subset_path = tech_dir / "s1" / "gt_subset.jsonl"
            write_jsonl(gt_subset_path, rows_s1)  # rows_s1 already limited

        # Save prompt metadata once per technique
        _save_prompt_meta(
            tech_dir,
            {
                "tech": tech,
                "model_id": args.model_id,
                "region": args.region,
                "max_tokens_s1": args.max_tokens_s1,
                "max_tokens_s2": args.max_tokens_s2,
                "temperature": args.temperature,
                "sc_temperature": args.sc_temperature,
                "s1_policy_used": bool(s1_policy),
                "s2_policy_used": bool(s2_policy),
                "boundary_note_used": bool(boundary) and ("boundary" in tech),
                "fewshots_s1_count": len(s1_shots),
                "fewshots_s2_count": len(s2_shots),
                "eda_root": str(args.eda_root) if args.eda_root else None,
                "s1_iou_threshold": args.s1_iou,
                # --- NEW: one-line summary of artifacts loaded ---
                "artifact_summary": (
                    "arts: boundary={b} conflicts={c} priors={p} lexicons={x} fewshot_bank(s1={s1},s2={s2})"
                ).format(
                    b=int(bool((prompt_arts or {}).get("boundary_prompts"))),
                    c=len(
                        ((prompt_arts or {}).get("conflicts") or {}).get("pairs", [])
                    ),
                    p=len((prompt_arts or {}).get("priors_prompt") or {}),
                    x=int(bool((prompt_arts or {}).get("lexicons"))),
                    s1=len(
                        ((prompt_arts or {}).get("fewshot_bank") or {}).get("s1", [])
                    ),
                    s2=len(
                        ((prompt_arts or {}).get("fewshot_bank") or {}).get("s2", [])
                    ),
                ),
            },
        )

        # For console preview, show the first built prompts once per technique
        printed_s1_preview = False
        printed_s2_preview = False

        # ------ S1 inference ------
        s1_out_rows = []
        s1_pruned_rows = []
        id2markers = {}  # for S2 conditioning
        total_raw, total_valid, total_pruned = 0, 0, 0
        saved_s1 = 0
        for rec in rows_s1:
            _id = rec.get("_id") or rec.get("doc_id")
            txt = rec.get("text", "")

            # ensure fs and zs are mutually exclusive
            use_fs = "fs" in tech
            use_zs = "zs" in tech
            if use_fs and not use_zs and s1_shots:
                # pick a balanced subset per prompt, e.g., 6
                fewshots_s1 = _pick_balanced_s1_fewshots(
                    s1_shots, k=min(6, len(s1_shots))
                )
            else:
                fewshots_s1 = []

            use_fs = "fs" in tech and "zs" not in tech
            fewshots_s1 = s1_shots if (use_fs and s1_shots) else []
            if use_fs and not s1_shots:
                logging.warning(
                    "[S1] fs requested but no fewshots loaded; falling back to zero-shot."
                )

            sys_blocks, user_block, n_samples, temp = s1_prompt(
                txt,
                policy=s1_policy if use_policy else "",
                fewshots=fewshots_s1,
                boundary_note=boundary if use_boundary else "",
                tech=tech,
                prompt_arts=prompt_arts if prompt_arts else None,
            )
            temp = args.sc_temperature if n_samples > 1 else args.temperature
            # Save/print prompts per policy
            if args.save_prompts == "all" or (
                args.save_prompts == "sample" and saved_s1 < 3
            ):
                _save_prompt_bundle(tech_dir, "s1", str(_id), sys_blocks, user_block)
                saved_s1 += 1
            if args.print_prompts_preview and not printed_s1_preview:
                sys_preview = "\\n\\n---\\n\\n".join(sys_blocks)
                user_preview = user_block
                print(
                    "[S1 prompt preview]\nSYSTEM:\n"
                    + sys_preview
                    + "\n\nUSER:\n"
                    + user_preview
                )
                printed_s1_preview = True

            spans = run_s1(
                rec,
                sys_blocks,
                user_block,
                args.model_id,
                args.region,
                args.max_tokens_s1,
                temp,
                n_samples,
            )
            total_raw += len(spans)
            if not spans:
                id2markers[_id] = []
                empty_markers = []
                s1_out_rows.append({"_id": _id, "markers": empty_markers})
                s1_pruned_rows.append({"_id": _id, "markers": empty_markers})
                continue

            # Post-process with evaluator-aligned token snap + cleanup + conflict NMS
            markers = postprocess_s1_spans(
                text=txt,
                spans=spans,  # raw model output (label/start/end)
                priors=(
                    json.loads(
                        (eda / "length_position_priors.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    if args.eda_root and (eda / "length_position_priors.json").exists()
                    else {}
                ),
                merge_gap=args.pp_merge_gap,
                dedup_iou=args.pp_dedup_iou,
                conflict_iou=args.pp_conflict_iou,
            )
            s1_out_rows.append({"_id": _id, "markers": markers})

            # limit per label for S2 prompt brevity
            by_lab = defaultdict(list)
            for m in markers:
                by_lab[m["type"]].append(m)

            total_valid += len(markers)
            pruned = []
            k = args.max_markers_per_label  # always Namespace here

            def _start_end(m):
                # accept either style; default to 0 if missing to avoid crashes
                s = m.get("start", m.get("startIndex", 0))
                e = m.get("end", m.get("endIndex", s))
                return int(s), int(e)

            lex = (prompt_arts or {}).get("lexicons", {})
            for lab, arr in by_lab.items():
                # sort by (start, end) and prefer longer spans first within same start
                arr = [m for m in arr if _valid_span(txt, *_start_end(m))]
                arr_sorted = sorted(arr, key=lambda m: _salience(m, txt, lex))
                pruned.extend(arr_sorted[:k])

            # store pruned markers using the *Bedrock/S2 prompt* schema (startIndex/endIndex)
            def _to_s2_marker(m):
                s, e = _start_end(m)
                return {
                    "type": m.get("type"),
                    "startIndex": s,
                    "endIndex": e,
                    "text": txt[s:e],
                }

            id2markers[_id] = [_to_s2_marker(m) for m in pruned]
            total_pruned += len(id2markers[_id])
            s1_pruned_rows.append({"_id": _id, "markers": id2markers[_id]})

        write_jsonl(s1_sub, s1_out_rows)
        write_jsonl(s1_pruned_sub, s1_pruned_rows)
        print(f"S1 done -> {s1_sub}")
        print(f"S1 pruned-for-S2 -> {s1_pruned_sub}")
        print(
            f"S1 debug: spans raw/valid/pruned = {total_raw}/{total_valid}/{total_pruned}"
        )

        # ------ S2 inference (conditioned on S1) ------
        s2_out_rows = []
        s2_prob_rows = []
        saved_s2 = 0
        for _id, doc2 in id2doc_s2.items():
            txt = doc2.get("text", "")
            # use S1 markers if present for the same doc_id
            mks = id2markers.get(_id, [])
            markers_json = json.dumps(mks, ensure_ascii=False)

            if use_fs and not use_zs and s2_shots:
                fewshots_s2 = _pick_balanced_s2_fewshots(
                    s2_shots, k=min(8, len(s2_shots))
                )
            else:
                fewshots_s2 = []

            sys_blocks, user_block, n_samples, temp = s2_prompt_with_markers(
                txt,
                policy=s2_policy if use_policy else "",
                fewshots=fewshots_s2,
                tech=tech,
                markers_json=markers_json,
                prompt_arts=prompt_arts if prompt_arts else None,
            )
            temp = args.sc_temperature if n_samples > 1 else args.temperature
            if args.save_prompts == "all" or (
                args.save_prompts == "sample" and saved_s2 < 3
            ):
                _save_prompt_bundle(tech_dir, "s2", str(_id), sys_blocks, user_block)
                saved_s2 += 1
            if args.print_prompts_preview and not printed_s2_preview:
                sys_preview = "\\n\\n---\\n\\n".join(sys_blocks)
                user_preview = user_block
                print(
                    "[S2 prompt preview]\nSYSTEM:\n"
                    + sys_preview
                    + "\n\nUSER:\n"
                    + user_preview
                )
                printed_s2_preview = True

            lbl, p_con, p_non = run_s2(
                doc2,
                sys_blocks,
                user_block,
                args.model_id,
                args.region,
                args.max_tokens_s2,
                temp,
                n_samples,
            )
            pred = "Yes" if lbl == "conspiracy" else "No"
            s2_out_rows.append({"_id": _id, "conspiracy": pred})
            s2_prob_rows.append(
                {
                    "_id": _id,
                    "label": lbl,
                    "p_conspiracy": round(p_con, 6),
                    "p_non": round(p_non, 6),
                }
            )

        write_jsonl(s2_sub, s2_out_rows)
        probs_path = tech_dir / "s2" / "probs.jsonl"
        write_jsonl(probs_path, s2_prob_rows)
        mean_p = None
        if s2_prob_rows:
            mean_p = sum(r["p_conspiracy"] for r in s2_prob_rows) / len(s2_prob_rows)
            frac_pos = sum(1 for r in s2_prob_rows if r["p_conspiracy"] >= 0.5) / len(
                s2_prob_rows
            )
            print(f"S2 prob stats: mean_p={mean_p:.3f} frac_p>=0.5={frac_pos:.3f}")

        # --- NEW: threshold from dev probs (auto) ---
        # --- threshold selection ---
        thr = 0.50
        if isinstance(args.s2_thresh, str) and args.s2_thresh.lower() == "auto":
            if s2_prob_rows:  # only tune if we actually have probs
                best_t, best_f1, stats = tune_threshold_dev(rows_s2, s2_prob_rows)
                thr = best_t
                print(
                    f"[S2] auto threshold tuned on dev: t={best_t:.2f} (dev f1={best_f1:.3f}, mean_p={stats['mean_p']:.3f}, n={stats['n']})"
                )
            else:
                logging.warning("[S2] no probability rows; falling back to 0.50")
        else:
            try:
                thr = float(args.s2_thresh)
            except Exception:
                logging.warning(
                    f"[S2] invalid --s2-thresh={args.s2_thresh}; using 0.50"
                )
                thr = 0.50

        # Rebuild submission using chosen threshold
        pred2 = [
            {
                "_id": r["_id"],
                "conspiracy": ("Yes" if r["p_conspiracy"] >= thr else "No"),
            }
            for r in s2_prob_rows
        ]
        write_jsonl(s2_sub, pred2)

        if mean_p is not None and mean_p < 0.15:
            logging.warning(
                "[S2] mean_p extremely low; likely 'all-No' drift. Check few-shots and markers."
            )

        print(f"S2 done -> {s2_sub}")

        # ---- NEW: also emit top-level Codabench files from this technique ----
        # S1 top-level file (strip 'text' from markers)
        codabench_s1 = [
            {"_id": r["_id"], "markers": _to_codabench_s1(r.get("markers", []))}
            for r in s1_out_rows
        ]
        write_jsonl("submission_s1.jsonl", codabench_s1)
        # S2 top-level file (final thresholded labels)
        write_jsonl("submission_s2.jsonl", pred2)
        print("Wrote top-level submissions: submission_s1.jsonl, submission_s2.jsonl")


if __name__ == "__main__":
    main()
