# starter/prompt_sweep_joint.py
import argparse
import json
import pathlib
import random
import re
import math  # <-- add this
import zipfile
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
from src.psycomark.llm.bedrock_chat import BedrockChat
from src.psycomark.llm.eda_support import (
    build_s1_policy,
    build_s2_policy,
    load_fewshots,
)
from prompt_builder import (
    load_json,
    build_s1_system,
    build_s1_user,
    build_s2_system,
    build_s2_user,
    extract_answer_json,
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


import re

_PURPOSE_PREFIXES = ("to ", "in order to ", "so that", "for ")


def _starts_purpose(text_slice: str) -> bool:
    s = text_slice.lstrip().lower()
    return any(s.startswith(p) for p in _PURPOSE_PREFIXES)


def _looks_like_citation(evidence_slice: str) -> bool:
    s = evidence_slice.lower()
    return (
        ("http" in s)
        or ("www." in s)
        or ("according to" in s)
        or ('"' in evidence_slice)
        or ("’" in evidence_slice)
        or ("‘" in evidence_slice)
        or ("“" in evidence_slice)
        or ("”" in evidence_slice)
    )


def _overlap_len(a, b) -> int:
    return max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def _trim_overlap_action_effect(a, b, text, max_overlap=2):
    """
    Try to keep both Action and Effect by trimming overlap to ≤ max_overlap.
    Mutates spans in place. Returns True if trimmed, else False.
    Assumes a and b are overlapping and labels are Action/Effect in any order.
    """
    # normalize order: act = Action, eff = Effect
    if a["label"] == "Action" and b["label"] == "Effect":
        act, eff = a, b
    elif a["label"] == "Effect" and b["label"] == "Action":
        act, eff = b, a
    else:
        return False

    ov = _overlap_len(act, eff)
    if ov <= max_overlap:
        return True  # already fine

    # Heuristic: if Effect has purpose prefix, prefer trimming Action to end at Effect.start
    eff_text = text[eff["start"] : eff["end"]]
    act_text = text[act["start"] : act["end"]]
    eff_has_purpose = _starts_purpose(eff_text)
    act_has_purpose = _starts_purpose(act_text)

    if eff_has_purpose and not act_has_purpose:
        # trim Action to end at (eff.start + max_overlap)
        new_end = min(act["end"], eff["start"] + max_overlap)
        if new_end > act["start"]:
            act["end"] = new_end
            return True

    # Else try trimming Effect to start at (act.end - max_overlap)
    new_start = max(eff["start"], act["end"] - max_overlap)
    if new_start < eff["end"]:
        eff["start"] = new_start
        return True

    return False


def _role_true_cmp(a, b, priors, text_len, text):
    """
    Return -1 if a preferred, +1 if b preferred, 0 if tie.
    Generic tie-breaker: prior_dist → longer → earlier.
    """
    da = _prior_dist(priors, a["label"], a["start"], a["end"], text_len)
    db = _prior_dist(priors, b["label"], b["start"], b["end"], text_len)
    if da != db:
        return -1 if da < db else 1
    la = a["end"] - a["start"]
    lb = b["end"] - b["start"]
    if la != lb:
        return -1 if la > lb else 1
    if a["start"] != b["start"]:
        return -1 if a["start"] < b["start"] else 1
    return 0


def _conflict_nms(spans, *, text, iou_thr=0.50, text_len=1, priors=None):
    """
    Suppress/trim overlaps across conflict pairs using role-aware rules.
    - Special handling for Action–Effect (purpose clause).
    - Actor–Victim must not overlap (keep minimal mention).
    - Evidence may overlap others if citation/quote/link.
    """
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

            # If no spatial overlap, skip fast
            if _overlap_len(a, b) == 0:
                continue

            # Evidence exceptions (allow overlap if looks like citation/quote/link)
            if "Evidence" in pair:
                ev = a if a["label"] == "Evidence" else b
                ev_text = text[ev["start"] : ev["end"]]
                if _looks_like_citation(ev_text):
                    # allow overlap; no suppression
                    continue
                # else fall through to standard resolution below

            # Action–Effect: try to keep both by trimming to ≤ 2 chars overlap
            if set(pair) == {"Action", "Effect"}:
                # Prefer role-true shapes: Effect should start with purpose introducer and be longer.
                # If both obviously role-true, attempt trim; else pick role-strong one.
                # Check role signals
                a_text = text[a["start"] : a["end"]]
                b_text = text[b["start"] : b["end"]]
                a_purpose = _starts_purpose(a_text)
                b_purpose = _starts_purpose(b_text)

                # If both valid roles (Action shortish; Effect has purpose)
                action_len = (
                    (a["end"] - a["start"])
                    if a["label"] == "Action"
                    else (b["end"] - b["start"])
                )
                effect_has_purpose = b_purpose if b["label"] == "Effect" else a_purpose
                if effect_has_purpose and action_len <= 16:
                    if _trim_overlap_action_effect(a, b, text, max_overlap=2):
                        continue  # both kept after trim

                # Otherwise choose role-true:
                # - Prefer the span that exhibits the expected role signal
                #   (Effect with purpose-prefix beats Action that contains 'to ...'; Action shorter beats long Action)
                if (
                    a["label"] == "Effect"
                    and _starts_purpose(a_text)
                    and not _starts_purpose(b_text)
                ):
                    # prefer a (Effect), drop b
                    keep[j] = False
                    continue
                if (
                    b["label"] == "Effect"
                    and _starts_purpose(b_text)
                    and not _starts_purpose(a_text)
                ):
                    keep[i] = False
                    break

                # Fallback to prior/length/earlier
                cmp = _role_true_cmp(a, b, priors, text_len, text)
                if cmp <= 0:
                    keep[j] = False
                else:
                    keep[i] = False
                    break
                continue  # next j

            # Actor–Victim: must not overlap; keep the smallest (role-true minimal mention)
            if set(pair) == {"Actor", "Victim"}:
                len_a = a["end"] - a["start"]
                len_b = b["end"] - b["start"]
                if len_a != len_b:
                    if len_a <= len_b:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
                else:
                    # tie → earlier
                    if a["start"] <= b["start"]:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
                continue

            # Generic conflict: IoU threshold → prior distance → longer → earlier
            if _iou(a, b) >= iou_thr:
                cmp = _role_true_cmp(a, b, priors, text_len, text)
                if cmp <= 0:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    # filter kept
    out = [s for k, s in zip(keep, spans) if k]

    # Final polish for Action–Effect pairs we kept: enforce ≤2 chars overlap
    # (In case they slipped through generic branch without trimming.)
    changed = True
    while changed:
        changed = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                ai, bj = out[i], out[j]
                if set((ai["label"], bj["label"])) == {"Action", "Effect"}:
                    if _overlap_len(ai, bj) > 2:
                        if _trim_overlap_action_effect(ai, bj, text, max_overlap=2):
                            changed = True
    return out


def _looks_like_evidence(txt: str) -> bool:
    t = txt.strip()
    has_url = re.search(r"https?://|www\.", t, re.I)
    has_domain = re.search(r"\b[A-Za-z0-9-]+\.(com|org|net|gov|edu)\b", t)
    has_quote = '"' in t or "“" in t or "”" in t or "'" in t
    has_attrb = re.search(
        r"\b(according to|said|reported|per|per\s+\w+|study|report)\b", t, re.I
    )
    has_number = re.search(
        r"\b\d[\d,]*(\.\d+)?\s*(%|million|billion|cases|people|USD|dollars|€)\b", t
    )
    return bool(
        has_url or has_domain or (has_quote and has_attrb) or (has_number and has_attrb)
    )


def _span_quality_gate(s: dict, full_text: str) -> bool:
    lab = (s.get("label") or s.get("type") or "").lower()
    a = s.get("start", s.get("startIndex"))
    b = s.get("end", s.get("endIndex"))
    if not isinstance(a, int) or not isinstance(b, int) or b <= a:
        return False
    t = full_text[a:b].strip()
    # Drop tiny/noisy spans
    if len(t) < 3:
        return False
    if len(t.split()) == 1 and re.fullmatch(
        r"(to|of|and|the|you|them|it|he|she|they|we|us)", t, re.I
    ):
        return False
    # Evidence gate
    if lab == "evidence" and not _looks_like_evidence(t):
        return False
    # Action must contain a verb character
    if lab == "action" and not re.search(
        r"\b\w+(ed|ing|s)\b|\b(be|do|have|go|make|take|give|get|use|say)\b", t, re.I
    ):
        return False
    # Actor/Victim should be nouny (avoid starting with auxiliaries)
    if lab in ("actor", "victim") and re.match(
        r"^(is|are|was|were|be|do|did|does|to)\b", t, re.I
    ):
        return False
    return True


_ALLOWED = {"Actor", "Action", "Effect", "Victim", "Evidence"}

_VERBISH = re.compile(
    r"\b(\w+ed|\w+ing|\w+s|be|is|are|was|were|do|does|did|have|has|had|make|take|give|get|use|say|go)\b",
    re.I,
)
_AUX_START = re.compile(r"^(is|are|was|were|be|do|did|does|to)\b", re.I)
_PURPOSE = re.compile(r"\b(to|in order to|so that)\b", re.I)
_URL_OR_DOMAIN = re.compile(
    r"(https?://|www\.)|([A-Za-z0-9-]+\.(com|org|net|gov|edu))", re.I
)
_ATTRIB = re.compile(
    r"\b(according to|said|reported|per|study|report|Reuters|AP|NYT|CDC|WHO)\b", re.I
)
_NUM_UNIT = re.compile(
    r"\b\d[\d,]*(\.\d+)?\s*(%|k|m|million|billion|cases|people|USD|dollars|€)\b", re.I
)


def _evidence_gate(txt: str) -> bool:
    t = (txt or "").strip()
    if not t:
        return False
    has_url = bool(_URL_OR_DOMAIN.search(t))
    has_quote = any(q in t for q in ['"', "“", "”", "'"])
    has_attr = bool(_ATTRIB.search(t))
    has_num = bool(_NUM_UNIT.search(t))
    # Need: URL/domain OR (quote & attribution) OR (numbers & attribution)
    return has_url or (has_quote and has_attr) or (has_num and has_attr)


def _quality_gate(m: dict, full_text: str, priors: dict | None) -> bool:
    lab = (m.get("label") or m.get("type") or "").strip()
    if lab not in _ALLOWED:
        return False
    s = m.get("start")
    e = m.get("end")
    if not isinstance(s, int) or not isinstance(e, int) or e <= s:
        return False
    t = full_text[s:e].strip()
    if len(t) < 3:
        return False
    if len(t.split()) == 1 and t.lower() in {
        "to",
        "of",
        "and",
        "the",
        "you",
        "them",
        "it",
        "he",
        "she",
        "they",
        "we",
        "us",
    }:
        return False
    # Label-specific checks
    if lab == "Evidence" and not _evidence_gate(t):
        return False
    if lab == "Action" and not _VERBISH.search(t):
        return False
    if lab in {"Actor", "Victim"} and _AUX_START.match(t):
        return False
    # Optional upper-bound using priors (cap at ~1.2*q90_len)
    if priors:
        pr = (priors.get("s1_priors") or priors).get(lab) or {}
        q90 = pr.get("q90_len")
        if isinstance(q90, int) and (e - s) > int(1.2 * max(8, q90)):
            # allow long Evidence (quotes/URLs) but clip others
            if lab != "Evidence":
                return False
    return True


def _split_action_effect(m: dict, full_text: str) -> list[dict]:
    """If Action contains a clear purpose clause, split tail to Effect."""
    if (m.get("label") or m.get("type")) != "Action":
        return [m]
    s, e = m["start"], m["end"]
    seg = full_text[s:e]
    mobj = _PURPOSE.search(seg)
    if not mobj:
        return [m]
    cut = s + mobj.start()
    # left = minimal Action head
    left = dict(m)
    left["end"] = cut
    # right = Effect tail (skip the conjunction token)
    right = dict(m)
    right["label"] = right["type"] = "Effect"
    right["start"] = cut + len(mobj.group(0))
    # trim space
    while right["start"] < e and full_text[right["start"]] == " ":
        right["start"] += 1
    if left["end"] - left["start"] < 3:  # too tiny
        return [m]
    if right["end"] - right["start"] < 3:
        return [left]
    return [left, right]


import re

# (keep your existing helpers/imports)

# --- 3) Heuristic split of Action→Effect on purpose markers ---
_PURPOSE = re.compile(r"\b(to|in order to|so that)\b", re.I)


def _split_action_effect(span: dict, full_text: str) -> list[dict]:
    """
    If an Action span contains a clear purpose/result clause ('to / in order to / so that'),
    split the tail into an Effect. Keeps minimal verb head as Action.
    """
    lab = (span.get("label") or span.get("type") or "").strip()
    if lab != "Action":
        return [span]
    s = span.get("start", span.get("startIndex"))
    e = span.get("end", span.get("endIndex"))
    if not isinstance(s, int) or not isinstance(e, int) or e <= s:
        return [span]
    seg = full_text[s:e]
    m = _PURPOSE.search(seg)
    if not m:
        return [span]
    cut = s + m.start()
    # Left: minimal Action head
    left = dict(span)
    left["label"] = left.get("label", span.get("type", "Action"))
    left["type"] = "Action"
    left["start"] = left.get("start", left.get("startIndex", s))
    left["end"] = left["endIndex"] = cut
    # Right: Effect tail (skip the marker token)
    right = dict(span)
    right["label"] = right["type"] = "Effect"
    right["start"] = right["startIndex"] = cut + len(m.group(0))
    while right["start"] < e and full_text[right["start"]] == " ":
        right["start"] += 1
    right["end"] = right["endIndex"] = e
    # Guard tiny fragments
    if (left["end"] - left["start"]) < 3:
        return [span]
    if (right["end"] - right["start"]) < 3:
        return [left]
    return [left, right]


def _prefer_role_true_minimal(a: dict, b: dict, full_text: str) -> str:
    """
    Return the label to keep (either a['label'] or b['label']) when two spans conflict.
    - If identical surface and labels are Victim/Effect: prefer Victim if 'harmed-group' words present; else Effect.
    - Else prefer the shorter span (more 'minimal').
    """
    ta = full_text[a["start"] : a["end"]]
    tb = full_text[b["start"] : b["end"]]
    if ta == tb and {a["label"], b["label"]} == {"Victim", "Effect"}:
        if re.search(
            r"\b(people|citizens|workers|children|victims?|students|minorities|civilians)\b",
            ta,
            re.I,
        ):
            return "Victim"
        return "Effect"
    len_a = a["end"] - a["start"]
    len_b = b["end"] - b["start"]
    return a["label"] if len_a <= len_b else b["label"]


def postprocess_s1_spans(
    text: str,
    spans: List[dict],
    priors: dict = None,
    merge_gap: int = 1,
    dedup_iou: float = 0.90,
    conflict_iou: float = 0.50,
):
    """Main post-proc: snap -> quality gate -> split AE -> merge tiny gaps -> dedup -> conflict NMS."""
    if not text or not spans:
        return []
    toks = _tokenize_eval(text)
    L = len(text)
    # 1) normalize schema + snap
    canon = []
    for m in spans:
        lab = (m.get("label") or m.get("type") or "").strip()
        if lab not in _ALLOWED:
            continue
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

    # 1.5) quality gate (drop tiny/noisy, Evidence without source, etc.)
    canon = [m for m in canon if _quality_gate(m, text, priors)]
    if not canon:
        return []
    # 1.6) split Action→Effect on purpose markers
    _tmp = []
    for m in canon:
        _tmp.extend(_split_action_effect(m, text))
    canon = [m for m in _tmp if _quality_gate(m, text, priors)]

    # 1b) role-aware min-lengths (after snap, before merges)
    _min_len = {"Actor": 3, "Victim": 3, "Action": 3, "Effect": 7, "Evidence": 5}
    canon = [m for m in canon if (m["end"] - m["start"]) >= _min_len.get(m["label"], 3)]
    if not canon:
        return []

    # 2) merge tiny gaps (same label)
    canon = _merge_tiny_gaps(canon, gap=merge_gap)
    # 3) de-dup near duplicates (same label)
    canon = _dedup_same_label(canon, iou_thr=dedup_iou, text_len=L, priors=priors)
    # 4) conflict-aware NMS (Action/Effect, Actor/Victim)
    canon = _conflict_nms(
        spans, text=text, iou_thr=0.50, text_len=len(text), priors=priors
    )
    # 4.1) tough conflicts with identical surfaces across labels
    if len(canon) > 1:
        keep = []
        used = [False] * len(canon)
        for i in range(len(canon)):
            if used[i]:
                continue
            a = canon[i]
            chosen = i
            for j in range(i + 1, len(canon)):
                if used[j]:
                    continue
                b = canon[j]
                # strong conflict if IoU >= conflict_iou OR identical surface
                iou = _iou(a, b)
                same_surface = (
                    text[a["start"] : a["end"]] == text[b["start"] : b["end"]]
                )
                if (iou >= conflict_iou) or same_surface:
                    keep_lab = _prefer_role_true_minimal(a, b, text)
                    chosen = i if keep_lab == a["label"] else j
                    used[i] = used[j] = True
                    break
            used[chosen] = True
            keep.append(canon[chosen])
        canon = keep
    # back to submission schema + attach text slice
    canon.sort(key=lambda m: (m["start"], m["end"]))
    out = [
        {
            "type": m["label"],
            "startIndex": m["start"],
            "endIndex": m["end"],
            "text": text[m["start"] : m["end"]],
        }
        for m in canon
    ]
    return out


# ---------- default prompts ----------
S1_BASE = """You are a careful annotator for PsyCoMark (SemEval-2026 Task 10, Subtask 1).
Task: extract character spans for the labels: Actor, Action, Effect, Victim, Evidence.

DEFINITIONS (disambiguation-first):
- Actor = the agent (person/group/institution) portrayed as initiating, planning, or controlling events. Includes plural/collectives (e.g., "the deep state", "they").
- Action = the deliberate action expressed as a VERB PHRASE (head verb + essential arguments), e.g., "air dropping billions", "directing Iran". Exclude outcomes/goals/purposes.
- Effect = the consequence, intended goal, or alleged hidden intent (often a NOUN PHRASE: “agenda”, “population control”) or a purpose clause (e.g., “to set up their deep state agenda”). If a clause signals purpose (to/so that/in order to), label that clause as Effect.
- Victim = the harmed/targeted entity (people/groups/institutions/public).
- Evidence = explicit cited support: links/quotes/numbers/named sources, or evidential attributions (“according to”, “leaked emails”, “the report shows…”). Exclude bare allegations or opinions.

OUTPUT (strict JSON list, no extra text):
[{"label":"Actor|Action|Effect|Victim|Evidence","start":int,"end":int}]

BOUNDARY RULES:
- Character offsets are 0-indexed; end is exclusive; spans must lie within TEXT.
- Keep spans tight: include core content words; EXCLUDE leading/trailing whitespace, quotes, and trailing punctuation.
- Determiners (“the”, “a”, “an”) only if required for disambiguation of a named group.
- Prepositions only if integral to the meaning (“set up” ≠ “set”; “in charge of”, “cover up”).

OVERLAP/CONFLICT POLICY:
- Action vs Effect: If an Action contains a purpose/result, split: label the verb phrase as Action and the purpose/result as Effect; allow overlap if unavoidable. Prefer minimal, non-redundant spans.
- Actor vs Victim: When the same entity appears in different roles, select the smallest mention specific to each role.
- Evidence can overlap any span if it is part of a quotation or citation.

SHORT REASONING (keep private): First identify Actors → Actions → Effects → Victims → Evidence. Then resolve overlaps with the policy above. Finally output only the JSON.

QUALITY PRIOR (span lengths):
- Actor: typically short NP; Action: compact VP; Effect: may be longer NP/clause; Victim: short NP; Evidence: quoted/cited fragments.

Return ONLY the JSON list; no prose, no keys other than label/start/end.
"""

S1_USER = """TASK: Extract spans for labels: Actor, Action, Effect, Victim, Evidence.
Return ONLY a strict JSON list (no prose, no keys other than label/start/end):
[{{"label":"Actor|Action|Effect|Victim|Evidence","start":int,"end":int}}]

TEXT:
{doc_text}
"""

S2_SYS = """You are a careful annotator for PsyCoMark (SemEval-2026 Task 10, Subtask 2).
Goal: decide whether the REDDIT COMMENT promotes a conspiracy narrative. You may use brief hidden reasoning, but return ONLY the final JSON.

CLASS DEFINITIONS (recall-aware, precise):
- "conspiracy": The comment asserts or clearly endorses a hidden-plot narrative: a small/elite/malevolent Actor secretly coordinating Actions toward a goal/Effect (e.g., “agenda”, “cover-up”, “population control”), often with unnamed “they”, coalition cues, or purposeful intent. Explicit or strongly implied counts.
- "non": Neutral reporting, jokes/irony, ordinary critique/debunking, or unrelated content. Fact-based questioning without hidden-agent framing is "non".

CALIBRATION (p_conspiracy + p_non = 1.0):
- Strong hidden-plot cues (secret coordination, agenda/cover-up, blaming a cabal/“they”) → p_conspiracy ≥ 0.6; explicit assertion → ≥ 0.9.
- Clear neutral/debunking or mundane content → p_non ≥ 0.8.
- Ambiguous phrasing without hidden-agent framing → lean non but avoid 0 or 1 extremes.

Use any provided DETECTED_MARKERS_JSON (Actor/Action/Effect/Victim/Evidence) as signals; absence does not force "non".

Return strict JSON ONLY:
{"label":"conspiracy|non","p_conspiracy":0.xx,"p_non":0.xx,"rationale":"<=2 sentences"}
Constraints:
- Ensure probabilities sum to 1.0 (within rounding) and label = argmax.
- Rationale: concise; DO NOT reveal chain-of-thought; mention key cues (e.g., “hidden agenda”, “secret coordination”) or “neutral critique”.
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
            lbl = "conspiracy" if lbl == "conspiracy" else "non"
            gold = (
                {
                    "label": "conspiracy",
                    "p_conspiracy": 0.8,
                    "p_non": 0.2,
                    "rationale": "asserts conspiracy.",
                }
                if lbl == "conspiracy"
                else {
                    "label": "non",
                    "p_conspiracy": 0.2,
                    "p_non": 0.8,
                    "rationale": "neutral/debunking.",
                }
            )
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
            block = "EXAMPLE:\nTEXT:\n" + shorten(txt, 800) + "\n"
            if mk_norm:
                block += (
                    "DETECTED_MARKERS_JSON:\n"
                    + json.dumps(mk_norm, ensure_ascii=False)
                    + "\n"
                )
            block += "JSON:\n" + json.dumps(gold, ensure_ascii=False)
        else:
            # Use pre-cropped, pre-remapped spans from _prepare_s1_fewshots
            spans = ex.get("spans") or []
            # keep spans compact and valid
            norm = []
            for m in spans:
                lab = (m.get("label") or "").strip()
                s = m.get("start")
                e = m.get("end")
                try:
                    s, e = int(s), int(e)
                except Exception:
                    continue
                if lab in _S1_ALLOWED and e > s and (e - s) <= 120:
                    norm.append({"label": lab, "start": s, "end": e})
            block = (
                "EXAMPLE:\nTEXT:\n"
                + shorten(txt, 800)
                + "\nJSON:\n"
                + json.dumps(norm, ensure_ascii=False)
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


# --- S1 few-shot guards (drop empties, dedupe, balance) ---

_S1_ALLOWED = {"Actor", "Action", "Effect", "Victim", "Evidence"}


def _ex_has_valid_spans_s1(ex, min_span_len=3, max_span_len=150):
    txt = (ex.get("text") or "").strip()
    if not txt:
        return False
    spans = ex.get("spans") or ex.get("markers") or []
    ok = False
    for sp in spans:
        lab = (sp.get("label") or sp.get("type") or "").strip()
        try:
            s = int(sp.get("start", sp.get("startIndex")))
            e = int(sp.get("end", sp.get("endIndex")))
        except Exception:
            continue
        if (
            lab in _S1_ALLOWED
            and e > s
            and (e - s) >= min_span_len
            and (e - s) <= max_span_len
        ):
            ok = True
            break
    return ok


def _dedup_by_text(examples):
    seen = set()
    out = []
    for ex in examples:
        t = (ex.get("text") or "").strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(ex)
    return out


_S1_ALLOWED = {"Actor", "Action", "Effect", "Victim", "Evidence"}
_CONSP_CUES = [
    "deep state",
    "hidden agenda",
    "agenda",
    "cover up",
    "cover-up",
    "plot",
    "cabal",
    "secret",
    "they",
    "rigged",
    "collude",
    "weaponiz",
    "false flag",
    "globalist",
    "conspiracy",
]


def _norm_text_key(t: str) -> str:
    # normalize for dedup: lowercase, collapse whitespace/punctuation spacing
    t = (t or "").lower()
    t = _re.sub(r"\s+", " ", t).strip()
    return t


def _ex_has_valid_spans_s1(ex, min_span_len=3, max_span_len=150):
    txt = (ex.get("text") or "").strip()
    if not txt:
        return False
    spans = ex.get("spans") or ex.get("markers") or []
    for sp in spans:
        lab = (sp.get("label") or sp.get("type") or "").strip()
        s = sp.get("start", sp.get("startIndex"))
        e = sp.get("end", sp.get("endIndex"))
        try:
            s, e = int(s), int(e)
        except Exception:
            continue
        if lab in _S1_ALLOWED and (e > s) and (min_span_len <= (e - s) <= max_span_len):
            return True
    return False


def _looks_conspiratorial(txt: str) -> bool:
    s = (txt or "").lower()
    # soft filter: keep if we hit any cue OR text mentions 2+ labels words
    if any(c in s for c in _CONSP_CUES):
        return True
    # allow “neutral” examples if they are short and clearly labeled
    return len(s) <= 220


def _dedup_by_text_and_id(examples):
    seen_keys = set()
    out = []
    for ex in examples:
        did = str(ex.get("doc_id") or ex.get("_id") or "")
        key = (did, _norm_text_key(ex.get("text") or ""))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append(ex)
    # also dedup by normalized text alone in case doc_id missing/reused
    seen_texts = set()
    final = []
    for ex in out:
        tk = _norm_text_key(ex.get("text") or "")
        if tk in seen_texts:
            continue
        seen_texts.add(tk)
        final.append(ex)
    return final


def _crop_example_text(ex, pad=120, max_spans_per_ex=4):
    """
    Create a cropped snippet around the MOST informative span (first one that passes checks),
    and remap all retained spans into the new [start,end) space.
    """
    txt = ex.get("text") or ""
    spans = ex.get("spans") or ex.get("markers") or []
    # keep only allowed labels and reasonable lengths; prefer diverse labels
    keep = []
    per_lab = {lab: 0 for lab in _S1_ALLOWED}
    for sp in spans:
        lab = (sp.get("label") or sp.get("type") or "").strip()
        s = sp.get("start", sp.get("startIndex"))
        e = sp.get("end", sp.get("endIndex"))
        try:
            s, e = int(s), int(e)
        except Exception:
            continue
        if lab in _S1_ALLOWED and e > s and (e - s) <= 120:
            if per_lab[lab] < 1:
                keep.append({"label": lab, "start": s, "end": e})
                per_lab[lab] += 1
        if len(keep) >= max_spans_per_ex:
            break
    if not keep:
        return None

    # choose a center span heuristically: prefer Action/Effect > Actor/Victim > Evidence
    order = {"Action": 0, "Effect": 1, "Actor": 2, "Victim": 3, "Evidence": 4}
    keep.sort(key=lambda m: (order.get(m["label"], 9), m["start"]))
    center = keep[0]
    L = len(txt)
    left = max(0, center["start"] - pad)
    right = min(L, center["end"] + pad)
    snippet = txt[left:right].strip()

    # remap offsets to snippet space; discard spans that fall completely outside
    remapped = []
    for m in keep:
        s, e = m["start"], m["end"]
        if e <= left or s >= right:
            continue
        ns = max(0, s - left)
        ne = max(0, e - left)
        if ne > ns:
            remapped.append({"label": m["label"], "start": ns, "end": ne})

    if not remapped or not _looks_conspiratorial(snippet):
        return None

    return {"doc_id": ex.get("doc_id"), "text": snippet, "spans": remapped}


def _balance_per_label_s1(examples, max_per_label=1, max_total=6):
    labs = ["Actor", "Action", "Effect", "Victim", "Evidence"]
    chosen, per_lab = [], {lab: 0 for lab in labs}

    def supports(ex, lab):
        for sp in ex.get("spans") or []:
            if sp.get("label") == lab:
                return True
        return False

    # ensure one per label when available
    for lab in labs:
        for ex in examples:
            if per_lab[lab] < max_per_label and supports(ex, lab):
                chosen.append(ex)
                per_lab[lab] += 1
                break

    # fill remaining by a light salience heuristic (cues + brevity)
    def salience(ex):
        t = (ex.get("text") or "").lower()
        cue_score = sum(1 for c in _CONSP_CUES if c in t)
        short_bonus = 1 if len(t) <= 220 else 0
        return -(cue_score + short_bonus)

    rest = [e for e in examples if e not in chosen]
    rest.sort(key=salience)
    for ex in rest:
        if len(chosen) >= max_total:
            break
        chosen.append(ex)
    return chosen[:max_total]


def _prepare_s1_fewshots(fewshots, max_per_label=1, max_total=6):
    if not fewshots:
        return []
    # keep examples with at least one valid span
    filt = [ex for ex in fewshots if _ex_has_valid_spans_s1(ex)]
    # dedup by doc_id and normalized text
    filt = _dedup_by_text_and_id(filt)
    # crop and remap offsets into short, focused snippets
    cropped = []
    for ex in filt:
        c = _crop_example_text(ex, pad=120, max_spans_per_ex=4)
        if c:
            cropped.append(c)
    # final dedup after crop (by normalized text)
    cropped = _dedup_by_text_and_id(cropped)
    if not cropped:
        return []
    # require on-domain or short neutral
    cropped = [ex for ex in cropped if _looks_conspiratorial(ex.get("text", ""))]
    if not cropped:
        return []
    # balance: one per label when possible, then fill
    return _balance_per_label_s1(
        cropped, max_per_label=max_per_label, max_total=max_total
    )


import re as _re


# ---------- prompt builders (tech-agnostic) ----------
def s1_prompt(
    doc_text,
    policy,
    fewshots,
    boundary_note: str,
    tech: str,
    prompt_arts: Dict[str, Any] = None,
):
    system = [S1_BASE]
    if policy:
        system.insert(0, policy)
    if "boundary" in tech and boundary_note:
        system.append("Boundary guidance:\n" + boundary_note)

    if "cot" in tech:
        system.append(
            "Checklist before labeling:\n"
            "- Identify agents (Actor), what they do (Action), consequences (Effect), victims, and any evidence.\n"
            "- Keep spans token-tight; exclude trailing punctuation/stopwords.\n"
            "- If Action and Effect overlap, choose the minimal span that best fits each role.\n"
            "- For Actor vs Victim overlaps, prefer the smaller specific mention for each role."
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

    n_samples, temp = (1, 0.0)
    if tech.startswith("sc"):
        n = _re.search(r"sc(\d+)", tech)
        n_samples = int(n.group(1)) if n else 5
        temp = 0.7

    prepend_examples = ("fs" in tech) or tech.startswith("sc")
    clean_fs = (
        _prepare_s1_fewshots(fewshots, max_per_label=1, max_total=6)
        if prepend_examples
        else []
    )
    fs_block = render_fewshots_block(clean_fs, is_s2=False) if clean_fs else ""

    user = (fs_block + "\n\n") if fs_block else ""
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
    # --- NEW: render; if nothing valid remains, drop to zero-shot gracefully ---
    fs_block = render_fewshots_block(fewshots, is_s2=True) if fewshots else ""
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
    if fs_block and ("fs" in tech or n_samples > 1):
        user_prefix = fs_block + "\n\n"

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


def _order_s1_fewshots(fs: list[dict]) -> list[dict]:
    """
    Reorder S1 few-shots as:
      1) one negative (spans==[]),
      2) one clean prototype per label (Actor,Action,Effect,Victim,Evidence),
      3) one conflict example (meta.reason starts 'ambiguous_pair'),
    with graceful fallbacks if any bucket is missing.
    Assumes examples are {"text":..., "spans":[...], "meta":{...}}.
    """
    if not fs:
        return []
    LABELS = ["Actor", "Action", "Effect", "Victim", "Evidence"]

    def _is_neg(ex):
        return isinstance(ex.get("spans"), list) and len(ex["spans"]) == 0

    def _is_conflict(ex):
        return (ex.get("meta") or {}).get("reason", "").startswith("ambiguous_pair_")

    def _labels(ex):
        return {m.get("label") for m in (ex.get("spans") or []) if isinstance(m, dict)}

    # 1) negative (first)
    neg = next((ex for ex in fs if _is_neg(ex)), None)
    # 2) prototypes: prefer meta.reason starting with 'prototype_clean', else 'per_label_top'
    protos = {}
    for lab in LABELS:
        # best candidate for this label
        cand = None
        for ex in fs:
            if lab in _labels(ex):
                r = (ex.get("meta") or {}).get("reason", "")
                if r.startswith("prototype_clean"):
                    cand = ex
                    break
        if cand is None:
            for ex in fs:
                if lab in _labels(ex):
                    r = (ex.get("meta") or {}).get("reason", "")
                    if r == "per_label_top":
                        cand = ex
                        break
        if cand:
            protos[lab] = cand
    proto_list = [protos[l] for l in LABELS if l in protos]
    # 3) one conflict if present
    conflict = next((ex for ex in fs if _is_conflict(ex)), None)
    # stitch with dedup by object id
    out, seen = [], set()

    def add(x):
        if x is None:
            return
        key = id(x)
        if key in seen:
            return
        seen.add(key)
        out.append(x)

    add(neg)
    for ex in proto_list:
        add(ex)
    add(conflict)
    # fallback: if we still have < 6, append remaining examples in original order
    for ex in fs:
        add(ex)
        if len(out) >= 8:
            break
    return out


# ---------- main ----------
# def main():
#    ap = argparse.ArgumentParser()
#    # Use dev_rehydrated.jsonl for both S1 and S2 by default
#    ap.add_argument("--test-file-s1", required=False, default="dev_rehydrated.jsonl")
#    ap.add_argument("--test-file-s2", required=False, default="dev_rehydrated.jsonl")
#    ap.add_argument("--eda-root", required=False, default=None)
#    ap.add_argument(
#        "--techniques",
#        default=ALL_TECHNIQUES,
#        help="Comma list applied jointly. Examples: zs,fs_boundary_policy,sc5,sc10",
#    )
#    ap.add_argument(
#        "--save-prompts",
#        choices=["none", "sample", "all"],
#        default="sample",
#        help="Save prompts to runs/<out>/prompts. 'sample' saves first 3 docs per task+tech.",
#    )
#    ap.add_argument(
#        "--print-prompts-preview",
#        action="store_true",
#        default=True,
#        help="Print a short preview (first 500 chars) of the final system+user prompts once per technique.",
#    )
#    ap.add_argument("--model-id", default=None)
#    ap.add_argument("--region", default=None)
#    ap.add_argument("--max-tokens-s1", type=int, default=1200)
#    ap.add_argument("--max-tokens-s2", type=int, default=900)
#    ap.add_argument("--temperature", type=float, default=0.0)
#    ap.add_argument("--sc-temperature", type=float, default=0.7)
#    ap.add_argument(
#        "--art-dir",
#        type=str,
#        default=None,
#        help="Folder with prompt artifacts (best_fewshot_examples.json, priors_prompt.json, fewshot_policy.json). If omitted, falls back to --eda-root.",
#    )
#    ap.add_argument(
#        "--s2-self-consistency",
#        type=int,
#        default=None,
#        help="Override scN from --techniques (e.g., sc5). If set, uses this value for all techniques.",
#    )
#    ap.add_argument("--s1-iou", type=float, default=0.5)
#    ap.add_argument("--out-root", default="runs/joint_llm")
#    ap.add_argument(
#        "--pp-merge-gap",
#        type=int,
#        default=1,
#        help="Merge same-label spans separated by <= gap chars.",
#    )
#    ap.add_argument(
#        "--pp-dedup-iou",
#        type=float,
#        default=0.90,
#        help="De-dup same-label spans with IoU>=thr (keep prior-closer).",
#    )
#    ap.add_argument(
#        "--pp-conflict-iou",
#        type=float,
#        default=0.50,
#        help="NMS IoU for conflict pairs (Actor/Victim, Action/Effect).",
#    )
#    ap.add_argument(
#        "--max-markers-per-label",
#        type=int,
#        default=3,
#        help="Limit markers passed to S2 to control prompt length.",
#    )
#    ap.add_argument(
#        "--limit-docs",
#        type=int,
#        default=None,
#        help="Process only the first N documents for each of S1/S2 (for quicker prompt sweeps).",
#    )
#    ap.add_argument(
#        "--s2-thresh",
#        default="auto",
#        help='Decision threshold for S2. Use "auto" to tune on dev probs, or a float like 0.45.',
#    )
#    args = ap.parse_args()
#
#    print("=== JOINT PROMPT SWEEP START ===")
#    # Print model info
#    print(f"Model ID: {args.model_id}, Region: {args.region}")
#
#    random.seed(42)
#
#    # EDA
#    prompt_arts = {}
#    s1_policy = s2_policy = ""
#    s1_shots = []
#    s2_shots = []
#    boundary = ""
#    # Prefer --art-dir if present; else fallback to --eda-root
#    art_dir = None
#    if args.eda_root:
#        eda = pathlib.Path(args.eda_root)
#        print(f"Loading EDA artifacts from: {eda}")
#        s1_policy = build_s1_policy(eda) or ""
#        s2_policy = build_s2_policy(eda) or ""
#        s1_shots = load_fewshots(eda, "s1", max_n=6) or []
#        s2_shots = load_fewshots(eda, "s2", max_n=8) or []
#        bctx = eda / "boundary_context.json"
#        if bctx.exists():
#            try:
#                boundary = json.loads(bctx.read_text(encoding="utf-8")).get("note", "")
#            except Exception:
#                boundary = ""
#        # --- NEW: load prompt artifacts & provide few-shot fallback ---
#        prompt_arts = _load_prompt_artifacts(eda)
#        if not s1_shots:
#            s1_shots = (prompt_arts.get("fewshot_bank") or {}).get("s1", [])[:6]
#        if not s2_shots:
#            s2_shots = (prompt_arts.get("fewshot_bank") or {}).get("s2", [])[:8]
#        art_dir = eda
#    if args.art_dir:
#        art_dir = pathlib.Path(args.art_dir)
#        if art_dir.exists():
#            # refresh artifacts from explicit art_dir if given
#            prompt_arts = _load_prompt_artifacts(art_dir)
#            if not s1_shots:
#                s1_shots = (prompt_arts.get("fewshot_bank") or {}).get("s1", [])[:6]
#            if not s2_shots:
#                s2_shots = (prompt_arts.get("fewshot_bank") or {}).get("s2", [])[:8]
#
#    # Data
#    rows_s1 = list(read_jsonl(args.test_file_s1))
#    rows_s2 = list(read_jsonl(args.test_file_s2))
#    if args.limit_docs is not None:
#        rows_s1 = rows_s1[: args.limit_docs]
#        rows_s2 = rows_s2[: args.limit_docs]
#        logging.info(
#            f"Limiting to first {args.limit_docs} docs for S1 ({len(rows_s1)}) and S2 ({len(rows_s2)})"
#        )
#    id2doc_s2 = {(r.get("_id") or r.get("doc_id")): r for r in rows_s2}
#
#    techniques = [t.strip() for t in args.techniques.split(",") if t.strip()]
#    # --- prompt builders (Sonnet 4.5 XML) ---
#    # Extract priors + conflict pairs from artifacts (if present)
#    priors = prompt_arts.get("priors_prompt") or {}
#    policy_json = prompt_arts.get("fewshot_policy") or {}
#    ambiguous_pairs = []
#    try:
#        cp = (policy_json.get("targets") or {}).get("ambiguous_pairs_top2") or []
#        ambiguous_pairs = [
#            tuple(sorted(p)) for p in cp if isinstance(p, (list, tuple)) and len(p) == 2
#        ]
#    except Exception:
#        ambiguous_pairs = [("Action", "Effect"), ("Actor", "Victim")]
#
#    for tech in techniques:
#        print(f"\n=== JOINT S1→S2 :: {tech} ===")
#
#        # per-tech flags (must be inside the loop)
#        has_fs = "fs" in tech
#        has_neg = "neg" in tech
#        use_cot = (
#            "cot" in tech
#        )  # kept for compatibility; builders already structure reasoning via <thinking>
#        use_boundary = "boundary" in tech
#        use_policy = "policy" in tech
#        sc_match = re.search(r"sc(\d+)", tech)
#        sc_n = int(sc_match.group(1)) if sc_match else 1
#        if isinstance(args.s2_self_consistency, int) and args.s2_self_consistency >= 1:
#            sc_n = args.s2_self_consistency
#
#        # few-shots (per-tech, with optional negatives)
#        fs_s1 = s1_shots[:] if (has_fs and s1_shots) else []
#        fs_s2 = s2_shots[:] if (has_fs and s2_shots) else []
#        if has_fs and has_neg and prompt_arts:
#            # S1 negatives: schema has "spans" (empty) for no-markers examples
#            s1_neg = [
#                ex
#                for ex in (prompt_arts.get("fewshot_bank") or {}).get("s1", [])
#                if isinstance(ex.get("spans"), list) and len(ex["spans"]) == 0
#            ][:1]
#            s2_neg = [
#                ex
#                for ex in (prompt_arts.get("fewshot_bank") or {}).get("s2", [])
#                if ex.get("gold", {}).get("label") == "non"
#            ][:1]
#            fs_s1 += s1_neg
#            fs_s2 += s2_neg
#
#        tech_dir = pathlib.Path(args.out_root) / tech
#        (tech_dir / "s1").mkdir(parents=True, exist_ok=True)
#        (tech_dir / "s2").mkdir(parents=True, exist_ok=True)
#        (tech_dir / "prompts").mkdir(parents=True, exist_ok=True)
#        # --- NEW: snapshot artifacts into the run folder ---
#        if prompt_arts:
#            _snapshot_artifacts(prompt_arts, tech_dir)
#        s1_sub = tech_dir / "s1" / "submission.jsonl"
#        s1_pruned_sub = tech_dir / "s1" / "submission_pruned.jsonl"
#        s2_sub = tech_dir / "s2" / "submission.jsonl"
#        # If limiting docs, create a ground-truth subset for fair S1 eval
#        gt_subset_path = None
#        if args.limit_docs is not None:
#            gt_subset_path = tech_dir / "s1" / "gt_subset.jsonl"
#            write_jsonl(gt_subset_path, rows_s1)  # rows_s1 already limited
#
#        # Save prompt metadata once per technique
#        _save_prompt_meta(
#            tech_dir,
#            {
#                "tech": tech,
#                "model_id": args.model_id,
#                "region": args.region,
#                "max_tokens_s1": args.max_tokens_s1,
#                "max_tokens_s2": args.max_tokens_s2,
#                "temperature": args.temperature,
#                "sc_temperature": args.sc_temperature,
#                "s1_policy_used": bool(s1_policy),
#                "s2_policy_used": bool(s2_policy),
#                "boundary_note_used": bool(boundary) and ("boundary" in tech),
#                "fewshots_s1_count": len(s1_shots),
#                "fewshots_s2_count": len(s2_shots),
#                "eda_root": str(args.eda_root) if args.eda_root else None,
#                "s1_iou_threshold": args.s1_iou,
#                # --- NEW: one-line summary of artifacts loaded ---
#                "artifact_summary": (
#                    "arts: boundary={b} conflicts={c} priors={p} lexicons={x} fewshot_bank(s1={s1},s2={s2})"
#                ).format(
#                    b=int(bool((prompt_arts or {}).get("boundary_prompts"))),
#                    c=len(
#                        ((prompt_arts or {}).get("conflicts") or {}).get("pairs", [])
#                    ),
#                    p=len((prompt_arts or {}).get("priors_prompt") or {}),
#                    x=int(bool((prompt_arts or {}).get("lexicons"))),
#                    s1=len(
#                        ((prompt_arts or {}).get("fewshot_bank") or {}).get("s1", [])
#                    ),
#                    s2=len(
#                        ((prompt_arts or {}).get("fewshot_bank") or {}).get("s2", [])
#                    ),
#                ),
#            },
#        )
#
#        # For console preview, show the first built prompts once per technique
#        printed_s1_preview = False
#        printed_s2_preview = False
#
#        # ------ S1 inference ------
#        s1_out_rows = []
#        s1_pruned_rows = []
#        id2markers = {}  # for S2 conditioning
#        total_raw, total_valid, total_pruned = 0, 0, 0
#        saved_s1 = 0
#        for rec in rows_s1:
#            _id = rec.get("_id") or rec.get("doc_id")
#            txt = rec.get("text", "")
#
#            # === New Sonnet 4.5 prompt (XML + <thinking>/<answer>) ===
#            # We do n_samples=1 for S1 to keep spans deterministic
#            fewshots_s1 = (
#                _pick_balanced_s1_fewshots(fs_s1, k=min(8, len(fs_s1))) if fs_s1 else []
#            )
#            # reorder: negative → one prototype per label → one conflict (if available)
#            fewshots_s1 = _order_s1_fewshots(fewshots_s1)
#            s1_system = build_s1_system(
#                priors=priors,
#                conflict_pairs=ambiguous_pairs,
#                boundary_note=(boundary if use_boundary else None),
#                policy_text=(s1_policy if use_policy else None),
#                include_cot=use_cot,
#            )
#            # (optional) assert sections exist for safety
#            if (
#                "<offset_scope>" not in s1_system
#                or "<forbidden_output>" not in s1_system
#            ):
#                logging.warning("[S1] system prompt missing offset/forbidden sections.")
#            s1_user = build_s1_user(
#                text_input=txt,
#                s1_fewshots=fewshots_s1,
#                include_cot=use_cot,
#            )
#            sys_blocks = [s1_system]
#            user_block = s1_user
#            n_samples = 1
#            temp = args.temperature
#            # Save/print prompts per policy
#            if args.save_prompts == "all" or (
#                args.save_prompts == "sample" and saved_s1 < 3
#            ):
#                _save_prompt_bundle(tech_dir, "s1", str(_id), sys_blocks, user_block)
#                saved_s1 += 1
#            if args.print_prompts_preview and not printed_s1_preview:
#                sys_preview = "\\n\\n---\\n\\n".join(sys_blocks)
#                user_preview = user_block
#                print(
#                    "[S1 prompt preview]\nSYSTEM:\n"
#                    + sys_preview
#                    + "\n\nUSER:\n"
#                    + user_preview
#                )
#                printed_s1_preview = True
#
#            spans = run_s1(
#                rec,
#                sys_blocks,
#                user_block,
#                args.model_id,
#                args.region,
#                args.max_tokens_s1,
#                temp,
#                n_samples,
#            )
#            total_raw += len(spans)
#            if not spans:
#                id2markers[_id] = []
#                empty_markers = []
#                s1_out_rows.append({"_id": _id, "markers": empty_markers})
#                s1_pruned_rows.append({"_id": _id, "markers": empty_markers})
#                continue
#
#            # Post-process with evaluator-aligned token snap + cleanup + conflict NMS
#            markers = postprocess_s1_spans(
#                text=txt,
#                spans=spans,  # raw model output (label/start/end)
#                priors=(
#                    json.loads(
#                        (eda / "length_position_priors.json").read_text(
#                            encoding="utf-8"
#                        )
#                    )
#                    if args.eda_root and (eda / "length_position_priors.json").exists()
#                    else {}
#                ),
#                merge_gap=args.pp_merge_gap,
#                dedup_iou=args.pp_dedup_iou,
#                conflict_iou=args.pp_conflict_iou,
#            )
#            s1_out_rows.append({"_id": _id, "markers": markers})
#
#            # limit per label for S2 prompt brevity
#            by_lab = defaultdict(list)
#            for m in markers:
#                by_lab[m["type"]].append(m)
#
#            total_valid += len(markers)
#            pruned = []
#            k = args.max_markers_per_label  # always Namespace here
#
#            def _start_end(m):
#                # accept either style; default to 0 if missing to avoid crashes
#                s = m.get("start", m.get("startIndex", 0))
#                e = m.get("end", m.get("endIndex", s))
#                return int(s), int(e)
#
#            lex = (prompt_arts or {}).get("lexicons", {})
#            for lab, arr in by_lab.items():
#                # --- NEW: filter bad Victim spans (money/objects/etc.) before salience sort ---
#                def _is_bad_victim(text_slice: str) -> bool:
#                    s = (text_slice or "").strip().lower()
#                    # numeric/money tokens or abstract objects are not victims
#                    if re.fullmatch(
#                        r"\$?\d[\d,]*(\.\d+)?(\s*(k|m|b|million|billion))?", s
#                    ):
#                        return True
#                    if s in {
#                        "bribe money",
#                        "taxes",
#                        "cash",
#                        "evidence",
#                        "agenda",
#                        "plan",
#                        "policy",
#                    }:
#                        return True
#                    return False
#
#                def _text_of(m):
#                    s, e = _start_end(m)
#                    return txt[s:e]
#
#                arr = [
#                    m
#                    for m in arr
#                    if _valid_span(txt, *_start_end(m))
#                    and not ((lab == "Victim") and _is_bad_victim(_text_of(m)))
#                ]
#                arr_sorted = sorted(arr, key=lambda m: _salience(m, txt, lex))
#                pruned.extend(arr_sorted[:k])
#
#            # store pruned markers using the *Bedrock/S2 prompt* schema (startIndex/endIndex)
#            def _to_s2_marker(m):
#                s, e = _start_end(m)
#                return {
#                    "type": m.get("type"),
#                    "startIndex": s,
#                    "endIndex": e,
#                    "text": txt[s:e],
#                }
#
#            id2markers[_id] = [_to_s2_marker(m) for m in pruned]
#            total_pruned += len(id2markers[_id])
#            s1_pruned_rows.append({"_id": _id, "markers": id2markers[_id]})
#
#        write_jsonl(s1_sub, s1_out_rows)
#        write_jsonl(s1_pruned_sub, s1_pruned_rows)
#        print(f"S1 done -> {s1_sub}")
#        print(f"S1 pruned-for-S2 -> {s1_pruned_sub}")
#        print(
#            f"S1 debug: spans raw/valid/pruned = {total_raw}/{total_valid}/{total_pruned}"
#        )
#
#        # ------ S2 inference (conditioned on S1) ------
#        s2_out_rows = []
#        s2_prob_rows = []
#        saved_s2 = 0
#        for _id, doc2 in id2doc_s2.items():
#            txt = doc2.get("text", "")
#            # use S1 markers if present for the same doc_id
#            mks = id2markers.get(_id, [])
#
#            fewshots_s2 = (
#                _pick_balanced_s2_fewshots(fs_s2, k=min(8, len(fs_s2))) if fs_s2 else []
#            )
#            s2_system = build_s2_system(
#                policy_text=(s2_policy if use_policy else None),
#                include_cot=use_cot,
#            )
#            s2_user = build_s2_user(
#                text_input=txt,
#                s1_output=mks,
#                s2_fewshots=fewshots_s2,
#                include_cot=use_cot,
#            )
#            sys_blocks = [s2_system]
#            user_block = s2_user
#            # self-consistency from scN or --s2-self-consistency
#            n_samples = max(1, int(sc_n))
#            temp = args.sc_temperature if n_samples > 1 else args.temperature
#            if args.save_prompts == "all" or (
#                args.save_prompts == "sample" and saved_s2 < 3
#            ):
#                _save_prompt_bundle(tech_dir, "s2", str(_id), sys_blocks, user_block)
#                saved_s2 += 1
#            if args.print_prompts_preview and not printed_s2_preview:
#                sys_preview = "\\n\\n---\\n\\n".join(sys_blocks)
#                user_preview = user_block
#                print(
#                    "[S2 prompt preview]\nSYSTEM:\n"
#                    + sys_preview
#                    + "\n\nUSER:\n"
#                    + user_preview
#                )
#                printed_s2_preview = True
#
#            lbl, p_con, p_non = run_s2(
#                doc2,
#                sys_blocks,
#                user_block,
#                args.model_id,
#                args.region,
#                args.max_tokens_s2,
#                temp,
#                n_samples,
#            )
#            pred = "Yes" if lbl == "conspiracy" else "No"
#            s2_out_rows.append({"_id": _id, "conspiracy": pred})
#            s2_prob_rows.append(
#                {
#                    "_id": _id,
#                    "label": lbl,
#                    "p_conspiracy": round(p_con, 6),
#                    "p_non": round(p_non, 6),
#                }
#            )
#
#        write_jsonl(s2_sub, s2_out_rows)
#        probs_path = tech_dir / "s2" / "probs.jsonl"
#        write_jsonl(probs_path, s2_prob_rows)
#        mean_p = None
#        if s2_prob_rows:
#            mean_p = sum(r["p_conspiracy"] for r in s2_prob_rows) / len(s2_prob_rows)
#            frac_pos = sum(1 for r in s2_prob_rows if r["p_conspiracy"] >= 0.5) / len(
#                s2_prob_rows
#            )
#            print(f"S2 prob stats: mean_p={mean_p:.3f} frac_p>=0.5={frac_pos:.3f}")
#
#        # --- NEW: threshold from dev probs (auto) ---
#        # --- threshold selection ---
#        thr = 0.50
#        if isinstance(args.s2_thresh, str) and args.s2_thresh.lower() == "auto":
#            if s2_prob_rows:  # only tune if we actually have probs
#                best_t, best_f1, stats = tune_threshold_dev(rows_s2, s2_prob_rows)
#                if stats.get("n", 0) <= 0:
#                    print(
#                        "[S2] No gold labels available for threshold tuning; keeping default 0.50."
#                    )
#                    thr = 0.50
#                else:
#                    thr = best_t
#                    print(
#                        f"[S2] auto threshold tuned on dev: t={best_t:.2f} (dev f1={best_f1:.3f}, mean_p={stats['mean_p']:.3f}, n={stats['n']})"
#                    )
#            else:
#                logging.warning("[S2] no probability rows; falling back to 0.50")
#        else:
#            try:
#                thr = float(args.s2_thresh)
#            except Exception:
#                logging.warning(
#                    f"[S2] invalid --s2-thresh={args.s2_thresh}; using 0.50"
#                )
#                thr = 0.50
#
#        # Rebuild submission using chosen threshold
#        pred2 = [
#            {
#                "_id": r["_id"],
#                "conspiracy": ("Yes" if r["p_conspiracy"] >= thr else "No"),
#            }
#            for r in s2_prob_rows
#        ]
#        write_jsonl(s2_sub, pred2)
#
#        if mean_p is not None and mean_p < 0.15:
#            logging.warning(
#                "[S2] mean_p extremely low; likely 'all-No' drift. Check few-shots and markers."
#            )
#
#        print(f"S2 done -> {s2_sub}")
#
#        # ---- NEW: also emit top-level Codabench files from this technique ----
#        # S1 top-level file (strip 'text' from markers)
#        codabench_s1 = [
#            {"_id": r["_id"], "markers": _to_codabench_s1(r.get("markers", []))}
#            for r in s1_out_rows
#        ]
#        write_jsonl("submission_s1.jsonl", codabench_s1)
#        # S2 top-level file (final thresholded labels)
#        write_jsonl("submission_s2.jsonl", pred2)
#        print("Wrote top-level submissions: submission_s1.jsonl, submission_s2.jsonl")
#        # --- NEW: zip each technique's submissions into ./submissions/submission_{tech}.zip ---
#        try:
#            submissions_dir = pathlib.Path("submissions")
#            submissions_dir.mkdir(parents=True, exist_ok=True)
#            zip_path = submissions_dir / f"submission_{tech}.zip"
#            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
#                # include top-level files as required by Codabench
#                if pathlib.Path("submission_s1.jsonl").exists():
#                    zf.write("submission_s1.jsonl", arcname="submission_s1.jsonl")
#                if pathlib.Path("submission_s2.jsonl").exists():
#                    zf.write("submission_s2.jsonl", arcname="submission_s2.jsonl")
#                # also include per-tech originals for traceability
#                if s1_sub.exists():
#                    zf.write(s1_sub, arcname=f"{tech}/s1/submission.jsonl")
#                if s1_pruned_sub.exists():
#                    zf.write(
#                        s1_pruned_sub, arcname=f"{tech}/s1/submission_pruned.jsonl"
#                    )
#                if s2_sub.exists():
#                    zf.write(s2_sub, arcname=f"{tech}/s2/submission.jsonl")
#                if probs_path.exists():
#                    zf.write(probs_path, arcname=f"{tech}/s2/probs.jsonl")
#            print(f"Packaged ZIP -> {zip_path}")
#        except Exception as e:
#            logging.warning(f"[ZIP] Failed to create technique ZIP: {e}")
#
#
# if __name__ == "__main__":
#    main()
#
