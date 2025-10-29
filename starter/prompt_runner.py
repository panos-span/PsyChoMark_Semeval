#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
starter/prompt_runner.py

A clean runner that calls Bedrock via Chat(add_system/add_user/generate),
reusing the big prompt/plumbing from starter/prompt_sweep_joint.py.

Features:
- EDA artifacts (priors, conflicts, boundary, fewshots) with snapshotting
- S1 XML prompts + boundary/policy injections + fewshots (incl. negatives)
- S1 span post-processing: token snap, merge gaps, dedup, conflict NMS
- S2 XML prompts with self-consistency and probability coercion
- Auto threshold tuning on dev probs
- Prompt previews/snapshots + Codabench outputs + per-tech ZIPs
"""

import argparse
import json
import logging
import pathlib
import random
import re
import sys
import zipfile
from collections import defaultdict
from tqdm.auto import tqdm
from typing import Any, Dict, List
from pathlib import Path

# --- Make repo root importable ---
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# --- Bedrock Chat (Sonnet 4.5 wrapper) ---
from src.psycomark.llm.bedrock_chat import BedrockChat

# --- Reuse the largest runner's utilities / builders ---
# (We import only what we need to avoid code duplication.)
from starter.prompt_sweep_joint import (  # noqa: E402
    _snapshot_artifacts,
    postprocess_s1_spans,
    read_jsonl,
    write_jsonl,
    _save_prompt_bundle,
    _tokenize_eval,
    _valid_span,
    _snap_to_tokens,
    _prior_dist,
    _save_prompt_meta,
    _to_codabench_s1,
    _pick_balanced_s1_fewshots,
    _pick_balanced_s2_fewshots,
    tune_threshold_dev,
)

from starter.prompt_builder import (  # noqa: E402
    build_s1_system,
    build_s1_user,
    extract_answer_json,
    build_s2_prompts_adapter,
    validate_and_repair_s1_spans,
    to_s2_marker,
    build_s1_verifier_prompts,
)


# ---------- tiny helpers you asked to include locally ----------
_LIST_RE = re.compile(r"\[.*?\]", re.S)

import pathlib as _pathlib
import os


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


def _as_text(out) -> str:
    """Coerce Bedrock/Claude output to a text string."""
    if out is None:
        return ""
    if isinstance(out, (bytes, bytearray)):
        try:
            return out.decode("utf-8", errors="ignore")
        except Exception:
            return str(out)
    if isinstance(out, dict):
        # Common shapes for Extended Thinking: {"answer": "...", "thinking": "..."}
        cand = out.get("answer") or out.get("text") or out.get("content")
        if isinstance(cand, (dict, list)):
            return json.dumps(cand, ensure_ascii=False)
        if cand is not None:
            return str(cand)
        return json.dumps(out, ensure_ascii=False)
    return str(out)


def list_json_extract(s: str) -> list:
    """
    Extract the first top-level JSON array from a string; return [] if missing/bad.
    """
    if not isinstance(s, str):
        return []
    m = _LIST_RE.search(s)
    if not m:
        return []
    try:
        val = json.loads(m.group(0))
        return val if isinstance(val, list) else []
    except Exception:
        return []


_CANON_LABEL = {
    k.lower(): k for k in ["Actor", "Action", "Effect", "Victim", "Evidence"]
}


def _norm_marker_for_print(m: dict) -> dict:
    lab = (m.get("label") or m.get("type") or "").strip()
    s = m.get("start", m.get("startIndex"))
    e = m.get("end", m.get("endIndex"))
    t = m.get("text")
    try:
        s = int(s) if s is not None else None
        e = int(e) if e is not None else None
    except Exception:
        s = e = None
    return {"label": lab, "start": s, "end": e, "text": t}


def _snippet(s: str, n: int = 180) -> str:
    s = (s or "").replace("\n", " ")
    return s[:n] + ("…" if len(s) > n else "")


# --- Normalize raw S1 items coming from the model (<answer> array) ---
def _coerce_s1_items(arr_raw) -> list[dict]:
    """
    Accepts list of possibly-messy items and returns clean list[dict].
    Valid outputs:
      - {"label": ..., "text": "...", ["start": int]}
      - {"label": ..., "start": int, "end": int}
    Anything else is dropped.
    """
    out = []
    if not isinstance(arr_raw, list):
        return out
    for it in arr_raw:
        # If the model put JSON as a string, try to parse once.
        if isinstance(it, str):
            try:
                it = json.loads(it)
            except Exception:
                continue
        if not isinstance(it, dict):
            continue
        # Canonicalize keys
        lab = (it.get("label") or it.get("type") or "").strip()
        if not lab:
            continue
        lab_c = _CANON_LABEL.get(lab.lower())
        if not lab_c:
            continue
        # (A) text-based form (preferred for later alignment)
        if isinstance(it.get("text"), str) and it["text"].strip():
            item = {"label": lab_c, "text": it["text"].strip()}
            # optional hint
            if isinstance(it.get("start"), int):
                item["start"] = it["start"]
            out.append(item)
            continue
        # (B) span-based fallback
        s = it.get("start", it.get("startIndex"))
        e = it.get("end", it.get("endIndex"))
        try:
            s, e = int(s), int(e)
        except Exception:
            continue
        if e <= s:
            continue
        out.append({"label": lab_c, "start": s, "end": e})
    return out


# --- BEGIN: few-shot adapters ---
def _adapt_bank_to_builder_s1(examples: list[dict]) -> list[dict]:
    """Your bank uses {'text','answer':[...]} but older builders expect 'spans'."""
    out = []
    for ex in examples or []:
        text = ex.get("text", "")
        ans = ex.get("answer", [])
        if isinstance(ans, dict):  # safety if someone put S2-style dict here
            ans = []
        out.append({"text": text, "spans": ans})
    return out


def _adapt_bank_to_builder_s2(examples: list[dict]) -> list[dict]:
    """Pass through; S2 builder usually reads {'text','answer':{'label','rationale'}}."""
    return examples or []


# --- END: few-shot adapters ---


def _all_occurrences(haystack: str, needle: str):
    if not needle:
        return
    i = 0
    while True:
        i = haystack.find(needle, i)
        if i == -1:
            break
        yield i
        i += 1


def align_text_span(item: dict, text: str, priors: dict | None):
    lab = (item.get("label") or "").strip()
    raw = (item.get("text") or "").strip()
    if not lab or not raw or not text:
        return None

    # exact matches for substring
    cand_starts = list(_all_occurrences(text, raw))
    if not cand_starts:
        # legacy fallback if model accidentally emitted start/end
        s = item.get("start")
        e = item.get("end")
        if isinstance(s, int) and isinstance(e, int) and e > s:
            toks = _tokenize_eval(text)
            snapped = _snap_to_tokens({"label": lab, "start": s, "end": e}, toks)
            return (
                snapped
                if snapped and _valid_span(text, snapped["start"], snapped["end"])
                else None
            )
        return None

    # pick candidate: prefer provided start hint if reasonable, else prior-closest
    hinted = item.get("start")
    if isinstance(hinted, int):
        chosen = min(cand_starts, key=lambda s: abs(s - hinted))
    else:
        L = len(text)

        def prior_cost(s0):
            return _prior_dist(priors or {}, lab, s0, s0 + len(raw), L)

        chosen = min(cand_starts, key=prior_cost)

    start, end = chosen, chosen + len(raw)
    toks = _tokenize_eval(text)
    snapped = _snap_to_tokens({"label": lab, "start": start, "end": end}, toks)
    if not snapped or not _valid_span(text, snapped["start"], snapped["end"]):
        return None
    return snapped  # {"label","start","end"}


def align_and_normalize_s1(model_items: list[dict], text: str, priors: dict | None):
    """
    Accepts items from <answer>:
      - new schema: {"label","text",["start"]}
      - legacy: {"label","start","end"} or {"type","startIndex","endIndex"}
    Returns [{"label","start","end"}], snapped to token boundaries.
    """
    if not isinstance(model_items, list):
        return []
    out = []

    # Preferred: text-based items
    for m in model_items:
        if m.get("text"):
            a = align_text_span(m, text, priors)
            if a:
                out.append(a)

    # Fallback: legacy spans if no text fields made it through
    if not out:
        for m in model_items:
            lab = (m.get("label") or m.get("type") or "").strip()
            s = m.get("start", m.get("startIndex"))
            e = m.get("end", m.get("endIndex"))
            try:
                s, e = int(s), int(e)
            except Exception:
                continue
            if e <= s:
                continue
            toks = _tokenize_eval(text)
            a = _snap_to_tokens({"label": lab, "start": s, "end": e}, toks)
            if a and _valid_span(text, a["start"], a["end"]):
                out.append(a)

    # Canonicalize
    return [
        {"label": m["label"], "start": int(m["start"]), "end": int(m["end"])}
        for m in out
        if m.get("end", 0) > m.get("start", 0)
    ]


def clip_and_validate(spans: list, text: str) -> list:
    """
    Normalize S1 spans to [{"label", "start", "end"}], clip to text bounds, drop invalid.
    Accepts either {label,start,end} or {type,startIndex,endIndex}.
    """
    L = len(text or "")
    out = []
    for m in spans or []:
        lab_raw = (m.get("label") or m.get("type") or "").strip()
        lab = _CANON_LABEL.get(lab_raw.lower())
        if not lab:
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
        out.append({"label": lab, "start": s, "end": e})
    return out


def align_text_markers_to_spans(
    doc_text: str, items: List[dict], case_sensitive: bool = False
) -> List[dict]:
    """
    Convert [{"label":..., "text":...}] to [{"label":..., "start":int, "end":int}]
    by exact substring matching (prefers non-overlapping, left-to-right).
    - If items already have start/end, they’re passed through.
    - If multiple matches exist, choose the first non-overlapping occurrence.
    """
    if not doc_text or not items:
        return []
    spans = []
    hay = doc_text if case_sensitive else doc_text.lower()

    # track used character positions to avoid overlaps
    used = [False] * len(doc_text)

    def _place_span(lab: str, s: int, e: int):
        if e <= s:
            return False
        # avoid overlaps
        if any(used[i] for i in range(s, e)):
            return False
        spans.append({"label": lab, "start": s, "end": e})
        for i in range(s, e):
            used[i] = True
        return True

    for it in items:
        lab = (it.get("label") or it.get("type") or "").strip()
        if not lab:
            continue

        # already a span? keep it (we'll still snap/dedup downstream)
        if "start" in it and "end" in it:
            try:
                s, e = int(it["start"]), int(it["end"])
                _place_span(lab, s, e)
            except Exception:
                pass
            continue

        txt = (it.get("text") or "").strip()
        if not txt:
            continue

        needle = txt if case_sensitive else txt.lower()
        pos = 0
        placed = False
        while True:
            idx = hay.find(needle, pos)
            if idx < 0:
                break
            j = idx + len(needle)
            if _place_span(lab, idx, j):
                placed = True
                break
            pos = idx + 1

        # (optional) fallback: try case-sensitive if insensitive failed
        if not placed and not case_sensitive:
            pos = 0
            while True:
                idx = doc_text.find(txt, pos)
                if idx < 0:
                    break
                j = idx + len(txt)
                if _place_span(lab, idx, j):
                    break
                pos = idx + 1

    return spans


from collections import Counter
import random, json, re, logging


def _verify_s1_spans_claude(
    *,
    text: str,
    spans: list[dict],
    bedrock: "BedrockChat",
    max_tokens: int = 800,
    temperature: float = 0.0,
    thr: float = 0.55,
) -> list[dict]:
    """
    Ask Claude to score each span (0..1) and optionally suggest a relabel among
    {Actor,Action,Effect,Victim,Evidence}. Keep spans with p>=thr.
    If a relabel is suggested with pr>=thr, apply it.
    """
    if not spans:
        return spans
    # Minimal, deterministic system prompt (no CoT in output)
    sys = (
        "<role>You are a strict validator for PsyCoMark S1 spans.</role>\n"
        "<labels>Actor, Action, Effect, Victim, Evidence</labels>\n"
        "<rules>Score each span in [0,1] for label correctness; suggest a relabel only if clearly better.</rules>\n"
        "<output>Return ONLY JSON: "
        '[{"i":int,"p":float,"label": "Actor|Action|Effect|Victim|Evidence"|null, "pr": float|null}]</output>'
    )
    # Compact user payload (truncate text to keep under context if needed)
    user = {
        "text": text,
        "spans": [
            {
                "i": i,
                "label": s.get("type") or s.get("label"),
                "start": s.get("startIndex", s.get("start")),
                "end": s.get("endIndex", s.get("end")),
                "t": text[
                    s.get("startIndex", s.get("start")) : s.get(
                        "endIndex", s.get("end")
                    )
                ],
            }
            for i, s in enumerate(spans)
        ],
    }
    out = bedrock.chat(
        system_prompt=sys,
        user_prompt=json.dumps(user, ensure_ascii=False),
        max_tokens=max_tokens,
        temperature=temperature,
    )
    # extract JSON array (tolerant)
    arr = []
    try:
        m = re.search(r"\[.*\]", out if isinstance(out, str) else "", re.S)
        if m:
            arr = json.loads(m.group(0))
    except Exception:
        arr = []
    if not arr:
        return spans
    # apply decisions
    keep = []
    for i, s in enumerate(spans):
        r = next((x for x in arr if x.get("i") == i), None)
        if not r:
            continue
        p = float(r.get("p", 0.0))
        if p < thr:
            continue
        new_lab = r.get("label")
        pr = r.get("pr")
        if new_lab and isinstance(pr, (int, float)) and pr >= thr:
            s = dict(s)
            s["type"] = new_lab
        keep.append(s)
    return keep


def _safe_label(x: str, allow_cant_tell: bool) -> str:
    x = (x or "").strip().lower()
    valid = {"conspiracy", "non"} | ({"cant_tell"} if allow_cant_tell else set())
    return x if x in valid else "non"


def run_s2_self_consistent(
    *,
    text: str,
    markers: list,
    fewshots_pool: list,
    tech: str,
    policy_text: str | None,  # kept in signature (ignored in new S2)
    prompt_arts: dict | None,  # kept in signature (ignored in new S2)
    model_id: str | None,
    region: str | None,
    bedrock,  # BedrockChat stateless client
    max_tokens: int,
    base_temperature: float,
    sc_temperature: float,
    sc_runs: int,
    allow_cant_tell: bool = False,  # NEW: expose cant_tell policy
    progress_factory=None,  # NEW: progress bar factory for SC runs
):
    """
    S2 self-consistency with vote aggregation.
    - Rebuild prompts per run (few-shot order shuffled).
    - No model probabilities required; we derive pseudo-probabilities from vote share.
    Returns: (final_label, p_con, p_non, first_sys, first_user)
    """
    temp = sc_temperature if sc_runs > 1 else base_temperature
    preds, labels = [], []
    first_sys = first_user = None
    pbar = progress_factory(sc_runs) if progress_factory else None

    for r in range(max(1, sc_runs)):
        fs = fewshots_pool[:] if fewshots_pool else []
        if fs:
            random.shuffle(fs)

        sys_prompt, user_prompt = build_s2_prompts_adapter(
            text=text,
            markers=markers,
            fewshots=fs,
            tech=tech,
            allow_cant_tell=allow_cant_tell,
        )
        if r == 0:
            first_sys, first_user = sys_prompt, user_prompt

        out = bedrock.chat(
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temp,
            stop_sequences=["</answer>"],
        )

        # --- Robust extraction: handle dict/bytes/str + object or array payloads ---
        def _as_text(x):
            if x is None:
                return ""
            if isinstance(x, (bytes, bytearray)):
                try:
                    return x.decode("utf-8", errors="ignore")
                except Exception:
                    return str(x)
            if isinstance(x, dict):
                cand = x.get("answer") or x.get("text") or x.get("content")
                return (
                    json.dumps(cand, ensure_ascii=False)
                    if isinstance(cand, (dict, list))
                    else (
                        str(cand)
                        if cand is not None
                        else json.dumps(x, ensure_ascii=False)
                    )
                )
            return str(x)

        raw = _as_text(out)
        js = None
        # 1) Prefer JSON inside <answer>...</answer> (object OR array)
        m = re.search(r"<answer>\s*(\{.*?\}|\[.*?\])\s*</answer>", raw, re.S)
        blob = m.group(1) if m else None
        # 2) Else fallback to the last JSON object/array in the text
        if not blob:
            all_blobs = re.findall(r"(\{.*?\}|\[.*?\])", raw, re.S)
            blob = all_blobs[-1] if all_blobs else None
        if blob:
            try:
                js = json.loads(blob)
            except Exception:
                # Tiny repair: remove trailing commas before } or ]
                blob2 = re.sub(r",\s*([\}\]])", r"\1", blob)
                try:
                    js = json.loads(blob2)
                except Exception:
                    js = None
        # If an array was returned, take the last dict element (common pattern)
        if isinstance(js, list):
            js = next((d for d in reversed(js) if isinstance(d, dict)), {}) or {}
        if not isinstance(js, dict):
            js = {}

        if pbar:
            pbar.update(1)

        lbl = _safe_label((js or {}).get("label"), allow_cant_tell=allow_cant_tell)
        labels.append(lbl)
        preds.append({"label": lbl})

    if pbar:
        pbar.close()

    # Majority vote
    maj_label, maj_count = Counter(labels).most_common(1)[0]
    vote_share = maj_count / max(1, len(labels))

    # Map vote share to pseudo-probs so your caller stays compatible
    if maj_label == "conspiracy":
        p_con, p_non = vote_share, 1.0 - vote_share
    elif maj_label == "non":
        p_con, p_non = 1.0 - vote_share, vote_share
    else:  # cant_tell
        p_con, p_non = 0.5, 0.5

    return maj_label, round(p_con, 6), round(p_non, 6), first_sys, first_user


def _order_s1_examples(exs):
    # score examples: prefer with both Action & Effect, then any positives, then negatives
    def score(ex):
        labs = {a.get("label") for a in (ex.get("spans") or [])}
        return 2 if ("Action" in labs and "Effect" in labs) else 1 if labs else 0

    # stable sort by (score desc, keep positives early)
    return sorted(exs, key=lambda e: score(e), reverse=True)


_ALLOWED_S1 = {"Actor", "Action", "Effect", "Victim", "Evidence"}


def _finalize_coda_spans(spans: list[dict], text: str) -> list[dict]:
    """
    Final safety pass for Codabench-shaped spans:
      [{"type":<label>, "startIndex":int, "endIndex":int, ...}]
    - Clamp to [0, len(text)]
    - Drop zero/neg-length
    - Enforce label set
    - Stable sort: (startIndex, length) for reproducible output
    """
    n = len(text)
    out = []
    for s in spans or []:
        if not isinstance(s, dict):
            continue
        lab = s.get("type")
        if lab not in _ALLOWED_S1:
            continue
        try:
            st = int(s.get("startIndex"))
            en = int(s.get("endIndex"))
        except Exception:
            continue
        st = max(0, min(n, st))
        en = max(0, min(n, en))
        if en <= st:
            continue
        rec = dict(s)
        rec["type"] = lab
        rec["startIndex"] = st
        rec["endIndex"] = en
        # keep text if consistent (optional)
        if "text" in rec and rec["text"] != text[st:en]:
            rec.pop("text", None)
        out.append(rec)
    out.sort(key=lambda r: (r["startIndex"], r["endIndex"] - r["startIndex"]))
    return out


def _dedup_examples(exs: list[dict]) -> list[dict]:
    """Avoid duplicate few-shot exemplars by _id or (text, hash of spans)."""
    seen, out = set(), []
    for e in exs or []:
        key = e.get("_id") or (
            e.get("text"),
            tuple(
                sorted(
                    (s.get("label"), s.get("start"), s.get("end"))
                    for s in (e.get("spans") or [])
                )
            ),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Run PsyCoMark S1→S2 with Claude Sonnet on Bedrock (Chat API)."
    )
    ap.add_argument("--test-file-s1", default="data/raw/dev_rehydrated.jsonl")
    ap.add_argument("--test-file-s2", default="data/raw/dev_rehydrated.jsonl")
    ap.add_argument("--eda-root", default=None, help="Folder with EDA artifacts.")
    ap.add_argument(
        "--techniques",
        default="fs_policy_boundary_cot_sc5",
        help="Comma list, e.g., fs_policy_boundary_cot_sc5, fs_neg_cot_sc10, zs_cot_sc5",
    )
    ap.add_argument("--model-id", default=None, help="Override Bedrock model id.")
    ap.add_argument("--region", default=None, help="AWS region (e.g., eu-central-1).")
    ap.add_argument("--max-tokens-s1", type=int, default=1500)
    ap.add_argument("--max-tokens-s2", type=int, default=900)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--sc-temperature", type=float, default=0.7)
    ap.add_argument("--s2-self-consistency", type=int, default=None)
    ap.add_argument("--s1-iou", type=float, default=0.5)
    ap.add_argument("--out-root", default="runs/joint_llm_v4")
    ap.add_argument("--pp-merge-gap", type=int, default=1)
    ap.add_argument("--pp-dedup-iou", type=float, default=0.90)
    ap.add_argument("--pp-conflict-iou", type=float, default=0.50)
    ap.add_argument("--max-markers-per-label", type=int, default=3)
    ap.add_argument("--limit-docs", type=int, default=None)
    ap.add_argument("--s2-thresh", default="auto")
    # ---- S1 verifier (optional) ----
    ap.add_argument(
        "--s1-verify",
        action="store_true",
        help="Enable 1-shot Claude verifier for S1 spans.",
    )
    ap.add_argument(
        "--s1-verify-thr",
        type=float,
        default=0.55,
        help="Keep spans with p>=thr; also gate relabels.",
    )
    ap.add_argument(
        "--s1-verify-max", type=int, default=4096, help="Max tokens for verifier call."
    )

    # NEW: explicit artifact inputs (override discovery under --eda-root)
    ap.add_argument(
        "--artifacts-file",
        type=Path,
        default=None,
        help="Path to prompt_artifacts.json (priors, conflicts). Overrides discovery under --eda-root.",
    )
    ap.add_argument(
        "--fewshot-bank",
        type=Path,
        default=None,
        help="Path to fewshot_bank.json with {'s1': [...], 's2': [...]} examples.",
    )

    ap.add_argument(
        "--save-prompts",
        choices=["none", "sample", "all"],
        default="sample",
        help="Save prompts to runs/<tech>/prompts (sample=first 3 docs/task).",
    )
    ap.add_argument("--print-prompts-preview", action="store_true", default=True)
    ap.add_argument(
        "--preview-s1",
        type=int,
        default=10,
        help="Preview the first N S1 answers in the console (0=disable).",
    )
    args = ap.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Determinism
    random.seed(42)

    def _load_json(p: Path | None):
        if not p:
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    # ---------- Resolve inputs ----------
    eda_root = Path(args.eda_root) if getattr(args, "eda_root", None) else None
    artifacts_path = (
        Path(args.artifacts_file)
        if args.artifacts_file
        else (eda_root / "prompt_artifacts.json" if eda_root else None)
    )
    fewshot_path = (
        Path(args.fewshot_bank)
        if args.fewshot_bank
        else (eda_root / "fewshot_bank.json" if eda_root else None)
    )

    artifacts = _load_json(artifacts_path) or {}
    fewshots = _load_json(fewshot_path) or {"s1": [], "s2": []}

    # Light validation
    s1_priors = artifacts.get("s1_priors", {}) or {}
    s1_conflicts = artifacts.get("s1_conflicts", []) or []
    s1_examples = fewshots.get("s1", []) or []
    s2_examples = fewshots.get("s2", []) or []

    if not isinstance(s1_examples, list):
        s1_examples = []
    if not isinstance(s2_examples, list):
        s2_examples = []

    print(
        f"[runner] priors={list(s1_priors.keys())[:5]}... | conflicts={s1_conflicts[:3]}"
    )
    print(f"[runner] fewshots: S1={len(s1_examples)} | S2={len(s2_examples)}")

    # Quick few-shot audit
    s1_pos = sum(
        1 for e in s1_examples if isinstance(e.get("answer"), list) and e["answer"]
    )
    s1_neg = sum(
        1 for e in s1_examples if isinstance(e.get("answer"), list) and not e["answer"]
    )
    print(f"[fewshot audit] S1 positives={s1_pos} negatives={s1_neg}")

    # Label coverage (rough glance)
    from collections import Counter

    labs = Counter(a.get("label") for e in s1_examples for a in (e.get("answer") or []))
    print(f"[fewshot audit] S1 label counts: {dict(labs)}")

    print("=== JOINT PROMPT (Chat API) START ===")
    random.seed(42)
    from os import getenv

    model_id = args.model_id or getenv("MODEL_ID")
    region = args.region or getenv("AWS_DEFAULT_REGION") or "us-east-1"

    # Initialize Bedrock client (keep your class/IDs)
    bc = BedrockChat(
        model_id=(model_id or "anthropic.claude-sonnet-4-5-20250929-v1:0"),
        region_name=(region or "eu-central-1"),
    )

    # ----- Build a prompt_arts shim so the rest of your code keeps working -----
    # Map our artifacts schema to what your builders expect.
    prompt_arts: Dict[str, Any] = {
        "priors_prompt": s1_priors,  # consumed by build_s1_system / postprocess
        "conflicts": {
            "pairs": s1_conflicts
        },  # your code expects .get("conflicts").get("pairs", [])
        "fewshot_bank": {"s1": s1_examples, "s2": s2_examples},
        # Optional: keep placeholders if your builder reads them
        "fewshot_policy": {"s1_policy": "", "s2_policy": ""},
        "boundary_prompts": {"note": ""},
    }

    # ----- Load data -----
    rows_s1 = list(read_jsonl(args.test_file_s1))
    rows_s2 = list(read_jsonl(args.test_file_s2))
    if args.limit_docs is not None:
        rows_s1 = rows_s1[: args.limit_docs]
        rows_s2 = rows_s2[: args.limit_docs]
        logging.info(
            f"Limiting to first {args.limit_docs} docs for S1 ({len(rows_s1)}) and S2 ({len(rows_s2)})"
        )
    id2doc_s2 = {(r.get("_id") or r.get("doc_id")): r for r in rows_s2}

    # ----- Techniques parsing -----
    techniques = [t.strip() for t in args.techniques.split(",") if t.strip()]

    for tech in techniques:
        print(f"\n=== JOINT S1→S2 :: {tech} ===")

        has_fs = "fs" in tech
        has_neg = "neg" in tech
        use_cot = "cot" in tech
        use_boundary = "boundary" in tech
        use_policy = "policy" in tech

        sc_match = re.search(r"sc(\d+)", tech)
        sc_n = int(sc_match.group(1)) if sc_match else 1
        if isinstance(args.s2_self_consistency, int) and args.s2_self_consistency >= 1:
            sc_n = args.s2_self_consistency

        # few-shot pools (balanced subsets; small to control prompt size)
        # Prefer explicit fewshot_bank if loaded; else fall back to prompt_arts internals.
        fs_s1_pool = _adapt_bank_to_builder_s1(s1_examples) if has_fs else []
        fs_s2_pool = _adapt_bank_to_builder_s2(s2_examples) if has_fs else []

        # Optionally add a single negative example per task when 'neg' is in technique.
        if has_fs and has_neg:
            # S1 negative = examples with empty "answer"
            s1_neg = [
                ex
                for ex in fs_s1_pool
                if isinstance(ex.get("spans"), list) and len(ex["spans"]) == 0
            ][:1]
            # S2 negative = examples with answer.label == "non"
            s2_neg = [
                ex
                for ex in fs_s2_pool
                if isinstance(ex.get("answer"), dict)
                and (ex["answer"].get("label") == "non")
            ][:1]
            fs_s1_pool = fs_s1_pool + s1_neg
            fs_s2_pool = fs_s2_pool + s2_neg

        tech_dir = pathlib.Path(args.out_root) / tech
        (tech_dir / "s1").mkdir(parents=True, exist_ok=True)
        (tech_dir / "s2").mkdir(parents=True, exist_ok=True)
        (tech_dir / "prompts").mkdir(parents=True, exist_ok=True)
        if prompt_arts:
            _snapshot_artifacts(prompt_arts, tech_dir)

        s1_sub = tech_dir / "s1" / "submission.jsonl"
        s1_pruned_sub = tech_dir / "s1" / "submission_pruned.jsonl"
        s2_sub = tech_dir / "s2" / "submission.jsonl"

        # save meta once
        _save_prompt_meta(
            tech_dir,
            {
                "tech": tech,
                "model_id": model_id,
                "region": args.region,
                "max_tokens_s1": args.max_tokens_s1,
                "max_tokens_s2": args.max_tokens_s2,
                "temperature": args.temperature,
                "sc_temperature": args.sc_temperature,
                "s1_policy_used": False,  # we keep empty policy strings by default
                "s2_policy_used": False,
                "boundary_note_used": False,
            },
        )

        printed_s1_preview = False
        printed_s2_preview = False

        # ===== S1 =====
        s1_out_rows = []
        s1_pruned_rows = []
        id2markers = {}
        total_raw = total_valid = total_pruned = 0
        saved_s1 = 0
        # inline S1 preview (print first N as we go)
        preview_limit = max(0, int(getattr(args, "preview_s1", 0)))
        previewed = 0

        # tqdm progress for S1 docs
        s1_iter = tqdm(rows_s1, desc="S1 docs", unit="doc", ncols=100, leave=False)
        for rec in s1_iter:
            _id = rec.get("_id") or rec.get("doc_id")
            s1_iter.set_postfix_str(str(_id))
            txt = (rec.get("text") or "").strip()
            if not txt:
                id2markers[_id] = []
                s1_out_rows.append({"_id": _id, "markers": []})
                s1_pruned_rows.append({"_id": _id, "markers": []})
                continue

            # Small, balanced take of S1 few-shots (up to 8) for this doc
            fewshots_s1 = (
                _pick_balanced_s1_fewshots(fs_s1_pool, k=min(8, len(fs_s1_pool)))
                if fs_s1_pool
                else []
            )
            # Deduplicate few-shots
            fewshots_s1 = _dedup_examples(fewshots_s1)
            # --- Guardrails to avoid "all []" priming and enforce coverage ---
            pool = fs_s1_pool or []
            # 1) Ensure ≥2 positive examples
            pos = [
                ex
                for ex in fewshots_s1
                if isinstance(ex.get("spans"), list) and len(ex["spans"]) > 0
            ]
            if len(pos) < 2:
                add = [
                    ex
                    for ex in pool
                    if isinstance(ex.get("spans"), list)
                    and len(ex["spans"]) > 0
                    and ex not in fewshots_s1
                ][: max(0, 2 - len(pos))]
                fewshots_s1.extend(add)
            # 2) Ensure at least one Victim example
            need_victim = not any(
                any(s.get("label") == "Victim" for s in (ex.get("spans") or []))
                for ex in fewshots_s1
            )
            if need_victim:
                v = next(
                    (
                        ex
                        for ex in pool
                        if any(
                            s.get("label") == "Victim" for s in (ex.get("spans") or [])
                        )
                        and ex not in fewshots_s1
                    ),
                    None,
                )
                if v:
                    # replace last slot to preserve length
                    if fewshots_s1:
                        fewshots_s1[-1] = v
                    else:
                        fewshots_s1.append(v)
            # 3) Ensure at least one Action–Effect pair example
            has_ae = any(
                {"Action", "Effect"}.issubset(
                    {s.get("label") for s in (ex.get("spans") or [])}
                )
                for ex in fewshots_s1
            )
            if not has_ae:
                ae = next(
                    (
                        ex
                        for ex in pool
                        if {"Action", "Effect"}.issubset(
                            {s.get("label") for s in (ex.get("spans") or [])}
                        )
                        and ex not in fewshots_s1
                    ),
                    None,
                )
                if ae:
                    if fewshots_s1:
                        fewshots_s1[0] = ae
                    else:
                        fewshots_s1.append(ae)

            # in build_s1_user before rendering:
            if isinstance(fewshots_s1, list) and fewshots_s1:
                fewshots_s1 = _order_s1_examples(fewshots_s1)

            if fewshots_s1 and all(
                (len(ex.get("spans") or []) == 0) for ex in fewshots_s1
            ):
                pos = [ex for ex in fs_s1_pool if len(ex.get("spans") or []) > 0]
                if pos:
                    fewshots_s1[0] = pos[
                        0
                    ]  # replace first negative with a positive exemplar

            # Guarantee at least two positive exemplars (prevents all-[] priming)
            pos_need = 2 - sum(
                1 for ex in fewshots_s1 if len(ex.get("spans") or []) > 0
            )
            if pos_need > 0:
                extras = [ex for ex in fs_s1_pool if len(ex.get("spans") or []) > 0]
                for ex in extras[:pos_need]:
                    for j, cur in enumerate(fewshots_s1):
                        if len(cur.get("spans") or []) == 0:
                            fewshots_s1[j] = ex
                            break

            # Build S1 prompts (inject priors/conflicts; boundary/policy off by default)
            s1_system = build_s1_system(
                priors=(prompt_arts.get("priors_prompt") or {}),
                conflicts=[
                    tuple(p)
                    for p in (prompt_arts.get("conflicts") or {}).get("pairs", [])
                ],
                use_cot=use_cot,
            )
            s1_user = build_s1_user(
                text_input=txt,
                s1_fewshots=(
                    fewshots_s1 if has_fs else []
                ),  # render few-shots if enabled
                include_cot=use_cot,
            )

            sys_blocks, user_block = [s1_system], s1_user

            # Preview/save
            if args.save_prompts == "all" or (
                args.save_prompts == "sample" and saved_s1 < 3
            ):
                _save_prompt_bundle(tech_dir, "s1", str(_id), sys_blocks, user_block)
                saved_s1 += 1
            if args.print_prompts_preview and not printed_s1_preview:
                tqdm.write(
                    "[S1 prompt preview]\nSYSTEM:\n"
                    + "\n\n---\n\n".join(sys_blocks)
                    + "\n\nUSER:\n"
                    + user_block
                )
                printed_s1_preview = True

            system_str = "\n\n".join(sys_blocks)  # Bedrock needs a single system string

            try:
                s1_raw = bc.chat(
                    system_prompt=system_str,
                    user_prompt=user_block,
                    max_tokens=args.max_tokens_s1,
                    temperature=args.temperature,
                    stop_sequences=["</answer>"],  # helpful for clean JSON
                )
                # bc.chat may return a string or a dict with {"answer": "..."} (Extended Thinking)
                if isinstance(s1_raw, dict):
                    raw_text = s1_raw.get("answer", "")
                else:
                    raw_text = s1_raw
                # Prefer <answer>[...] extraction; fallback to first top-level array
                arr = extract_answer_json(raw_text) or list_json_extract(raw_text) or []
                # --- Optional verifier using builder prompts (fast, cheap) ---
                if getattr(args, "s1_verify", False) and arr:
                    # build candidate list in canonical shape for the verifier
                    cands = []
                    for m in arr:
                        lab = (m.get("label") or m.get("type") or "").strip()
                        s = m.get("start", m.get("startIndex"))
                        e = m.get("end", m.get("endIndex"))
                        try:
                            s, e = int(s), int(e)
                        except Exception:
                            continue
                        if e <= s or s < 0:
                            continue
                        cands.append(
                            {
                                "label": lab,
                                "start": s,
                                "end": e,
                                "text": txt[s:e],
                            }
                        )

                    if cands:
                        v_sys, v_user = build_s1_verifier_prompts(
                            text=txt, candidate_spans=cands
                        )
                        v_out = bc.chat(
                            system_prompt=v_sys,
                            user_prompt=v_user,
                            max_tokens=getattr(args, "s1_verify_max", 800),
                            temperature=0.0,
                        )
                        # tolerant JSON object extraction: {"keep":[...], "reject":[...]}
                        keep_idx = []
                        try:
                            v_txt = (
                                v_out if isinstance(v_out, str) else json.dumps(v_out)
                            )
                            mobj = re.search(r"\{.*\}", v_txt, re.S)
                            if mobj:
                                js = json.loads(mobj.group(0))
                                keep_idx = js.get("keep") or []
                                if not isinstance(keep_idx, list):
                                    keep_idx = []
                        except Exception:
                            keep_idx = []
                        if keep_idx:
                            # filter original arr using keep indices
                            arr = [
                                arr[i]
                                for i in keep_idx
                                if isinstance(i, int) and 0 <= i < len(arr)
                            ]
                        elif getattr(args, "print_prompts_preview", False):
                            # If previewing prompts, note a full rejection for debugging
                            print(
                                f"[S1 verifier] no keeps out of {len(cands)} candidates"
                            )

                arr = _coerce_s1_items(arr)  # <-- sanitize to list[dict]
                # Validate/repair to evaluator-aligned spans
                # 1) Try precise alignment from verbatim text (robust to indices noise)
                spans = align_and_normalize_s1(
                    arr, txt, (prompt_arts.get("priors_prompt") or {})
                )
                # 2) If still empty, try the legacy validator with a wider window
                if not spans:
                    spans = validate_and_repair_s1_spans(
                        arr, txt, win=32, use_tokens=True
                    )

            except Exception as e:
                logging.warning(f"[S1] generate() failed -> empty spans: {e}")
                spans = []

            # Post-process to evaluator-aligned spans
            merge_gap = getattr(args, "pp_merge_gap", 1)
            dedup_iou = getattr(args, "pp_dedup_iou", 0.90)
            conflict_iou = getattr(args, "pp_conflict_iou", 0.50)
            markers = postprocess_s1_spans(
                text=txt,
                spans=spans,
                priors=(prompt_arts.get("priors_prompt") or {}),
                merge_gap=merge_gap,
                dedup_iou=dedup_iou,
                conflict_iou=conflict_iou,
            )
            # --- Optional verifier pass ---
            if args.s1_verify and markers:
                markers = _verify_s1_spans_claude(
                    text=txt,
                    spans=markers,
                    bedrock=bc,
                    max_tokens=getattr(args, "s1_verify_max", 4096),
                    temperature=getattr(args, "temperature", 0.0),
                    thr=getattr(args, "s1_verify_thr", 0.5),
                )

            # 🔒 Final hard clamp/sort to guarantee submission-safe spans
            markers = _finalize_coda_spans(markers, txt)
            # Per-doc instrumentation (helps detect where loss happens)
            if not spans or not markers:
                logging.debug(
                    f"[S1 dbg] doc={_id} extracted={len(spans)} kept_after_pp={len(markers)}"
                )
            total_valid += len(markers)

            # prune per label for S2 prompt size
            by_lab = defaultdict(list)
            for m in markers:
                by_lab[m["type"]].append(m)

            pruned = []
            k = getattr(args, "max_markers_per_label", 5)
            for lab, arr_ in by_lab.items():
                arr_sorted = sorted(
                    arr_, key=lambda m: (-(m["endIndex"] - m["startIndex"]))
                )
                pruned.extend(arr_sorted[:k])

            # normalize schema for S2 prompt
            id2markers[_id] = [to_s2_marker(m, txt) for m in pruned]
            total_pruned += len(id2markers[_id])

            # Update counters AFTER finalize so metrics reflect the kept spans
            total_raw += len(spans)
            total_valid += len(markers)

            s1_out_rows.append({"_id": _id, "markers": markers})
            s1_pruned_rows.append({"_id": _id, "markers": id2markers[_id]})

            # ---- inline preview for first N docs ----
            if previewed < preview_limit:
                tqdm.write(
                    f"\n[S1 #{previewed+1}] _id={_id}\n"
                    f"  text: {_snippet(txt, 180)}\n"
                    f"  markers ({len(markers)}): {json.dumps([_norm_marker_for_print(m) for m in markers], ensure_ascii=False)}"
                )
                previewed += 1

        write_jsonl(s1_sub, s1_out_rows)
        write_jsonl(s1_pruned_sub, s1_pruned_rows)
        print(f"S1 done -> {s1_sub}")
        print(f"S1 pruned-for-S2 -> {s1_pruned_sub}")
        print(
            f"S1 debug: spans raw/valid/pruned = {total_raw}/{total_valid}/{total_pruned}"
        )

        # ===== S2 =====
        s2_out_rows, s2_prob_rows = [], []
        saved_s2 = 0
        printed_s2_preview = False

        # tqdm progress for S2 docs (need a list for deterministic total)
        s2_items = list(id2doc_s2.items())
        s2_iter = tqdm(s2_items, desc="S2 docs", unit="doc", ncols=100, leave=False)
        for _id, doc2 in s2_iter:
            s2_iter.set_postfix_str(str(_id))
            txt = (doc2.get("text") or "").strip()
            mks = id2markers.get(_id, [])

            # few-shot pool per doc (up to 8)
            fewshots_s2 = (
                _pick_balanced_s2_fewshots(fs_s2_pool, k=min(8, len(fs_s2_pool)))
                if fs_s2_pool
                else []
            )

            # Self-consistency runs for S2
            final_label, avg_con, avg_non, dbg_sys, dbg_user = run_s2_self_consistent(
                text=txt,
                markers=mks,
                fewshots_pool=fewshots_s2,
                tech=tech,
                policy_text=None,  # keep empty unless you have policy text
                prompt_arts=prompt_arts,
                bedrock=bc,
                max_tokens=args.max_tokens_s2,
                base_temperature=args.temperature,
                sc_temperature=args.sc_temperature,
                sc_runs=max(1, sc_n),
                allow_cant_tell=False,
                progress_factory=lambda total: tqdm(
                    total=total, desc="SC", unit="run", ncols=60, leave=False
                ),
                model_id=(model_id or "anthropic.claude-sonnet-4-5-20250929-v1:0"),
                region=region,
            )

            if args.save_prompts == "all" or (
                args.save_prompts == "sample" and saved_s2 < 3
            ):
                _save_prompt_bundle(tech_dir, "s2", str(_id), [dbg_sys], dbg_user)
                saved_s2 += 1
            if args.print_prompts_preview and not printed_s2_preview:
                tqdm.write(
                    "[S2 prompt preview]\nSYSTEM:\n"
                    + dbg_sys
                    + "\n\nUSER:\n"
                    + dbg_user
                )
                printed_s2_preview = True

            s2_out_rows.append(
                {
                    "_id": _id,
                    "conspiracy": ("Yes" if final_label == "conspiracy" else "No"),
                }
            )
            s2_prob_rows.append(
                {
                    "_id": _id,
                    "label": final_label,
                    "p_conspiracy": round(avg_con, 6),
                    "p_non": round(avg_non, 6),
                }
            )

        # write raw probs, then choose threshold
        probs_path = tech_dir / "s2" / "probs.jsonl"
        write_jsonl(probs_path, s2_prob_rows)

        thr = 0.50
        if isinstance(args.s2_thresh, str) and args.s2_thresh.lower() == "auto":
            best_t, best_f1, stats = tune_threshold_dev(rows_s2, s2_prob_rows)
            if stats.get("n", 0) > 0:
                thr = best_t
                print(
                    f"[S2] tuned threshold on dev: t={thr:.2f} (dev f1={best_f1:.3f}, mean_p={stats['mean_p']:.3f})"
                )
            else:
                print("[S2] no gold labels; keep default thr=0.50")
        else:
            try:
                thr = float(args.s2_thresh)
            except Exception:
                logging.warning(
                    f"[S2] invalid --s2-thresh={args.s2_thresh}; using 0.50"
                )
                thr = 0.50

        # final submission using chosen threshold
        pred2 = [
            {
                "_id": r["_id"],
                "conspiracy": ("Yes" if r["p_conspiracy"] >= thr else "No"),
            }
            for r in s2_prob_rows
        ]
        write_jsonl(s2_sub, pred2)
        print(f"S2 done -> {s2_sub}")

        # top-level Codabench files
        codabench_s1 = [
            {"_id": r["_id"], "markers": _to_codabench_s1(r.get("markers", []))}
            for r in s1_out_rows
        ]

        def _assert_submission_safe(
            rows: list[dict], text_by_id: dict[str, str] | None = None
        ):
            for r in rows:
                _id = r.get("_id")
                for m in r.get("markers", []):
                    assert isinstance(m, dict)
                    assert m.get("type") in _ALLOWED_S1
                    assert isinstance(m.get("startIndex"), int) and isinstance(
                        m.get("endIndex"), int
                    )
                    assert m["startIndex"] < m["endIndex"]

        _text_map = {
            (rec.get("_id") or rec.get("doc_id")): (rec.get("text") or "")
            for rec in rows_s1
        }
        _assert_submission_safe(
            [
                {"_id": r["_id"], "markers": _to_codabench_s1(r.get("markers", []))}
                for r in s1_out_rows
            ],
            _text_map,
        )

        write_jsonl("submission_s1.jsonl", codabench_s1)
        write_jsonl("submission_s2.jsonl", pred2)
        print("Wrote top-level submissions: submission_s1.jsonl, submission_s2.jsonl")

        # per-tech ZIP
        try:
            submissions_dir = pathlib.Path("submissions")
            submissions_dir.mkdir(parents=True, exist_ok=True)
            zip_path = submissions_dir / f"submission_{tech}.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                if pathlib.Path("submission_s1.jsonl").exists():
                    zf.write("submission_s1.jsonl", arcname="submission_s1.jsonl")
                if pathlib.Path("submission_s2.jsonl").exists():
                    zf.write("submission_s2.jsonl", arcname="submission_s2.jsonl")
                if s1_sub.exists():
                    zf.write(s1_sub, arcname=f"{tech}/s1/submission.jsonl")
                if s1_pruned_sub.exists():
                    zf.write(
                        s1_pruned_sub, arcname=f"{tech}/s1/submission_pruned.jsonl"
                    )
                if s2_sub.exists():
                    zf.write(s2_sub, arcname=f"{tech}/s2/submission.jsonl")
                if probs_path.exists():
                    zf.write(probs_path, arcname=f"{tech}/s2/probs.jsonl")
            print(f"Packaged ZIP -> {zip_path}")
        except Exception as e:
            logging.warning(f"[ZIP] Failed to create technique ZIP: {e}")


if __name__ == "__main__":
    main()
