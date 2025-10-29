#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_prompt_artifacts.py

Generates prompt artifacts for SemEval-2026 PsyCoMark:
- S1 priors (length percentiles, start-mode)
- S1 conflict pairs (most overlapping label pairs)
- Few-shot banks for S1 (span extraction) and S2 (doc classification)

Outputs:
- JSON artifact with priors + conflicts (path via --output-file)
- fewshot_bank.json with {"s1":[...], "s2":[...]} (path via --fewshot-out)
"""

from __future__ import annotations
import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import re

import numpy as np

S1_LABELS = {"Actor", "Action", "Effect", "Victim", "Evidence"}
_CANON = {
    "actor": "Actor",
    "action": "Action",
    "effect": "Effect",
    "victim": "Victim",
    "evidence": "Evidence",
}

EVIDENCE_CUE_RE = re.compile(
    r"(according to|reported|reports|says|said|stated|\".+?\"|https?://)", re.I
)
# Heuristic verb-headed matcher (no POS): “to VERB”, “VERB-ing/ed/es/s”
VERB_HEAD_RE = re.compile(r"^\s*(?:to\s+)?[A-Za-z]+(?:ed|ing|es|s)?\b")


def _closest_sentence_bounds(
    text: str, left: int, right: int, limit: int = 140
) -> tuple[int, int]:
    """Expand [left,right] to nearest sentence-ish boundaries without exceeding ±limit."""
    l0 = text.rfind(".", max(0, left - limit), left)
    l1 = text.rfind("!", max(0, left - limit), left)
    l2 = text.rfind("?", max(0, left - limit), left)
    new_left = max(0, max(l0, l1, l2) + 1) if max(l0, l1, l2) != -1 else left
    r0 = text.find(".", right, min(len(text), right + limit))
    r1 = text.find("!", right, min(len(text), right + limit))
    r2 = text.find("?", right, min(len(text), right + limit))
    candidates = [x for x in (r0, r1, r2) if x != -1]
    new_right = min(len(text), (min(candidates) + 1) if candidates else right)
    return new_left, new_right


def _diversity_key_of(ex: Dict[str, Any], key: str) -> str:
    v = ex.get(key)
    if v is None:
        v = ex.get("subreddit") or ex.get("source") or "NA"
    return str(v)


def _score_curriculum(spans: List[Dict[str, Any]]) -> int:
    """Score examples for ordering: conflict > positive > negative."""
    labs = {m.get("label") for m in (spans or [])}
    return 2 if ("Action" in labs and "Effect" in labs) else (1 if labs else 0)


def _mine_hard_negatives(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prefer negatives with trigger words but no gold spans (harder)."""
    TRIG = re.compile(
        r"(according to|report|says|said|quote|http|evidence|prove|proof|secret|conspiracy)",
        re.I,
    )
    out = []
    for ex in docs:
        if _normalize_markers(ex):  # has spans -> skip
            continue
        t = ex.get("text") or ""
        if TRIG.search(t):
            out.append(ex)
    return out


def _normalize_markers(ex: Dict[str, Any]) -> List[Dict[str, int | str]]:
    """
    Return a list of dicts with keys: label,start,end from multiple possible schemas.
    Accepts:
      - ex["markers"] : [{"label","start","end",...}]
      - ex["spans"]   : same as above
      - nested: ex["answer"] (list) if present in fewshot-like items
    """
    cand = ex.get("markers")
    if not cand:
        cand = ex.get("spans")
    if not cand and isinstance(ex.get("answer"), list):
        cand = ex["answer"]
    out = []
    for m in cand or []:
        try:
            raw = m.get("label")
            lab = _CANON.get(str(raw).strip().lower())
            s = int(m.get("start"))
            e = int(m.get("end"))
            if lab in S1_LABELS and 0 <= s < e:
                out.append({"label": lab, "start": s, "end": e})
        except Exception:
            continue
    return out


def _pick_doc_with_label(
    docs: List[Dict[str, Any]], label: str
) -> Dict[str, Any] | None:
    for ex in docs:
        if any(m["label"] == label for m in _normalize_markers(ex)):
            return ex
    return None


def _pick_doc_with_labels(
    docs: List[Dict[str, Any]], labels: set[str]
) -> Dict[str, Any] | None:
    for ex in docs:
        labs = {m["label"] for m in _normalize_markers(ex)}
        if labels.issubset(labs):
            return ex
    return None


# ---------------- IO ----------------


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load .jsonl into a list[dict]."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {path}: {e}")
        return []


def save_json(data: Dict[str, Any], path: Path) -> None:
    """Save dict to pretty .json."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved JSON → {path}")
    except IOError as e:
        print(f"Error saving to {path}: {e}")


# ------------- Normalization helpers -------------


def _normalize_markers(ex: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Accept 'markers' (PsyCoMark) or 'spans' (generic).
    Return normalized list with label/start/end for S1 labels.
    """
    out = []
    marks = ex.get("markers")
    if isinstance(marks, list):
        for m in marks:
            if not isinstance(m, dict):
                continue
            lab = m.get("label")
            s, e = m.get("start"), m.get("end")
            if lab in S1_LABELS and isinstance(s, int) and isinstance(e, int) and e > s:
                out.append({"label": lab, "start": s, "end": e})
    elif isinstance(ex.get("spans"), list):
        for m in ex["spans"]:
            if not isinstance(m, dict):
                continue
            lab = m.get("label") or m.get("type") or m.get("tag")
            s = m.get("start") or m.get("startIndex")
            e = m.get("end") or m.get("endIndex")
            if lab in S1_LABELS and isinstance(s, int) and isinstance(e, int) and e > s:
                out.append({"label": lab, "start": s, "end": e})
    return out


def _doc_label(ex: Dict[str, Any]) -> str:
    """Robust S2 label reader."""
    lab = (
        ex.get("doc_label")
        or ex.get("label")
        or (ex.get("gold") or {}).get("label")
        or "non"
    )
    return str(lab).strip().lower()


# ------------- Priors & conflicts -------------


def calculate_statistical_priors(
    data: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    For each S1 label:
      - q50_len, q90_len
      - mode_pos (mode of binned relative start in 0.1 steps)
      - mean_len (info only)
    """
    buckets = defaultdict(lambda: {"lengths": [], "positions": []})

    for item in data:
        text = item.get("text") or ""
        n = len(text)
        if n <= 0:
            continue
        for m in _normalize_markers(item):
            length = m["end"] - m["start"]
            if length <= 0:
                continue
            relpos = m["start"] / max(1, n)
            # bin to 0.1 steps for stability
            b = round(math.floor(relpos * 10) / 10.0, 1)
            buckets[m["label"]]["lengths"].append(length)
            buckets[m["label"]]["positions"].append(b)

    priors: Dict[str, Dict[str, Any]] = {}
    for lab, vals in buckets.items():
        L = vals["lengths"]
        P = vals["positions"]
        if not L:
            continue
        q50 = int(np.percentile(L, 50))
        q90 = int(np.percentile(L, 90))
        mean_len = float(np.mean(L))
        # mode of positions (ties broken deterministically)
        if P:
            cnt = Counter(P)
            most = max(sorted(cnt.items()), key=lambda kv: (kv[1], kv[0]))[0]
        else:
            most = 0.5
        priors[lab] = {
            "q50_len": q50,
            "q90_len": q90,
            "mean_len": round(mean_len, 1),
            "mode_pos": most,
        }
    print("Calculated S1 priors for labels:", sorted(priors.keys()))
    return priors


def _overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return max(a[0], b[0]) < min(a[1], b[1])


def analyze_span_conflicts(
    data: List[Dict[str, Any]], top_n: int = 2
) -> List[List[str]]:
    """
    Find most frequent overlapping label pairs.
    Returns list of ["Action","Effect"], ...
    """
    counts = Counter()
    for item in data:
        spans = sorted(_normalize_markers(item), key=lambda m: m["start"])
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                m1, m2 = spans[i], spans[j]
                if _overlap((m1["start"], m1["end"]), (m2["start"], m2["end"])):
                    pair = tuple(sorted([m1["label"], m2["label"]]))
                    counts[pair] += 1
    pairs = [list(p) for p, _ in counts.most_common(top_n)]
    print(f"Top {len(pairs)} conflict pairs:", pairs)
    return pairs


# ------------- Few-shot builders -------------


def _safe_window(
    text: str, start: int, end: int, pad: int = 120, hard_cap: int | None = None
) -> Tuple[str, int]:
    """
    Returns (snippet, offset) with snippet=text[left:right], offset=left.
    Offsets start/end corrected by subtracting 'offset'.
    """
    n = len(text)
    left = max(0, start - pad)
    right = min(n, end + pad)
    # snap to sentence-ish boundaries (stable prompts, cleaner examples)
    left, right = _closest_sentence_bounds(text, left, right, limit=140)
    snippet = text[left:right].strip()
    if hard_cap and len(snippet) > hard_cap:
        snippet = snippet[:hard_cap].rstrip()
    return snippet, left


def build_s1_fewshot_snippets(
    docs: List[Dict[str, Any]],
    want: int,
    seed: int = 42,
    victim_min: int = 1,
    conflict_min: int = 1,
    max_per_diverse: int = 2,
    diversity_key: str = "subreddit",
    max_chars: int = 360,
) -> List[Dict[str, Any]]:
    """
    Mix of negatives/simple/complex with offset-correct markers.
    Emits:
      { "text": snippet, "answer": [ {label,start,end} ... ] }
    """
    random.seed(seed)
    neg, simple, complex_ = [], [], []
    for ex in docs:
        mks = _normalize_markers(ex)
        n = len(mks)
        if n == 0:
            neg.append(ex)
        elif n <= 2:
            simple.append(ex)
        else:
            complex_.append(ex)

    t_neg = int(want * 0.25)
    t_simple = int(want * 0.40)
    t_complex = want - t_neg - t_simple

    def _take(items, k):
        return random.sample(items, min(k, len(items)))

    # mine harder negatives first, then fill regular
    hard_negs = _mine_hard_negatives(neg)
    picked = (
        _take(hard_negs, min(len(hard_negs), max(1, t_neg // 2)))
        + _take(
            [x for x in neg if x not in hard_negs],
            t_neg - min(len(hard_negs), max(1, t_neg // 2)),
        )
        + _take(simple, t_simple)
        + _take(complex_, t_complex)
    )

    if len(picked) < want:
        remaining = [ex for ex in docs if ex not in picked]
        remaining.sort(key=lambda ex: 0 if len(_normalize_markers(ex)) >= 1 else 1)
        picked += remaining[: want - len(picked)]

    # --- Victim quota (search full docs, not just picked)
    have_victim = any(
        any(m["label"] == "Victim" for m in _normalize_markers(ex)) for ex in picked
    )
    if not have_victim and victim_min > 0:
        ex_v = _pick_doc_with_label(docs, "Victim")
        if ex_v:
            picked[0] = ex_v

    # --- Conflict example: ensure ≥1 doc with both Action and Effect ---
    def _has_action_effect(ex):
        labs = {m["label"] for m in _normalize_markers(ex)}
        return ("Action" in labs) and ("Effect" in labs)

    if conflict_min > 0 and not any(_has_action_effect(ex) for ex in picked):
        ex_c = _pick_doc_with_labels(docs, {"Action", "Effect"})
        if ex_c:
            picked[-1] = ex_c  # ensure at least one A–E conflict doc

    # --- Diversity cap (e.g., by subreddit)
    bucket_count = defaultdict(int)
    diversified = []
    have_victim_doc = False
    for ex in picked:
        k = _diversity_key_of(ex, diversity_key)
        if bucket_count[k] < max_per_diverse:
            bucket_count[k] += 1
            diversified.append(ex)
            if any(m["label"] == "Victim" for m in _normalize_markers(ex)):
                have_victim_doc = True
    # top-up if we dropped too many
    if len(diversified) < want:
        for ex in docs:
            if ex in diversified:
                continue
            k = _diversity_key_of(ex, diversity_key)
            if bucket_count[k] < max_per_diverse:
                bucket_count[k] += 1
                diversified.append(ex)
            if len(diversified) >= want:
                break
    picked = diversified[:want]
    # --- Enforce Victim DOC quota (>= victim_min) by mining full pool ---
    victim_docs = [
        ex for ex in docs if any(m["label"] == "Victim" for m in _normalize_markers(ex))
    ]
    have_victim_count = sum(
        1
        for ex in picked
        if any(m["label"] == "Victim" for m in _normalize_markers(ex))
    )
    if victim_min and have_victim_count < victim_min:
        need = victim_min - have_victim_count
        # Candidates to replace: items in picked that do NOT contain Victim
        replace_idxs = [
            i
            for i, ex in enumerate(picked)
            if not any(m["label"] == "Victim" for m in _normalize_markers(ex))
        ]
        # Victim docs not already in picked
        to_insert = [ex for ex in victim_docs if ex not in picked][:need]
        for i, ex_v in zip(replace_idxs, to_insert):
            picked[i] = ex_v

    items = []
    victim_needed = max(0, int(victim_min))
    for idx, ex in enumerate(picked[:want]):
        text = ex.get("text") or ""
        mks = _normalize_markers(ex)
        if not mks:  # negative
            snippet, _ = _safe_window(
                text, 0, min(len(text), 120), pad=0, hard_cap=max_chars
            )
            items.append({"text": snippet, "answer": []})
            continue
        # choose spans; if this slot must teach Victim or A–E, bias selection
        # decide from the CURRENT example's labels; avoid indexing picked[-1]/[0]
        # choose spans; guarantee exactly one Victim across items if available
        need_victim = (victim_needed > 0) and any(m["label"] == "Victim" for m in mks)
        need_conflict = (idx == len(picked[:want]) - 1) and {
            "Action",
            "Effect",
        }.issubset({m["label"] for m in mks})
        chosen = []
        if need_conflict:
            ae = [m for m in mks if m["label"] in ("Action", "Effect")]
            # pick one Action + one Effect (shortest spans first)
            acts = sorted(
                [m for m in ae if m["label"] == "Action"],
                key=lambda m: m["end"] - m["start"],
            )
            effs = sorted(
                [m for m in ae if m["label"] == "Effect"],
                key=lambda m: m["end"] - m["start"],
            )
            if acts:
                chosen.append(acts[0])
            if effs:
                chosen.append(effs[0])
        if need_victim and not chosen:
            vs = sorted(
                [m for m in mks if m["label"] == "Victim"],
                key=lambda m: m["end"] - m["start"],
            )
            if vs:
                chosen.append(vs[0])
            victim_needed -= 1
        if not chosen:
            chosen = mks[:2]  # fallback compact set
        s0 = min(m["start"] for m in chosen)
        e0 = max(m["end"] for m in chosen)
        snippet, offset = _safe_window(text, s0, e0, pad=120, hard_cap=max_chars)
        rel = []
        for m in chosen:
            s_rel = m["start"] - offset
            e_rel = m["end"] - offset
            if 0 <= s_rel < e_rel <= len(snippet):
                # --- Evidence: expand leftwards to include cue if present ---
                if m["label"] == "Evidence":
                    # look back up to 40 chars for a cue inside the snippet
                    look_left = max(0, s_rel - 40)
                    prefix = snippet[look_left:s_rel]
                    if EVIDENCE_CUE_RE.search(prefix + snippet[s_rel:e_rel]):
                        # snap to start of the cue match if it begins in prefix
                        m2 = EVIDENCE_CUE_RE.search(prefix + snippet[s_rel:e_rel])
                        cue_start = look_left + (m2.start() if m2 else 0)
                        if cue_start < s_rel:
                            s_rel = cue_start
                    # minimum length for evidence spans
                    if (e_rel - s_rel) < 10:
                        # try to extend to the next token
                        e_rel = min(len(snippet), e_rel + 5)
                # --- Action: prefer verb-headed; if not, try to tighten to first verb-like token ---
                if m["label"] == "Action" and not VERB_HEAD_RE.match(
                    snippet[s_rel:e_rel]
                ):
                    m2 = re.search(
                        r"(?:to\s+)?[A-Za-z]+(?:ed|ing|es|s)?\b", snippet[s_rel:e_rel]
                    )
                    if m2:
                        s_rel = s_rel + m2.start()
                rel.append({"label": m["label"], "start": s_rel, "end": e_rel})
        items.append({"text": snippet, "answer": rel if rel else []})

    # Ensure not all negatives if positives exist in source
    if any(len(_normalize_markers(d)) > 0 for d in docs) and all(
        len(e["answer"]) == 0 for e in items
    ):
        for ex in docs:
            mks = _normalize_markers(ex)
            if not mks:
                continue
            text = ex.get("text") or ""
            s0, e0 = mks[0]["start"], mks[0]["end"]
            snippet, offset = _safe_window(text, s0, e0, pad=120)
            rel = [{"label": mks[0]["label"], "start": s0 - offset, "end": e0 - offset}]
            items[0] = {"text": snippet, "answer": rel}
            break
    # Light preference: if negatives > 40% and positives exist, swap one more negative for a positive
    neg_idx = [i for i, e in enumerate(items) if not e.get("answer")]
    if len(neg_idx) > len(items) * 0.4:
        for ex in docs:
            mks = _normalize_markers(ex)
            if mks:
                s0, e0 = mks[0]["start"], mks[0]["end"]
                snippet, offset = _safe_window(ex.get("text") or "", s0, e0, pad=120)
                items[neg_idx[0]] = {
                    "text": snippet,
                    "answer": [
                        {
                            "label": mks[0]["label"],
                            "start": s0 - offset,
                            "end": e0 - offset,
                        }
                    ],
                }
                break
    # Curriculum ordering: conflict > positive > negative
    items = sorted(
        items, key=lambda e: _score_curriculum(e.get("answer", [])), reverse=True
    )
    # debug: victim count in final answers
    _vict_cnt = sum(
        1 for e in items for a in (e.get("answer") or []) if a.get("label") == "Victim"
    )
    print(
        f"[fewshot] S1 Victim spans in final items: {_vict_cnt} (target ≥ {victim_min})"
    )
    return items


def build_s2_fewshot_examples(
    docs: List[Dict[str, Any]],
    want: int,
    seed: int = 42,
    max_chars: int = 1200,
) -> List[Dict[str, Any]]:
    """
    Aim for ~50/50 conspiracy vs non.
    Emits:
      { "text": doc_or_snippet, "answer": {"label": "...", "rationale": "..."} }
    """
    random.seed(seed)

    def lab(ex):
        v = _doc_label(ex)
        return v if v in ("conspiracy", "non") else "non"

    cons = [d for d in docs if lab(d) == "conspiracy"]
    nonc = [d for d in docs if lab(d) != "conspiracy"]
    half = want // 2

    sel = random.sample(cons, min(half, len(cons))) + random.sample(
        nonc, min(half, len(nonc))
    )
    if len(sel) < want:
        remaining = [d for d in docs if d not in sel]
        sel += remaining[: want - len(sel)]

    out = []
    for ex in sel[:want]:
        t = (ex.get("text") or "").strip()
        if len(t) > max_chars:
            t = t[:max_chars].rstrip()
        L = lab(ex)
        # Lightweight rationale templates
        rat = ex.get("rationale")
        if not rat:
            if L == "conspiracy":
                rat = "Claims covert coordination/hidden agenda beyond publicly stated facts."
            else:
                rat = "Descriptive/informational tone without hidden-agenda claims."
        out.append(
            {
                "text": t,
                "answer": {"label": L, "rationale": rat[:300]},
            }
        )
    return out


# ------------- CLI -------------


def main():
    ap = argparse.ArgumentParser(
        description="Generate PsyCoMark prompt artifacts + few-shot banks."
    )
    ap.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Annotated train .jsonl (text + markers/doc_label).",
    )
    ap.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Where to save priors/conflicts JSON.",
    )
    ap.add_argument(
        "--fewshot-out",
        type=Path,
        required=True,
        help="Where to save fewshot_bank.json",
    )
    ap.add_argument(
        "--s1-shots", type=int, default=12, help="Number of S1 few-shot examples."
    )
    ap.add_argument(
        "--s2-shots", type=int, default=10, help="Number of S2 few-shot examples."
    )
    ap.add_argument(
        "--s1-victim-min", type=int, default=2, help="Min Victim examples in S1 bank."
    )
    ap.add_argument(
        "--s1-conflict-min",
        type=int,
        default=1,
        help="Min Action–Effect conflict examples.",
    )
    ap.add_argument(
        "--diversity-key",
        type=str,
        default="subreddit",
        help="Field for diversity (e.g., subreddit).",
    )
    ap.add_argument(
        "--max-per-diverse",
        type=int,
        default=2,
        help="Max examples per diversity bucket.",
    )
    ap.add_argument(
        "--s1-max-chars",
        type=int,
        default=1200,
        help="Cap S1 snippet length (post window).",
    )
    ap.add_argument(
        "--s2-max-chars", type=int, default=1200, help="Cap S2 text length."
    )
    ap.add_argument(
        "--top-n-conflicts", type=int, default=2, help="Top-N overlapping label pairs."
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed for determinism.")
    args = ap.parse_args()

    random.seed(args.seed)

    print(f"Loading training data from: {args.input_file}")
    training = load_jsonl(args.input_file)
    if not training:
        print("Input file is empty or could not be loaded. Exiting.")
        return

    print("\n--- Generating S1 priors & conflicts ---")
    priors = calculate_statistical_priors(training)
    conflicts = analyze_span_conflicts(training, top_n=args.top_n_conflicts)

    artifacts = {
        "s1_priors": priors,
        "s1_conflicts": conflicts,
        "metadata": {
            "source_file": str(args.input_file),
            "num_docs_analyzed": len(training),
            "seed": args.seed,
        },
    }
    save_json(artifacts, args.output_file)

    print("\n--- Building few-shot banks ---")
    s1_bank = build_s1_fewshot_snippets(
        training,
        want=args.s1_shots,
        seed=args.seed,
        victim_min=args.s1_victim_min,
        conflict_min=args.s1_conflict_min,
        max_per_diverse=args.max_per_diverse,
        diversity_key=args.diversity_key,
        max_chars=args.s1_max_chars,
    )
    s2_bank = build_s2_fewshot_examples(
        training,
        want=args.s2_shots,
        seed=args.seed,
        max_chars=args.s2_max_chars,
    )
    fewshot_bank = {"s1": s1_bank, "s2": s2_bank}
    save_json(fewshot_bank, args.fewshot_out)

    # Coverage/logging
    def _lab_counts_s1(items):
        from collections import Counter

        c = Counter()
        for e in items:
            for a in e.get("answer") or []:
                c[a.get("label")] += 1
        return dict(c)

    print("[fewshot] S1 label counts:", _lab_counts_s1(s1_bank))
    has_victim = any(
        any(a.get("label") == "Victim" for a in e.get("answer") or []) for e in s1_bank
    )
    has_conflict = any(
        {"Action", "Effect"}.issubset({a.get("label") for a in (e.get("answer") or [])})
        for e in s1_bank
    )
    print(f"[fewshot] S1 Victim present: {has_victim}")
    print(f"[fewshot] S1 has Action–Effect conflict: {has_conflict}")

    print("\n✅ Artifact generation complete.")


if __name__ == "__main__":
    main()
