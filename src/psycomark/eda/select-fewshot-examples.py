#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select-fewshot-examples.py

Strategic few-shot selector for SemEval-2026 PsyCoMark.
- S1: ensures a mix of negatives/simple/complex examples with subreddit diversity.
- S2: aims for ~50/50 conspiracy vs non (with graceful fallback).
"""

from __future__ import annotations
import argparse
import json
import random
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Any, Tuple

# ---------------- IO ----------------


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load .jsonl into a list[dict]."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {path}: {e}")
        return []


def save_jsonl(data: List[Dict[str, Any]], path: Path) -> None:
    """Save list[dict] to .jsonl."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"✅ Saved {len(data)} examples → {path}")
    except IOError as e:
        print(f"Error saving to {path}: {e}")


# ------------- Helpers --------------

S1_LABELS = {"Actor", "Action", "Effect", "Victim", "Evidence"}


def _get_subreddit(ex: Dict[str, Any]) -> str:
    return (ex.get("subreddit") or "").strip()


def _get_markers(ex: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Support both 'markers' (PsyCoMark) and 'spans' (generic).
    Normalize each to {'label', 'start', 'end'} when possible.
    """
    marks = ex.get("markers")
    if isinstance(marks, list):
        return [m for m in marks if isinstance(m, dict) and m.get("label")]
    spans = ex.get("spans")
    if isinstance(spans, list):
        # Try to normalize common span schemas
        norm = []
        for m in spans:
            if not isinstance(m, dict):
                continue
            lab = m.get("label") or m.get("type") or m.get("tag")
            s = m.get("start") or m.get("startIndex")
            e = m.get("end") or m.get("endIndex")
            if lab and isinstance(s, int) and isinstance(e, int):
                norm.append({"label": lab, "start": s, "end": e})
        return norm
    return []


def _count_s1_spans(ex: Dict[str, Any]) -> int:
    """Count valid S1 markers only."""
    return sum(1 for m in _get_markers(ex) if m.get("label") in S1_LABELS)


def _get_s2_label(ex: Dict[str, Any]) -> str:
    """
    Robustly read document label for S2.
    Priority: doc_label → label → gold.label; default 'non'
    """
    lbl = (
        ex.get("doc_label")
        or ex.get("label")
        or (ex.get("gold") or {}).get("label")
        or "non"
    )
    return str(lbl).strip().lower()


def _take_n_diverse(
    items: List[Dict[str, Any]], n: int, key_fn
) -> List[Dict[str, Any]]:
    """
    Take up to n items preferring diversity by key_fn (e.g., subreddit).
    Simple greedy: pick one per key round-robin until filled.
    """
    if n <= 0 or not items:
        return []
    by_key = defaultdict(list)
    for it in items:
        by_key[key_fn(it)].append(it)
    keys = list(by_key.keys())
    random.shuffle(keys)
    out = []
    idx = 0
    while len(out) < n and keys:
        k = keys[idx % len(keys)]
        if by_key[k]:
            out.append(by_key[k].pop(random.randrange(len(by_key[k]))))
        # drop empty keys to speed up
        keys = [kk for kk in keys if by_key[kk]]
        idx += 1
    return out


# ------------- S1 selection -------------


def select_diverse_s1_examples(
    pool: List[Dict[str, Any]], num_examples: int
) -> List[Dict[str, Any]]:
    """
    Stratify by complexity:
      - negative: 0 spans
      - simple:   1–2 spans
      - complex:  3+ spans
    Target mix: 25% negative, 40% simple, 35% complex (rounded; with fallback).
    Enforce lightweight subreddit diversity.
    """
    if num_examples >= len(pool):
        return list(pool)

    buckets = {"negative": [], "simple": [], "complex": []}
    for ex in pool:
        n = _count_s1_spans(ex)
        if n == 0:
            buckets["negative"].append(ex)
        elif 1 <= n <= 2:
            buckets["simple"].append(ex)
        else:
            buckets["complex"].append(ex)

    # initial targets
    targets = {
        "negative": int(num_examples * 0.25),
        "simple": int(num_examples * 0.40),
        "complex": int(num_examples * 0.35),
    }
    # fix rounding
    while sum(targets.values()) < num_examples:
        targets[random.choice(list(targets.keys()))] += 1

    selected: List[Dict[str, Any]] = []

    # pick with subreddit diversity
    for name in ("negative", "simple", "complex"):
        want = min(targets[name], len(buckets[name]))
        if want > 0:
            picks = _take_n_diverse(buckets[name], want, key_fn=_get_subreddit)
            selected.extend(picks)

    # if any bucket was under-filled, top up from remaining examples (prioritize positives)
    remaining = [ex for ex in pool if ex not in selected]
    if len(selected) < num_examples and remaining:
        # Prioritize simple/complex then negatives for topping up
        remaining.sort(key=lambda ex: (0 if _count_s1_spans(ex) >= 1 else 1))
        need = num_examples - len(selected)
        selected.extend(random.sample(remaining, k=min(need, len(remaining))))

    random.shuffle(selected)
    return selected[:num_examples]


# ------------- S2 selection -------------


def select_diverse_s2_examples(
    pool: List[Dict[str, Any]], num_examples: int
) -> List[Dict[str, Any]]:
    """
    Aim for ~50/50 conspiracy vs non with subreddit diversity and fallback.
    """
    if num_examples >= len(pool):
        return list(pool)

    buckets = defaultdict(list)
    for ex in pool:
        lab = _get_s2_label(ex)
        if lab not in ("conspiracy", "non"):
            lab = "non"
        buckets[lab].append(ex)

    half = num_examples // 2
    selected: List[Dict[str, Any]] = []

    cons = _take_n_diverse(buckets["conspiracy"], half, key_fn=_get_subreddit)
    nonc = _take_n_diverse(buckets["non"], half, key_fn=_get_subreddit)
    selected.extend(cons)
    selected.extend(nonc)

    # Fill remainder if odd num_examples or buckets were sparse
    remaining = [ex for ex in pool if ex not in selected]
    need = num_examples - len(selected)
    if need > 0 and remaining:
        # Prefer whichever class is currently under-represented
        cur = Counter(_get_s2_label(x) for x in selected)
        remaining.sort(key=lambda ex: (cur[_get_s2_label(ex)], random.random()))
        selected.extend(remaining[:need])

    random.shuffle(selected)
    return selected[:num_examples]


# ------------- Random baseline -------------


def select_random_examples(
    pool: List[Dict[str, Any]], num_examples: int
) -> List[Dict[str, Any]]:
    if num_examples >= len(pool):
        return list(pool)
    return random.sample(pool, num_examples)


# ------------- CLI -------------


def main():
    ap = argparse.ArgumentParser(
        description="Strategically select few-shot examples for PsyCoMark prompts."
    )
    ap.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Path to the .jsonl with candidate examples.",
    )
    ap.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Path to save selected .jsonl few-shot examples.",
    )
    ap.add_argument(
        "--num-examples",
        type=int,
        default=8,
        help="Total number of few-shot examples to select.",
    )
    ap.add_argument(
        "--task",
        type=str,
        choices=["s1", "s2"],
        required=True,
        help="Target task: s1 (extraction) or s2 (classification).",
    )
    ap.add_argument(
        "--strategy",
        type=str,
        choices=["random", "diverse"],
        default="diverse",
        help="Selection strategy.",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed for determinism.")
    args = ap.parse_args()

    random.seed(args.seed)

    print(f"Loading examples from: {args.input_file}")
    pool = load_jsonl(args.input_file)
    if not pool:
        print("Input file is empty or could not be loaded. Exiting.")
        return

    print(
        f"Selecting {args.num_examples} examples for task '{args.task}' using '{args.strategy}' strategy…"
    )

    if args.strategy == "random":
        selected = select_random_examples(pool, args.num_examples)
    else:
        selected = (
            select_diverse_s1_examples(pool, args.num_examples)
            if args.task == "s1"
            else select_diverse_s2_examples(pool, args.num_examples)
        )

    if not selected:
        print("⚠️ No examples selected. Falling back to random selection.")
        selected = select_random_examples(pool, args.num_examples)

    save_jsonl(selected, args.output_file)


if __name__ == "__main__":
    main()
