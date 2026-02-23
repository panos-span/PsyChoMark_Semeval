#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psycomark.data.pipeline — Data preparation (Safe & RAG-Optimized).

Ported from the repository-level `data_pipeline.py`.

Key properties:
1. S1: Keeps 15% of clean negatives AND includes 'cant_tell' documents.
2. S2: Strictly binary (conspiracy vs non). Filters out 'cant_tell'.
3. Offset safety: Preserves RAW text (only trims edges) to maintain span offsets.

Usage:
  python -m psycomark.data.pipeline --input data/raw/train_rehydrated.jsonl --output-dir data/clean_v2
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

from loguru import logger

# --- 1. Lexicons (Preserved) ---
ABSOLUTIST = [
    "always",
    "never",
    "everyone",
    "no one",
    "impossible",
    "undeniable",
    "without a doubt",
    "completely",
    "totally",
    "entirely",
    "absolutely",
    "certainly",
    "no doubt",
    "truth",
    "fact",
    "proven",
]

HEDGES = [
    "maybe",
    "perhaps",
    "possibly",
    "likely",
    "unlikely",
    "appears",
    "seems",
    "suggests",
    "might",
    "could",
    "may",
    "arguably",
    "allegedly",
    "reportedly",
    "claimed",
]

# --- 2. ReX Categories (Preserved) ---
CUES_MUNDANE = re.compile(
    r"\b(office|wages|manager|boss|weather|traffic|customer service|refund|price|scam|bs|myth|work from home|policy|HR|commission|fee)\b",
    re.I,
)
CUES_DEBUNK = re.compile(
    r"\b(debunk|no evidence|false claim|conspiracy theory|not true|lies|fact check|hoax|bullshit|ridiculous)\b",
    re.I,
)
CUES_EVANGELIST = re.compile(
    r"\b(wake up|the truth|universal truth|mission|listen to|watch this|series|documentary|red pill|must watch|share this)\b",
    re.I,
)
CUES_INSIDER = re.compile(
    r"\b(deep state|cabal|regime|globalist|agenda|controlled by|owned by|false flag|psyop|elites|intelligence|shadow|new world order)\b",
    re.I,
)


# --- 3. Utilities ---

def calculate_linguistic_intensity(text: str) -> dict:
    tokens = re.findall(r"\w+", text.lower())
    total = len(tokens) or 1
    abs_count = sum(1 for t in tokens if t in ABSOLUTIST)
    hed_count = sum(1 for t in tokens if t in HEDGES)
    return {"abs_rate": abs_count / total, "hed_rate": hed_count / total}


def check_confusion_overlap(spans: List[Dict]) -> bool:
    actions = [s for s in spans if s["label"] == "Action"]
    effects = [s for s in spans if s["label"] == "Effect"]
    if not actions or not effects:
        return False

    for a in actions:
        for e in effects:
            start = max(a["start"], e["start"])
            end = min(a["end"], e["end"])
            if end > start:
                return True
    return False


def detect_s2_subtype(text: str, label: str) -> str:
    if label == "non":
        if CUES_DEBUNK.search(text):
            return "non_debunking"
        if CUES_MUNDANE.search(text):
            return "non_mundane"
        return "non_reporting"

    if label == "conspiracy":
        if CUES_EVANGELIST.search(text):
            return "con_evangelist"
        if CUES_INSIDER.search(text):
            return "con_insider"
        return "con_general"

    if label == "cant_tell":
        return "ambiguous"

    return "unknown"


def score_s2_richness(text: str, label: str, spans: List[Dict]) -> dict:
    subtype = detect_s2_subtype(text, label)
    ling_stats = calculate_linguistic_intensity(text)
    is_confusing = check_confusion_overlap(spans)

    score = 0.0
    if 200 < len(text) < 2000:
        score += 1.0
    if subtype != "unknown":
        score += 1.0
    if is_confusing:
        score -= 5.0

    if label == "conspiracy" and ling_stats["abs_rate"] > 0.015:
        score += 2.0
    elif label == "non" and ling_stats["hed_rate"] > 0.015:
        score += 2.0

    is_hard_negative = (label == "non" and len(spans) > 0) or (subtype == "non_debunking")
    if is_hard_negative:
        score += 3.0

    return {
        "s2_score": round(score, 3),
        "s2_subtype": subtype,
        "is_hard_negative": is_hard_negative,
    }


# --- 4. S1 Consensus Logic ---

def merge_s1_spans(all_spans: List[Dict], num_annotators: int) -> List[Dict]:
    if not all_spans:
        return []
    threshold = (num_annotators // 2) + 1
    final_spans: list[dict] = []

    spans_by_label = collections.defaultdict(list)
    for s in all_spans:
        spans_by_label[s["label"]].append(s)

    for label, spans in spans_by_label.items():
        spans.sort(key=lambda x: (x["start"], x["end"]))
        clusters: list[list[dict]] = []

        for span in spans:
            placed = False
            for cluster in clusters:
                rep = cluster[0]
                if span["start"] < rep["end"] and span["end"] > rep["start"]:
                    cluster.append(span)
                    placed = True
                    break
            if not placed:
                clusters.append([span])

        for cluster in clusters:
            if len(cluster) >= threshold:
                best_span = max(cluster, key=lambda x: (x["end"] - x["start"], -x["start"]))
                best_span = best_span.copy()
                best_span["why"] = None
                best_span.pop("startIndex", None)
                best_span.pop("endIndex", None)
                final_spans.append(best_span)

    final_spans.sort(key=lambda x: (x["start"], x["end"]))
    return final_spans


# --- 5. Main Pipeline ---

def normalize_input_row(row: Dict) -> Dict:
    raw_label = row.get("conspiracy", "non")

    if raw_label == "Yes":
        label = "conspiracy"
    elif raw_label == "No":
        label = "non"
    elif raw_label == "Can't tell":
        label = "cant_tell"
    else:
        label = "DROP_ME"

    return {
        "doc_id": str(row.get("_id") or row.get("doc_id") or ""),
        "text": row.get("text", "").strip(),
        "label": label,
        "spans": row.get("markers", []),
        "subreddit": row.get("subreddit", ""),
    }


def process_disaggregated_group(group: List[Dict]) -> Dict | None:
    if not group:
        return None
    base = group[0]

    labels = [r["label"] for r in group]
    counts = Counter(labels)
    consensus_label, _ = sorted(counts.most_common(1), key=lambda x: x[0])[0]

    if consensus_label == "DROP_ME":
        return None
    if len(counts) > 1 and counts.most_common(2)[0][1] == counts.most_common(2)[1][1]:
        return None

    all_spans: list[dict] = []
    for r in group:
        for s in r["spans"]:
            s_clean = {
                "label": s.get("type") or s.get("label"),
                "text": s.get("text"),
                "start": int(s.get("startIndex") or s.get("start", 0)),
                "end": int(s.get("endIndex") or s.get("end", 0)),
            }
            if s_clean["label"]:
                all_spans.append(s_clean)

    final_spans = merge_s1_spans(all_spans, len(group))
    s2_meta = score_s2_richness(base["text"], consensus_label, final_spans)

    raw_span_count = sum(len(r["spans"]) for r in group)
    final_span_count = len(final_spans)
    has_conflict = raw_span_count > 0 and final_span_count == 0

    return {
        "doc_id": base["doc_id"],
        "text": base["text"],
        "label": consensus_label,
        "spans": final_spans,
        "subreddit": base["subreddit"],
        "has_conflict": has_conflict,
        **s2_meta,
    }


def select_balanced_s2_subset(rows: List[Dict], target_total: int = 2000) -> List[Dict]:
    """Stratified sampling with deterministic sort (by -score, doc_id)."""
    buckets = collections.defaultdict(list)
    for r in rows:
        if r["s2_score"] > 0:
            buckets[r["s2_subtype"]].append(r)

    def det_sort(items: list[dict]) -> list[dict]:
        return sorted(items, key=lambda x: (-x["s2_score"], x["doc_id"]))

    selected: list[dict] = []

    k_hn = int(target_total * 0.20)
    k_non = int(target_total * 0.13)
    k_con = int(target_total * 0.12)

    hn = [r for r in rows if r["is_hard_negative"] and r["s2_score"] > 0]
    selected.extend(det_sort(hn)[:k_hn])

    for st in ("non_mundane", "non_debunking"):
        selected.extend(det_sort(buckets[st])[:k_non])
    for st in ("con_evangelist", "con_insider", "con_general"):
        selected.extend(det_sort(buckets[st])[:k_con])

    current_ids = {r["doc_id"] for r in selected}
    rem = [r for r in rows if r["doc_id"] not in current_ids and r["s2_score"] > 0]

    needed = target_total - len(selected)
    if needed > 0:
        selected.extend(det_sort(rem)[:needed])

    return selected


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw/train_rehydrated.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/clean_v2"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--s2-target", type=int, default=2000, help="Target docs for S2")
    args = parser.parse_args(argv)

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[dict]] = collections.defaultdict(list)
    try:
        with open(args.input, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = normalize_input_row(json.loads(line))
                    if len(row["text"]) > 10:
                        groups[row["doc_id"]].append(row)
                except Exception:
                    continue
    except FileNotFoundError:
        logger.error("Input file not found.")
        return

    processed: list[dict] = []
    for doc_id in sorted(groups.keys()):
        res = process_disaggregated_group(groups[doc_id])
        if res:
            processed.append(res)

    processed.sort(key=lambda x: (-len(x["spans"]), -int(x["is_hard_negative"]), x["doc_id"]))

    # --- Save S1 ---
    s1_path = args.output_dir / "train_clean_s1.jsonl"
    count = 0
    with open(s1_path, "w", encoding="utf-8") as f:
        for r in processed:
            if r["spans"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                count += 1
            elif r["has_conflict"]:
                continue
            elif random.random() < 0.15:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                count += 1
    logger.info(f"Wrote S1 data ({count} docs) to {s1_path} (includes 'cant_tell')")

    # --- Save S2 (strictly binary) ---
    s2_candidates = [r for r in processed if r["label"] in ("conspiracy", "non")]
    s2_curated = select_balanced_s2_subset(s2_candidates, target_total=args.s2_target)
    s2_path = args.output_dir / "train_clean_s2.jsonl"

    with open(s2_path, "w", encoding="utf-8") as f:
        for r in s2_curated:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Wrote S2 data ({len(s2_curated)} docs) to {s2_path} (excludes 'cant_tell')")


if __name__ == "__main__":
    main()
