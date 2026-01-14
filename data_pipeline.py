#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_pipeline.py (Safe & RAG-Optimized)

1. S1: Keeps 15% of 'Clean Negatives' (0 markers).
2. S2: Preserves markers for Hard Negative detection.
3. SAFETY: Preserves RAW text to maintain span offset validity.
"""

import json
import re
import argparse
import collections
import random
from pathlib import Path
from typing import Dict, List, Counter
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
                return True  # Any overlap is confusing
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
        score -= 5.0  # Penalty for broken spans

    if label == "conspiracy" and ling_stats["abs_rate"] > 0.015:
        score += 2.0
    elif label == "non" and ling_stats["hed_rate"] > 0.015:
        score += 2.0

    # Hard Negative Boost
    is_hard_negative = (label == "non" and len(spans) > 0) or (
        subtype == "non_debunking"
    )
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
    final_spans = []

    spans_by_label = collections.defaultdict(list)
    for s in all_spans:
        spans_by_label[s["label"]].append(s)

    for label, spans in spans_by_label.items():
        # Deterministic Sort for Clustering
        spans.sort(key=lambda x: (x["start"], x["end"]))
        clusters = []
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
                # Deterministic Selection (Longest first, then earliest start)
                best_span = max(
                    cluster, key=lambda x: (x["end"] - x["start"], -x["start"])
                ).copy()
                best_span["why"] = None
                # Clean up keys for RAG
                best_span.pop("startIndex", None)
                best_span.pop("endIndex", None)
                final_spans.append(best_span)

    # Final deterministic sort of the list
    final_spans.sort(key=lambda x: (x["start"], x["end"]))
    return final_spans


# --- 5. Main Pipeline ---


def normalize_input_row(row: Dict) -> Dict:
    raw_label = row.get("conspiracy", "non")
    label = (
        "conspiracy"
        if raw_label == "Yes"
        else "non" if raw_label == "No" else "DROP_ME"
    )

    return {
        "doc_id": str(row.get("_id") or row.get("doc_id") or ""),
        "text": row.get("text", "").strip(),  # Trim only, NO aggressive cleaning
        "label": label,
        "spans": row.get("markers", []),
        "subreddit": row.get("subreddit", ""),
    }


def process_disaggregated_group(group: List[Dict]) -> Dict | None:
    if not group:
        return None
    base = group[0]

    # Strict Consensus
    labels = [r["label"] for r in group]
    counts = Counter(labels)
    # Sort for deterministic tie breaking if counts equal (though we drop ties anyway)
    consensus_label, _ = sorted(counts.most_common(1), key=lambda x: x[0])[0]

    if consensus_label == "DROP_ME":
        return None
    if len(counts) > 1 and counts.most_common(2)[0][1] == counts.most_common(2)[1][1]:
        return None  # Drop exact ties

    all_spans = []
    for r in group:
        for s in r["spans"]:
            # Preserve RAW offsets
            s_clean = {
                "label": s.get("type") or s.get("label"),
                "text": s.get("text"),  # Raw text
                "start": int(s.get("startIndex") or s.get("start", 0)),
                "end": int(s.get("endIndex") or s.get("end", 0)),
            }
            if s_clean["label"]:
                all_spans.append(s_clean)

    final_spans = merge_s1_spans(all_spans, len(group))
    s2_meta = score_s2_richness(base["text"], consensus_label, final_spans)

    # Check if raw inputs had spans but consensus killed them
    raw_span_count = sum(len(r["spans"]) for r in group)
    final_span_count = len(final_spans)

    # If we started with spans but ended with none, we have a CONFLICT.
    has_conflict = raw_span_count > 0 and final_span_count == 0

    return {
        "doc_id": base["doc_id"],
        "text": base["text"],  # Raw text
        "label": consensus_label,
        "spans": final_spans,
        **s2_meta,
        "subreddit": base["subreddit"],
        "has_conflict": has_conflict,
    }


def select_balanced_s2_subset(rows: List[Dict], target_total=300) -> List[Dict]:
    """
    Stratified Sampling with DETERMINISTIC Sorting.
    Sort Key: (-s2_score, doc_id)
    This ensures identical results every run, regardless of input order.
    """
    buckets = collections.defaultdict(list)
    for r in rows:
        if r["s2_score"] > 0:
            buckets[r["s2_subtype"]].append(r)

    selected = []

    def deterministic_sort(items):
        return sorted(items, key=lambda x: (-x["s2_score"], x["doc_id"]))

    hn = [r for r in rows if r["is_hard_negative"] and r["s2_score"] > 0]
    selected.extend(deterministic_sort(hn)[:60])

    for st in ["non_mundane", "non_debunking"]:
        selected.extend(deterministic_sort(buckets[st])[:40])
    for st in ["con_evangelist", "con_insider", "con_general"]:
        selected.extend(deterministic_sort(buckets[st])[:35])

    current_ids = {r["doc_id"] for r in selected}
    rem = [r for r in rows if r["doc_id"] not in current_ids and r["s2_score"] > 0]
    selected.extend(deterministic_sort(rem)[: (target_total - len(selected))])

    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default="data/raw/train_rehydrated.jsonl")
    parser.add_argument("--output-dir", type=Path, default="data/clean")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1. Set Seed for any random ops (though we rely mostly on deterministic sorting now)
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    groups = collections.defaultdict(list)
    try:
        with open(args.input, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = normalize_input_row(json.loads(line))
                    if len(row["text"]) > 10:
                        groups[row["doc_id"]].append(row)
                except:
                    continue
    except FileNotFoundError:
        logger.error("Input file not found.")
        return

    processed = []
    for doc_id in sorted(groups.keys()):
        res = process_disaggregated_group(groups[doc_id])
        if res:
            processed.append(res)

    # Sort deterministically
    processed.sort(
        key=lambda x: (-len(x["spans"]), -int(x["is_hard_negative"]), x["doc_id"])
    )

    # 3. Save S1 (With Negative Injection)
    s1_path = args.output_dir / "train_clean_s1.jsonl"
    count = 0
    with open(s1_path, "w", encoding="utf-8") as f:
        for r in processed:
            # OPTIMIZATION: Keep all positives, plus 15% random negatives
            if r["spans"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                count += 1
            # 2. SKIP if there was a conflict (Don't teach the model to ignore this)
            elif r["has_conflict"]:
                continue
            # 3. Sample Clean Negatives (True Empty)
            elif random.random() < 0.15:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                count += 1
    logger.info(f"Wrote S1 data ({count} docs) to {s1_path}")

    # 4. Save S2 (Curated)
    s2_curated = select_balanced_s2_subset(processed, target_total=300)
    s2_path = args.output_dir / "train_clean_s2.jsonl"
    with open(s2_path, "w", encoding="utf-8") as f:
        for r in s2_curated:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Wrote S2 data ({len(s2_curated)} docs) to {s2_path}")


if __name__ == "__main__":
    main()
