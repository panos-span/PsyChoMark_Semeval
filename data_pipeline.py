"""
PsyCoMark Data Processing Pipeline (Final Reproducible Version).

Role:
1. Aggregates multiple annotators into a 'Gold Standard' (Consensus).
2. Curates a BALANCED dataset for S2 RAG using Stratified Sampling.
3. SCORES documents by Linguistic Intensity to select archetypal examples.
4. GUARANTEES REPRODUCIBILITY via deterministic tie-breaking (Score + DocID).
"""

import json
import re
import argparse
import collections
import random
from pathlib import Path
from typing import Dict, List, Counter
from loguru import logger

# --- 1. Lexicons (Forensic Intensity) ---

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

# --- 2. ReX Categories (Sub-typing) ---

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

# --- 3. Quality & Scoring Logic ---


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
                inter = end - start
                union = (a["end"] - a["start"]) + (e["end"] - e["start"]) - inter
                iou = inter / union
                if iou > 0.5:
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
    return "unknown"


def score_s2_richness(text: str, label: str, spans: List[Dict]) -> dict:
    subtype = detect_s2_subtype(text, label)
    ling_stats = calculate_linguistic_intensity(text)
    is_confusing = check_confusion_overlap(spans)

    score = 0.0
    length = len(text)
    if 200 < length < 2000:
        score += 1.0
    if subtype != "unknown":
        score += 1.0

    # Quality Control
    if is_confusing:
        score -= 5.0

    # Archetypal Boosting
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
        "linguistics": ling_stats,
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
                if "startIndex" in best_span:
                    del best_span["startIndex"]
                if "endIndex" in best_span:
                    del best_span["endIndex"]
                final_spans.append(best_span)

    # Final deterministic sort of the list
    final_spans.sort(key=lambda x: (x["start"], x["end"]))
    return final_spans


# --- 5. Main Pipeline ---


def normalize_input_row(row: Dict) -> Dict:
    raw_label = row.get("conspiracy", "non")
    if raw_label == "Yes":
        label = "conspiracy"
    elif raw_label == "No":
        label = "non"
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

    # Strict Consensus
    labels = [r["label"] for r in group]
    counts = Counter(labels)
    # Sort for deterministic tie breaking if counts equal (though we drop ties anyway)
    consensus_label, _ = sorted(counts.most_common(1), key=lambda x: x[0])[0]

    if consensus_label == "DROP_ME":
        return None
    if len(counts) > 1 and counts.most_common(2)[0][1] == counts.most_common(2)[1][1]:
        return None

    all_spans = []
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

    return {
        "doc_id": base["doc_id"],
        "text": base["text"],
        "label": consensus_label,
        "spans": final_spans,
        **s2_meta,
        "subreddit": base["subreddit"],
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
        # High score first, then alphabetical ID
        return sorted(items, key=lambda x: (-x["s2_score"], x["doc_id"]))

    # 1. Hard Negatives
    hn_candidates = [r for r in rows if r["is_hard_negative"] and r["s2_score"] > 0]
    selected.extend(deterministic_sort(hn_candidates)[:60])

    # 2. Mundane & Debunking
    for subtype in ["non_mundane", "non_debunking"]:
        cands = deterministic_sort(buckets[subtype])
        selected.extend(cands[:40])

    # 3. Conspiracy Archetypes
    for subtype in ["con_evangelist", "con_insider", "con_general"]:
        cands = deterministic_sort(buckets[subtype])
        selected.extend(cands[:35])

    # 4. Fill remaining
    current_ids = {r["doc_id"] for r in selected}
    remaining = [
        r for r in rows if r["doc_id"] not in current_ids and r["s2_score"] > 0
    ]
    selected.extend(deterministic_sort(remaining)[: (target_total - len(selected))])

    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default="train_rehydrated.jsonl")
    parser.add_argument("--output-dir", type=Path, default="data/clean")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    args = parser.parse_args()

    # 1. Set Seed for any random ops (though we rely mostly on deterministic sorting now)
    random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    groups = collections.defaultdict(list)
    logger.info(f"Reading {args.input}...")
    try:
        with open(args.input, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = normalize_input_row(json.loads(line))
                    if len(row["text"]) > 10:
                        groups[row["doc_id"]].append(row)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        logger.error("Input file not found.")
        return

    processed_rows = []
    # Process in deterministic order of IDs
    for doc_id in sorted(groups.keys()):
        res = process_disaggregated_group(groups[doc_id])
        if res:
            processed_rows.append(res)

    # 3. Save S1 - Deterministic Sort
    # Key: (Count desc, HardNeg desc, DocID asc)
    processed_rows.sort(
        key=lambda x: (-len(x["spans"]), -int(x["is_hard_negative"]), x["doc_id"])
    )

    s1_path = args.output_dir / "train_clean_s1.jsonl"
    with open(s1_path, "w", encoding="utf-8") as f:
        for r in processed_rows:
            if r["spans"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Wrote S1 data to {s1_path}")

    # 4. Save S2 - Deterministic Selection
    curated_s2 = select_balanced_s2_subset(processed_rows, target_total=300)

    s2_path = args.output_dir / "train_clean_s2.jsonl"
    with open(s2_path, "w", encoding="utf-8") as f:
        for r in curated_s2:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Wrote S2 curated data ({len(curated_s2)} docs) to {s2_path}")


if __name__ == "__main__":
    main()
