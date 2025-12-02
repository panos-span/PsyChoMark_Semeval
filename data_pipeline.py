# data_pipeline.py
"""
PsyCoMark Data Processing Pipeline (Optimized for RAG).

Enhancements:
1. Lexical Bank: Extracts common terms per label.
2. Narrative Scoring: Boosts docs with complete Actor->Action->Effect chains.
3. Length Filter: Removes noise (too short/long texts).
"""

import sys
import json
import re
import argparse
from loguru import logger
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any

# --- Constants & Regex ---
S2_CUE_RE = re.compile(
    r"(deep state|globalist|elite|agenda|cover[- ]?up|false flag|"
    r"hoax|they\s+want|they're trying|new world order|"
    r"pedo|chemtrail|MK[-\s]?Ultra|shadow government)",
    re.I,
)
S2_DEBUNK_RE = re.compile(
    r"\b(debunk|myth|not true|no evidence|conclusion is wrong|"
    r"conspiracy theory(?:ies)? as such)\b",
    re.I,
)

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "that",
    "this",
    "these",
    "those",
    "it",
    "he",
    "she",
    "they",
    "them",
    "their",
    "his",
    "her",
    "its",
}

# --- Scoring Functions ---


def score_s1_complexity(spans: List[Dict]) -> float:
    """
    Rates a document for S1 RAG suitability.

    IMPROVED LOGIC:
    - Huge Bonus for COMPLETE CHAINS (Actor + Action + Effect).
    - Penalty for ORPHANED ACTIONS (Action without Actor).
    """
    if not spans:
        return 0.0
    labels = {s.get("label") for s in spans}
    count = len(spans)

    score = count * 0.1

    # 1. Narrative Completeness (The Holy Grail of Few-Shots)
    if {"Actor", "Action", "Effect"}.issubset(labels):
        score += 3.0  # Gold standard
    elif {"Actor", "Action"}.issubset(labels):
        score += 1.5  # Solid causality

    # 2. Rare Class Bonus
    if "Victim" in labels:
        score += 1.0

    # 3. Orphan Penalty (Confusing for models)
    if "Action" in labels and "Actor" not in labels:
        score -= 0.5

    return round(score, 3)


def score_s2_richness(text: str, label: str, spans: List[Dict]) -> dict:
    """Rates a document for S2 RAG suitability."""
    cues = len(S2_CUE_RE.findall(text))
    is_debunk = bool(S2_DEBUNK_RE.search(text))
    length = len(text)
    has_markers = len(spans) > 0
    score = 0.0

    if label == "conspiracy":
        score += cues * 0.5
        score += min(length / 500.0, 2.0)
        if has_markers:
            score += 3.0
    elif label == "non":
        if is_debunk:
            score += 5.0
        elif cues > 0:
            score += 3.0
        else:
            score += 1.0

    return {"s2_score": round(score, 3), "is_hard_negative": is_debunk}


def normalize_row(row: Dict) -> Dict:
    """Standardizes keys."""
    doc_id = str(row.get("_id") or row.get("doc_id") or "")
    text = row.get("text", "").strip()

    raw_label = row.get("conspiracy", "").strip()
    label = "non"
    if raw_label == "Yes":
        label = "conspiracy"
    elif raw_label == "No":
        label = "non"
    else:
        label = "cant_tell"

    raw_markers = row.get("markers", [])
    spans = []
    for m in raw_markers:
        m_type = m.get("type") or m.get("label")
        m_text = m.get("text")
        s = m.get("startIndex") or m.get("start")
        e = m.get("endIndex") or m.get("end")
        if m_type and m_text:
            span_obj = {"label": m_type, "text": m_text}
            if s is not None and e is not None:
                span_obj["start"] = int(s)
                span_obj["end"] = int(e)
            spans.append(span_obj)

    return {
        "doc_id": doc_id,
        "text": text,
        "label": label,
        "spans": spans,
        "subreddit": row.get("subreddit", ""),
    }


# --- Artifact Generation ---


def generate_priors(rows: List[Dict]) -> Dict[str, Any]:
    """Calculates label distribution and conflicts."""
    s1_label_counts = Counter()
    s2_label_counts = Counter()
    conflict_pairs = Counter()

    for r in rows:
        if r["label"] in ["conspiracy", "non"]:
            s2_label_counts[r["label"]] += 1
        spans = r["spans"]
        if not spans:
            continue

        for s in spans:
            s1_label_counts[s["label"]] += 1

        valid_spans = [s for s in spans if "start" in s and "end" in s]
        if len(valid_spans) > 1:
            for i, s1 in enumerate(valid_spans):
                for j, s2 in enumerate(valid_spans):
                    if i >= j:
                        continue
                    if s1["start"] < s2["end"] and s2["start"] < s1["end"]:
                        pair = tuple(sorted([s1["label"], s2["label"]]))
                        if pair[0] != pair[1]:
                            conflict_pairs[pair] += 1

    total_spans = sum(s1_label_counts.values()) or 1
    s1_priors = {k: round(v / total_spans, 3) for k, v in s1_label_counts.items()}
    top_conflicts = [f"{k[0]} vs {k[1]}" for k, _ in conflict_pairs.most_common(5)]

    return {
        "s1_priors": s1_priors,
        "s1_conflicts": top_conflicts,
        "s2_class_balance": dict(s2_label_counts),
    }


def generate_lexical_bank(rows: List[Dict], top_k: int = 20) -> Dict[str, List[str]]:
    """
    Extracts the top K most frequent terms for each label.
    Useful for 'Playbook' injection.
    """
    banks = defaultdict(Counter)

    for r in rows:
        for s in r["spans"]:
            # Simple cleaning: lowercase, keep only alpha words > 2 chars
            words = [
                w.lower()
                for w in s["text"].split()
                if w.isalpha() and len(w) > 2 and w.lower() not in STOPWORDS
            ]
            for w in words:
                banks[s["label"]][w] += 1

    output = {}
    for label, counter in banks.items():
        # Get top K terms
        output[label] = [word for word, count in counter.most_common(top_k)]

    return output


# --- LSH Helpers (MinHash) ---
def get_minhash_signature(text: str, num_perm: int = 128) -> List[int]:
    shingles = set()
    words = text.lower().split()
    for i in range(len(words) - 2):
        shingles.add(" ".join(words[i : i + 3]))
    if not shingles:
        return [0] * num_perm
    signature = []
    for i in range(num_perm):
        min_hash = float("inf")
        for shingle in shingles:
            hash_val = int(
                hashlib.md5(f"{shingle}_{i}".encode("utf-8")).hexdigest(), 16
            )
            if hash_val < min_hash:
                min_hash = hash_val
        signature.append(min_hash)
    return signature


def get_lsh_buckets(signature: List[int], bands: int = 8) -> List[str]:
    rows = len(signature) // bands
    buckets = []
    for i in range(bands):
        band = tuple(signature[i * rows : (i + 1) * rows])
        bucket_id = int(hashlib.md5(str(band).encode("utf-8")).hexdigest(), 16)
        buckets.append(f"{i}_{bucket_id}")
    return buckets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default="./", help="Dir with train_rehydrated.jsonl"
    )
    parser.add_argument("--output-root", type=Path, default="./data/rag")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    raw_path = args.data_dir / "train_rehydrated.jsonl"
    if not raw_path.exists():
        logger.error(f"File not found: {raw_path}")
        return

    logger.info(f"Processing {raw_path}...")

    # 1. Load & Dedupe
    unique_rows = []
    seen_hashes = set()

    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                norm = normalize_row(row)

                # Length Filter (New Optimization)
                # Drop extremely short (<50 chars) or huge (>10k chars) docs to reduce noise
                if len(norm["text"]) < 50 or len(norm["text"]) > 10000:
                    continue

                sig = get_minhash_signature(norm["text"])
                buckets = tuple(get_lsh_buckets(sig))
                if buckets in seen_hashes:
                    continue
                seen_hashes.add(buckets)

                unique_rows.append(norm)
            except Exception:
                pass

    logger.info(f"Unique, filtered docs: {len(unique_rows)}")

    # 2. Generate Artifacts
    priors_data = generate_priors(unique_rows)
    with open(args.output_root / "priors.json", "w") as f:
        json.dump(priors_data, f, indent=2)

    lex_bank = generate_lexical_bank(unique_rows)
    with open(args.output_root / "lexical_bank.json", "w") as f:
        json.dump(lex_bank, f, indent=2)

    logger.info("Saved priors.json and lexical_bank.json")

    # 3. Build S1 Dataset (Sorted by improved complexity)
    s1_rows = []
    for r in unique_rows:
        if r["spans"]:
            r["s1_score"] = score_s1_complexity(r["spans"])
            s1_rows.append(r)
    s1_rows.sort(key=lambda x: x["s1_score"], reverse=True)

    with open(args.output_root / "train_clean_s1.jsonl", "w", encoding="utf-8") as f:
        for r in s1_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 4. Build S2 Dataset
    s2_rows = []
    for r in unique_rows:
        if r["label"] in ["conspiracy", "non"]:
            res = score_s2_richness(r["text"], r["label"], r["spans"])
            r.update(res)
            s2_rows.append(r)
    s2_rows.sort(key=lambda x: x["s2_score"], reverse=True)

    with open(args.output_root / "train_clean_s2.jsonl", "w", encoding="utf-8") as f:
        for r in s2_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("Pipeline Complete.")


if __name__ == "__main__":
    main()
