# data_pipeline.py
"""
PsyCoMark Data Processing Pipeline (for official train/dev splits).

This script cleans and verifies the official PsyCoMark train/dev splits.
Its main purpose is to identify and remove any cross-split duplicates (data leakage)
from the training set to ensure robust evaluation.

It takes the raw rehydrated JSONL files as input and produces a versioned
output directory containing:
- Cleaned and leakage-free train/dev splits in JSONL format.
- A report on removed cross-split duplicates.
- Data priors (class weights, length priors) calculated *only* from the clean training data.
- A manifest.json file for provenance.

Example usage:
python data_pipeline.py --data-dir ./data/raw --output-root ./data/derived --seed 42
"""
import os
import sys
import json
import time
import re
import hashlib
import datetime
import argparse
import logging
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple
from rehydrate_data import preprocess  # canonical text preprocessor used by offsets

import numpy as np
import pandas as pd

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# --- Constants ---
ALLOWED_MARKERS = {"Actor", "Action", "Effect", "Victim", "Evidence"}
DOC_LABELS = ["conspiracy", "non", "cant_tell"]

# ==============================================================================
# SECTION 1: CORE UTILITY FUNCTIONS (Unchanged)
# ==============================================================================
# (All functions from the previous script like `load_jsonl`, `build_dataframe`,
# `create_duplicate_components`, etc., remain here. They are unchanged.)


def load_jsonl(p: Path) -> List[Dict]:
    """Loads a JSONL file into a list of dictionaries."""
    data = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    logging.warning(f"Skipping malformed JSON line in {p}")
    return data


def get_doc_id(rec: Dict) -> Optional[str]:
    for k in ("_id", "id", "doc_id", "reddit_id", "submission_id", "source_id"):
        if k in rec:
            return str(rec[k])
    return None


def get_subreddit(rec: Dict) -> Optional[str]:
    for k in ("subreddit", "community", "source_subreddit"):
        if k in rec:
            return rec[k]
    return None


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_text(rec: Dict) -> Optional[str]:
    for k in ("text", "plain_text", "content", "submission_statement", "ss_text"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def normalize_doc_label(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip().lower()
    mapping = {
        "yes": "conspiracy",
        "y": "conspiracy",
        "consp": "conspiracy",
        "conspiracy": "conspiracy",
        "no": "non",
        "n": "non",
        "non": "non",
        "not conspiracy": "non",
        "can't tell": "cant_tell",
        "cant tell": "cant_tell",
        "cant_tell": "cant_tell",
        "uncertain": "cant_tell",
        "unknown": "cant_tell",
    }
    return mapping.get(v, None)


def get_doc_label(rec: Dict) -> Optional[str]:
    for k in (
        "conspiracy",
        "conspiracy_label",
        "binary_label",
        "doc_label",
        "label",
        "is_conspiracy",
    ):
        if k in rec:
            lab = rec[k]
            if isinstance(lab, dict):
                lab = lab.get("label") or lab.get("value")
            return normalize_doc_label(lab)
    return None


def get_markers(rec: Dict) -> List[Dict]:
    container = next(
        (
            rec.get(k)
            for k in ("markers", "spans", "annotations")
            if isinstance(rec.get(k), list)
        ),
        None,
    )
    if container is None:
        return []
    out = []
    aliases = {
        "actors": "Actor",
        "actions": "Action",
        "effects": "Effect",
        "victims": "Victim",
        "evidences": "Evidence",
    }
    for m in container:
        if not isinstance(m, dict):
            continue
        label = m.get("label") or m.get("type") or m.get("name")
        if isinstance(label, dict):
            label = label.get("label") or label.get("value")
        if not label:
            continue
        label_norm = aliases.get(
            str(label).strip().lower(), str(label).strip().capitalize()
        )
        if label_norm not in ALLOWED_MARKERS:
            continue
        try:
            s = m.get("startIndex", m.get("start", m.get("begin")))
            e = m.get("endIndex", m.get("end", m.get("finish")))
            t = m.get("text", m.get("span_text"))
            s = int(s) if s is not None else None
            e = int(e) if e is not None else None
            out.append(
                {
                    "label": label_norm,
                    "start": s,
                    "end": e,
                    "text": t if isinstance(t, str) else None,
                }
            )
        except (ValueError, TypeError):
            continue
    return out


def normalize_for_dup(s: str) -> str:
    return " ".join((s or "").lower().split())


def text_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def tokenize_for_simhash(s: str) -> List[str]:
    return re.findall(r"\w+", s.lower())


def simhash64(tokens: List[str]) -> int:
    v = [0] * 64
    for t in tokens:
        h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
        for i in range(64):
            v[i] += 1 if ((h >> i) & 1) else -1
    out = 0
    for i in range(64):
        if v[i] >= 0:
            out |= 1 << i
    return out


def lsh_buckets(simhash_val: int, bands: int) -> List[Tuple[int, int]]:
    assert 64 % bands == 0
    r = 64 // bands
    return [
        (b, (simhash_val & (((1 << r) - 1) << (b * r))) >> (b * r))
        for b in range(bands)
    ]


class UF:
    def __init__(self, n):
        self.p, self.r = list(range(n)), [0] * n

    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if self.r[ra] < self.r[rb]:
                self.p[ra] = rb
            elif self.r[ra] > self.r[rb]:
                self.p[rb] = ra
            else:
                self.p[rb] = ra
                self.r[ra] += 1


# ==============================================================================
# SECTION 2: HIGH-LEVEL PIPELINE FUNCTIONS (Unchanged)
# ==============================================================================


def build_dataframe(records: List[Dict], name: str) -> pd.DataFrame:
    # (Slight modification to log the name of the dataset being built)
    # ... implementation is identical to before ...
    rows = []
    for rec in records:
        rows.append(
            {
                "doc_id": get_doc_id(rec),
                "subreddit": get_subreddit(rec),
                "text": get_text(rec),
                "doc_label": get_doc_label(rec),
                "markers": get_markers(rec),
            }
        )
    df = pd.DataFrame(rows)
    if df["doc_id"].isna().any():
        df["doc_id"] = df["doc_id"].fillna(
            df.index.map(lambda i: f"{name}_doc_{i:07d}")
        )
    before = len(df)
    df = (
        df.sort_values("doc_id")
        .drop_duplicates("doc_id", keep="first")
        .reset_index(drop=True)
    )
    if len(df) < before:
        logging.info(f"[unique doc_id in {name}] {before} -> {len(df)}")
    return df


def create_duplicate_components(
    df: pd.DataFrame, lsh_bands: int, lsh_ham: int
) -> pd.DataFrame:
    # (This function is unchanged, it's just called on the combined dataset now)
    # ... implementation is identical to before ...
    logging.info("Building duplicate components (exact + near)...")
    docs = df[["doc_id", "text"]].copy()
    docs["norm_text"] = docs["text"].fillna("").apply(normalize_for_dup)
    docs["exact_hash"] = docs["norm_text"].apply(text_hash)
    docs["tokens"] = docs["norm_text"].apply(tokenize_for_simhash)
    docs["simhash64"] = docs["tokens"].apply(simhash64)
    N = len(docs)
    uf = UF(N)
    logging.info("... finding exact duplicate unions")
    for _, idxs in docs.groupby("exact_hash").indices.items():
        if len(idxs) > 1:
            for i in range(1, len(idxs)):
                uf.union(idxs[0], idxs[i])
    logging.info("... finding near duplicate unions via LSH")
    simvals = docs["simhash64"].tolist()
    cand, bucket_map = set(), defaultdict(list)
    for idx, val in enumerate(simvals):
        for key in lsh_buckets(val, bands=lsh_bands):
            bucket_map[key].append(idx)
    for _, idxs in bucket_map.items():
        if len(idxs) > 1:
            idxs.sort()
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    cand.add((idxs[i], idxs[j]))
    for i, j in cand:
        if (int(simvals[i]) ^ int(simvals[j])).bit_count() <= lsh_ham:
            uf.union(i, j)
    docs["dup_comp"] = [uf.find(i) for i in range(N)]
    return docs[["doc_id", "dup_comp"]]


# ==============================================================================
# SECTION 3: MAIN EXECUTION LOGIC (Heavily Modified)
# ==============================================================================


def main(args):
    """Main pipeline execution function."""
    start_time = time.time()
    np.random.seed(args.seed)
    random.seed(args.seed)
    STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTDIR = args.output_root / f"psycomark_official_split_{STAMP}"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Starting pipeline run. Seed={args.seed}, Output Dir: {OUTDIR}")

    # --- 1. Load Official Train and Dev Data ---
    train_rehydrated_path = args.data_dir / "train_rehydrated.jsonl"
    dev_rehydrated_path = args.data_dir / "dev_rehydrated.jsonl"
    if not train_rehydrated_path.exists() or not dev_rehydrated_path.exists():
        logging.error(
            f"Input data not found. Ensure 'train_rehydrated.jsonl' and 'dev_rehydrated.jsonl' are in {args.data_dir}"
        )
        sys.exit(1)

    logging.info(f"Loading train data from {train_rehydrated_path}...")
    train_records = load_jsonl(train_rehydrated_path)
    train_df = build_dataframe(train_records, name="train")

    logging.info(f"Loading dev data from {dev_rehydrated_path}...")
    dev_records = load_jsonl(dev_rehydrated_path)
    dev_df = build_dataframe(dev_records, name="dev")

    for df, nm in [(train_df, "train"), (dev_df, "dev")]:
        df["text"] = df["text"].fillna("").map(preprocess)
        n = df["text"].str.len()
        assert (n >= 160).mean() > 0.98, f"{nm} length gate drift: too many <160 chars"
        assert (
            n <= 1000
        ).mean() > 0.98, f"{nm} length gate drift: too many >1000 chars"

    # --- 2. Identify Cross-Split Duplicates ---
    logging.info("Auditing official split for cross-set leakage...")
    train_df["split_source"] = "train"
    dev_df["split_source"] = "dev"
    all_df = pd.concat([train_df, dev_df], ignore_index=True)

    dup_components_df = create_duplicate_components(
        all_df, args.lsh_bands, args.lsh_ham
    )
    df_with_comps = all_df.merge(dup_components_df, on="doc_id")

    # Find components that span both train and dev
    comp_splits = df_with_comps.groupby("dup_comp")["split_source"].unique()
    leaky_comps = comp_splits[comp_splits.apply(lambda x: len(x) > 1)].index

    leaky_train_docs_ids = set(
        df_with_comps[
            (df_with_comps["dup_comp"].isin(leaky_comps))
            & (df_with_comps["split_source"] == "train")
        ]["doc_id"]
    )

    leak_report = df_with_comps[df_with_comps["doc_id"].isin(leaky_train_docs_ids)][
        ["doc_id", "dup_comp", "text"]
    ]
    if not leak_report.empty:
        leak_report.to_csv(OUTDIR / "leakage_removed_train.csv", index=False)

    dev_dup = df_with_comps[df_with_comps["split_source"] == "dev"]
    dev_internal_dup = len(dev_dup) - dev_dup["dup_comp"].nunique()
    logging.info(f"Dev internal duplicates (kept as-is): {dev_internal_dup}")

    if leaky_train_docs_ids:
        logging.warning(
            f"Found {len(leaky_train_docs_ids)} train documents that are duplicates of dev documents. Removing them from the training set to prevent data leakage."
        )
    else:
        logging.info("Verification successful: No cross-split duplicates found.")

    # --- 3. Deduplicate Training Set Internally ---
    logging.info("Deduplicating training set internally...")
    # Consider only the training docs that didn't leak
    clean_train_candidates = df_with_comps[
        (df_with_comps["split_source"] == "train")
        & (~df_with_comps["doc_id"].isin(leaky_train_docs_ids))
    ].copy()

    # For each duplicate component within the clean train candidates, keep only the longest one
    clean_train_candidates["n_markers"] = clean_train_candidates["markers"].map(
        lambda x: len(x or [])
    )
    clean_train_candidates["text_len"] = clean_train_candidates["text"].str.len()
    kept_train_docs = clean_train_candidates.sort_values(
        ["n_markers", "text_len", "doc_id"], ascending=[False, False, True]
    ).drop_duplicates(subset=["dup_comp"], keep="first")

    kept_train_ids = set(kept_train_docs["doc_id"])

    # --- 4. Create Final DataFrames ---
    final_train_df = train_df[train_df["doc_id"].isin(kept_train_ids)].drop(
        columns=["split_source"]
    )
    final_dev_df = dev_df.drop(columns=["split_source"])  # Dev set is untouched

    # doc-level count of removed within-train duplicates
    num_within_train_removed_docs = len(clean_train_candidates) - len(kept_train_docs)

    # (optional) how many components had >1 train doc (i.e., had internal dups)
    num_dup_components = (clean_train_candidates.groupby("dup_comp").size() > 1).sum()

    logging.info(
        f"Removed {num_within_train_removed_docs} internal duplicate documents from the training set "
        f"(across {num_dup_components} duplicate components)."
    )

    logging.info(f"Final sizes: Train={len(final_train_df)}, Dev={len(final_dev_df)}")

    # --- 5. Export Artifacts ---
    logging.info("Exporting final artifacts...")
    train_path = OUTDIR / "train.jsonl"
    dev_path = OUTDIR / "dev.jsonl"
    final_train_df.to_json(train_path, orient="records", lines=True)
    final_dev_df.to_json(dev_path, orient="records", lines=True)

    logging.info("Exporting data priors (from clean train set only)...")
    lambda_len = 0.15
    span_rows = []
    for _, r in final_train_df.iterrows():
        for m in r["markers"] or []:
            s, e, lab = m.get("start"), m.get("end"), m.get("label")
            if (
                isinstance(s, int)
                and isinstance(e, int)
                and e > s
                and lab in ALLOWED_MARKERS
            ):
                span_rows.append({"label": lab, "char_len": e - s})

    sp = pd.DataFrame(span_rows)
    all_labels = sorted(ALLOWED_MARKERS)
    if not sp.empty:
        span_counts = sp["label"].value_counts().reindex(all_labels, fill_value=0)
        class_weights = (
            (1.0 / np.sqrt(span_counts.replace(0, np.nan))).fillna(0.0).to_dict()
        )
        q90 = (
            sp.groupby("label")["char_len"]
            .quantile(0.9)
            .reindex(all_labels, fill_value=0)
            .round()
            .astype(int)
            .to_dict()
        )
        length_priors = {"q90_per_label": q90, "lambda": lambda_len}
    else:
        class_weights = {k: 0.0 for k in all_labels}
        length_priors = {
            "q90_per_label": {k: 0 for k in all_labels},
            "lambda": lambda_len,
        }

    (
        json.dumps(class_weights),
        json.dumps(length_priors),
    )  # no-op to surface errors early

    with open(OUTDIR / "class_weights.json", "w") as f:
        json.dump(class_weights, f, indent=2)
    with open(OUTDIR / "length_priors.json", "w") as f:
        json.dump(length_priors, f, indent=2)

    manifest = {
        "created_utc": int(time.time()),
        "stamp": STAMP,
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "notes": "Processed official train/dev splits. Cleaned cross-split leakage and internal train duplicates.",
        "sizes": {
            "train_raw": len(train_df),
            "dev_raw": len(dev_df),
            "train_final": len(final_train_df),
            "dev_final": len(final_dev_df),
            "train_docs_removed_leakage": len(leaky_train_docs_ids),
            "num_within_train_removed_docs": num_within_train_removed_docs,
        },
        "lsh": {"bands": args.lsh_bands, "ham": args.lsh_ham},
        "inputs": {
            "train_path": str(train_rehydrated_path),
            "dev_path": str(dev_rehydrated_path),
            "train_sha256": sha256_file(train_rehydrated_path),
            "dev_sha256": sha256_file(dev_rehydrated_path),
        },
        "artifacts": {
            p.name: str(p.resolve()) for p in OUTDIR.glob("*") if p.is_file()
        },
    }
    with open(OUTDIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    latest_ptr = Path(args.output_root) / "psycomark_latest.txt"
    latest_ptr.parent.mkdir(parents=True, exist_ok=True)
    latest_ptr.write_text(str(OUTDIR.resolve()))

    logging.info(
        f"Pipeline finished successfully in {time.time() - start_time:.2f} seconds."
    )
    logging.info(f"Artifacts saved to: {OUTDIR.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PsyCoMark Data Processing Pipeline for official splits."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default="./",
        help="Directory containing raw data ('train_rehydrated.jsonl' and 'dev_rehydrated.jsonl')",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default="./data/derived",
        help="Root directory to save versioned outputs",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--lsh-bands",
        type=int,
        default=8,
        help="LSH bands for near-duplicate detection",
    )
    parser.add_argument(
        "--lsh-ham",
        type=int,
        default=4,
        help="LSH Hamming distance threshold for near-duplicates",
    )

    args = parser.parse_args()
    main(args)
