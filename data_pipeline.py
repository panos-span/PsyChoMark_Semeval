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
from collections import defaultdict, Counter
from typing import Dict, List, Optional
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
# === Cant_tell policy & CV folds helpers ===
ALLOWED_S2_LABELS = {"conspiracy", "non"}  # cant_tell excluded from S2 training

VALID_LSH_BANDS = (1, 2, 4, 8, 16, 32, 64)


def normalize_lsh_bands(b: int) -> int:
    """Return the nearest valid divisor of 64 for LSH bands."""
    if b in VALID_LSH_BANDS:
        return b
    # choose nearest by absolute distance, tie -> smaller
    return sorted(VALID_LSH_BANDS, key=lambda x: (abs(x - b), x))[0]


# ==============================================================================
# SECTION 1: CORE UTILITY FUNCTIONS
# ==============================================================================
def _filter_docclf(df):
    """Return a doc-classification view with cant_tell removed."""
    keep = df["doc_label"].isin(ALLOWED_S2_LABELS)
    # keep minimal columns + markers if you want later multi-tasking
    cols = [
        c
        for c in ["doc_id", "text", "doc_label", "markers", "subreddit"]
        if c in df.columns
    ]
    return df.loc[keep, cols].copy()


def _stratified_component_folds(df_with_comp, k=5, seed=42):
    """
    Assigns fold id ∈ [0..k-1] per dup_comp, stratified by majority doc_label.
    Expects columns: dup_comp, doc_label. Returns {dup_comp: fold}.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    # component -> majority label
    comp_lab = (
        df_with_comp.groupby("dup_comp")["doc_label"]
        .agg(lambda s: s.value_counts().idxmax())
        .reset_index()
    )
    # buckets per label
    buckets = {}
    for lab, g in comp_lab.groupby("doc_label", dropna=False):
        comps = g["dup_comp"].tolist()
        rng.shuffle(comps)
        buckets[lab] = comps

    # round-robin assign per bucket
    comp2fold = {}
    for lab, comps in buckets.items():
        for i, comp in enumerate(comps):
            comp2fold[comp] = i % k
    return comp2fold


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


def lsh_buckets(simhash_val: int, bands: int = 16):
    # instead of: assert 64 % bands == 0
    if 64 % bands != 0:
        # auto-normalize defensively if called directly
        nb = normalize_lsh_bands(bands)
        if nb != bands:
            bands = nb
    r = 64 // bands  # bits per band
    mask = (1 << r) - 1
    for b in range(bands):
        yield (b, (simhash_val >> (b * r)) & mask)


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
    """
    Builds a DataFrame from raw records, ensuring unique doc_ids.
    Fills missing doc_ids with synthetic ones based on index.

    Args:
        records: List of record dictionaries.
        name: Name prefix for synthetic doc_ids if needed.
    Returns:
        A pandas DataFrame with columns: doc_id, subreddit, text, doc_label, markers.
    """
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
    """
    Identifies duplicate components (exact + near duplicates) in the DataFrame.
    Uses exact text matching and SimHash LSH for near-duplicates.
    Args:
        df: DataFrame with at least 'doc_id' and 'text' columns.
        lsh_bands: Number of bands for LSH.
        lsh_ham: Hamming distance threshold for near-duplicates.
    Returns:
        DataFrame with 'doc_id' and 'dup_comp' (duplicate component ID) columns.
    """
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


def offset_integrity(df, text_col="text", markers_col="markers", sample_limit=200):
    """
    Returns stats + a small sample of problems so we can inspect drift.
    Stats:
      - total_spans
      - ok_bounds: start/end within [0, len(text)]
      - have_text: markers that include their literal text
      - ok_exact: exact match between text[start:end] and marker['text'] (if provided)
    """
    stats = dict(total_spans=0, ok_bounds=0, have_text=0, ok_exact=0)
    issues = []
    for _, row in df.iterrows():
        t = row.get(text_col) or ""
        tlen = len(t)
        spans = row.get(markers_col) or []
        for m in spans:
            s, e = m.get("start"), m.get("end")
            if not (isinstance(s, int) and isinstance(e, int)):
                continue
            stats["total_spans"] += 1
            if 0 <= s < e <= tlen:
                stats["ok_bounds"] += 1
                gold_txt = m.get("text")
                if gold_txt:
                    stats["have_text"] += 1
                    if t[s:e] == gold_txt:
                        stats["ok_exact"] += 1
                    elif len(issues) < sample_limit:
                        issues.append(
                            {
                                "doc_id": row.get("doc_id"),
                                "label": m.get("label"),
                                "start": s,
                                "end": e,
                                "expected": gold_txt,
                                "found": t[s:e],
                            }
                        )
            elif len(issues) < sample_limit:
                issues.append(
                    {
                        "doc_id": row.get("doc_id"),
                        "label": m.get("label"),
                        "start": s,
                        "end": e,
                        "reason": "out_of_bounds",
                        "text_len": tlen,
                    }
                )

    # derived rates (as floats)
    def r(num, den):
        return float(num) / float(den) if den else 0.0

    summary = {
        **stats,
        "rate_ok_bounds": r(stats["ok_bounds"], stats["total_spans"]),
        "rate_ok_exact_of_with_text": (
            r(stats["ok_exact"], stats["have_text"]) if stats["have_text"] else None
        ),
    }
    return summary, issues


def choose_rep_label_aware(df_cluster):
    """
    df_cluster: DataFrame of duplicate component (same dup_comp).
    Prefers: component-majority label -> more markers -> moderately longer text (avoid extremes).
    Returns the INDEX of the chosen row.
    """
    labels = df_cluster["doc_label"].fillna("non").tolist()
    majority_label, _ = Counter(labels).most_common(1)[0]

    def score_row(r):
        # normalize features
        n_markers = len(r.get("markers") or [])
        text_len = len(r.get("text") or "")
        # light penalty for extreme length (beyond 95th percentile later if available)
        return (
            10.0 * (1 if r.get("doc_label") == majority_label else 0)
            + 2.0 * n_markers
            + 0.001 * text_len
        )

    scores = df_cluster.apply(score_row, axis=1)
    return scores.idxmax()


def fit_lognorm_params(x: List[float]) -> Dict[str, float]:
    """
    Fits log-normal parameters (mu, sigma) to the positive values in x.
    x: array-like of positive values
    Returns: dict with 'mu' and 'sigma'
    """
    x = np.asarray([v for v in x if v > 0], dtype=float)
    if len(x) < 5:
        return {"mu": 0.0, "sigma": 1.0}
    lx = np.log(x)
    return {"mu": float(lx.mean()), "sigma": float(lx.std(ddof=1) or 1.0)}


def fit_beta_params(pos: List[float]) -> Dict[str, float]:
    """
    Fits beta distribution parameters (alpha, beta) to the values in pos (0 < pos < 1).
    pos: array-like of values in (0, 1)
    Returns: dict with 'alpha' and 'beta'
    """
    pos = np.asarray([v for v in pos if 0.0 <= v <= 1.0], dtype=float)
    if len(pos) < 5:
        return {"alpha": 1.0, "beta": 1.0}
    m = pos.mean()
    v = pos.var(ddof=1) or 1e-6
    # method-of-moments
    k = (m * (1 - m) / v) - 1
    alpha = max(m * k, 0.5)
    beta = max((1 - m) * k, 0.5)
    return {"alpha": float(alpha), "beta": float(beta)}


# ==============================================================================
# SECTION 3: MAIN EXECUTION LOGIC (Heavily Modified)
# ==============================================================================


def main(args):
    """Main pipeline execution function."""
    start_time = time.time()
    np.random.seed(args.seed)
    random.seed(args.seed)
    # sanity for LSH bands
    orig_bands = args.lsh_bands
    args.lsh_bands = normalize_lsh_bands(args.lsh_bands)
    if args.lsh_bands != orig_bands:
        logging.warning(
            f"lsh_bands={orig_bands} is invalid (must divide 64). "
            f"Using normalized value: {args.lsh_bands}."
        )
    # sanity for ham threshold
    if not (0 <= args.lsh_ham <= 64):
        logging.warning(f"lsh_ham={args.lsh_ham} out of range [0,64]. Clamping.")
        args.lsh_ham = max(0, min(64, args.lsh_ham))

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
        if not args.skip_preprocess:
            logging.info(f"Applying text preprocessing to {nm} set...")
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

    # Choose representative per duplicate component
    kept_idx = []
    audit_rows = []

    for comp_id, df_comp in clean_train_candidates.groupby("dup_comp", sort=False):
        if len(df_comp) == 1:
            rep_idx = df_comp.index[0]
        else:
            rep_idx = choose_rep_label_aware(df_comp)
        kept_idx.append(rep_idx)

        # Audit rows (kept vs dropped) for sanity checks
        for idx, row in df_comp.iterrows():
            audit_rows.append(
                {
                    "dup_comp": comp_id,
                    "doc_id": row["doc_id"],
                    "doc_label": row.get("doc_label", "non"),
                    "n_markers": row["n_markers"],
                    "text_len": row["text_len"],
                    "kept": bool(idx == rep_idx),
                }
            )

    kept_train_docs = clean_train_candidates.loc[kept_idx].copy()
    kept_train_ids = set(kept_train_docs["doc_id"])

    # Optional: write an audit file to inspect the decisions
    pd.DataFrame(audit_rows).sort_values(
        ["dup_comp", "kept"], ascending=[True, False]
    ).to_csv(OUTDIR / "dup_audit.csv", index=False)

    # Old heuristic: keep the longest text in each dup_comp (unstable)
    # kept_train_docs = clean_train_candidates.sort_values(
    #     ["n_markers", "text_len", "doc_id"], ascending=[False, False, True]
    # ).drop_duplicates(subset=["dup_comp"], keep="first")

    # kept_train_ids = set(kept_train_docs["doc_id"])

    # --- 4. Create Final DataFrames ---
    final_train_df = train_df[train_df["doc_id"].isin(kept_train_ids)].drop(
        columns=["split_source"]
    )
    final_dev_df = dev_df.drop(columns=["split_source"])  # Dev set is untouched

    # doc-level count of removed within-train duplicates
    num_within_train_removed_docs = len(clean_train_candidates) - len(kept_train_docs)

    # (optional) how many components had >1 train doc (i.e., had internal dups)
    num_dup_components = (clean_train_candidates.groupby("dup_comp").size() > 1).sum()

    # Map doc_id -> dup_comp so we can build CV folds on final train
    dup_map = df_with_comps[["doc_id", "dup_comp"]].drop_duplicates()
    final_train_df = final_train_df.merge(dup_map, on="doc_id", how="left")

    logging.info(
        f"Removed {num_within_train_removed_docs} internal duplicate documents from the training set "
        f"(across {num_dup_components} duplicate components)."
    )

    logging.info(f"Final sizes: Train={len(final_train_df)}, Dev={len(final_dev_df)}")

    # Set _id field for compatibility
    final_train_df["_id"] = final_train_df["doc_id"]
    final_dev_df["_id"] = final_dev_df["doc_id"]

    # --- 5. Export Artifacts ---
    logging.info("Exporting final artifacts...")
    train_path = OUTDIR / "train.jsonl"
    dev_path = OUTDIR / "dev.jsonl"
    final_train_df.to_json(train_path, orient="records", lines=True)
    final_dev_df.to_json(dev_path, orient="records", lines=True)

    # --- S2 doc-classification views (cant_tell excluded) ---
    train_docclf = _filter_docclf(final_train_df)
    dev_docclf = _filter_docclf(final_dev_df)

    train_docclf_path = OUTDIR / "train_docclf.jsonl"
    dev_docclf_path = OUTDIR / "dev_docclf.jsonl"
    train_docclf.to_json(
        train_docclf_path, orient="records", lines=True, force_ascii=False
    )
    dev_docclf.to_json(dev_docclf_path, orient="records", lines=True, force_ascii=False)

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

        # collect lengths & normalized starts
        per_label = {lab: {"lens": [], "pos": []} for lab in ALLOWED_MARKERS}
        for _, r in final_train_df.iterrows():
            tlen = len(r["text"] or "")
            for m in r["markers"] or []:
                s, e, lab = m.get("start"), m.get("end"), m.get("label")
                if (
                    isinstance(s, int)
                    and isinstance(e, int)
                    and e > s
                    and lab in ALLOWED_MARKERS
                    and tlen > 0
                ):
                    per_label[lab]["lens"].append(e - s)
                    per_label[lab]["pos"].append(s / tlen)

        length_position_priors = {}
        for lab, d in per_label.items():
            length_position_priors[lab] = {
                "length_lognorm": fit_lognorm_params(d["lens"]),
                "start_beta": fit_beta_params(d["pos"]),
                "q90_len": int(np.quantile(d["lens"], 0.9)) if d["lens"] else 0,
                "coverage_rate": float(
                    len(d["lens"]) / max(1, len(final_train_df))
                ),  # docs with ≥1 span of lab
                "avg_spans_per_doc": (
                    float(np.mean(pd.Series(d["lens"]).groupby(level=0).count()))
                    if d["lens"]
                    else 0.0
                ),
            }

    else:
        class_weights = {k: 0.0 for k in all_labels}
        length_position_priors = {
            "q90_per_label": {k: 0 for k in all_labels},
            "lambda": lambda_len,
        }

    (
        json.dumps(class_weights),
        json.dumps(length_position_priors),
    )  # no-op to surface errors early

    with open(OUTDIR / "class_weights.json", "w") as f:
        json.dump(class_weights, f, indent=2)
    with open(OUTDIR / "length_position_priors.json", "w") as f:
        json.dump(length_position_priors, f, indent=2)
    # with open(OUTDIR / "length_priors.json", "w") as f:
    #    json.dump(length_priors, f, indent=2)

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
        "lsh": {"bands": int(args.lsh_bands), "ham": int(args.lsh_ham)},
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

    # Cant_tell policy details
    manifest.setdefault("policy", {})
    manifest["policy"]["cant_tell"] = {
        "s2_training_excluded": True,
        "fewshot_negative_allowed": True,
        "train_filtered_out": int((final_train_df["doc_label"] == "cant_tell").sum()),
        "dev_filtered_out": int((final_dev_df["doc_label"] == "cant_tell").sum()),
    }
    manifest["artifacts"]["train_docclf.jsonl"] = str(train_docclf_path.resolve())
    manifest["artifacts"]["dev_docclf.jsonl"] = str(dev_docclf_path.resolve())

    manifest["artifacts"]["length_position_priors.json"] = str(
        (OUTDIR / "length_position_priors.json").resolve()
    )

    # --- 5-fold CV by duplicate components (prevents leakage) ---
    # Use final_train_df (one doc per dup_comp after internal dedup)
    if "dup_comp" in final_train_df.columns:
        comp2fold = _stratified_component_folds(
            final_train_df[["dup_comp", "doc_label"]], k=5, seed=args.seed
        )
        folds_df = final_train_df[["doc_id", "dup_comp", "doc_label"]].copy()
        folds_df["fold"] = folds_df["dup_comp"].map(comp2fold).astype(int)

        folds_path = OUTDIR / "folds.jsonl"
        folds_df.to_json(folds_path, orient="records", lines=True, force_ascii=False)

        # summary counts per fold/label
        folds_summary = (
            folds_df.groupby(["fold", "doc_label"])["doc_id"]
            .count()
            .unstack(fill_value=0)
            .to_dict(orient="index")
        )
        with open(OUTDIR / "folds_summary.json", "w", encoding="utf-8") as f:
            json.dump(folds_summary, f, indent=2)

        manifest["artifacts"]["folds.jsonl"] = str(folds_path.resolve())
        manifest["artifacts"]["folds_summary.json"] = str(
            (OUTDIR / "folds_summary.json").resolve()
        )
    else:
        logging.warning("dup_comp missing on final_train_df; CV folds not created.")

    # Drift report: label and subreddit distribution before vs after deduplication
    logging.info("Generating deduplication drift report...")

    def value_counts_norm(s):
        vc = s.value_counts(dropna=False)
        return (vc / vc.sum()).to_dict()

    dedup_drift = {
        "before": {
            "label": value_counts_norm(
                clean_train_candidates["doc_label"].fillna("non")
            ),
            "subreddit": value_counts_norm(
                clean_train_candidates.get("subreddit", pd.Series()).fillna("NA")
            ),
        },
        "after": {
            "label": value_counts_norm(final_train_df["doc_label"].fillna("non")),
            "subreddit": value_counts_norm(
                final_train_df.get("subreddit", pd.Series()).fillna("NA")
            ),
        },
    }
    manifest["dedup_drift"] = dedup_drift

    # Save drift reports as CSV for easy inspection
    pd.DataFrame(
        [
            {"phase": "before", "label": k, "p": v}
            for k, v in dedup_drift["before"]["label"].items()
        ]
        + [
            {"phase": "after", "label": k, "p": v}
            for k, v in dedup_drift["after"]["label"].items()
        ]
    ).to_csv(OUTDIR / "dedup_drift_label.csv", index=False)

    # --- 6. Offset Integrity Check ---
    logging.info("Performing offset integrity checks...")
    train_off, train_issues = offset_integrity(train_df)
    dev_off, dev_issues = offset_integrity(dev_df)

    manifest["offset_integrity"] = {
        "train": train_off,
        "dev": dev_off,
        "skip_preprocess": args.skip_preprocess,
    }

    # Save a small sample for quick inspection
    with open(OUTDIR / "offset_issues_sample_train.json", "w", encoding="utf-8") as f:
        json.dump(train_issues, f, ensure_ascii=False, indent=2)
    with open(OUTDIR / "offset_issues_sample_dev.json", "w", encoding="utf-8") as f:
        json.dump(dev_issues, f, ensure_ascii=False, indent=2)

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
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        default=True,
        help="If set (default), assumes input JSONL is already preprocessed/rehydrated and will NOT re-apply preprocess().",
    )

    args = parser.parse_args()
    main(args)
