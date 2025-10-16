#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_and_insights.py
- Loads latest derived split (via pointer)
- Produces:
  * overlap_pair_stats.json
  * overlap_pair_stats_ci.json (doc-level bootstrap CIs)
  * boundary_context.json
  * first_occurrence_cdf.csv
  * label_coverage.json
  * mean_iou_matrix.png
  * span_position_analysis.png
  * absolutist_language_rate_by_doc_label.png
  * hedges_rate_by_doc_label.png
  * absolutist_hedge_summary.csv
  * absolutist_hedge_by_subreddit.csv (if enough n)
  * lexical_effect_sizes.csv
  * hard_examples.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Optional deps (present in your env)
from scipy.stats import entropy, mannwhitneyu

# -----------------------------
# Config
# -----------------------------
ALLOWED_MARKERS = {"Actor", "Action", "Effect", "Victim", "Evidence"}
SEED = 42
rng = np.random.default_rng(SEED)

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
    "no doubts",
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
]

# -----------------------------
# Utils
# -----------------------------


def find_latest_dir(pointer: Path) -> Path:
    if not pointer.exists():
        raise FileNotFoundError(
            f"Pointer file not found: {pointer}. Run data_pipeline.py first."
        )
    d = Path(pointer.read_text().strip())
    if not d.exists():
        raise FileNotFoundError(f"Derived folder listed in pointer not found: {d}")
    return d


def alias_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure an _id column exists (alias of doc_id if needed)."""
    df = df.copy()
    if "_id" not in df.columns and "doc_id" in df.columns:
        df["_id"] = df["doc_id"]
    return df


def safe_markers(row) -> List[dict]:
    m = row.get("markers", [])
    return m if isinstance(m, list) else []


def iou_char(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    s1, e1 = a
    s2, e2 = b
    inter = max(0, min(e1, e2) - max(s1, s2))
    if inter <= 0:
        return 0.0
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / union if union > 0 else 0.0


def word_tokens(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", s or "")


def make_word_boundary_regex(terms: List[str]):
    return re.compile(r"(?i)(?<!\w)(" + "|".join(map(re.escape, terms)) + r")(?!\w)")


def bh_correct(pvals: np.ndarray) -> np.ndarray:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    cummin = 1.0
    for i, idx in enumerate(order[::-1], start=1):
        rank = m - i + 1
        val = pvals[idx] * m / rank
        cummin = min(cummin, val)
        adj[idx] = cummin
    return adj


def cliffs_delta(a: List[float], b: List[float]) -> float:
    gt = lt = 0
    for x in a:
        for y in b:
            if x > y:
                gt += 1
            elif x < y:
                lt += 1
    n1, n2 = len(a), len(b)
    return (gt - lt) / (n1 * n2) if n1 and n2 else 0.0


# -----------------------------
# Core analyses
# -----------------------------


def spans_from_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        text = r.get("text") or ""
        for m in safe_markers(r):
            lab = m.get("label")
            s, e = m.get("start"), m.get("end")
            if lab in ALLOWED_MARKERS and isinstance(s, int) and isinstance(e, int):
                rows.append(
                    {
                        "doc_id": r.get("doc_id"),
                        "label": lab,
                        "start": int(s),
                        "end": int(e),
                        "text_len": len(text),
                    }
                )
    return pd.DataFrame(rows)


def overlap_stats_for_pairs(spans_df: pd.DataFrame) -> Dict[str, dict]:
    per_pair = defaultdict(list)
    for doc_id, g in spans_df.groupby("doc_id"):
        spans = g[["label", "start", "end"]].to_records(index=False)
        spans = [(str(l), int(s), int(e)) for (l, s, e) in spans]
        for i in range(len(spans)):
            lab1, s1, e1 = spans[i]
            for j in range(i + 1, len(spans)):
                lab2, s2, e2 = spans[j]
                if lab1 == lab2:
                    continue
                iou = iou_char((s1, e1), (s2, e2))
                if iou == 0:
                    continue
                pair = tuple(sorted([lab1, lab2]))
                starts_first = 1 if s1 < s2 else 0
                contain = int(s1 <= s2 and e1 >= e2) or int(s2 <= s1 and e2 >= e1)
                per_pair[pair].append(
                    {
                        "iou": iou,
                        "a_starts_first": starts_first,
                        "contain": contain,
                    }
                )
    out = {}
    for pair, rows in per_pair.items():
        xs = [r["iou"] for r in rows]
        out["/".join(pair)] = {
            "n": len(xs),
            "mean_iou": float(np.mean(xs)),
            "median_iou": float(np.median(xs)),
            "iou@0.1": float(np.mean([x >= 0.1 for x in xs])),
            "iou@0.5": float(np.mean([x >= 0.5 for x in xs])),
            "starts_first_rate": float(np.mean([r["a_starts_first"] for r in rows])),
            "contain_rate": float(np.mean([r["contain"] for r in rows])),
        }
    return out


def pairwise_bootstrap_ci(spans_df: pd.DataFrame, out_path: Path, B=1000):
    # Gather IoUs per (pair, doc) for doc-level bootstrap
    per_pair_per_doc = defaultdict(list)
    for doc_id, g in spans_df.groupby("doc_id"):
        recs = [
            (str(L), int(S), int(E))
            for (L, S, E) in g[["label", "start", "end"]].to_records(index=False)
        ]
        pair_to_ious = defaultdict(list)
        for i in range(len(recs)):
            lab1, s1, e1 = recs[i]
            for j in range(i + 1, len(recs)):
                lab2, s2, e2 = recs[j]
                if lab1 == lab2:
                    continue
                v = iou_char((s1, e1), (s2, e2))
                if v > 0:
                    pair = tuple(sorted([lab1, lab2]))
                    pair_to_ious[pair].append(v)
        for pair, ious in pair_to_ious.items():
            per_pair_per_doc[pair].append({"doc_id": doc_id, "ious": ious})

    def bootstrap_ci(vals: List[float], B=B, alpha=0.05):
        if not vals:
            return (None, None)
        V = np.asarray(vals, float)
        n = len(V)
        stats = []
        for _ in range(B):
            s = rng.choice(V, size=n, replace=True)
            stats.append(float(np.mean(s)))
        lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
        return float(lo), float(hi)

    out = {}
    for pair, rows in per_pair_per_doc.items():
        doc_means = [np.mean(r["ious"]) for r in rows]
        doc_rate01 = [np.mean([x >= 0.1 for x in r["ious"]]) for r in rows]
        doc_rate05 = [np.mean([x >= 0.5 for x in r["ious"]]) for r in rows]
        out["/".join(pair)] = {
            "n_docs": len(rows),
            "mean_iou": float(np.mean(doc_means)),
            "mean_iou_ci": list(bootstrap_ci(doc_means)),
            "iou@0.1": float(np.mean(doc_rate01)),
            "iou@0.1_ci": list(bootstrap_ci(doc_rate01)),
            "iou@0.5": float(np.mean(doc_rate05)),
            "iou@0.5_ci": list(bootstrap_ci(doc_rate05)),
        }

    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def extract_boundary_context(
    spans_df: pd.DataFrame, texts: Dict[str, str], out_path: Path, k_chars=5, top=25
):
    ctx = {
        lab: {
            "before_chars": Counter(),
            "after_chars": Counter(),
            "before_1w": Counter(),
            "after_1w": Counter(),
            "before_2w": Counter(),
            "after_2w": Counter(),
        }
        for lab in ALLOWED_MARKERS
    }
    for _, r in spans_df.iterrows():
        t = texts.get(r["doc_id"], "") or ""
        s, e, L = int(r["start"]), int(r["end"]), r["label"]
        before_c = t[max(0, s - k_chars) : s]
        after_c = t[e : min(len(t), e + k_chars)]
        if before_c:
            ctx[L]["before_chars"][before_c] += 1
        if after_c:
            ctx[L]["after_chars"][after_c] += 1
        before_w = word_tokens(t[:s])[-2:]
        after_w = word_tokens(t[e:])[:2]
        if before_w:
            ctx[L]["before_1w"][" ".join(before_w[-1:])] += 1
            if len(before_w) >= 2:
                ctx[L]["before_2w"][" ".join(before_w[-2:])] += 1
        if after_w:
            ctx[L]["after_1w"][" ".join(after_w[:1])] += 1
            if len(after_w) >= 2:
                ctx[L]["after_2w"][" ".join(after_w[:2])] += 1

    bc = {}
    for lab, d in ctx.items():
        bc[lab] = {
            "before_chars": [w for w, _ in d["before_chars"].most_common(top)],
            "after_chars": [w for w, _ in d["after_chars"].most_common(top)],
            "before_1w": [w for w, _ in d["before_1w"].most_common(top)],
            "after_1w": [w for w, _ in d["after_1w"].most_common(top)],
            "before_2w": [w for w, _ in d["before_2w"].most_common(top)],
            "after_2w": [w for w, _ in d["after_2w"].most_common(top)],
        }
    out_path.write_text(json.dumps(bc, indent=2), encoding="utf-8")
    return bc


def mean_iou_matrix_plot(df_all: pd.DataFrame, out_png: Path):
    labels_sorted = sorted(list(ALLOWED_MARKERS))
    sum_iou, cnt_iou = defaultdict(float), defaultdict(int)
    for _, row in df_all.iterrows():
        spans = sorted(
            [
                (m["label"], m["start"], m["end"])
                for m in safe_markers(row)
                if m.get("label") in ALLOWED_MARKERS and isinstance(m.get("start"), int)
            ],
            key=lambda x: (x[1], x[2]),
        )
        for i in range(len(spans)):
            li, si, ei = spans[i]
            for j in range(i + 1, len(spans)):
                lj, sj, ej = spans[j]
                if sj >= ei:
                    break
                v = iou_char((si, ei), (sj, ej))
                if v > 0:
                    a, b = sorted([li, lj])
                    key = (a, b)
                    sum_iou[key] += v
                    cnt_iou[key] += 1

    mat = np.zeros((len(labels_sorted), len(labels_sorted)), dtype=float)
    for i, a in enumerate(labels_sorted):
        for j, b in enumerate(labels_sorted):
            if i == j:
                mat[i, j] = 1.0
            else:
                key = tuple(sorted([a, b]))
                if cnt_iou[key] > 0:
                    mat[i, j] = sum_iou[key] / cnt_iou[key]

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        mat,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        xticklabels=labels_sorted,
        yticklabels=labels_sorted,
        vmin=0,
        vmax=1,
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.title("Mean IoU of Overlapping Spans", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", dpi=200)
    plt.close()


def first_occurrence_and_coverage(
    spans_df: pd.DataFrame, df_all: pd.DataFrame, out_cdf_csv: Path, out_cov_json: Path
):
    # coverage: fraction of docs that contain each label at least once
    per_doc_counts = (
        spans_df.groupby(["doc_id", "label"]).size().rename("n").reset_index()
    )
    coverage = (
        per_doc_counts["label"].value_counts() / per_doc_counts["doc_id"].nunique()
    ).to_dict()

    # first occurrence CDFs
    first_pos = []
    lens = df_all.set_index("doc_id")["text"].str.len().fillna(0).to_dict()
    for (doc, lab), g in spans_df.groupby(["doc_id", "label"]):
        tlen = max(1, int(lens.get(doc, 0)))
        pos = int(g["start"].min()) / tlen
        first_pos.append({"label": lab, "first_pos": pos})
    if first_pos:
        first_pos_df = pd.DataFrame(first_pos)
        cdf = (
            first_pos_df.groupby("label")["first_pos"]
            .quantile([0.1, 0.25, 0.5, 0.75, 0.9])
            .unstack()
        )
        cdf.to_csv(out_cdf_csv)

    out_cov_json.write_text(
        json.dumps(
            {
                "coverage_rate": coverage,
                "avg_spans_per_doc": per_doc_counts.groupby("label")["n"]
                .mean()
                .to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def absolutist_hedge_analyses(df_all: pd.DataFrame, out_dir: Path):
    abs_pat = make_word_boundary_regex(ABSOLUTIST)
    hed_pat = make_word_boundary_regex(HEDGES)

    df = df_all.copy()
    df["char_len"] = df["text"].str.len().fillna(0).clip(lower=1)
    df["abs_cnt"] = df["text"].apply(lambda s: len(abs_pat.findall(s or "")))
    df["hed_cnt"] = df["text"].apply(lambda s: len(hed_pat.findall(s or "")))
    df["abs_per_1k"] = 1000.0 * df["abs_cnt"] / df["char_len"]
    df["hed_per_1k"] = 1000.0 * df["hed_cnt"] / df["char_len"]

    # Save summary
    summ = (
        df.groupby("doc_label")[["abs_per_1k", "hed_per_1k"]]
        .agg(["mean", "median", "std", "count"])
        .round(4)
    )
    summ.to_csv(out_dir / "absolutist_hedge_summary.csv")

    # Plots
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="doc_label", y="abs_per_1k")
    sns.stripplot(
        data=df, x="doc_label", y="abs_per_1k", dodge=False, alpha=0.25, size=2
    )
    plt.title("Absolutist language rate (per 1k chars) by document label")
    plt.xlabel("Document label")
    plt.ylabel("Absolutist per 1k chars")
    plt.tight_layout()
    plt.savefig(out_dir / "absolutist_language_rate_by_doc_label.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="doc_label", y="hed_per_1k")
    sns.stripplot(
        data=df, x="doc_label", y="hed_per_1k", dodge=False, alpha=0.25, size=2
    )
    plt.title("Hedges/uncertainty rate (per 1k chars) by document label")
    plt.xlabel("Document label")
    plt.ylabel("Hedges per 1k chars")
    plt.tight_layout()
    plt.savefig(out_dir / "hedges_rate_by_doc_label.png", dpi=200)
    plt.close()

    # Subreddit averages (n>=20) for domain effects
    if "subreddit" in df.columns:
        sub_stats = (
            df.groupby(["subreddit", "doc_label"])
            .agg(
                n=("doc_id", "nunique"),
                abs_per_1k=("abs_per_1k", "mean"),
                hed_per_1k=("hed_per_1k", "mean"),
            )
            .reset_index()
        )
        sub_stats[sub_stats["n"] >= 20].to_csv(
            out_dir / "absolutist_hedge_by_subreddit.csv", index=False
        )

    # Effect sizes + BH
    groups = {
        lab: g["abs_per_1k"].dropna().tolist() for lab, g in df.groupby("doc_label")
    }
    labs = sorted(groups.keys())
    tests, pvals = [], []
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            a, b = groups[labs[i]], groups[labs[j]]
            if len(a) == 0 or len(b) == 0:
                continue
            stat, p = mannwhitneyu(a, b, alternative="two-sided")
            delta = cliffs_delta(a, b)
            tests.append(
                {
                    "metric": "abs_per_1k",
                    "a": labs[i],
                    "b": labs[j],
                    "p": float(p),
                    "cliffs_delta": float(delta),
                }
            )
            pvals.append(p)
    if pvals:
        adj = bh_correct(np.array(pvals))
        for k, v in zip(tests, adj):
            k["p_bh"] = float(v)
        pd.DataFrame(tests).to_csv(out_dir / "lexical_effect_sizes.csv", index=False)


def hard_examples_selection(
    train_df: pd.DataFrame, dev_df: pd.DataFrame, out_path: Path
):
    df_all = pd.concat(
        [train_df.assign(split="train"), dev_df.assign(split="dev")], ignore_index=True
    )

    def max_ae(markers: List[dict]) -> float:
        acts = [
            (m["start"], m["end"])
            for m in markers
            if m.get("label") == "Action" and isinstance(m.get("start"), int)
        ]
        effs = [
            (m["start"], m["end"])
            for m in markers
            if m.get("label") == "Effect" and isinstance(m.get("start"), int)
        ]
        if not acts or not effs:
            return 0.0
        mx = 0.0
        for s1, e1 in acts:
            for s2, e2 in effs:
                if max(s1, s2) < min(e1, e2):
                    inter = max(0, min(e1, e2) - max(s1, s2))
                    union = (e1 - s1) + (e2 - s2) - inter
                    mx = max(mx, inter / union if union > 0 else 0.0)
        return mx

    hard = {}
    df_all["max_ae_iou"] = df_all["markers"].apply(
        lambda m: max_ae(m if isinstance(m, list) else [])
    )
    high_iou = df_all[df_all["max_ae_iou"] > 0.7]
    for _, r in high_iou.iterrows():
        did = r["doc_id"]
        hard.setdefault(did, {"reasons": [], "text": r.get("text", "")})
        hard[did]["reasons"].append(f"High Action/Effect IoU ({r['max_ae_iou']:.2f})")

    # Subreddit label entropy
    sub_counts = df_all.groupby(["subreddit", "doc_label"]).size().unstack(fill_value=0)
    sub_probs = sub_counts.div(sub_counts.sum(axis=1), axis=0).fillna(0)
    sub_ent = sub_probs.apply(lambda row: entropy(row, base=2), axis=1)
    df_all["subreddit_entropy"] = df_all["subreddit"].map(sub_ent)
    amb = df_all[df_all["subreddit_entropy"] > 1.5]
    for _, r in amb.iterrows():
        did = r["doc_id"]
        hard.setdefault(did, {"reasons": [], "text": r.get("text", "")})
        hard[did]["reasons"].append(
            f"High Subreddit Entropy ({r['subreddit_entropy']:.2f})"
        )

    # Cheap baseline (TF-IDF LR) for confident errors on dev
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        df_bin_tr = train_df[train_df["doc_label"].isin(["conspiracy", "non"])]
        df_bin_dev = dev_df[dev_df["doc_label"].isin(["conspiracy", "non"])]
        if not df_bin_dev.empty and not df_bin_tr.empty:
            vec = TfidfVectorizer(max_features=5000, stop_words="english")
            Xtr = vec.fit_transform(df_bin_tr["text"])
            ytr = df_bin_tr["doc_label"]
            Xdv = vec.transform(df_bin_dev["text"])
            clf = LogisticRegression(
                random_state=42, class_weight="balanced", max_iter=200
            )
            clf.fit(Xtr, ytr)
            probs = clf.predict_proba(Xdv)
            preds = clf.classes_[np.argmax(probs, axis=1)]
            mis = preds != df_bin_dev["doc_label"].to_numpy()
            conf = np.max(probs[mis], axis=1) if mis.any() else np.array([])
            hard_dev = df_bin_dev[mis].copy()
            hard_dev["error_confidence"] = conf
            conf_err = hard_dev[hard_dev["error_confidence"] > 0.8]
            for _, r in conf_err.iterrows():
                did = r["doc_id"]
                hard.setdefault(did, {"reasons": [], "text": r.get("text", "")})
                hard[did]["reasons"].append(
                    f"Baseline Confident Error (Conf: {r['error_confidence']:.2f})"
                )
    except Exception as e:
        print(f"[warn] Skipped baseline error mining due to: {e}")

    final = [{"doc_id": k, **v} for k, v in hard.items()]
    out_path.write_text(
        json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[hard examples] wrote {len(final)} rows → {out_path}")


def span_position_plot(sdf: pd.DataFrame, df_all: pd.DataFrame, out_path: Path):
    """
    Plot KDE of normalized span center positions by label.

    sdf columns expected (char offsets): doc_id, label, start, end
    df_all columns must include: doc_id, text
    """

    # --- 0) Guardrails on required columns
    for col in ("doc_id", "label", "start", "end"):
        if col not in sdf.columns:
            print(f"[span_position_plot] Missing '{col}' in spans; skipping plot.")
            return
    if ("doc_id" not in df_all.columns) or ("text" not in df_all.columns):
        print(
            "[span_position_plot] df_all must have 'doc_id' and 'text'; skipping plot."
        )
        return

    # --- 1) Coerce id types to string for a reliable join/map
    sdf = sdf.copy()
    df_all = df_all.copy()
    sdf["doc_id"] = sdf["doc_id"].astype(str)
    df_all["doc_id"] = df_all["doc_id"].astype(str)

    # --- 2) Compute text_len via a dict map (works even if there are dup ids)
    text_len_map = (
        df_all[["doc_id", "text"]]
        .drop_duplicates("doc_id")
        .assign(_len=lambda d: d["text"].astype(str).str.len())
        .set_index("doc_id")["_len"]
        .to_dict()
    )
    sdf["text_len"] = sdf["doc_id"].map(text_len_map)

    # If mapping failed for some rows (rare), fill with 1 so we can proceed safely
    if "text_len" not in sdf.columns:
        sdf["text_len"] = 1
    else:
        sdf["text_len"] = sdf["text_len"].fillna(1).astype(int).clip(lower=1)

    # --- 3) Filter to valid rows
    if "label" not in sdf.columns:
        print("[span_position_plot] No 'label' column; skipping plot.")
        return
    # Now text_len definitely exists; drop where label is missing
    sdf = sdf.dropna(subset=["label"]).copy()
    if sdf.empty:
        print("[span_position_plot] No spans after label filtering; skipping plot.")
        return

    # --- 4) Compute normalized span center position
    sdf["start"] = sdf["start"].astype(int)
    sdf["end"] = sdf["end"].astype(int)
    sdf["norm_center_pos"] = ((sdf["start"] + sdf["end"]) / 2.0) / sdf[
        "text_len"
    ].astype(float)

    # --- 5) Plot
    plt.figure(figsize=(12, 7))
    try:
        hue_order = sorted([x for x in sdf["label"].dropna().unique().tolist()])
        if not hue_order:
            print("[span_position_plot] No labels to plot; skipping.")
            return

        sns.kdeplot(
            data=sdf,
            x="norm_center_pos",
            hue="label",
            hue_order=hue_order,
            fill=True,
            common_norm=False,
            alpha=0.20,
        )
        plt.title("Normalized Position of Marker Spans within Documents", fontsize=16)
        plt.xlabel("Normalized Document Position (0 = start, 1 = end)")
        plt.ylabel("Density")
        plt.xlim(0, 1)
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        print(f"[span_position_plot] Saved to {out_path}")
    finally:
        plt.close()


# -----------------------------
# Main
# -----------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived-root", default="data/derived")
    parser.add_argument("--latest-pointer", default="data/derived/psycomark_latest.txt")
    args = parser.parse_args()

    derived_root = Path(args.derived_root)
    latest_ptr = Path(args.latest_pointer)
    out_dir = find_latest_dir(latest_ptr)

    print(f"--- Loading data from latest pipeline run: {out_dir.name} ---")
    train_df = pd.read_json(out_dir / "train.jsonl", lines=True)
    dev_df = pd.read_json(out_dir / "dev.jsonl", lines=True)

    # Ensure _id alias present
    train_df = alias_ids(train_df)
    dev_df = alias_ids(dev_df)

    # Manifest summary (robust to key changes)
    man_path = out_dir / "manifest.json"
    if man_path.exists():
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
        print("\n=== Pipeline Run Summary ===")
        sz = manifest.get("sizes", {})
        print(
            f"Train Set: {sz.get('train_raw','?')} raw -> {sz.get('train_final','?')} final"
        )
        print(
            f"  - Docs removed due to dev set leakage: {sz.get('train_docs_removed_leakage','?')}"
        )
        # name fix vs earlier versions
        removed_internal = sz.get(
            "num_within_train_removed_docs",
            sz.get("train_docs_removed_internal_dups", "N/A"),
        )
        print(f"  - Docs removed as internal duplicates: {removed_internal}")
        print(
            f"Dev Set:   {sz.get('dev_raw','?')} raw -> {sz.get('dev_final','?')} final (dev set is preserved)"
        )
        print("============================")
    else:
        print("[warn] manifest.json not found; skipping summary.")

    # Combined DF
    train_df["split"] = "train"
    dev_df["split"] = "dev"
    df_all = pd.concat([train_df, dev_df], ignore_index=True)
    print("\nData loaded and prepared for analysis.")

    # Build spans (train only for priors + doc-level stats)
    spans_train = spans_from_df(train_df)
    spans_all = spans_from_df(df_all)
    texts_train = {r["doc_id"]: (r["text"] or "") for _, r in train_df.iterrows()}

    # ---- Overlap stats (simple) ----
    pair_stats = overlap_stats_for_pairs(spans_all)
    (out_dir / "overlap_pair_stats.json").write_text(
        json.dumps(pair_stats, indent=2), encoding="utf-8"
    )
    top_pairs = sorted(
        pair_stats.items(), key=lambda kv: kv[1]["iou@0.5"], reverse=True
    )[:5]
    print("Top pairs by IoU@0.5:", top_pairs)

    # ---- Overlap stats with doc-level bootstrap CIs ----
    pairwise_bootstrap_ci(spans_train, out_dir / "overlap_pair_stats_ci.json", B=1000)

    # ---- Boundary context ----
    extract_boundary_context(
        spans_train, texts_train, out_dir / "boundary_context.json", k_chars=5, top=25
    )

    # ---- Mean IoU matrix (plot) ----
    mean_iou_matrix_plot(df_all, out_dir / "mean_iou_matrix.png")

    # ---- First-occurrence CDFs + coverage ----
    first_occurrence_and_coverage(
        spans_all,
        df_all,
        out_dir / "first_occurrence_cdf.csv",
        out_dir / "label_coverage.json",
    )

    # ---- Absolutist / Hedge analyses (+ plots + effect sizes) ----
    absolutist_hedge_analyses(df_all, out_dir)

    # ---- Hard example mining ----
    hard_examples_selection(train_df, dev_df, out_dir / "hard_examples.json")

    # ---- Span position plot ----
    span_position_plot(spans_all, df_all, out_dir / "span_position_analysis.png")

    (out_dir / "lexicons.json").write_text(
        json.dumps({"ABSOLUTIST": ABSOLUTIST, "HEDGES": HEDGES}, indent=2),
        encoding="utf-8",
    )
    print("[analysis_and_insights] Wrote lexicons.json")

    print(
        "\n[analysis_and_insights] Finished. Artifacts written to:", out_dir.resolve()
    )


if __name__ == "__main__":
    main()
