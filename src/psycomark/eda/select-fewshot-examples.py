#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_fewshot_examples.py

Produces best_fewshot_examples.json and fewshot_policy.json in the latest derived folder:
- S2: balanced conspiracy/non, optional cant_tell-as-negative with rationale
- S1: prior-aware (length/start), targets ambiguous pairs via overlap stats,
      adds small quota of outliers, enforces subreddit diversity, short snippets.

Inputs (from latest derived run):
  train.jsonl, dev.jsonl
  overlap_pair_stats_ci.json or overlap_pair_stats.json (optional, used if present)
  length_position_priors.json (optional; robust fallbacks)
  boundary_context.json (optional; not required)
  hard_examples.json (optional; helps S1/S2 selection, but not required)

Outputs:
  best_fewshot_examples.json
  fewshot_policy.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# -----------------------------
# Defaults / Policy
# -----------------------------
ALLOWED_S1 = {"Actor", "Action", "Effect", "Victim", "Evidence"}

# Policy (FROZEN unless overridden with flags)
CANT_TELL_IN_S2 = False
CANT_TELL_RATIONALE_DEFAULT = "Insufficient evidence for a concrete conspiracy claim; statements are ambiguous or hedged."


# -----------------------------
# Helpers
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


def top_pairs(
    pair_stats: Dict[str, dict], key="iou@0.5", topk=2
) -> List[Tuple[str, str]]:
    if not pair_stats:
        return []
    items = [(p, d.get(key, 0.0)) for p, d in pair_stats.items()]
    items = sorted(items, key=lambda x: x[1], reverse=True)
    return [tuple(sorted(p.split("/"))) for p, _ in items[:topk]]


def load_json(path: Path, default=None):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {} if default is None else default


def prior_len_z(priors, label, span_len) -> float:
    """Return |z| distance under lognormal prior; robust to missing fits."""
    p = priors.get(label, {})
    if "length_lognorm" in p:
        mu = p["length_lognorm"].get("mu", 0.0)
        sig = max(1e-6, p["length_lognorm"].get("sigma", 1.0))
        return abs((math.log(max(1, span_len)) - mu) / sig)
    # fallback: compare to q90 if present -> pseudo z in [0, ~1.5+]
    q90 = p.get("q90_len") or p.get("q90", None)
    if q90:
        return max(0.0, (span_len - float(q90)) / (float(q90) + 1e-6))
    return 1.0  # neutral


def prior_pos_dist(priors, label, start_pos) -> float:
    """Distance to Beta mode for start position; robust to missing fits."""
    p = priors.get(label, {})
    if "start_beta" in p:
        a = p["start_beta"].get("alpha", 1.0)
        b = p["start_beta"].get("beta", 1.0)
        mode = (
            (a - 1) / (a + b - 2)
            if (a > 1 and b > 1)
            else (a / (a + b) if (a > 0 and b > 0) else 0.5)
        )
        return abs(start_pos - mode)
    return 0.5  # neutral


def doc_has_target_overlap(
    markers: List[dict], target_pairs: List[Tuple[str, str]]
) -> int:
    spans = [
        (m["label"], m["start"], m["end"])
        for m in (markers or [])
        if m.get("label") in ALLOWED_S1
    ]
    for i in range(len(spans)):
        l1, s1, e1 = spans[i]
        for j in range(i + 1, len(spans)):
            l2, s2, e2 = spans[j]
            if (
                tuple(sorted([l1, l2])) in target_pairs
                and iou_char((s1, e1), (s2, e2)) >= 0.5
            ):
                return 1
    return 0


def markers_compact(markers: List[dict], max_per_label=2) -> List[dict]:
    out = []
    per_lab = defaultdict(int)
    for m in markers or []:
        lbl = m.get("label")
        if lbl not in ALLOWED_S1:
            continue
        if per_lab[lbl] >= max_per_label:
            continue
        if {"label", "start", "end"} <= m.keys():
            out.append({"label": lbl, "start": int(m["start"]), "end": int(m["end"])})
            per_lab[lbl] += 1
    return out


# -----------------------------
# S2 selection
# -----------------------------
def pick_balanced_by_label(
    df,
    k_per_label=2,
    label_col="doc_label",
    text_col="text",
    diversity_col="subreddit",
    seed=42,
    len_min=160,
    len_max=1000,
):
    out = []
    rng = np.random.default_rng(seed)
    for lab in ["conspiracy", "non"]:
        sub = df[df[label_col] == lab].copy()
        if sub.empty:
            continue
        sub["_rand"] = rng.random(size=len(sub))
        if diversity_col in sub.columns:
            sub = sub.sort_values([diversity_col, "_rand"])
        else:
            sub = sub.sort_values(["_rand"])
        picked = []
        seen = defaultdict(int)
        for _, r in sub.iterrows():
            txt = r.get(text_col, "") or ""
            if not (len_min <= len(txt) <= len_max):
                continue
            sr = r.get(diversity_col, "")
            if seen[sr] >= 2:  # light diversity cap
                continue
            picked.append(r.to_dict())
            seen[sr] += 1
            if len(picked) >= k_per_label:
                break
        out.extend(picked)
    return out


def coerce_s2_item(r, label_override=None, rationale=None, tag=None):
    item = {
        "doc_id": r.get("doc_id"),
        "text": r.get("text", ""),
        "label": label_override if label_override else r.get("doc_label"),
        "rationale": rationale if rationale else "",
    }
    if tag:
        item["source_label"] = tag
    return item


# -----------------------------
# S1 scoring / selection
# -----------------------------
def build_s1_candidates(
    train_df: pd.DataFrame, priors: dict, target_pairs: List[Tuple[str, str]]
) -> pd.DataFrame:
    cand = []
    for _, r in train_df.iterrows():
        t = r.get("text") or ""
        tlen = max(1, len(t))
        has_tpair = doc_has_target_overlap(r.get("markers"), target_pairs)
        for m in r.get("markers") or []:
            L = m.get("label")
            if L not in ALLOWED_S1:
                continue
            s, e = int(m["start"]), int(m["end"])
            span_len = max(1, e - s)
            start_pos = s / tlen
            score = 0.0
            # priors (closer is better) -> convert to [0,1]
            score += 1.5 * (1.0 - min(3.0, prior_len_z(priors, L, span_len)) / 3.0)
            score += 1.5 * (1.0 - min(1.0, prior_pos_dist(priors, L, start_pos)))
            # overlap bonus for docs with target pair conflict
            score += 1.0 * has_tpair
            # compact-length bonus vs q90 (if provided)
            q90 = priors.get(L, {}).get("q90_len") or priors.get(L, {}).get(
                "q90_per_label", {}
            ).get(L)
            if q90 is not None:
                score += 0.25 * (span_len <= float(q90))
            cand.append(
                {
                    "doc_id": r["doc_id"],
                    "subreddit": r.get("subreddit", "NA"),
                    "label": L,
                    "start": s,
                    "end": e,
                    "score": float(score),
                    "text": t,
                }
            )
    return pd.DataFrame(cand)


def pick_s1_for_label(
    df_lab: pd.DataFrame,
    k=2,
    outlier_k=1,
    priors: dict | None = None,
    subreddit_diversity=True,
):
    if df_lab.empty:
        return []
    topk = df_lab.sort_values("score", ascending=False)
    chosen = []
    if subreddit_diversity and "subreddit" in df_lab.columns:
        seen = set()
        for _, row in topk.iterrows():
            if row["subreddit"] in seen:
                continue
            chosen.append(row.to_dict())
            seen.add(row["subreddit"])
            if len(chosen) >= k:
                break
    if len(chosen) < k:
        chosen = topk.head(k).to_dict(orient="records")

    # add outliers (furthest from priors)
    def prior_z_len(label, span_len):
        return prior_len_z(priors or {}, label, span_len)

    def prior_dist_pos(label, start_pos):
        return prior_pos_dist(priors or {}, label, start_pos)

    df_lab = df_lab.copy()
    df_lab["z_len"] = df_lab.apply(
        lambda r: prior_z_len(r["label"], r["end"] - r["start"]), axis=1
    )
    df_lab["pos_dist"] = df_lab.apply(
        lambda r: prior_dist_pos(r["label"], r["start"] / max(1, len(r["text"]))),
        axis=1,
    )
    outliers = df_lab.sort_values(["z_len", "pos_dist"], ascending=False).head(
        outlier_k
    )
    chosen += outliers.to_dict(orient="records")
    return chosen[: k + outlier_k]


def make_snippet(item: dict, pad=120) -> dict:
    """Crop around span, normalize start/end to snippet space."""
    t = item["text"]
    s, e = int(item["start"]), int(item["end"])
    left = max(0, s - pad)
    right = min(len(t), e + pad)
    snippet = (t[left:s] + t[s:e] + t[e:right]).strip()
    new_start = len(t[left:s])
    new_end = new_start + (e - s)
    return {
        "doc_id": item["doc_id"],
        "text": snippet,
        "spans": [{"label": item["label"], "start": new_start, "end": new_end}],
        "meta": {
            "source_window": [left, right],
            "reason": "prior_closeness+ambiguous_pair_bonus",
        },
    }


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--derived-root", default="data/derived")
    ap.add_argument("--latest-pointer", default="data/derived/psycomark_latest.txt")
    ap.add_argument("--shots-s2-per-class", type=int, default=8)
    ap.add_argument("--shots-s1-per-label", type=int, default=2)
    ap.add_argument("--shots-s1-outliers", type=int, default=1)
    ap.add_argument(
        "--cant-tell-negs",
        type=int,
        default=2,
        help="How many cant_tell to inject as negative S2 (label=non).",
    )
    ap.add_argument("--cant-tell-rationale", default=CANT_TELL_RATIONALE_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--preserve-existing-s1",
        action="store_true",
        help="If set, keep existing S1 exemplars and only update S2.",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = find_latest_dir(Path(args.latest_pointer))
    print(f"--- Using latest derived run: {out_dir.name} ---")

    # Load data
    train_df = pd.read_json(out_dir / "train.jsonl", lines=True)
    dev_df = pd.read_json(out_dir / "dev.jsonl", lines=True)
    train_df = alias_ids(train_df)
    dev_df = alias_ids(dev_df)

    # Pools / convenience views
    df_all = pd.concat([train_df, dev_df], ignore_index=True).copy()

    # Load optional stats
    priors = load_json(out_dir / "length_position_priors.json", default={})
    pair_stats = load_json(out_dir / "overlap_pair_stats_ci.json", default={})
    if not pair_stats:
        pair_stats = load_json(out_dir / "overlap_pair_stats.json", default={})
    boundary = load_json(out_dir / "boundary_context.json", default={})

    # Hard examples (optional)
    hard_path = out_dir / "hard_examples.json"
    hard_df = pd.read_json(hard_path) if hard_path.exists() else pd.DataFrame()

    # Existing fewshots (optional)
    fs_path = out_dir / "best_fewshot_examples.json"
    existing = load_json(fs_path, default={"s1": [], "s2": []})

    # ----------------- S2 selection (binary only) -----------------
    # Build S2 pool (exclude cant_tell)
    pool_s2 = df_all[df_all["doc_label"].isin(["conspiracy", "non"])].copy()

    # Prefer hard examples if available; else use whole pool
    if not hard_df.empty:
        # Ensure doc_id is string on both sides
        hard_df["doc_id"] = hard_df["doc_id"].astype(str)
        df_all["doc_id"] = df_all["doc_id"].astype(str)

        # Keep the 'text' already in hard_df; only join NEW cols
        right = df_all.set_index("doc_id")[["doc_label", "subreddit"]]
        base = hard_df.set_index("doc_id").join(right, how="left").reset_index()

        # If any hard rows lack text, fill from df_all
        if "text" not in base.columns:
            base["text"] = ""
        text_map = df_all.set_index("doc_id")["text"].astype(str).to_dict()
        need_text = base["text"].isna() | (base["text"].astype(str).str.len() == 0)
        base.loc[need_text, "text"] = (
            base.loc[need_text, "doc_id"].map(text_map).fillna("")
        )

        # Keep only binary labels
        base = base[base["doc_label"].isin(["conspiracy", "non"])].copy()
        pool_for_pick = base
    else:
        pool_for_pick = pool_s2

    print(f"[S2] candidate pool (binary): {len(pool_for_pick)}")

    s2_raw = pick_balanced_by_label(
        pool_for_pick,
        k_per_label=args.shots_s2_per_class,
        label_col="doc_label",
        text_col="text",
        diversity_col="subreddit",
        seed=args.seed,
        len_min=160,
        len_max=1000,
    )
    s2_main = [coerce_s2_item(r) for r in s2_raw]

    # Optional: inject cant_tell as negative with rationale
    s2_ct = []
    if args.cant_tell_negs > 0:
        all_raw = df_all[["doc_id", "text", "doc_label", "subreddit"]].copy()
        ct_pool = all_raw[all_raw["doc_label"] == "cant_tell"].copy()
        if not ct_pool.empty:
            rng = np.random.default_rng(args.seed)
            ct_pool["_rand"] = rng.random(size=len(ct_pool))
            order = (
                ["subreddit", "_rand"] if "subreddit" in ct_pool.columns else ["_rand"]
            )
            for _, r in ct_pool.sort_values(order).head(args.cant_tell_negs).iterrows():
                s2_ct.append(
                    coerce_s2_item(
                        r,
                        label_override="non",
                        rationale=args.cant_tell_rationale,
                        tag="cant_tell",
                    )
                )
    # Freeze policy
    assert all(
        x["label"] in {"conspiracy", "non"} for x in (s2_main + s2_ct)
    ), "Policy breach: S2 must be binary."
    assert (
        CANT_TELL_IN_S2 is False
    ), "Policy breach: cant_tell cannot appear as S2 label."

    s2_fewshots = s2_main + s2_ct

    # ----------------- S1 selection (spans) -----------------
    if args.preserve_existing_s1 and existing.get("s1"):
        s1_examples = existing["s1"]
        print(f"[S1] Preserving existing S1 few-shots: {len(s1_examples)}")
    else:
        pairs = top_pairs(pair_stats, key="iou@0.5", topk=2)
        print(
            f"[S1] Target ambiguous pairs: {pairs if pairs else 'None (fallback to priors only)'}"
        )
        # Build candidates from train set
        cands = build_s1_candidates(train_df, priors=priors, target_pairs=pairs)
        s1_examples = []
        for L in sorted(ALLOWED_S1):
            df_lab = cands[cands["label"] == L].copy()
            picks = pick_s1_for_label(
                df_lab,
                k=args.shots_s1_per_label,
                outlier_k=args.shots_s1_outliers,
                priors=priors,
                subreddit_diversity=True,
            )
            for p in picks:
                s1_examples.append(make_snippet(p, pad=120))

    # ----------------- Write outputs -----------------
    out_fs = {"s1": s1_examples, "s2": s2_fewshots}
    fs_path.write_text(
        json.dumps(out_fs, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    policy_meta = {
        "cant_tell": {
            "s2_training_excluded": True,
            "fewshot_negatives_added": int(
                sum(1 for ex in s2_fewshots if ex.get("source_label") == "cant_tell")
            ),
            "rationale": args.cant_tell_rationale,
        },
        "seed": args.seed,
        "shots": {
            "s2_per_class": args.shots_s2_per_class,
            "s1_per_label": args.shots_s1_per_label,
            "s1_outliers_per_label": args.shots_s1_outliers,
        },
        "targets": {
            "ambiguous_pairs_top2": pairs if "pairs" in locals() else [],
        },
    }
    (out_dir / "fewshot_policy.json").write_text(
        json.dumps(policy_meta, indent=2), encoding="utf-8"
    )

    print(f"\n[select_fewshot_examples] Wrote:")
    print(f"  - {fs_path}")
    print(f"  - {out_dir / 'fewshot_policy.json'}")
    print(f"  S1 count: {len(s1_examples)}  | S2 count: {len(s2_fewshots)}")


if __name__ == "__main__":
    main()
