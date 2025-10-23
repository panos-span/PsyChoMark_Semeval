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
import re
from string import punctuation
import math
import random
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# -----------------------------
# Defaults / Policy
# -----------------------------
ALLOWED_S1 = {"Actor", "Action", "Effect", "Victim", "Evidence"}
_STOP = {
    "the",
    "a",
    "an",
    "to",
    "of",
    "and",
    "in",
    "on",
    "for",
    "with",
    "at",
    "by",
    "from",
    "that",
    "this",
    "it",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "as",
    "or",
    "if",
    "but",
    "so",
    "do",
    "did",
    "does",
}
_PUNCT_RE = re.compile(rf"^[{re.escape(punctuation)}\s]+$")

# Policy (FROZEN unless overridden with flags)
CANT_TELL_IN_S2 = False
CANT_TELL_RATIONALE_DEFAULT = "Insufficient evidence for a concrete conspiracy claim; statements are ambiguous or hedged."


# -----------------------------
# Helpers
# -----------------------------


def _is_all_stopwords(txt: str) -> bool:
    toks = [t for t in re.split(r"\s+", txt.strip()) if t]
    if not toks:
        return True
    return all(t.lower() in _STOP for t in toks)


def _valid_span(sp, text: str, min_len: int = 3, max_len: int = 150) -> bool:
    try:
        s = int(sp.get("start", -1))
        e = int(sp.get("end", -1))
        lbl = sp.get("label")
    except Exception:
        return False
    if lbl not in ALLOWED_S1 or e <= s:
        return False
    if (e - s) < min_len or (e - s) > max_len:
        return False
    # extract raw span safely
    if not (0 <= s < len(text)) or not (0 < e <= len(text)):
        return False
    snip = text[s:e]
    if not snip or _PUNCT_RE.match(snip):
        return False
    if _is_all_stopwords(snip):
        return False
    return True


def _example_has_valid_span(ex) -> bool:
    text = ex.get("text", "") or ""
    spans = ex.get("spans", []) or []
    return any(_valid_span(sp, text) for sp in spans)


def load_lexicons(out_dir: Path):
    # defaults if file missing
    abs_default = [
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
    hed_default = [
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
    path = out_dir / "lexicons.json"
    if path.exists():
        try:
            js = json.loads(path.read_text(encoding="utf-8"))
            A = js.get("ABSOLUTIST") or abs_default
            H = js.get("HEDGES") or hed_default
            return A, H
        except Exception:
            pass
    return abs_default, hed_default


# --- NEW: Prior- & boundary-aware pickers for S1 ---
def _dist_to_priors(priors: dict, label: str, s: int, e: int, tlen: int) -> float:
    span_len = max(1, e - s)
    start_pos = s / max(1, tlen)
    # lower is better
    dz = prior_len_z(priors, label, span_len)  # ~|z|
    dp = prior_pos_dist(priors, label, start_pos)  # ~distance in [0,1]
    return 1.5 * dz + 1.0 * dp


def pick_s1_prior_examples(df_lab: pd.DataFrame, priors: dict, k_near=1, k_outlier=1):
    if df_lab.empty:
        return []
    df = df_lab.copy()
    df["tlen"] = df["text"].str.len().fillna(1).astype(int)
    df["prior_dist"] = df.apply(
        lambda r: _dist_to_priors(priors, r["label"], r["start"], r["end"], r["tlen"]),
        axis=1,
    )
    # near-prior: lowest distance
    near = (
        df.sort_values("prior_dist", ascending=True)
        .head(k_near)
        .to_dict(orient="records")
    )
    # outlier: highest distance
    out = (
        df.sort_values("prior_dist", ascending=False)
        .head(k_outlier)
        .to_dict(orient="records")
    )
    return near + out


def _has_boundary_cue(ctx: dict, label: str, text: str, s: int, e: int) -> bool:
    cues = []
    for key in ("before_1w", "after_1w", "before_2w", "after_2w"):
        cues.extend(ctx.get(label, {}).get(key, [])[:5])  # top-5 per side
    window = text[max(0, s - 40) : min(len(text), e + 40)].lower()
    return any(c and c.lower() in window for c in cues if isinstance(c, str))


def pick_s1_boundary_examples(df_lab: pd.DataFrame, boundary_ctx: dict, k=1):
    if df_lab.empty:
        return []
    rows = []
    for _, r in df_lab.iterrows():
        if _has_boundary_cue(
            boundary_ctx, r["label"], r["text"] or "", int(r["start"]), int(r["end"])
        ):
            rows.append(r.to_dict())
    rng = np.random.default_rng(42)
    rng.shuffle(rows)
    return rows[:k]


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


# --- add near other helpers ---
def pick_negative_s1_snippet(dev_df, min_len=120, max_len=320, seed=42):
    """Return a dict with text and empty spans to teach 'no markers'."""
    pool = dev_df[
        dev_df["markers"].apply(lambda m: not isinstance(m, list) or len(m) == 0)
    ]
    if pool.empty:
        return None
    rows = pool.sample(n=min(50, len(pool)), random_state=seed, replace=False)
    for _, r in rows.iterrows():
        t = (r.get("text") or "").strip()
        if min_len <= len(t) <= max_len:
            return {
                "doc_id": r.get("doc_id"),
                "text": t,
                "spans": [],  # empty JSON to demonstrate valid 'no markers'
                "meta": {"reason": "negative_no_markers"},
            }
    return None


def _shorten_rationale(r: str, max_chars=160) -> str:
    r = (r or "").strip().split("\n")[0]
    return (r[:max_chars] + "…") if len(r) > max_chars else r


def enforce_min_yes_fewshots(s2_list, min_yes=4, df_all=None, seed=42):
    yes = [x for x in s2_list if (x.get("label") or "").lower() == "conspiracy"]
    if len(yes) >= min_yes:
        # clean rationales
        for x in s2_list:
            if "rationale" in x:
                x["rationale"] = _shorten_rationale(x["rationale"])
        return s2_list

    # sample additional 'conspiracy' rows from data
    needed = min_yes - len(yes)
    if df_all is not None:
        pool = df_all[df_all["doc_label"] == "conspiracy"].copy()
        if not pool.empty:
            pool = pool.sample(n=min(needed * 3, len(pool)), random_state=seed)
            added = 0
            for _, r in pool.iterrows():
                if any(x["doc_id"] == r["doc_id"] for x in s2_list):
                    continue
                s2_list.append(
                    {
                        "doc_id": r["doc_id"],
                        "text": r.get("text", ""),
                        "label": "conspiracy",
                        "rationale": "Text clearly alleges a covert plan with actors and evidence.",
                    }
                )
                added += 1
                if added >= needed:
                    break
    # clean rationales for all
    for x in s2_list:
        if "rationale" in x:
            x["rationale"] = _shorten_rationale(x["rationale"])
    return s2_list


# -----------------------------
# S1 scoring / selection
# -----------------------------
MIN_PER_LABEL = 2
ALL_LABS = {"Actor", "Action", "Effect", "Victim", "Evidence"}


def _label_counts(exs):
    c = Counter()
    for ex in exs:
        for m in ex.get("spans", []):
            lab = m.get("label")
            if lab in ALL_LABS:
                c[lab] += 1
    return c


def backfill_missing_labels(train_df, s1_examples, max_per_label=MIN_PER_LABEL):
    """
    Relaxed backfill: if any label has < max_per_label examples, sample additional
    (sanity-filtered) spans from train_df to guarantee coverage.
    """
    have = _label_counts(s1_examples)
    need = [lab for lab in ALL_LABS if have[lab] < max_per_label]
    if not need:
        return []

    added = []
    seen_texts = {ex.get("text", "") for ex in s1_examples}

    for _, r in train_df.iterrows():
        if not need:
            break
        text = r.get("text") or ""
        tlen = len(text)
        for m in r.get("markers") or []:
            L = m.get("label")
            if L not in need:
                continue
            s, e = int(m.get("start", 0)), int(m.get("end", 0))
            span_len = max(1, e - s)

            # relaxed but sane filters
            if span_len < 3:
                continue
            if s == 0 and e >= (tlen - 3):
                continue
            if text in seen_texts:
                continue

            left = max(0, s - 120)
            right = min(tlen, e + 120)
            snippet = text[left:right].strip()
            off_s, off_e = s - left, e - left

            ex = {
                "doc_id": r.get("doc_id"),
                "text": snippet,
                "spans": [{"label": L, "start": off_s, "end": off_e}],
                "meta": {"reason": "backfill_relaxed"},
            }
            added.append(ex)
            seen_texts.add(text)
            have[L] += 1
            if have[L] >= max_per_label:
                need.remove(L)
                if not need:
                    break
    return added


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

            # --- QUALITY FILTERS ---
            if span_len < 4:
                continue
            if s == 0 and e >= (tlen - 5):
                continue
            span_txt = t[s:e].strip().lower()
            if span_txt.endswith((".", ",", ";", "’", "”")) or span_txt.startswith(
                ("the ", "a ", "an ")
            ):
                continue
            # -----------------------

            start_pos = s / tlen
            score = 0.0
            # priors (closer is better)
            score += 1.5 * (1.0 - min(3.0, prior_len_z(priors, L, span_len)) / 3.0)
            score += 1.5 * (1.0 - min(1.0, prior_pos_dist(priors, L, start_pos)))
            # overlap bonus for docs with target pair conflict
            score += 1.0 * has_tpair
            # compact-length bonus vs q90
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


def compute_prior_features(priors, label, start, end, text_len):
    length = max(1, int(end) - int(start))
    start_pos = int(start) / max(1, int(text_len))
    return {
        "len": int(length),
        "start_pos": float(start_pos),
        "z_len": float(prior_len_z(priors, label, length)),
        "pos_dist": float(prior_pos_dist(priors, label, start_pos)),
    }


def detect_boundary_hit(boundary_ctx, label, text, start, end):
    if not boundary_ctx:
        return {"hit": False, "cues": []}
    window = (text or "").lower()[max(0, start - 50) : min(len(text), end + 50)]
    cues = []
    for key in ("before_1w", "after_1w", "before_2w", "after_2w"):
        for c in (boundary_ctx.get(label, {}).get(key, []) or [])[:5]:
            if isinstance(c, str) and c.lower() in window:
                cues.append(c)
    return {"hit": len(cues) > 0, "cues": list(dict.fromkeys(cues))}


# --- S2 heuristic pickers ---
def _marker_density(mks):  # compact proxy: more markers -> more conspiratorial framing
    return 0 if not isinstance(mks, list) else min(10, len(mks))


def is_hedged_no(row_text: str, hedges) -> bool:
    s = (row_text or "").lower()
    return any(h in s for h in hedges)


def is_speculative_yes(row_text: str, absolutist) -> bool:
    s = (row_text or "").lower()
    return any(a in s for a in absolutist) and (
        "they" in s or "agenda" in s or "cover up" in s
    )


def pick_s2_buckets(
    df_all,
    ABSOLUTIST,
    HEDGES,
    k_yes=3,
    k_no=3,
    k_hedged_no=2,
    k_spec_yes=2,
    seed=42,
):
    # clear YES/NO by label
    yy = df_all[df_all["doc_label"] == "conspiracy"].copy()
    nn = df_all[df_all["doc_label"] == "non"].copy()
    # enrich with markers if available
    if "markers" in df_all.columns:
        yy["_md"] = yy["markers"].apply(_marker_density)
        nn["_md"] = nn["markers"].apply(_marker_density)
        yy = yy.sort_values("_md", ascending=False)
        nn = nn.sort_values("_md", ascending=True)

    clear_yes = yy.head(k_yes).to_dict(orient="records")
    clear_no = nn.head(k_no).to_dict(orient="records")

    # hedged NO (non + hedges)
    hed_pool = nn[nn["text"].apply(lambda t: is_hedged_no(t, HEDGES))]
    hedged_no = (
        hed_pool.sample(n=min(k_hedged_no, len(hed_pool)), random_state=seed).to_dict(
            orient="records"
        )
        if not hed_pool.empty
        else []
    )

    # speculative YES (conspiracy + absolutist language)
    spec_pool = yy[yy["text"].apply(lambda t: is_speculative_yes(t, ABSOLUTIST))]
    spec_yes = (
        spec_pool.sample(n=min(k_spec_yes, len(spec_pool)), random_state=seed).to_dict(
            orient="records"
        )
        if not spec_pool.empty
        else []
    )

    return clear_yes, clear_no, hedged_no, spec_yes


def pick_hard_S2(hard_df, df_all, max_borderline=3, max_misleading=3):
    if hard_df is None or hard_df.empty:
        return []
    J = hard_df.copy()
    J["doc_id"] = J["doc_id"].astype(str)
    right = df_all.set_index("doc_id")[["text", "doc_label", "subreddit"]]
    common = (set(J.columns) & set(right.columns)) - {"doc_id"}
    if common:
        right = right.drop(columns=list(common))
    J = J.merge(right, on="doc_id", how="left")
    J = J[J["doc_label"].isin(["conspiracy", "non"])]

    borderline = J[
        J["reasons"].apply(lambda rs: any("High" in r or "Entropy" in r for r in rs))
    ].head(max_borderline)
    misleading = J[
        J["reasons"].apply(lambda rs: any("Baseline Confident Error" in r for r in rs))
    ].head(max_misleading)

    out = []
    for _, r in borderline.iterrows():
        out.append(
            {
                "doc_id": r["doc_id"],
                "text": r["text"],
                "label": r["doc_label"],
                "rationale": "Borderline framing; avoid over-reading ambiguity.",
            }
        )
    for _, r in misleading.iterrows():
        # keep the gold label but make rationale explicit
        lab = r["doc_label"]
        rat = (
            "Speculative framing asserted as fact."
            if lab == "conspiracy"
            else "Non-conspiratorial despite suggestive phrasing."
        )
        out.append(
            {"doc_id": r["doc_id"], "text": r["text"], "label": lab, "rationale": rat}
        )
    return out


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
        "--max-n-fewshot",
        type=int,
        default=6,
        help="Max number of S1 fewshot examples to include (after balancing)",
    )
    ap.add_argument(
        "--cant-tell-negs",
        type=int,
        default=2,
        help="How many cant_tell to inject as negative S2 (label=non).",
    )
    ap.add_argument("--cant-tell-rationale", default=CANT_TELL_RATIONALE_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--s2-thresh",
        default="auto",
        help="Probability threshold for 'Yes'. Float in [0,1] or 'auto' to tune on dev.",
    )

    ap.add_argument(
        "--preserve-existing-s1",
        action="store_true",
        default=False,
        help="If set, keep existing S1 exemplars and only update S2.",
    )
    args = ap.parse_args()
    max_n = args.max_n_fewshot

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = find_latest_dir(Path(args.latest_pointer))
    ABSOLUTIST, HEDGES = load_lexicons(out_dir)
    print(f"--- Using latest derived run: {out_dir.name} ---")

    # Load data
    train_df = pd.read_json(out_dir / "train.jsonl", lines=True)
    dev_df = pd.read_json(out_dir / "dev.jsonl", lines=True)
    train_df = alias_ids(train_df)
    dev_df = alias_ids(dev_df)

    # Pools / convenience views
    df_all = pd.concat([train_df, dev_df], ignore_index=True).copy()
    # print df_all columns
    print(f"Data loaded: train={len(train_df)}, dev={len(dev_df)}, total={len(df_all)}")
    print(f"Data columns: {df_all.columns.tolist()}")

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

    # Balanced buckets: clear Yes/No + hedged No + speculative Yes
    cy, cn, hed_no, spec_yes = pick_s2_buckets(
        df_all,
        k_yes=3,
        k_no=3,
        k_hedged_no=2,
        k_spec_yes=2,
        seed=args.seed,
        ABSOLUTIST=ABSOLUTIST,
        HEDGES=HEDGES,
    )

    def _mk(item, label_override=None, rationale=""):
        return coerce_s2_item(item, label_override=label_override, rationale=rationale)

    s2_main = []
    s2_main += [
        _mk(
            r, rationale="Explicit claim of covert actors/actions with supporting cues."
        )
        for r in cy
    ]
    s2_main += [
        _mk(r, rationale="Statement is non-conspiratorial and descriptive.") for r in cn
    ]
    s2_main += [
        _mk(
            r,
            label_override="non",
            rationale="Hedged/uncertain language without endorsement.",
        )
        for r in hed_no
    ]
    s2_main += [
        _mk(
            r,
            label_override="conspiracy",
            rationale="Speculative assertion framed as fact with absolutist cues.",
        )
        for r in spec_yes
    ]

    s2_hard = pick_hard_S2(hard_df, df_all, max_borderline=2, max_misleading=2)
    s2_main += s2_hard

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

    # attach compact markers to S2 fewshots so the S2 prompt examples mirror runtime conditioning
    def attach_markers(ex):
        did = ex["doc_id"]
        row = df_all[df_all["doc_id"] == did].head(1)
        mks = (
            markers_compact(row.iloc[0].get("markers", []), max_per_label=2)
            if not row.empty
            else []
        )
        ex["markers"] = mks
        return ex

    s2_fewshots = [attach_markers(x) for x in (s2_main + s2_ct)]
    s2_fewshots = enforce_min_yes_fewshots(
        s2_fewshots, min_yes=6, df_all=df_all, seed=args.seed
    )

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
            # keep one diversity-aware top example
            base = pick_s1_for_label(
                df_lab, k=1, outlier_k=0, priors=priors, subreddit_diversity=True
            )
            # add near-prior + outlier (1+1)
            prior_set = pick_s1_prior_examples(df_lab, priors, k_near=1, k_outlier=1)
            # add one boundary-cue exemplar if available
            bset = pick_s1_boundary_examples(df_lab, boundary, k=1)
            picks = (base + prior_set + bset)[
                : (args.shots_s1_per_label + args.shots_s1_outliers)
            ]
            for p in picks:
                s1_examples.append(make_snippet(p, pad=120))

    # Add overlap exemplars for top ambiguous pairs (Action/Effect, Actor/Victim)
    if pairs:
        for pair in pairs:
            a, b = pair
            df_ab = cands[(cands["label"].isin([a, b]))].copy()
            # keep docs that contain BOTH labels with IoU >= 0.3 within a 240-char window
            seen = set()
            for doc_id, g in df_ab.groupby("doc_id"):
                if doc_id in seen:
                    continue
                g = g.sort_values(["start", "end"])
                spans = g.to_dict(orient="records")
                ok = False
                for i in range(len(spans)):
                    for j in range(i + 1, len(spans)):
                        if spans[i]["label"] == spans[j]["label"]:
                            continue
                        s1, e1 = int(spans[i]["start"]), int(spans[i]["end"])
                        s2, e2 = int(spans[j]["start"]), int(spans[j]["end"])
                        inter = max(0, min(e1, e2) - max(s1, s2))
                        union = (e1 - s1) + (e2 - s2) - inter
                        iou = (inter / union) if union > 0 else 0.0
                        if iou >= 0.30 and abs(max(e1, e2) - min(s1, s2)) <= 240:
                            # add a snippet covering both spans; include both spans in JSON
                            left = max(0, min(s1, s2) - 120)
                            right = min(len(spans[i]["text"]), max(e1, e2) + 120)
                            t = spans[i]["text"]
                            snippet = (t[left:right]).strip()
                            off1s, off1e = s1 - left, e1 - left
                            off2s, off2e = s2 - left, e2 - left
                            s1_examples.append(
                                {
                                    "doc_id": doc_id,
                                    "text": snippet,
                                    "spans": [
                                        {
                                            "label": spans[i]["label"],
                                            "start": off1s,
                                            "end": off1e,
                                        },
                                        {
                                            "label": spans[j]["label"],
                                            "start": off2s,
                                            "end": off2e,
                                        },
                                    ],
                                    "meta": {"reason": f"ambiguous_pair_{a}_{b}"},
                                }
                            )
                            seen.add(doc_id)
                            ok = True
                            break
                    if ok:
                        break

    # --- NEW: drop S1 few-shots that have no valid spans (or junk spans) ---
    _before = len(s1_examples)
    s1_examples = [ex for ex in s1_examples if _example_has_valid_span(ex)]
    print(
        f"[S1] filtered few-shots: kept={len(s1_examples)} dropped={_before - len(s1_examples)} (invalid/empty)"
    )

    # ---- Build audit log for reproducibility ----
    audit = {
        "seed": args.seed,
        "shots": {
            "s1_per_label": args.shots_s1_per_label,
            "s1_outliers_per_label": args.shots_s1_outliers,
            "s2_per_class": args.shots_s2_per_class,
        },
        "s1_examples": [],
        "s2_examples": [],
    }

    # S1 audit
    for ex in s1_examples:
        rec = {
            "doc_id": ex.get("doc_id"),
            "reason": ex.get("meta", {}).get("reason", ""),
            "text_len": len(ex.get("text", "")),
        }
        rec["spans"] = []
        for sp in ex.get("spans", []):
            L = sp.get("label")
            s = int(sp.get("start", 0))
            e = int(sp.get("end", 0))
            pf = compute_prior_features(priors, L, s, e, rec["text_len"])
            b = detect_boundary_hit(boundary, L, ex.get("text", ""), s, e)
            rec["spans"].append(
                {
                    "label": L,
                    "start": s,
                    "end": e,
                    **pf,
                    "boundary_hit": b["hit"],
                    "boundary_cues": b["cues"],
                }
            )
        audit["s1_examples"].append(rec)

    # S2 audit
    for ex in s2_fewshots:
        audit["s2_examples"].append(
            {
                "doc_id": ex.get("doc_id"),
                "label": ex.get("label"),
                "markers_count": len(ex.get("markers", [])),
                "source_label": ex.get("source_label", ""),
            }
        )

    # Negative S1 exemplar (teaches: empty JSON is valid)
    # neg_s1 = pick_negative_s1_snippet(dev_df, seed=args.seed)
    # if neg_s1:
    #    s1_examples.append(neg_s1)

    # --- NEW: guarantee per-label coverage via relaxed backfill (Actor/Action gaps etc.) ---
    extras = backfill_missing_labels(
        train_df=train_df,
        s1_examples=s1_examples,
        max_per_label=args.shots_s1_per_label,  # usually 2
    )
    if extras:
        s1_examples.extend(extras)
        print(f"[S1] Backfill added {len(extras)} examples to meet per-label coverage.")

    # --- Ensure all 5 S1 markers are covered at least 2x ---
    min_per_label = 2
    label_counter = Counter()
    for ex in s1_examples:
        for m in ex.get("spans", []):
            label = m.get("label")
            if label:
                label_counter[label] += 1

    # Log current status
    print("🔍 Fewshot label counts before trimming:", dict(label_counter))

    # Filter to balance if needed
    final_s1_examples = []
    used = set()
    per_label_buffer = {lab: [] for lab in ALLOWED_S1}

    for ex in s1_examples:
        added = False
        for m in ex.get("spans", []):
            lab = m.get("label")
            if lab in ALLOWED_S1 and ex["text"] not in used:
                per_label_buffer[lab].append(ex)
                used.add(ex["text"])
                added = True
        if not added:
            final_s1_examples.append(ex)

    # Now collect balanced
    balanced = []
    for lab in ALLOWED_S1:
        balanced.extend(per_label_buffer[lab][:min_per_label])

    # Add remainder until max_n
    seen_texts = set(e["text"] for e in balanced)
    for ex in s1_examples:
        if len(balanced) >= max_n:
            break
        if ex["text"] not in seen_texts:
            balanced.append(ex)
            seen_texts.add(ex["text"])

    # Overwrite
    s1_examples = balanced[:max_n]
    print(
        "✅ Final S1 fewshot label counts:",
        dict(
            Counter(
                l
                for ex in s1_examples
                for l in [m["label"] for m in ex.get("spans", [])]
            )
        ),
    )

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

    print("\n[select_fewshot_examples] Wrote:")
    print(f"  - {fs_path}")
    print(f"  - {out_dir / 'fewshot_policy.json'}")
    print(f"  S1 count: {len(s1_examples)}  | S2 count: {len(s2_fewshots)}")


if __name__ == "__main__":
    main()
