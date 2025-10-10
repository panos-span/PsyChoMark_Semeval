#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rule-based post-processor for span predictions.
- Resolves Action↔Effect and Actor↔Victim overlaps using priors & overlap stats.
- Keeps both spans when pairwise IoU is common but uncertain per stats.
- Writes post-processed JSONL and a small summary.

Usage:
  python postprocess_spans.py ^
     --pred path/to/dev_raw.jsonl ^
     --data path/to/dev.jsonl ^
     --priors path/to/length_position_priors.json ^
     --pairs-ci path/to/overlap_pair_stats_ci.json ^
     --cdf path/to/first_occurrence_cdf.csv ^
     --out runs/s1_merge/dev_pp.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import os
import numpy as np
import pandas as pd

ALLOWED = {"Actor", "Action", "Effect", "Victim", "Evidence"}


# -----------------------------
# IO helpers
# -----------------------------
def load_jsonl(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                continue


def normalize_spans_record(rec: dict):
    """
    Accept multiple input schemas and return a normalized (_id, spans, base_text).

    Accepts:
      - {"_id"/"doc_id"/"id", "markers": [ {label,start,end,text?,score?} ... ]}
      - {"_id"/"doc_id"/"id", "prediction": [ {label/startIndex,...} or {"spans":[...]} ]}
    Returns:
      did: str | None
      spans: List[dict] | None   (each: {label,start,end,text?,score?})
      base_text: str | ""
    """
    did = rec.get("_id") or rec.get("doc_id") or rec.get("id")
    base_text = rec.get("text", "") or ""
    spans = None

    # unified schema (markers)
    if isinstance(rec.get("markers"), list):
        spans = rec["markers"]

    # older/alternative schema (prediction)
    elif "prediction" in rec:
        pred = rec["prediction"]
        if isinstance(pred, dict) and isinstance(pred.get("spans"), list):
            spans = pred["spans"]
        elif isinstance(pred, list):
            spans = pred

    if did is None or spans is None:
        return None, None, base_text

    # normalize fields
    out = []
    for s in spans:
        if not isinstance(s, dict):
            continue
        lab = s.get("label") or s.get("type")
        st = s.get("start") or s.get("startIndex")
        en = s.get("end") or s.get("endIndex")
        if lab is None or st is None or en is None:
            continue
        item = {"label": str(lab), "start": int(st), "end": int(en)}
        if "text" in s:
            item["text"] = s["text"]
        if "score" in s:
            try:
                item["score"] = float(s["score"])
            except Exception:
                pass
        out.append(item)

    return did, out, base_text


# -----------------------------
# math & rule helpers
# -----------------------------
def iou(a, b):
    s1, e1 = a
    s2, e2 = b
    inter = max(0, min(e1, e2) - max(s1, s2))
    if inter <= 0:
        return 0.0
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / union if union > 0 else 0.0


def beta_mode(alpha, beta):
    if alpha > 1 and beta > 1:
        return (alpha - 1) / (alpha + beta - 2)
    return alpha / (alpha + beta) if (alpha > 0 and beta > 0) else 0.5


def prior_z_len(priors, label, span_len):
    p = priors.get(label, {}).get("length_lognorm", {"mu": 0.0, "sigma": 1.0})
    mu, sig = p["mu"], max(1e-6, p["sigma"])
    return abs((math.log(max(1, span_len)) - mu) / sig)


def prior_dist_pos(priors, label, start_pos):
    sb = priors.get(label, {}).get("start_beta", {"alpha": 1.0, "beta": 1.0})
    mode = beta_mode(sb["alpha"], sb["beta"])
    return abs(start_pos - mode)


def decide_action_effect(spanA, spanE, priors, text_len, pair_stats_ci):
    """
    If IoU>=0.6 -> decide by prior closeness.
    Else keep both.
    Minor tie-break: if corpus IoU@0.5 is high (>=0.5), prefer earlier as Action.
    """
    sA, eA = spanA["start"], spanA["end"]
    sE, eE = spanE["start"], spanE["end"]
    i = iou((sA, eA), (sE, eE))
    doc_pos_A = sA / max(1, text_len)
    doc_pos_E = sE / max(1, text_len)

    stats = pair_stats_ci.get("Action/Effect", {})
    iou05_rate = stats.get("iou@0.5", 0.5)

    if i < 0.6:
        return "keep_both"

    dA = 0.6 * prior_dist_pos(priors, "Action", doc_pos_A) + 0.4 * prior_z_len(
        priors, "Action", eA - sA
    )
    dE = 0.6 * prior_dist_pos(priors, "Effect", doc_pos_E) + 0.4 * prior_z_len(
        priors, "Effect", eE - sE
    )
    if dA < dE:
        return "prefer_action"
    elif dE < dA:
        return "prefer_effect"
    else:
        return "prefer_action" if (iou05_rate >= 0.5 and sA <= sE) else "keep_both"


def decide_actor_victim(
    spanActor, spanVictim, priors, text_len, cdf_q1=None, pair_stats_ci=None
):
    """
    Prefer Actor when normalized start is in first quartile (from CDF if available, else 0.25).
    If containment, keep smaller. If IoU<0.5 keep both.
    """
    sX, eX = spanActor["start"], spanActor["end"]
    sY, eY = spanVictim["start"], spanVictim["end"]
    i = iou((sX, eX), (sY, eY))
    q1 = float(cdf_q1.get("Actor", 0.25)) if cdf_q1 else 0.25
    posA = sX / max(1, text_len)
    contains = (sX <= sY and eX >= eY) or (sY <= sX and eY >= eX)

    if contains:
        return "keep_smaller"
    if i < 0.5:
        return "keep_both"
    return "prefer_actor" if posA <= q1 else "prefer_victim"


def apply_rules_one(spans, text, priors, pair_stats_ci, cdf_q1):
    """Apply pairwise rules to a list of spans; return filtered list and stats."""
    spans = [s for s in spans if s.get("label") in ALLOWED]
    if not spans:
        return spans, {"decisions": 0, "removed": 0}

    changed = 0
    removed = 0
    keep = [True] * len(spans)
    n = len(spans)

    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            Li, Lj = spans[i]["label"], spans[j]["label"]
            pair = tuple(sorted([Li, Lj]))

            if pair == ("Action", "Effect"):
                A = spans[i] if Li == "Action" else spans[j]
                E = spans[j] if Li == "Action" else spans[i]
                dec = decide_action_effect(A, E, priors, len(text), pair_stats_ci)
                if dec == "prefer_action":
                    idx = j if Li == "Action" else i
                    keep[idx] = False
                    changed += 1
                    removed += 1
                elif dec == "prefer_effect":
                    idx = i if Li == "Action" else j
                    keep[idx] = False
                    changed += 1
                    removed += 1
                # keep_both → no-op

            elif pair == ("Actor", "Victim"):
                Act = spans[i] if Li == "Actor" else spans[j]
                Vic = spans[j] if Li == "Actor" else spans[i]
                dec = decide_actor_victim(
                    Act, Vic, priors, len(text), cdf_q1, pair_stats_ci
                )
                if dec == "prefer_actor":
                    idx = j if Li == "Actor" else i
                    keep[idx] = False
                    changed += 1
                    removed += 1
                elif dec == "prefer_victim":
                    idx = i if Li == "Actor" else j
                    keep[idx] = False
                    changed += 1
                    removed += 1
                elif dec == "keep_smaller":
                    li = spans[i]["end"] - spans[i]["start"]
                    lj = spans[j]["end"] - spans[j]["start"]
                    if li >= lj:
                        keep[i] = False
                    else:
                        keep[j] = False
                    changed += 1
                    removed += 1
            # other pairs: leave as-is

    new_spans = [s for s, k in zip(spans, keep) if k]
    return new_spans, {"decisions": changed, "removed": removed}


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pred",
        required=True,
        help="JSONL: accepts 'markers' or 'prediction' schemas.",
    )
    ap.add_argument("--data", required=True, help="dev/train jsonl (to access text)")
    ap.add_argument("--priors", required=True, help="length_position_priors.json")
    ap.add_argument("--pairs", required=False, help="overlap_pair_stats.json")
    ap.add_argument("--pairs-ci", required=False, help="overlap_pair_stats_ci.json")
    ap.add_argument("--cdf", required=False, help="first_occurrence_cdf.csv (optional)")
    ap.add_argument(
        "--out", required=False, help="output JSONL (default adds _pp before extension)"
    )
    args = ap.parse_args()

    # Load text by id to backfill
    df_data = pd.read_json(args.data, lines=True)
    id_to_text = {}
    for _, r in df_data.iterrows():
        did = r.get("doc_id") or r.get("_id") or r.get("id")
        if did is not None:
            id_to_text[did] = r.get("text", "") or ""

    PRIORS = json.loads(Path(args.priors).read_text(encoding="utf-8"))
    pair_stats_ci = {}
    if args.pairs_ci and Path(args.pairs_ci).exists():
        pair_stats_ci.update(
            json.loads(Path(args.pairs_ci).read_text(encoding="utf-8"))
        )
    elif args.pairs and Path(args.pairs).exists():
        pair_stats_ci.update(json.loads(Path(args.pairs).read_text(encoding="utf-8")))

    # optional first-occurrence quartiles
    cdf_q1 = None
    if args.cdf and Path(args.cdf).exists():
        cdf = pd.read_csv(args.cdf, index_col=0)
        if "0.25" in cdf.columns:
            cdf_q1 = {lab: float(cdf.loc[lab, "0.25"]) for lab in cdf.index}

    in_path = Path(args.pred)
    out_path = (
        Path(args.out) if args.out else in_path.with_name(in_path.stem + "_pp.jsonl")
    )
    os.makedirs(out_path.parent.as_posix(), exist_ok=True)

    changes = 0
    removed = 0
    total = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for rec in load_jsonl(in_path):
            did, spans, base_text = normalize_spans_record(rec)
            if did is None or spans is None:
                continue

            # backfill text; and fill per-span "text" slices if absent
            text = base_text or id_to_text.get(did, "")
            if text:
                for s in spans:
                    if (
                        "text" not in s
                        and isinstance(s.get("start"), int)
                        and isinstance(s.get("end"), int)
                    ):
                        st, en = s["start"], s["end"]
                        if 0 <= st < en <= len(text):
                            s["text"] = text[st:en]

            new_spans, stats = apply_rules_one(
                spans, text, PRIORS, pair_stats_ci, cdf_q1
            )
            changes += stats["decisions"]
            removed += stats["removed"]
            total += 1

            # preserve original, write unified 'prediction'
            out_rec = dict(rec)
            out_rec["prediction_raw"] = spans
            out_rec["prediction"] = new_spans
            if "text" not in out_rec:
                out_rec["text"] = text
            if "_id" not in out_rec and "doc_id" in out_rec:
                out_rec["_id"] = out_rec["doc_id"]

            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    summary = {
        "input": str(in_path),
        "output": str(out_path),
        "docs": total,
        "decisions": changes,
        "spans_removed": removed,
    }
    with open(out_path.with_suffix(".summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
