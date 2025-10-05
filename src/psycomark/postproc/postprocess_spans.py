#!/usr/bin/env python3
"""
Rule-based post-processor for span predictions.
- Resolves Action↔Effect and Actor↔Victim overlaps using priors & overlap stats.
- Keeps both spans when pairwise IoU is common but uncertain per stats.
- Writes post-processed JSONL and a small summary.

Usage:
  python postprocess_spans.py \
     --pred path/to/bedrock_preds_s1_*.jsonl \
     --data path/to/dev.jsonl \
     --priors path/to/length_position_priors.json \
     --pairs  path/to/overlap_pair_stats.json \
     --pairs-ci path/to/overlap_pair_stats_ci.json \
     --cdf path/to/first_occurrence_cdf.csv  # optional
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

ALLOWED = {"Actor", "Action", "Effect", "Victim", "Evidence"}


def load_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


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
    If IoU>=0.6 and Action/Effect 'starts_first_rate' ~ tie (≈0.5),
    prefer the label whose boundary better matches start Beta mode AND length lognormal (smaller composite distance).
    Else, if IoU@0.5 rate < 0.5 in corpus, keep both.
    """
    sA, eA = spanA["start"], spanA["end"]
    sE, eE = spanE["start"], spanE["end"]
    i = iou((sA, eA), (sE, eE))
    doc_pos_A = sA / max(1, text_len)
    doc_pos_E = sE / max(1, text_len)

    stats = pair_stats_ci.get("Action/Effect", {})
    iou05_rate = stats.get("iou@0.5", 0.5)
    # keep-both policy if overlaps rarely exceed 0.5 in corpus
    if i < 0.6:
        return "keep_both"

    # tie on who starts first? we use closeness to priors
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
        # still undecided; if corpus says high IoU@0.5 is common, pick earlier span as Action heuristic
        if iou05_rate >= 0.5:
            return "prefer_action" if sA <= sE else "prefer_effect"
        else:
            return "keep_both"


def decide_actor_victim(
    spanActor, spanVictim, priors, text_len, cdf_q1=None, pair_stats_ci=None
):
    """
    Prefer Actor when normalized start position is in the first quartile (per CDF if available; else 0.25),
    else keep Victim. Suppress duplicates when containment is likely.
    """
    sX, eX = spanActor["start"], spanActor["end"]
    sY, eY = spanVictim["start"], spanVictim["end"]
    i = iou((sX, eX), (sY, eY))
    q1 = float(cdf_q1.get("Actor", 0.25)) if cdf_q1 else 0.25
    posA = sX / max(1, text_len)
    # containment heuristic: if one fully contains the other, keep the smaller span
    contains = (sX <= sY and eX >= eY) or (sY <= sX and eY >= eX)
    if contains:
        return "keep_smaller"
    if i < 0.5:
        return "keep_both"
    return "prefer_actor" if posA <= q1 else "prefer_victim"


def apply_rules_one(doc_pred, priors, pair_stats_ci, cdf_q1):
    text = doc_pred.get("text") or ""
    spans = [s for s in (doc_pred.get("prediction") or []) if s.get("label") in ALLOWED]
    if not spans:
        return spans, {"decisions": 0, "removed": 0}

    changed = 0
    removed = 0
    keep = [True] * len(spans)

    # iterate pairwise and apply specific rules
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            if not (keep[i] and keep[j]):
                continue
            Li, Lj = spans[i]["label"], spans[j]["label"]
            pair = tuple(sorted([Li, Lj]))
            if pair == ("Action", "Effect"):
                A = spans[i] if Li == "Action" else spans[j]
                E = spans[j] if Li == "Action" else spans[i]
                dec = decide_action_effect(A, E, priors, len(text), pair_stats_ci)
                if dec == "prefer_action":
                    # drop Effect
                    idx = j if Li == "Action" else i
                    keep[idx] = False
                    removed += 1
                    changed += 1
                elif dec == "prefer_effect":
                    idx = i if Li == "Action" else j
                    keep[idx] = False
                    removed += 1
                    changed += 1
                # else keep_both -> no change
            elif pair == ("Actor", "Victim"):
                Act = spans[i] if Li == "Actor" else spans[j]
                Vic = spans[j] if Li == "Actor" else spans[i]
                # optional: only if substantial overlap
                dec = decide_actor_victim(
                    Act,
                    Vic,
                    priors,
                    len(text),
                    cdf_q1=cdf_q1,
                    pair_stats_ci=pair_stats_ci,
                )
                if dec == "prefer_actor":
                    idx = j if Li == "Actor" else i
                    keep[idx] = False
                    removed += 1
                    changed += 1
                elif dec == "prefer_victim":
                    idx = i if Li == "Actor" else j
                    keep[idx] = False
                    removed += 1
                    changed += 1
                elif dec == "keep_smaller":
                    # drop the longer span
                    li = spans[i]["end"] - spans[i]["start"]
                    lj = spans[j]["end"] - spans[j]["start"]
                    if li >= lj:
                        keep[i] = False
                    else:
                        keep[j] = False
                    removed += 1
                    changed += 1
            # other pairs: leave as-is

    new_spans = [s for s, k in zip(spans, keep) if k]
    return new_spans, {"decisions": changed, "removed": removed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pred",
        required=True,
        help="JSONL from Bedrock runner with fields: doc_id, prediction, raw/error",
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

    texts = {
        r["doc_id"]: (r["text"] or "")
        for _, r in pd.read_json(args.data, lines=True).iterrows()
    }
    PRIORS = json.loads(Path(args.priors).read_text())
    pair_stats_ci = {}
    if args.pairs_ci and Path(args.pairs_ci).exists():
        pair_stats_ci.update(json.loads(Path(args.pairs_ci).read_text()))
    elif args.pairs and Path(args.pairs).exists():
        pair_stats_ci.update(json.loads(Path(args.pairs).read_text()))

    # optional first-occurrence quartiles
    cdf_q1 = None
    if args.cdf and Path(args.cdf).exists():
        cdf = pd.read_csv(args.cdf, index_col=0)
        # column '0.25' from previous export; if not present, default to 0.25
        if "0.25" in cdf.columns:
            cdf_q1 = {lab: float(cdf.loc[lab, "0.25"]) for lab in cdf.index}

    in_path = Path(args.pred)
    out_path = (
        Path(args.out) if args.out else in_path.with_name(in_path.stem + "_pp.jsonl")
    )

    changes = 0
    removed = 0
    total = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for rec in load_jsonl(in_path):
            doc_id = rec.get("doc_id")
            # copy original prediction list under 'prediction_raw' for A/B
            rec["prediction_raw"] = rec.get("prediction", [])
            # attach text (if not already)
            rec_text = texts.get(doc_id, "")
            rec["text"] = rec.get("text") or rec_text
            new_spans, stats = apply_rules_one(
                {"prediction": rec["prediction"], "text": rec["text"]},
                PRIORS,
                pair_stats_ci,
                cdf_q1,
            )
            rec["prediction"] = new_spans
            changes += stats["decisions"]
            removed += stats["removed"]
            total += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

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
