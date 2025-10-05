#!/usr/bin/env python3
import argparse, json, numpy as np
from scipy.optimize import minimize


def load_probs(path):
    ids, p1 = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ids.append(d["_id"])
            p1.append(float(d["p_conspiracy"]))
    return ids, np.asarray(p1)


def load_labels(path):
    y = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            y[d["_id"]] = 1 if d["conspiracy"] in ("Yes", "conspiracy") else 0
    return y


def nll_temp(t, p, y):
    # temperature on logit: logit' = logit/ T  ⇒ p' = σ(logit/T)
    eps = 1e-7
    logit = np.log(p + eps) - np.log(1 - p + eps)
    p_adj = 1 / (1 + np.exp(-logit / np.maximum(t, 1e-3)))
    return -np.mean(y * np.log(p_adj + eps) + (1 - y) * np.log(1 - p_adj + eps))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True, help="dev_probs.jsonl")
    ap.add_argument(
        "--labels",
        required=True,
        help="dev submission-like jsonl with _id & conspiracy",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ids, p = load_probs(args.probs)
    ymap = load_labels(args.labels)
    y = np.asarray([ymap[i] for i in ids])

    res = minimize(lambda t: nll_temp(t[0], p, y), x0=[1.0], bounds=[(0.05, 5.0)])
    T = float(res.x[0])

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"temperature": T}, f, indent=2)
    print(f"Saved temperature={T:.3f} → {args.out}")
