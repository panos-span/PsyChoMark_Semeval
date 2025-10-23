#!/usr/bin/env python3
import argparse, json, pathlib, subprocess, sys
from typing import Dict, List
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def _read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                yield json.loads(ln)


def eval_s1(gt_file: str, pred_file: str, iou: float = 0.5) -> Dict[str, float]:
    """
    Wrapper around token-IoU evaluator (same as Codabench alignment).
    """
    out_json = pathlib.Path("scores_s1.json")
    if out_json.exists():
        out_json.unlink()
    subprocess.check_call(
        [
            sys.executable,
            "starter/eval_token.py",
            "--ground_truth_file",
            gt_file,
            "--prediction_file",
            pred_file,
            "--scores_output_file",
            str(out_json),
            "--iou_threshold",
            str(iou),
        ]
    )
    return json.loads(out_json.read_text(encoding="utf-8"))


def eval_s2(gt_file: str, pred_file: str) -> Dict[str, float]:
    gold_ids, y_true = [], []
    for r in _read_jsonl(gt_file):
        lab = (r.get("doc_label") or r.get("conspiracy") or "").strip().lower()
        if lab not in ("conspiracy", "non", "yes", "no"):
            continue
        gold_ids.append(r.get("_id") or r.get("doc_id"))
        y_true.append(1 if lab in ("conspiracy", "yes") else 0)

    pred_map = {
        r.get("_id"): 1 if r.get("conspiracy") == "Yes" else 0
        for r in _read_jsonl(pred_file)
    }
    y_pred = [pred_map.get(i, 0) for i in gold_ids]

    acc = accuracy_score(y_true, y_pred)
    p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except ValueError:
        tn = fp = fn = tp = 0
    return {
        "Accuracy": acc,
        "Precision_binary": p_bin,
        "Recall_binary": r_bin,
        "F1_binary": f1_bin,
        "F1_macro": f1_macro,
        "F1_weighted": f1_weighted,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-s1", help="Ground truth JSONL with 'markers' (dev/train).")
    ap.add_argument("--pred-s1", default="submission_s1.jsonl")
    ap.add_argument(
        "--gt-s2", help="Ground truth JSONL with 'doc_label' or 'conspiracy'."
    )
    ap.add_argument("--pred-s2", default="submission_s2.jsonl")
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    if args.gt_s1 and args.pred_s1:
        s1 = eval_s1(args.gt_s1, args.pred_s1, iou=args.iou)
        print("[S1] Token-IoU scores:", json.dumps(s1, indent=2))

    if args.gt_s2 and args.pred_s2:
        s2 = eval_s2(args.gt_s2, args.pred_s2)
        print("[S2] Metrics:", json.dumps(s2, indent=2))


if __name__ == "__main__":
    main()
