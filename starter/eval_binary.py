import argparse
import csv
import logging
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import orjson

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib optional
    plt = None

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    roc_auc_score,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

YES, NO = "Yes", "No"


def _has_labels(jsonl_path: Path, sample=200):
    """Return True if file appears to contain ground-truth labels."""
    if not jsonl_path or not jsonl_path.exists():
        return False
    n, y = 0, 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            r = orjson.loads(ln)
            lab = r.get("doc_label") or r.get("conspiracy")
            if lab not in (None, "", "null"):
                y += 1
            n += 1
            if n >= sample:
                break
    return y > 0


def _load_gold(jsonl_path: Path):
    ids, y = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            r = orjson.loads(ln)
            _id = r.get("_id") or r.get("doc_id")
            lab = r.get("doc_label") or r.get("conspiracy")
            if isinstance(lab, str):
                ll = lab.strip().lower()
                if ll in ("conspiracy", "yes"):
                    y.append(1)
                elif ll in ("non", "no"):
                    y.append(0)
                else:
                    continue  # skip cant_tell / unknown
            else:
                continue
            ids.append(_id)
    return ids, y


def _load_submission(path: Path):
    pred = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            r = orjson.loads(ln)
            _id = r.get("_id")
            lab = r.get("conspiracy")
            pred[_id] = 1 if lab == YES else 0
    return pred


def _load_probs(path: Path):
    if not path or not path.exists():
        return {}
    p = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            r = orjson.loads(ln)
            _id = r.get("_id")
            p[_id] = float(r.get("p_conspiracy", 0.5))
    return p


def _metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    auc = float("nan")
    if y_prob is not None and len(set(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            pass
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return dict(
        acc=acc,
        prec=p,
        rec=r,
        f1=f1,
        f1_macro=f1_macro,
        f1_weighted=f1_w,
        auc=auc,
        cm=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)),
    )


def _write_confusion_csv(path: Path, rows):
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "tn", "fp", "fn", "tp"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logging.info(f"Confusion matrix written to {path}")


def _save_confusion_plot(cm_counts: dict, out_dir: Path, stem: str = "confusion"):
    if plt is None:
        logging.warning(
            "matplotlib not available; skipping confusion matrix plot creation"
        )
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = np.array(
        [[cm_counts["tn"], cm_counts["fp"]], [cm_counts["fn"], cm_counts["tp"]]]
    )
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted No", "Predicted Yes"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actual No", "Actual Yes"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                matrix[i, j],
                ha="center",
                va="center",
                color="black",
                fontsize=12,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path = out_dir / f"{stem}_confusion.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logging.info(f"Confusion matrix plot saved to {out_path}")


def _save_roc_plot(y_true, y_prob, out_dir: Path, stem: str = "roc"):
    if plt is None:
        logging.warning("matplotlib not available; skipping ROC plot creation")
        return
    if y_prob is None or len(set(y_true)) < 2:
        logging.info("ROC plot skipped (needs probabilities and both classes present)")
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(fpr, tpr, label="ROC curve")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path = out_dir / f"{stem}_roc.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logging.info(f"ROC curve saved to {out_path}")


def eval_on_dev(
    gold_file: Path,
    submission_file: Path,
    probs_file: Path = None,
    confusion_csv: Path = None,
):
    logging.info("Mode A: Evaluating submission against DEV gold.")
    ids, y = _load_gold(gold_file)
    pred = _load_submission(submission_file)
    prob = _load_probs(probs_file) if probs_file else {}
    y_pred = [pred.get(i, 0) for i in ids]
    y_prob = [prob.get(i, 0.5) for i in ids] if prob else None
    if y_prob is not None:
        # Use probabilities to derive predictions at a configurable threshold
        thr = getattr(eval_on_dev, "_threshold", 0.5)
        y_pred = [1 if p >= thr else 0 for p in y_prob]
    else:
        # Fall back to hard labels
        y_pred = [pred.get(i, 0) for i in ids]

    m = _metrics(y, y_pred, y_prob)
    logging.info(
        f"DEV -> acc={m['acc']:.3f} p={m['prec']:.3f} r={m['rec']:.3f} f1={m['f1']:.3f} auc={m['auc']:.3f}"
    )
    logging.info(
        f"DEV (macro/weighted) -> f1_macro={m['f1_macro']:.3f} f1_weighted={m['f1_weighted']:.3f}"
    )
    cm = m["cm"]
    logging.info(
        f"DEV confusion matrix -> tn={cm['tn']} fp={cm['fp']} fn={cm['fn']} tp={cm['tp']}"
    )
    _save_confusion_plot(cm, submission_file.parent, submission_file.stem)
    _save_roc_plot(y, y_prob, submission_file.parent, submission_file.stem)
    # Save a compact Codabench-ish scores.json next to submission
    try:
        out_scores = {
            "Accuracy": float(m["acc"]),
            "F1_binary": float(m["f1"]),
            "F1_macro": float(m["f1_macro"]),
            "F1_weighted": float(m["f1_weighted"]),
            "AUC": float(m["auc"]),
            "TN": m["cm"]["tn"],
            "FP": m["cm"]["fp"],
            "FN": m["cm"]["fn"],
            "TP": m["cm"]["tp"],
        }
        (submission_file.parent / "scores.json").write_bytes(
            orjson.dumps(out_scores, option=orjson.OPT_INDENT_2)
        )
        logging.info(f"Scores written to {submission_file.parent / 'scores.json'}")
    except Exception as e:
        logging.warning(f"Could not write scores.json: {e}")
    if confusion_csv:
        _write_confusion_csv(confusion_csv, [dict(split="dev", **cm)])
    return m


def _load_train_docclf(path: Path):
    xs, ys, ids = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            r = orjson.loads(ln)
            lab = (r.get("doc_label") or "").strip().lower()
            if lab not in ("conspiracy", "non"):  # exclude cant_tell
                continue
            _id = r.get("_id") or r.get("doc_id")
            xs.append({"_id": _id, "text": r.get("text", "")})
            ys.append(1 if lab == "conspiracy" else 0)
            ids.append(_id)
    return xs, ys, ids


def _load_folds(path: Path):
    fold_of = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            r = orjson.loads(ln)
            fold_of[r["doc_id"]] = int(r["fold"])
    return fold_of


def run_cv(
    train_docclf: Path,
    folds_path: Path,
    runner: Path,
    model_id: str = None,
    region: str = None,
    eda_root: Path = None,
    max_tokens=800,
    temperature=0.0,
    confusion_csv: Path = None,
):
    logging.info("Mode B: DEV unlabeled -> running 5-fold CV on TRAIN.")
    X, y, ids = _load_train_docclf(train_docclf)
    fold_of = _load_folds(folds_path)

    folds = defaultdict(list)
    for i, _id in enumerate(ids):
        if _id in fold_of:
            folds[fold_of[_id]].append(i)

    per_fold = []
    conf_rows = []
    for fold, idxs in sorted(folds.items()):
        with tempfile.TemporaryDirectory() as td:
            test = [{"_id": ids[i], "text": X[i]["text"]} for i in idxs]
            tpath = Path(td) / f"fold{fold}_test.jsonl"
            spath = Path(td) / f"fold{fold}_sub.jsonl"
            ppath = Path(td) / f"fold{fold}_probs.jsonl"
            with open(tpath, "w", encoding="utf-8") as f:
                for r in test:
                    f.write(orjson.dumps(r).decode() + "\n")

            cmd = [
                "uv",
                "run",
                "python",
                str(runner),
                "--test-file",
                str(tpath),
                "--submission-file",
                str(spath),
                "--probs-file",
                str(ppath),
            ]
            if model_id:
                cmd += ["--model-id", model_id]
            if region:
                cmd += ["--region", region]
            if eda_root:
                cmd += ["--eda-root", str(eda_root)]
            cmd += ["--max-tokens", str(max_tokens), "--temperature", str(temperature)]

            logging.info(f"[fold {fold}] running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

            pred = _load_submission(spath)
            prob = _load_probs(ppath)

            y_true = [y[i] for i in idxs]
            y_pred = [pred.get(ids[i], 0) for i in idxs]
            y_prob = [prob.get(ids[i], 0.5) for i in idxs]

            m = _metrics(y_true, y_pred, y_prob)
            logging.info(
                f"[fold {fold}] acc={m['acc']:.3f} p={m['prec']:.3f} r={m['rec']:.3f} f1={m['f1']:.3f} auc={m['auc']:.3f}"
            )
            cm = m["cm"]
            logging.info(
                f"[fold {fold}] confusion matrix -> tn={cm['tn']} fp={cm['fp']} fn={cm['fn']} tp={cm['tp']}"
            )
            per_fold.append(m)
            conf_rows.append(dict(split=f"fold_{fold}", **cm))

    # aggregate
    keys = ["acc", "prec", "rec", "f1", "auc"]
    mean_std = {
        k: (np.nanmean([m[k] for m in per_fold]), np.nanstd([m[k] for m in per_fold]))
        for k in keys
    }
    logging.info("\n== CV (mean ± std) ==")
    for k, (mu, sig) in mean_std.items():
        logging.info(f"{k}: {mu:.3f} ± {sig:.3f}")
    if confusion_csv and conf_rows:
        totals = {k: sum(row[k] for row in conf_rows) for k in ("tn", "fp", "fn", "tp")}
        conf_rows.append(dict(split="overall", **totals))
        _write_confusion_csv(confusion_csv, conf_rows)
    return per_fold, mean_std


def main():
    ap = argparse.ArgumentParser()
    # Dev-style evaluation (if labeled)
    ap.add_argument(
        "--gold-file", default=None, help="If labeled, evaluate submission directly."
    )
    ap.add_argument("--submission-file", default=None)
    ap.add_argument("--probs-file", default=None)
    # CV evaluation (fallback)
    ap.add_argument("--train-docclf", default=None)
    ap.add_argument("--folds", default=None)
    ap.add_argument("--runner", default="starter/llm_infer_binary.py")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--region", default=None)
    ap.add_argument("--eda-root", default=None)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument(
        "--confusion-csv",
        default="confusion_matrix.csv",
        help="Path to write confusion matrix counts as CSV (set empty to skip).",
    )
    ap.add_argument(
        "--sweep-threshold",
        action="store_true",
        help="If set (and probs available), sweep threshold ∈ [0.05..0.95] and log best F1.",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold on p_conspiracy when --probs-file is provided.",
    )
    args = ap.parse_args()

    gold_path = Path(args.gold_file) if args.gold_file else None
    sub_path = Path(args.submission_file) if args.submission_file else None
    probs_path = Path(args.probs_file) if args.probs_file else None
    conf_csv_path = Path(args.confusion_csv) if args.confusion_csv else None

    # Prefer DEV eval if labeled gold is provided
    if gold_path and _has_labels(gold_path):
        # plumb threshold into eval function
        setattr(eval_on_dev, "_threshold", float(args.threshold))
        if not sub_path or not sub_path.exists():
            raise SystemExit("submission file missing for DEV eval")
        eval_on_dev(gold_path, sub_path, probs_path, conf_csv_path)
        if args.sweep_threshold and probs_path and probs_path.exists():
            ids, y = _load_gold(gold_path)
            prob = _load_probs(probs_path)
            y_prob = [prob.get(i, 0.5) for i in ids]
            best = (0.0, 0.5)  # f1, thr
            for t in [i / 100 for i in range(5, 96, 5)]:
                y_pred = [1 if p >= t else 0 for p in y_prob]
                m = _metrics(y, y_pred, y_prob)
                if m["f1"] > best[0]:
                    best = (m["f1"], t)
            logging.info(f"[SWEEP] best F1={best[0]:.3f} at threshold={best[1]:.2f}")
        eval_on_dev(gold_path, sub_path, probs_path, conf_csv_path)
        return

    # Otherwise, run CV on train using pipeline outputs
    train_path = Path(args.train_docclf) if args.train_docclf else None
    folds_path = Path(args.folds) if args.folds else None
    if not (train_path and folds_path and train_path.exists() and folds_path.exists()):
        raise SystemExit(
            "DEV unlabeled and no train/folds given. Provide --train-docclf and --folds."
        )
    run_cv(
        train_docclf=train_path,
        folds_path=folds_path,
        runner=Path(args.runner),
        model_id=args.model_id,
        region=args.region,
        eda_root=Path(args.eda_root) if args.eda_root else None,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        confusion_csv=conf_csv_path,
    )


if __name__ == "__main__":
    main()
