#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
import numpy as np


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


def prior_penalty(priors, label, span_len, start_pos):
    lp = priors.get(label, {})
    mu, sig = lp.get("length_lognorm", {}).get("mu", 0.0), max(
        1e-6, lp.get("length_lognorm", {}).get("sigma", 1.0)
    )
    z_len = abs((math.log(max(1, span_len)) - mu) / sig)
    sb = lp.get("start_beta", {"alpha": 1.0, "beta": 1.0})
    mode = (
        (sb["alpha"] - 1) / (sb["alpha"] + sb["beta"] - 2)
        if sb["alpha"] > 1 and sb["beta"] > 1
        else sb["alpha"] / (sb["alpha"] + sb["beta"])
    )
    start_dist = abs(start_pos - mode)
    return 0.4 * z_len + 0.6 * start_dist  # weights can be tuned


def nms(spans, thr):
    spans = sorted(spans, key=lambda x: x["score"], reverse=True)
    keep = []
    for s in spans:
        if all(iou((s["start"], s["end"]), (k["start"], k["end"])) < thr for k in keep):
            keep.append(s)
    return keep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="JSONL files with {_id, markers:[{label,start,end,score}]}",
    )
    ap.add_argument(
        "--texts", required=True, help="jsonl with _id,text (use dev.jsonl)"
    )
    ap.add_argument("--priors", required=True)
    ap.add_argument("--out_raw", required=True)
    ap.add_argument("--out_pp", required=True)
    ap.add_argument("--pairs_ci", required=True)
    args = ap.parse_args()

    import json

    PRIORS = json.loads(Path(args.priors).read_text())
    PAIRS = json.loads(Path(args.pairs_ci).read_text())

    # load texts (for normalized position)
    texts = {}
    for d in load_jsonl(args.texts):
        texts[d["_id"]] = d.get("text", "")

    # merge per _id
    by_id = {}
    for src in args.sources:
        for d in load_jsonl(src):
            L = by_id.setdefault(d["_id"], [])
            for m in d.get("markers", []):
                if not {"label", "start", "end"}.issubset(m):
                    continue
                # default score if missing
                m["score"] = float(m.get("score", 0.5))
                L.append(m)

    # label-level thresholds (can be tuned)
    NMS_THR = {
        "Actor": 0.6,
        "Victim": 0.6,
        "Action": 0.5,
        "Effect": 0.5,
        "Evidence": 0.5,
    }

    out = []
    for _id, spans in by_id.items():
        t = texts.get(_id, "")
        tlen = max(1, len(t))
        # score with prior penalty
        scored = []
        for m in spans:
            span_len = m["end"] - m["start"]
            start_pos = m["start"] / tlen
            pen = prior_penalty(PRIORS, m["label"], span_len, start_pos)
            m2 = dict(m)
            m2["score"] = float(m["score"] - pen)
            scored.append(m2)
        # per-label NMS
        merged = []
        for lab in ["Actor", "Victim", "Action", "Effect", "Evidence"]:
            lab_sp = [s for s in scored if s["label"] == lab]
            merged.extend(nms(lab_sp, NMS_THR.get(lab, 0.5)))
        out.append({"_id": _id, "markers": merged, "text": t})

    # write raw merged
    with open(args.out_raw, "w", encoding="utf-8") as f:
        for d in out:
            f.write(json.dumps({"_id": d["_id"], "markers": d["markers"]}) + "\n")

    # --- run cross-label rules post-processor ---
    import subprocess, sys
    from pathlib import Path

    # resolve postprocessor path robustly: project_root/src/psycomark/postproc/postprocess_spans.py
    POSTPROC = Path(__file__).resolve().parents[1] / "postproc" / "postprocess_spans.py"

    # ensure output folder exists
    Path(args.out_pp).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(POSTPROC),
        "--pred",
        args.out_raw,
        "--data",
        args.texts,
        "--priors",
        args.priors,
        "--pairs-ci",
        args.pairs_ci,
        "--out",
        args.out_pp,
    ]
    print("Running postprocess:", " ".join(cmd))
    subprocess.check_call(cmd)
