# starter/prompt_sweep.py
import argparse
import json
import os
import pathlib
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

# repo root on path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.psycomark.llm.bedrock_chat import Chat
from src.psycomark.llm.eda_support import (
    build_s1_policy,
    build_s2_policy,
    load_fewshots,
)


# ------------------- utility -------------------
def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def strict_json_extract(s: str):
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def list_json_extract(s: str):
    m = re.search(r"\[.*\]", s, flags=re.S)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, list) else []
    except Exception:
        return []


def shorten(txt: str, n=1600):
    return txt if len(txt) <= n else txt[:n] + "..."


# ------------------- defaults -------------------
S2_SYS = """You are a careful annotator for PsyCoMark (SemEval-2026 Task 10, Subtask 2).
Decide if the given Reddit comment promotes a conspiracy narrative.
Use short chain-of-thought privately; return ONLY the final JSON specified.
Label rules:
- "conspiracy" if the author asserts or strongly endorses a conspiratorial claim.
- "non" otherwise (including neutral reporting, critique, jokes, or unrelated content).
Do NOT use subreddit as a sole proxy. Focus on the content of the text.
Return strict JSON: {"label":"conspiracy|non","rationale":"<=2 sentences","confidence":0..1}"""

S2_USER = """TASK: Document-level classification (conspiracy vs non). Return JSON only.

TEXT:
{doc_text}

RETURN JSON:
{{"label":"conspiracy|non","rationale":"...","confidence":0.xx}}"""

S1_BASE = """You are a careful annotator for PsyCoMark (SemEval-2026 Task 10, Subtask 1).
Extract character spans (0-indexed, end-exclusive) for labels: Actor, Action, Effect, Victim, Evidence.
Return ONLY a JSON list of objects:
[{"label":"Actor|Action|Effect|Victim|Evidence","start":int,"end":int}]
Guidelines:
- Keep spans tight; exclude punctuation/stopwords.
- Prefer semantic correctness; avoid invented text.
- If none fit a label, omit it."""

S1_USER = """TASK: Extract spans for labels: Actor, Action, Effect, Victim, Evidence.
Return ONLY strict JSON list:
[{{"label":"Actor|Action|Effect|Victim|Evidence","start":int,"end":int}}]

TEXT:
{doc_text}"""


# ------------------- technique registry -------------------
def render_fewshots_block(examples: List[Dict[str, Any]], is_s2: bool) -> str:
    if not examples:
        return ""
    blocks = []
    for ex in examples:
        txt = ex.get("text") or ex.get("doc_text") or ""
        if is_s2:
            gold = (
                ex.get("gold")
                or ex.get("json")
                or {"label": "non", "rationale": "baseline", "confidence": 0.7}
            )
        else:
            gold = ex.get("gold") or ex.get("json") or []
        blocks.append(
            f"EXAMPLE:\nTEXT:\n{shorten(txt, 800)}\nJSON:\n{json.dumps(gold, ensure_ascii=False)}"
        )
    return "\n\n".join(blocks)


def s2_prompt(doc_text, policy, fewshots, tech: str):
    # returns: system_texts: List[str], user_text: str, n_samples, temp
    tech = tech.lower()
    fs_block = render_fewshots_block(fewshots, is_s2=True)

    # defaults
    system = [S2_SYS]
    user = ""
    n_samples, temp = 1, 0.0

    if tech in ("zs",):
        pass
    elif tech in ("zs_policy",):
        if policy:
            system.insert(0, policy)
    elif tech in ("fs",):
        user = (fs_block + "\n\n") if fs_block else ""
    elif tech in ("fs_policy",):
        if policy:
            system.insert(0, policy)
        user = (fs_block + "\n\n") if fs_block else ""
    elif tech in ("cot_fs_policy", "cot"):
        if policy:
            system.insert(0, policy)
        # CoT hint already in S2_SYS; keep fewshots
        user = (fs_block + "\n\n") if fs_block else ""
    elif tech.startswith("sc"):  # self-consistency
        if policy:
            system.insert(0, policy)
        user = (fs_block + "\n\n") if fs_block else ""
        n_samples = (
            int(re.search(r"sc(\d+)", tech).group(1))
            if re.search(r"sc(\d+)", tech)
            else 5
        )
        temp = 0.7
    else:
        # fallback = fs_policy
        if policy:
            system.insert(0, policy)
        user = (fs_block + "\n\n") if fs_block else ""

    user += S2_USER.format(doc_text=doc_text)
    return system, user, n_samples, temp


def s1_prompt(doc_text, policy, fewshots, boundary_note: str, tech: str):
    # returns: system_texts, user_text, n_samples, temp
    tech = tech.lower()
    fs_block = render_fewshots_block(fewshots, is_s2=False)
    system = [S1_BASE]
    if tech in ("zs",):
        pass
    elif tech in ("fs",):
        pass
    elif tech in ("fs_boundary", "fs_boundary_policy"):
        if policy:
            system.insert(0, policy)
        if boundary_note:
            system.append("Boundary guidance:\n" + boundary_note)
    elif tech.startswith("sc"):  # self-consistency for spans
        if policy:
            system.insert(0, policy)
        if boundary_note:
            system.append("Boundary guidance:\n" + boundary_note)
    else:
        # default to fs_boundary_policy
        if policy:
            system.insert(0, policy)
        if boundary_note:
            system.append("Boundary guidance:\n" + boundary_note)

    n_samples = 1
    temp = 0.0
    if tech.startswith("sc"):
        n_samples = (
            int(re.search(r"sc(\d+)", tech).group(1))
            if re.search(r"sc(\d+)", tech)
            else 5
        )
        temp = 0.7

    user = ""
    if tech in ("fs", "fs_boundary", "fs_boundary_policy") or tech.startswith("sc"):
        user += (fs_block + "\n\n") if fs_block else ""
    user += S1_USER.format(doc_text=doc_text)
    return system, user, n_samples, temp


# ------------------- S2 running & aggregation -------------------
def run_s2(
    doc, system_blocks, user_block, model_id, region, max_tokens, temperature, n_samples
):
    # Multi-sample with majority vote (self-consistency)
    preds = []
    for _ in range(n_samples):
        chat = Chat(
            model_id=model_id,
            region=region,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        for s in system_blocks:
            chat.add_system(s)
        chat.add_user(user_block)
        out = chat.generate()
        js = strict_json_extract(out) or {
            "label": "non",
            "rationale": "repair",
            "confidence": 0.51,
        }
        lbl = js.get("label", "non")
        conf = float(js.get("confidence", 0.5))
        preds.append((lbl, conf))
    # aggregate
    labels = [p[0] for p in preds]
    vote = Counter(labels).most_common()
    top_label = vote[0][0]
    # average confidence for chosen label
    confs = [c for l, c in preds if l == top_label]
    agg_conf = (
        sum(confs) / len(confs) if confs else sum([c for _, c in preds]) / len(preds)
    )
    return top_label, agg_conf


# ------------------- S1 running & aggregation -------------------
LABELS = {"Actor", "Action", "Effect", "Victim", "Evidence"}


def clip_and_validate(spans, text):
    L = len(text)
    out = []
    for m in spans:
        lab = m.get("label")
        if lab not in LABELS:
            continue
        try:
            s = int(m.get("start"))
            e = int(m.get("end"))
        except Exception:
            continue
        s = max(0, min(L, s))
        e = max(0, min(L, e))
        if e <= s:
            continue
        out.append({"label": lab, "start": s, "end": e})
    return out


def iou(a, b):
    inter = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = (a["end"] - a["start"]) + (b["end"] - b["start"]) - inter
    return (inter / union) if union > 0 else 0.0


def merge_by_freq(all_runs: List[List[Dict[str, int]]], min_votes=2, iou_thr=0.8):
    # simple consensus: keep spans that appear in >= min_votes with high IoU
    kept = []
    for lab in LABELS:
        # collect all spans of that label
        spans = [s for run in all_runs for s in run if s["label"] == lab]
        spans = sorted(spans, key=lambda x: (x["start"], x["end"]))
        # greedy clustering
        clusters = []
        for s in spans:
            placed = False
            for cl in clusters:
                if iou(s, cl["rep"]) >= iou_thr:
                    cl["members"].append(s)
                    # update representative to median span
                    cl["rep"] = cl["members"][len(cl["members"]) // 2]
                    placed = True
                    break
            if not placed:
                clusters.append({"rep": s, "members": [s]})
        for cl in clusters:
            if len(cl["members"]) >= min_votes:
                kept.append(cl["rep"])
    return kept


def run_s1(
    doc, system_blocks, user_block, model_id, region, max_tokens, temperature, n_samples
):
    text = doc.get("text", "")
    if n_samples == 1:
        chat = Chat(
            model_id=model_id,
            region=region,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        for s in system_blocks:
            chat.add_system(s)
        chat.add_user(user_block)
        out = chat.generate()
        spans = clip_and_validate(list_json_extract(out), text)
        return spans
    else:
        runs = []
        for _ in range(n_samples):
            chat = Chat(
                model_id=model_id,
                region=region,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            for s in system_blocks:
                chat.add_system(s)
            chat.add_user(user_block)
            out = chat.generate()
            spans = clip_and_validate(list_json_extract(out), text)
            runs.append(spans)
        return merge_by_freq(runs, min_votes=max(2, n_samples // 3), iou_thr=0.8)


# ------------------- evaluation -------------------
def eval_s2(gt_file, sub_file) -> Dict[str, Any]:
    # expects your local eval_binary.py present
    try:
        out = subprocess.check_output(
            [
                sys.executable,
                "starter/eval_binary.py",
                gt_file,
                sub_file,
                "scores.json",
            ],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(e.output)
    # read scores.json if created
    scores_path = pathlib.Path("scores.json")
    if scores_path.exists():
        return json.loads(scores_path.read_text(encoding="utf-8"))
    return {}


def eval_s1(gt_file, sub_file, iou=0.5) -> Dict[str, Any]:
    try:
        out = subprocess.check_output(
            [
                sys.executable,
                "starter/eval_token.py",
                "--ground_truth_file",
                gt_file,
                "--prediction_file",
                sub_file,
                "--scores_output_file",
                "scores_s1.json",
                "--iou_threshold",
                str(iou),
            ],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(e.output)
    p = pathlib.Path("scores_s1.json")
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


# ------------------- main -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["s1", "s2"], required=True)
    ap.add_argument("--test-file", required=True)
    ap.add_argument("--eda-root", required=False, default=None)
    ap.add_argument(
        "--techniques",
        required=False,
        default="zs,fs_policy,sc5",
        help="Comma list. S2: zs,zs_policy,fs,fs_policy,cot_fs_policy,sc5,sc10 | S1: zs,fs,fs_boundary_policy,sc5",
    )
    ap.add_argument("--model-id", default=None)  # falls back to env
    ap.add_argument("--region", default=None)  # falls back to env
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--sc-temperature", type=float, default=0.7)
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--s1-iou", type=float, default=0.5)
    args = ap.parse_args()

    random.seed(42)

    test_rows = list(read_jsonl(args.test_file))

    # EDA support
    policy, fewshots, boundary = "", [], ""
    if args.eda_root:
        eda = pathlib.Path(args.eda_root)
        if args.task == "s2":
            policy = build_s2_policy(eda) or ""
            fewshots = load_fewshots(eda, "s2", max_n=8) or []
        else:
            policy = build_s1_policy(eda) or ""
            fewshots = load_fewshots(eda, "s1", max_n=6) or []
            bctx = eda / "boundary_context.json"
            if bctx.exists():
                try:
                    boundary = json.loads(bctx.read_text(encoding="utf-8")).get(
                        "note", ""
                    )
                except Exception:
                    boundary = ""

    techniques = [t.strip() for t in args.techniques.split(",") if t.strip()]

    summary = []
    for tech in techniques:
        print(f"\n=== Running {args.task.upper()} :: {tech} ===")
        out_dir = pathlib.Path(args.out_root) / f"{args.task}_llm" / tech
        out_dir.mkdir(parents=True, exist_ok=True)
        sub_file = out_dir / "submission.jsonl"

        rows_out = []
        t0 = time.time()

        if args.task == "s2":
            for rec in test_rows:
                doc_text = rec.get("text", "")
                sys_blocks, user_block, n_samples, temp = s2_prompt(
                    doc_text, policy, fewshots, tech
                )
                # override temp for SC
                temp = args.sc_temperature if n_samples > 1 else args.temperature
                lbl, conf = run_s2(
                    rec,
                    sys_blocks,
                    user_block,
                    args.model_id,
                    args.region,
                    args.max_tokens,
                    temp,
                    n_samples,
                )
                pred = "Yes" if lbl == "conspiracy" else "No"
                _id = rec.get("_id") or rec.get("doc_id")
                rows_out.append({"_id": _id, "conspiracy": pred})
        else:
            for rec in test_rows:
                doc_text = rec.get("text", "")
                sys_blocks, user_block, n_samples, temp = s1_prompt(
                    doc_text, policy, fewshots, boundary, tech
                )
                temp = args.sc_temperature if n_samples > 1 else args.temperature
                spans = run_s1(
                    rec,
                    sys_blocks,
                    user_block,
                    args.model_id,
                    args.region,
                    args.max_tokens,
                    temp,
                    n_samples,
                )
                _id = rec.get("_id") or rec.get("doc_id")
                # codabench format
                markers = [
                    {
                        "type": m["label"],
                        "startIndex": m["start"],
                        "endIndex": m["end"],
                        "text": doc_text[m["start"] : m["end"]],
                    }
                    for m in spans
                ]
                rows_out.append({"_id": _id, "markers": markers})

        write_jsonl(sub_file, rows_out)
        dt = time.time() - t0
        print(f"wrote: {sub_file}  ({len(rows_out)} docs, {dt:.1f}s)")

        # evaluate
        if args.task == "s2":
            scores = eval_s2(args.test_file, str(sub_file))
            row = {"task": "s2", "tech": tech, **scores}
        else:
            scores = eval_s1(args.test_file, str(sub_file), iou=args.s1_iou)
            row = {"task": "s1", "tech": tech, **scores}
        summary.append(row)

    # write summary CSV + pretty print
    summ_path = (
        pathlib.Path(args.out_root) / f"{args.task}_llm" / "prompt_sweep_summary.csv"
    )
    # normalize keys
    all_keys = sorted({k for r in summary for k in r.keys()})
    import csv

    with open(summ_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        w.writerows(summary)
    print(f"\nSummary saved -> {summ_path}")

    # pretty table
    def fmt(x):
        if isinstance(x, float):
            return f"{x:.4f}"
        return str(x)

    cols = ["task", "tech"]
    if args.task == "s2":
        cols += [
            "F1_weighted",
            "F1_macro",
            "Accuracy",
            "Precision_macro",
            "Recall_macro",
        ]
    else:
        cols += [
            "F1_Aggregate_Token",
            "F1_Macro_Token",
            "Precision_Aggregate_Token",
            "Recall_Aggregate_Token",
        ]
    cols = [c for c in cols if c in all_keys]
    if cols:
        # header
        print("\n" + " | ".join(cols))
        print("-" * (len(" | ".join(cols)) + 5))
        for r in summary:
            print(" | ".join(fmt(r.get(c, "")) for c in cols))


if __name__ == "__main__":
    main()
