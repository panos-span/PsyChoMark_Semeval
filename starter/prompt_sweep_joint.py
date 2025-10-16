# starter/prompt_sweep_joint.py
import argparse
import json
import pathlib
import random
import re
import subprocess
import sys
import os
from collections import defaultdict
from typing import Any, Dict, List
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

# repo root on path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pathlib as _pathlib
from src.psycomark.llm.bedrock_chat import Chat
from src.psycomark.llm.eda_support import (
    build_s1_policy,
    build_s2_policy,
    load_fewshots,
)


# ---------- .env loader (no deps) ----------
def _load_dotenv_into_environ():
    root = _pathlib.Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    # Map non-standard names to AWS_* so boto3 sees them
    if "ACCESS_KEY_ID" in os.environ and "AWS_ACCESS_KEY_ID" not in os.environ:
        os.environ["AWS_ACCESS_KEY_ID"] = os.environ["ACCESS_KEY_ID"]
    if "SECRET_ACCESS_KEY" in os.environ and "AWS_SECRET_ACCESS_KEY" not in os.environ:
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["SECRET_ACCESS_KEY"]


_load_dotenv_into_environ()
# ------------------------------------------


# ---------- utils ----------
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


# ---------- default prompts ----------
S1_BASE = """You are a careful annotator for PsyCoMark (SemEval-2026 Task 10, Subtask 1).
Task: extract character spans that best reflect the following labels: Actor, Action, Effect, Victim, Evidence.

Output format (strict JSON list, no extra text):
[{"label":"Actor|Action|Effect|Victim|Evidence","start":int,"end":int}]

Rules and rubric:
- Offsets are 0-indexed; end is exclusive. Spans must lie within the provided TEXT.
- Keep spans tight and semantically meaningful. Exclude leading/trailing whitespace and punctuation.
- Prefer minimal spans that uniquely identify the entity/event (e.g., "agency" not "the agency" unless needed to disambiguate).
- Do not invent text not present in the input. Omit a label if not evidenced in the text.
- Overlaps are allowed when justified (e.g., an Action containing a sub-phrase), but avoid redundant duplicates.
- Boundary hints: include core content words; exclude trailing stopwords unless essential (e.g., prepositions integral to meaning).
"""

S1_USER = """TASK: Extract spans for labels: Actor, Action, Effect, Victim, Evidence.
Return ONLY a strict JSON list (no prose, no keys other than label/start/end):
[{{"label":"Actor|Action|Effect|Victim|Evidence","start":int,"end":int}}]

TEXT:
{doc_text}
"""

S2_SYS = """You are a careful annotator for PsyCoMark (SemEval-2026 Task 10, Subtask 2).
Goal: decide whether the REDDIT COMMENT promotes a conspiracy narrative.
Use brief private reasoning, but return ONLY the final JSON below.

Label decision rubric:
- "conspiracy": The author asserts or clearly endorses a conspiratorial claim.
- "non": Neutral reporting, critique/debunking, jokes/irony, or unrelated content.
Avoid using subreddit as a proxy; rely on text content and framing.

Probability rubric (p_conspiracy, p_non should sum to 1.0):
- 0.90–1.00: Explicit assertion/endorsement of a conspiracy.
- 0.60–0.80: Strong implication or supportive framing without explicit claim.
- ~0.50: Ambiguous/uncertain.
- 0.00–0.20: Clearly non-conspiratorial (neutral/debunking/irrelevant).

Return strict JSON ONLY:
{"label":"conspiracy|non","p_conspiracy":0.xx,"p_non":0.xx,"rationale":"<=2 sentences"}
Constraints:
- Ensure p_conspiracy + p_non = 1.0 (within rounding). Set label to the higher of the two.
- Keep rationale concise and non-revealing of chain-of-thought.
"""

S2_USER = """TASK: Document-level classification (conspiracy vs non). Return JSON only.

TEXT:
{doc_text}

DETECTED_MARKERS_JSON:
{markers_json}

HOW_TO_USE_MARKERS:
- Treat Actor/Action/Effect/Victim/Evidence as signals of conspiratorial framing.
- Absence of markers ≠ non-conspiracy; decide based on content.
- If uncertain, choose "non".

RETURN JSON:
{{"label":"conspiracy|non","p_conspiracy":0.xx,"p_non":0.xx,"rationale":"..."}}"""


# ---------- fewshots rendering ----------
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


# ---------- quick GT validation ----------
def _file_has_markers(jsonl_path: str, sample: int = 200) -> bool:
    try:
        n, y = 0, 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                r = json.loads(ln)
                mks = r.get("markers") or []
                if isinstance(mks, list) and len(mks) > 0:
                    y += 1
                n += 1
                if n >= sample:
                    break
        return y > 0
    except Exception:
        return False


# ---------- S1 helpers ----------
LABELS = {"Actor", "Action", "Effect", "Victim", "Evidence"}
_CANON_LABEL = {lab.lower(): lab for lab in LABELS}


def clip_and_validate(spans, text):
    L = len(text)
    out = []
    for m in spans:
        lab_raw = (m.get("label") or "").strip()
        lab = _CANON_LABEL.get(lab_raw.lower())
        if not lab:
            continue
        try:
            s_val = m.get("start")
            e_val = m.get("end")
            if s_val is None or e_val is None:
                # Fallback to startIndex/endIndex if model used that schema
                s_val = m.get("startIndex")
                e_val = m.get("endIndex")
            s = int(s_val)
            e = int(e_val)
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
    kept = []
    for lab in LABELS:
        spans = [s for run in all_runs for s in run if s["label"] == lab]
        spans = sorted(spans, key=lambda x: (x["start"], x["end"]))
        clusters = []
        for s in spans:
            placed = False
            for cl in clusters:
                if iou(s, cl["rep"]) >= iou_thr:
                    cl["members"].append(s)
                    cl["rep"] = cl["members"][len(cl["members"]) // 2]
                    placed = True
                    break
            if not placed:
                clusters.append({"rep": s, "members": [s]})
        for cl in clusters:
            if len(cl["members"]) >= min_votes:
                kept.append(cl["rep"])
    return kept


# ---------- prompt builders (tech-agnostic) ----------
def s1_prompt(doc_text, policy, fewshots, boundary_note: str, tech: str):
    fs_block = render_fewshots_block(fewshots, is_s2=False)
    system = [S1_BASE]
    if policy:
        system.insert(0, policy)
    if "boundary" in tech and boundary_note:
        system.append("Boundary guidance:\n" + boundary_note)
    n_samples = 1
    temp = 0.0
    if tech.startswith("sc"):  # self-consistency
        n = re.search(r"sc(\d+)", tech)
        n_samples = int(n.group(1)) if n else 5
        temp = 0.7
    user = ""
    if "fs" in tech or tech.startswith("sc"):
        user += (fs_block + "\n\n") if fs_block else ""
    user += S1_USER.format(doc_text=doc_text)
    return system, user, n_samples, temp


def s2_prompt_with_markers(doc_text, policy, fewshots, tech: str, markers_json: str):
    fs_block = render_fewshots_block(fewshots, is_s2=True)
    system = [S2_SYS]
    if "policy" in tech and policy:
        system.insert(0, policy)
    n_samples = 1
    temp = 0.0
    if tech.startswith("sc"):
        n = re.search(r"sc(\d+)", tech)
        n_samples = int(n.group(1)) if n else 5
        temp = 0.7
    user_prefix = ""
    if "fs" in tech or tech.startswith("sc"):
        user_prefix = (fs_block + "\n\n") if fs_block else ""
    user = user_prefix + S2_USER.format(doc_text=doc_text, markers_json=markers_json)
    return system, user, n_samples, temp


# ---------- run S1 (then S2) for one doc ----------
def run_s1(
    doc, sys_blocks, user_block, model_id, region, max_tokens, temperature, n_samples
):
    text = doc.get("text", "")

    if n_samples == 1:
        chat = Chat(
            model_id=model_id,
            region=region,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        for s in sys_blocks:
            chat.add_system(s)
        chat.add_user(user_block)
        try:
            out = chat.generate()
            spans = clip_and_validate(list_json_extract(out), text)
            return spans
        except Exception as e:
            logging.warning(
                f"[S1] generate() failed for doc -> fallback empty spans: {e}"
            )
            return []
    else:
        runs = []
        for _ in range(n_samples):
            chat = Chat(
                model_id=model_id,
                region=region,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            for s in sys_blocks:
                chat.add_system(s)
            chat.add_user(user_block)
            try:
                out = chat.generate()
                runs.append(clip_and_validate(list_json_extract(out), text))
            except Exception as e:
                logging.warning(
                    f"[S1] generate() failed (SC) -> counting as empty: {e}"
                )
                runs.append([])
        return merge_by_freq(runs, min_votes=max(2, n_samples // 3), iou_thr=0.8)


def _coerce_probabilities(js: Dict[str, Any]) -> Dict[str, float]:
    """Return a dict with label, p_conspiracy, p_non parsed from model output."""
    label = (js.get("label") or "non").strip().lower()
    p_con = js.get("p_conspiracy")
    p_non = js.get("p_non")

    def _float_or_none(val):
        try:
            return float(val)
        except Exception:
            return None

    p_con = _float_or_none(p_con)
    p_non = _float_or_none(p_non)

    if p_con is None and p_non is None:
        conf = _float_or_none(js.get("confidence"))
        if conf is None:
            conf = 0.5
        if label == "conspiracy":
            p_con = conf
            p_non = 1.0 - conf
        else:
            p_con = 1.0 - conf
            p_non = conf

    if p_con is None:
        if p_non is None:
            p_con = 0.5
            p_non = 0.5
        else:
            p_con = max(0.0, min(1.0, 1.0 - p_non))
    if p_non is None:
        p_non = max(0.0, min(1.0, 1.0 - p_con))

    total = p_con + p_non
    if total > 0:
        p_con /= total
        p_non /= total
    else:
        p_con = p_non = 0.5

    if label not in ("conspiracy", "non"):
        label = "conspiracy" if p_con >= p_non else "non"

    return {"label": label, "p_conspiracy": p_con, "p_non": p_non}


def run_s2(
    doc, sys_blocks, user_block, model_id, region, max_tokens, temperature, n_samples
):
    preds = []
    for _ in range(n_samples):
        chat = Chat(
            model_id=model_id,
            region=region,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        for s in sys_blocks:
            chat.add_system(s)
        chat.add_user(user_block)
        try:
            out = chat.generate()
            js = strict_json_extract(out) or {
                "label": "non",
                "rationale": "repair",
                "confidence": 0.51,
            }
        except Exception as e:
            logging.warning(f"[S2] generate() failed -> defaulting to non: {e}")
            js = {"label": "non", "rationale": "fallback", "confidence": 0.55}
        preds.append(_coerce_probabilities(js))

    avg_p_con = sum(p["p_conspiracy"] for p in preds) / len(preds)
    avg_p_non = sum(p["p_non"] for p in preds) / len(preds)
    total = avg_p_con + avg_p_non
    if total > 0:
        avg_p_con /= total
        avg_p_non /= total
    final_label = "conspiracy" if avg_p_con >= avg_p_non else "non"
    return final_label, avg_p_con, avg_p_non


# ---------- evaluation ----------
def eval_s1(gt_file, sub_file, iou=0.5) -> Dict[str, Any]:
    if not _file_has_markers(gt_file):
        logging.warning(
            f"S1 eval skipped or unreliable: no ground-truth 'markers' found in {gt_file}. "
            f"Use a labeled file (e.g., train with markers) for S1 evaluation."
        )
        return {}
    try:
        subprocess.check_output(
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
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_s1_metrics_artifacts(scores: Dict[str, Any], out_dir: pathlib.Path, tech: str):
    """Save per-label precision/recall/F1 CSV and bar plot for S1 if matplotlib available."""
    if not scores:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    # Build row per label
    labels = sorted(
        [
            lab_name
            for lab_name in ["Actor", "Action", "Effect", "Victim", "Evidence"]
            if f"F1_{lab_name}_Token" in scores
        ]
    )
    rows = []
    for lab in labels:
        rows.append(
            {
                "label": lab,
                "precision": scores.get(f"Precision_{lab}_Token", 0.0),
                "recall": scores.get(f"Recall_{lab}_Token", 0.0),
                "f1": scores.get(f"F1_{lab}_Token", 0.0),
            }
        )
    # Add aggregate
    rows.append(
        {
            "label": "_AGG_",
            "precision": scores.get("Precision_Aggregate_Token", 0.0),
            "recall": scores.get("Recall_Aggregate_Token", 0.0),
            "f1": scores.get("F1_Aggregate_Token", 0.0),
        }
    )
    # Write CSV
    import csv

    csv_path = out_dir / f"s1_metrics_{tech}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["label", "precision", "recall", "f1"])
        w.writeheader()
        w.writerows(rows)
    logging.info(f"S1 metrics CSV saved -> {csv_path}")
    # Plot
    if plt is None:
        logging.warning("matplotlib not installed; skipping S1 metrics plot")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    labs_plot = [r["label"] for r in rows if r["label"] != "_AGG_"]
    f1_vals = [r["f1"] for r in rows if r["label"] != "_AGG_"]
    ax.bar(labs_plot, f1_vals, color="#4c72b0")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 (Token IoU>=threshold)")
    ax.set_title(f"S1 Per-Label F1 :: {tech}")
    for i, v in enumerate(f1_vals):
        ax.text(
            i,
            v + 0.01 if v < 0.95 else v - 0.05,
            f"{v:.2f}",
            ha="center",
            va="bottom" if v < 0.95 else "top",
            fontsize=9,
        )
    fig.tight_layout()
    plot_path = out_dir / f"s1_f1_{tech}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logging.info(f"S1 F1 plot saved -> {plot_path}")


def eval_s2(gold_file, sub_file) -> Dict[str, Any]:
    """
    Evaluate S2 predictions locally using the updated eval_binary.py
    which expects flagged arguments.
    """
    # Instead of shelling out, compute metrics inline so sweep prints per-technique results.
    gold_ids, y_true = [], []
    with open(gold_file, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            r = json.loads(ln)
            lab = (r.get("doc_label") or r.get("conspiracy") or "").strip().lower()
            if lab not in ("conspiracy", "non", "yes", "no"):
                continue
            y_true.append(1 if lab in ("conspiracy", "yes") else 0)
            gold_ids.append(r.get("_id") or r.get("doc_id"))
    pred_map = {}
    with open(sub_file, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            r = json.loads(ln)
            pred_map[r.get("_id")] = 1 if r.get("conspiracy") == "Yes" else 0
    y_pred = [pred_map.get(i, 0) for i in gold_ids]
    # Binary metrics
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
    # Confusion matrix (ensure both classes exist)
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except ValueError:
        tn = fp = fn = tp = 0
    # AUC cannot be computed without probabilities; leave NaN
    auc = float("nan")
    return {
        "Accuracy": acc,
        "Precision_binary": p_bin,
        "Recall_binary": r_bin,
        "F1_binary": f1_bin,
        "F1_macro": f1_macro,
        "F1_weighted": f1_weighted,
        "AUC": auc,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--test-file-s1",
        required=True,
        help="Labeled file with text for S1 scoring (e.g., $latest/train.jsonl or a CV fold subset).",
    )
    ap.add_argument(
        "--test-file-s2",
        required=True,
        help="Labeled file for S2 scoring (e.g., $latest/train_docclf.jsonl or a CV fold subset).",
    )
    ap.add_argument("--eda-root", required=False, default=None)
    ap.add_argument(
        "--techniques",
        default="fs_boundary_policy,sc5",
        help="Comma list applied jointly. Examples: zs,fs_boundary_policy,sc5,sc10",
    )
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--region", default=None)
    ap.add_argument("--max-tokens-s1", type=int, default=1200)
    ap.add_argument("--max-tokens-s2", type=int, default=900)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--sc-temperature", type=float, default=0.7)
    ap.add_argument("--s1-iou", type=float, default=0.5)
    ap.add_argument("--out-root", default="runs/joint_llm")
    ap.add_argument(
        "--max-markers-per-label",
        type=int,
        default=3,
        help="Limit markers passed to S2 to control prompt length.",
    )
    ap.add_argument(
        "--limit-docs",
        type=int,
        default=None,
        help="Process only the first N documents for each of S1/S2 (for quicker prompt sweeps).",
    )
    args = ap.parse_args()

    random.seed(42)

    # EDA
    s1_policy = s2_policy = ""
    s1_shots = []
    s2_shots = []
    boundary = ""
    if args.eda_root:
        eda = pathlib.Path(args.eda_root)
        s1_policy = build_s1_policy(eda) or ""
        s2_policy = build_s2_policy(eda) or ""
        s1_shots = load_fewshots(eda, "s1", max_n=6) or []
        s2_shots = load_fewshots(eda, "s2", max_n=8) or []
        bctx = eda / "boundary_context.json"
        if bctx.exists():
            try:
                boundary = json.loads(bctx.read_text(encoding="utf-8")).get("note", "")
            except Exception:
                boundary = ""

    # Data
    rows_s1 = list(read_jsonl(args.test_file_s1))
    rows_s2 = list(read_jsonl(args.test_file_s2))
    if args.limit_docs is not None:
        rows_s1 = rows_s1[: args.limit_docs]
        rows_s2 = rows_s2[: args.limit_docs]
        logging.info(
            f"Limiting to first {args.limit_docs} docs for S1 ({len(rows_s1)}) and S2 ({len(rows_s2)})"
        )
    id2doc_s2 = {(r.get("_id") or r.get("doc_id")): r for r in rows_s2}

    techniques = [t.strip() for t in args.techniques.split(",") if t.strip()]

    summary_rows = []
    for tech in techniques:
        print(f"\n=== JOINT S1→S2 :: {tech} ===")

        tech_dir = pathlib.Path(args.out_root) / tech
        (tech_dir / "s1").mkdir(parents=True, exist_ok=True)
        (tech_dir / "s2").mkdir(parents=True, exist_ok=True)
        s1_sub = tech_dir / "s1" / "submission.jsonl"
        s1_pruned_sub = tech_dir / "s1" / "submission_pruned.jsonl"
        s2_sub = tech_dir / "s2" / "submission.jsonl"
        # If limiting docs, create a ground-truth subset for fair S1 eval
        gt_subset_path = None
        if args.limit_docs is not None:
            gt_subset_path = tech_dir / "s1" / "gt_subset.jsonl"
            write_jsonl(gt_subset_path, rows_s1)  # rows_s1 already limited

        # ------ S1 inference ------
        s1_out_rows = []
        s1_pruned_rows = []
        id2markers = {}  # for S2 conditioning
        total_raw, total_valid, total_pruned = 0, 0, 0
        for rec in rows_s1:
            _id = rec.get("_id") or rec.get("doc_id")
            txt = rec.get("text", "")

            sys_blocks, user_block, n_samples, temp = s1_prompt(
                txt, s1_policy, s1_shots, boundary, tech
            )
            temp = args.sc_temperature if n_samples > 1 else args.temperature
            spans = run_s1(
                rec,
                sys_blocks,
                user_block,
                args.model_id,
                args.region,
                args.max_tokens_s1,
                temp,
                n_samples,
            )
            total_raw += len(spans)
            if not spans:
                id2markers[_id] = []
                empty_markers = []
                s1_out_rows.append({"_id": _id, "markers": empty_markers})
                s1_pruned_rows.append({"_id": _id, "markers": empty_markers})
                continue

            markers = [
                {
                    "type": m["label"],
                    "startIndex": m["start"],
                    "endIndex": m["end"],
                    "text": txt[m["start"] : m["end"]],
                }
                for m in spans
            ]
            s1_out_rows.append({"_id": _id, "markers": markers})

            # limit per label for S2 prompt brevity
            by_lab = defaultdict(list)
            for m in markers:
                by_lab[m["type"]].append(m)

            total_valid += len(markers)
            pruned = []
            k = args.max_markers_per_label  # always Namespace here

            def _start_end(m):
                # accept either style; default to 0 if missing to avoid crashes
                s = m.get("start", m.get("startIndex", 0))
                e = m.get("end", m.get("endIndex", s))
                return int(s), int(e)

            for lab, arr in by_lab.items():
                # sort by (start, end) and prefer longer spans first within same start
                arr_sorted = sorted(
                    arr,
                    key=lambda x: (
                        _start_end(x)[0],
                        -(_start_end(x)[1] - _start_end(x)[0]),
                    ),
                )
                # take top-k
                pruned.extend(arr_sorted[:k])

            # store pruned markers using the *Bedrock/S2 prompt* schema (startIndex/endIndex)
            def _to_s2_marker(m):
                s, e = _start_end(m)
                return {
                    "type": m.get("type"),
                    "startIndex": s,
                    "endIndex": e,
                    "text": txt[s:e],
                }

            id2markers[_id] = [_to_s2_marker(m) for m in pruned]
            total_pruned += len(id2markers[_id])
            s1_pruned_rows.append({"_id": _id, "markers": id2markers[_id]})

        write_jsonl(s1_sub, s1_out_rows)
        write_jsonl(s1_pruned_sub, s1_pruned_rows)
        s1_scores = eval_s1(
            str(gt_subset_path) if gt_subset_path else args.test_file_s1,
            str(s1_sub),
            iou=args.s1_iou,
        )
        print(f"S1 done -> {s1_sub}")
        print(f"S1 pruned-for-S2 -> {s1_pruned_sub}")
        print(
            f"S1 debug: spans raw/valid/pruned = {total_raw}/{total_valid}/{total_pruned}"
        )
        if s1_scores:
            # Print a concise S1 summary
            agg = s1_scores.get("F1_Aggregate_Token") or s1_scores.get("F1_Aggregate")
            macro = s1_scores.get("F1_Macro_Token") or s1_scores.get("F1_Macro")
            print(
                f"S1 metrics: F1_aggregate={agg:.3f} F1_macro={macro:.3f}"
                if agg and macro
                else f"S1 metrics: {s1_scores}"
            )
            # Save artifacts
            save_s1_metrics_artifacts(s1_scores, tech_dir / "s1", tech)
        # ------ S2 inference (conditioned on S1) ------
        s2_out_rows = []
        s2_prob_rows = []
        for _id, doc2 in id2doc_s2.items():
            txt = doc2.get("text", "")
            # use S1 markers if present for the same doc_id
            mks = id2markers.get(_id, [])
            markers_json = json.dumps(mks, ensure_ascii=False)

            sys_blocks, user_block, n_samples, temp = s2_prompt_with_markers(
                txt, s2_policy, s2_shots, tech, markers_json
            )
            temp = args.sc_temperature if n_samples > 1 else args.temperature
            lbl, p_con, p_non = run_s2(
                doc2,
                sys_blocks,
                user_block,
                args.model_id,
                args.region,
                args.max_tokens_s2,
                temp,
                n_samples,
            )
            pred = "Yes" if lbl == "conspiracy" else "No"
            s2_out_rows.append({"_id": _id, "conspiracy": pred})
            s2_prob_rows.append(
                {
                    "_id": _id,
                    "label": lbl,
                    "p_conspiracy": round(p_con, 6),
                    "p_non": round(p_non, 6),
                }
            )

        write_jsonl(s2_sub, s2_out_rows)
        probs_path = tech_dir / "s2" / "probs.jsonl"
        write_jsonl(probs_path, s2_prob_rows)
        if s2_prob_rows:
            mean_p = sum(r["p_conspiracy"] for r in s2_prob_rows) / len(s2_prob_rows)
            frac_pos = sum(1 for r in s2_prob_rows if r["p_conspiracy"] >= 0.5) / len(
                s2_prob_rows
            )
            print(f"S2 prob stats: mean_p={mean_p:.3f} frac_p>=0.5={frac_pos:.3f}")
        s2_scores = eval_s2(args.test_file_s2, str(s2_sub))
        print(f"S2 done -> {s2_sub}")
        print(
            "S2 metrics: acc={acc:.3f} f1_bin={f1b:.3f} f1_macro={f1m:.3f} f1_weighted={f1w:.3f} tn={tn} fp={fp} fn={fn} tp={tp}".format(
                acc=s2_scores.get("Accuracy", float("nan")),
                f1b=s2_scores.get("F1_binary", float("nan")),
                f1m=s2_scores.get("F1_macro", float("nan")),
                f1w=s2_scores.get("F1_weighted", float("nan")),
                tn=s2_scores.get("TN"),
                fp=s2_scores.get("FP"),
                fn=s2_scores.get("FN"),
                tp=s2_scores.get("TP"),
            )
        )

        # record summary
        summary_rows.append(
            {
                "tech": tech,
                **{f"S1_{k}": v for k, v in s1_scores.items()},
                **{f"S2_{k}": v for k, v in s2_scores.items()},
            }
        )

    # save summary
    summ_path = pathlib.Path(args.out_root) / "joint_prompt_sweep_summary.csv"
    all_keys = sorted({k for r in summary_rows for k in r.keys()})
    import csv

    with open(summ_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nSummary saved -> {summ_path}")

    # pretty print
    def fmt(x):
        if isinstance(x, float):
            return f"{x:.4f}"
        return str(x)

    cols = [
        "tech",
        "S1_F1_Aggregate_Token",
        "S1_F1_Macro_Token",
        "S2_F1_weighted",
        "S2_F1_macro",
        "S2_Accuracy",
    ]
    cols = [c for c in cols if c in all_keys]
    if cols:
        print("\n" + " | ".join(cols))
        print("-" * (len(" | ".join(cols)) + 5))
        for r in summary_rows:
            print(" | ".join(fmt(r.get(c, "")) for c in cols))


if __name__ == "__main__":
    main()
