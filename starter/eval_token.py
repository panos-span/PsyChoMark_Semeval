import json
import os
import sys
import argparse
import re
import logging
from typing import Dict, List, Set, Tuple

# --- Configuration ---
DEFAULT_TEST_FILE = "test.jsonl"
DEFAULT_SUBMISSION_FILE = "submission_span.jsonl"
DEFAULT_SCORES_FILE = "scores.json"

MARKER_TYPES = {"Action", "Actor", "Effect", "Evidence", "Victim"}
DEFAULT_IOU_THRESHOLD = 0.5


# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# --- Tokenization and Span Conversion ---
def tokenize_text(text: str) -> List[Tuple[int, int]]:
    """
    Simple, robust tokenization: words or single punctuation symbols.
    Returns a list of (start_char, end_char) tuples.
    """
    token_spans = []
    for match in re.finditer(r"(\w+|[^\w\s])", text):
        token_spans.append((match.start(), match.end()))
    return token_spans


def char_span_to_token_set(
    char_start: int, char_end: int, token_spans: List[Tuple[int, int]]
) -> Set[int]:
    """
    Convert a character span to indices of tokens it overlaps.
    Overlap if: start_A < end_B AND end_A > start_B
    """
    covered = set()
    for idx, (t_start, t_end) in enumerate(token_spans):
        if char_start < t_end and char_end > t_start:
            covered.add(idx)
    return covered


def calculate_token_iou(set_a: Set[int], set_b: Set[int]) -> float:
    """IoU over token index sets."""
    if not set_a and not set_b:
        return 1.0
    inter = set_a & set_b
    union = set_a | set_b
    if not union:
        return 0.0
    return len(inter) / len(union)


# --- Data Handling and Evaluation ---
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate span extraction with TOKEN-BASED Overlap F1 (IoU >= threshold)."
    )
    parser.add_argument(
        "--ground_truth_file",
        nargs="?",
        default=DEFAULT_TEST_FILE,
        help="Ground-truth JSONL with 'text' and gold 'markers'.",
    )
    parser.add_argument(
        "--prediction_file",
        nargs="?",
        default=DEFAULT_SUBMISSION_FILE,
        help="Predicted JSONL (Codabench submission format).",
    )
    parser.add_argument(
        "--scores_output_file",
        nargs="?",
        default=DEFAULT_SCORES_FILE,
        help="Output JSON with scores (Codabench-style keys).",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help=f"Minimum token IoU to count a TP (default {DEFAULT_IOU_THRESHOLD}).",
    )
    parser.add_argument(
        "--strict-extra-preds",
        action="store_true",
        help="If set, count predictions for IDs missing in ground truth as false positives.",
    )
    return parser.parse_args()


def load_jsonl(file_path):
    """Load JSONL; returns list of dicts or None on missing file."""
    data = []
    if not os.path.exists(file_path):
        print(f"Error: Required file not found at {file_path}", file=sys.stderr)
        if len(sys.argv) > 1:
            sys.exit(1)
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                logging.warning(
                    f"Skipping invalid JSON line in {file_path}: {line[:120]}..."
                )
    return data


def extract_markers(data: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Map _id -> list of predicted markers, each with a 'matched' flag.
    Expects submission format: {"_id": ..., "markers":[{"type", "startIndex", "endIndex"}]}
    """
    prepared: Dict[str, List[Dict]] = {}
    for item in data:
        doc_id = item.get("_id") or item.get("doc_id")
        if not doc_id:
            continue
        markers_list = [
            {
                "start": m.get("startIndex"),
                "end": m.get("endIndex"),
                "type": m.get("type"),
                "matched": False,
            }
            for m in item.get("markers", [])
            if m.get("type") in MARKER_TYPES
            and isinstance(m.get("startIndex"), int)
            and isinstance(m.get("endIndex"), int)
            and m.get("startIndex") < m.get("endIndex")
        ]
        prepared[doc_id] = markers_list
    return prepared


def prepare_true_data(data: List[Dict]) -> Dict[str, Dict]:
    """
    Prepare ground truth map: _id -> {"token_spans": [...], "markers": [...]}
    Tokenization is done once per doc on the ground-truth TEXT.
    """
    prepared: Dict[str, Dict] = {}
    for item in data:
        doc_id = item.get("_id") or item.get("doc_id")
        text = item.get("text", "")
        if not doc_id or not text:
            # Skip GT rows without text (cannot tokenize)
            continue

        token_spans = tokenize_text(text)

        markers_list = []
        for m in item.get("markers", []):
            m_type = m.get("type") or m.get("label")
            s = (
                m.get("startIndex")
                if m.get("startIndex") is not None
                else m.get("start")
            )
            e = m.get("endIndex") if m.get("endIndex") is not None else m.get("end")
            try:
                s = int(s)
                e = int(e)
            except Exception:
                continue
            if (
                m_type in MARKER_TYPES
                and isinstance(s, int)
                and isinstance(e, int)
                and s < e
            ):
                markers_list.append(
                    {
                        "start": s,
                        "end": e,
                        "type": m_type,
                        "matched": False,
                    }
                )

        prepared[doc_id] = {"token_spans": token_spans, "markers": markers_list}
    return prepared


def evaluate(true_data, pred_data, iou_threshold, strict_extra_preds=False):
    """
    Compute Token-Based Overlap F1 across all docs/types.
    """
    if true_data is None or pred_data is None:
        return {"Error": "Data loading failed."}, {}

    true_docs = prepare_true_data(true_data)
    pred_markers_map = extract_markers(pred_data)

    logging.info(f"GT docs usable (have text): {len(true_docs)}")
    logging.info(f"Pred docs: {len(pred_markers_map)}")

    total_tp = total_fp = total_fn = 0
    type_metrics = {t: {"tp": 0, "fp": 0, "fn": 0} for t in MARKER_TYPES}

    gt_ids = set(true_docs.keys())
    pred_ids = set(pred_markers_map.keys())

    missing_in_pred = gt_ids - pred_ids
    extra_in_pred = pred_ids - gt_ids

    if missing_in_pred:
        logging.info(
            f"Predictions missing for {len(missing_in_pred)} GT docs (these can still get FN via unmatched GT spans)."
        )
    if extra_in_pred:
        if strict_extra_preds:
            logging.info(
                f"Counting {len(extra_in_pred)} extra predicted doc IDs as FPs (strict mode)."
            )
        else:
            logging.info(
                f"Ignoring {len(extra_in_pred)} extra predicted doc IDs (default Codabench-like behavior)."
            )

    # Evaluate only over GT docs (Codabench style)
    for doc_id in gt_ids:
        true_doc = true_docs.get(doc_id)
        if not true_doc:
            continue

        true_spans = true_doc["markers"]
        token_spans = true_doc["token_spans"]
        pred_spans = pred_markers_map.get(doc_id, [])

        # Match loop
        for tspan in true_spans:
            true_set = char_span_to_token_set(tspan["start"], tspan["end"], token_spans)

            best_iou = -1.0
            best_idx = -1

            for idx, pspan in enumerate(pred_spans):
                if pspan["matched"] or pspan["type"] != tspan["type"]:
                    continue
                pred_set = char_span_to_token_set(
                    pspan["start"], pspan["end"], token_spans
                )
                iou = calculate_token_iou(true_set, pred_set)
                if iou > best_iou:
                    best_iou, best_idx = iou, idx

            if best_iou >= iou_threshold and best_idx != -1:
                total_tp += 1
                type_metrics[tspan["type"]]["tp"] += 1
                tspan["matched"] = True
                pred_spans[best_idx]["matched"] = True

        # Count FN and FP within this doc
        for tspan in true_spans:
            if not tspan["matched"]:
                total_fn += 1
                type_metrics[tspan["type"]]["fn"] += 1

        for pspan in pred_spans:
            if not pspan["matched"] and pspan["type"] in MARKER_TYPES:
                total_fp += 1
                type_metrics[pspan["type"]]["fp"] += 1

    # Strict mode: extra doc IDs count as FP (every predicted span)
    if strict_extra_preds and extra_in_pred:
        for doc_id in extra_in_pred:
            for pspan in pred_markers_map.get(doc_id, []):
                if pspan["type"] in MARKER_TYPES:
                    total_fp += 1
                    type_metrics[pspan["type"]]["fp"] += 1

    # Aggregate metrics
    final_results: Dict[str, float] = {}
    all_f1 = []

    for mtype in sorted(MARKER_TYPES):
        tp = type_metrics[mtype]["tp"]
        fp = type_metrics[mtype]["fp"]
        fn = type_metrics[mtype]["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        all_f1.append(f1)
        final_results[f"P ({mtype})"] = p
        final_results[f"R ({mtype})"] = r
        final_results[f"F1 ({mtype})"] = f1

    f1_macro = sum(all_f1) / len(MARKER_TYPES) if MARKER_TYPES else 0.0
    p_agg = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    r_agg = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_agg = 2 * p_agg * r_agg / (p_agg + r_agg) if (p_agg + r_agg) > 0 else 0.0

    final_results["F1 (Macro)"] = f1_macro
    final_results["P (Agg)"] = p_agg
    final_results["R (Agg)"] = r_agg
    final_results["F1 (Agg)"] = f1_agg

    # Pretty-printable summary
    formatted = {"--- Per-Type Results (Token IoU) ---": "---"}
    for mtype in sorted(MARKER_TYPES):
        formatted[f"P ({mtype})"] = f"{final_results[f'P ({mtype})']:.4f}"
        formatted[f"R ({mtype})"] = f"{final_results[f'R ({mtype})']:.4f}"
        formatted[f"F1 ({mtype})"] = f"{final_results[f'F1 ({mtype})']:.4f}"

    formatted["--- Aggregate Results ---"] = "---"
    formatted["IoU Threshold"] = (
        DEFAULT_IOU_THRESHOLD if iou_threshold is None else iou_threshold
    )
    formatted["True Positives (Agg)"] = total_tp
    formatted["False Positives (Agg)"] = total_fp
    formatted["False Negatives (Agg)"] = total_fn
    formatted["Precision (Agg)"] = f"{p_agg:.4f}"
    formatted["Recall (Agg)"] = f"{r_agg:.4f}"
    formatted["F1-Score (Agg/Micro)"] = f"{f1_agg:.4f}"
    formatted["F1-Score (Macro)"] = f"{f1_macro:.4f}"

    return final_results, formatted


def save_scores_to_codabench(results, output_file):
    """
    Save scores in a Codabench-friendly JSON.
    """
    scores = dict()
    scores["F1_Aggregate_Token"] = results.get("F1 (Agg)", 0.0)
    scores["Precision_Aggregate_Token"] = results.get("P (Agg)", 0.0)
    scores["Recall_Aggregate_Token"] = results.get("R (Agg)", 0.0)
    scores["F1_Macro_Token"] = results.get("F1 (Macro)", 0.0)

    for m_type in sorted(MARKER_TYPES):
        scores[f"F1_{m_type}_Token"] = results.get(f"F1 ({m_type})", 0.0)
        scores[f"Precision_{m_type}_Token"] = results.get(f"P ({m_type})", 0.0)
        scores[f"Recall_{m_type}_Token"] = results.get(f"R ({m_type})", 0.0)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    logging.info(f"Token-based scores saved to {output_file} (Codabench-compatible).")


if __name__ == "__main__":
    args = parse_args()

    TEST_FILE = args.ground_truth_file
    SUB_FILE = args.prediction_file
    SCORES_FILE = args.scores_output_file
    IOU = args.iou_threshold
    STRICT = args.strict_extra_preds

    print(
        f"Starting TOKEN-BASED evaluation (IoU >= {IOU}).\n"
        f"Ground Truth (must contain text): {TEST_FILE}\n"
        f"Predictions (char offsets):       {SUB_FILE}\n"
        f"Strict extra preds counted as FP: {STRICT}"
    )

    gt = load_jsonl(TEST_FILE)
    pr = load_jsonl(SUB_FILE)

    if gt is None or pr is None:
        print("Evaluation terminated due to file loading errors.")
        default = {
            "F1 (Agg)": 0.0,
            "P (Agg)": 0.0,
            "R (Agg)": 0.0,
            "F1 (Macro)": 0.0,
        }
        for t in MARKER_TYPES:
            default[f"F1 ({t})"] = 0.0
            default[f"P ({t})"] = 0.0
            default[f"R ({t})"] = 0.0
        save_scores_to_codabench(default, SCORES_FILE)
        sys.exit(1)

    raw, pretty = evaluate(gt, pr, iou_threshold=IOU, strict_extra_preds=STRICT)
    save_scores_to_codabench(raw, SCORES_FILE)

    print("\n--- Token-Based Evaluation Results ---")
    for k, v in pretty.items():
        if k.startswith("---"):
            print(k)
        else:
            print(f"{k:<30}: {v}")
    print("--------------------------------------")
