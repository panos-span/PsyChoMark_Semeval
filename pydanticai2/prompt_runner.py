#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prompt_runner.py — The Competition Execution Engine.

Updates:
- RESUME CAPABILITY: Automatically skips docs already in output files.
- LIVE LOGGING: No more buffering. Logs appear instantly.
- S1 EVALUATION: Calculates Macro F1 against ground truth.
- S2 SKIPPING: Skips 'Can't Tell' docs for S2 to save cost.
"""

from datetime import datetime
import os
import sys
import json
import asyncio
import argparse
import random
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
from sklearn.metrics import f1_score
import pathlib

# Third-party
import mlflow
from tqdm.asyncio import tqdm
from loguru import logger

# -----------------------
# CLI & Environment setup
# -----------------------

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def normalize_label(label: Any) -> str:
    s = str(label).lower().strip().replace(".", "")
    if s in ["yes", "true", "conspiracy", "conspiracy theory", "1"]:
        return "conspiracy"
    if s in ["no", "false", "non", "not conspiracy", "0"]:
        return "non"
    return "ambiguous"


# --- Cost Estimation Helper ---
class TokenMeter:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.INPUT_RATE = 3.00
        self.OUTPUT_RATE = 15.00

    def add(self, usage: Dict[str, int]):
        if not usage:
            return
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)

    def total_cost(self) -> float:
        in_cost = (self.input_tokens / 1_000_000) * self.INPUT_RATE
        out_cost = (self.output_tokens / 1_000_000) * self.OUTPUT_RATE
        return in_cost + out_cost


class OpenAITokenMeter:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        # OpenAI GPT-4o / GPT-5.2 Estimated Pricing (adjust as per actual release)
        # Current GPT-4o: $2.50 / 1M input, $10.00 / 1M output
        self.INPUT_RATE = 1.75
        self.OUTPUT_RATE = 14.00

    def add(self, usage: Dict[str, int]):
        if not usage:
            return
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)

    def total_cost(self) -> float:
        in_cost = (self.input_tokens / 1_000_000) * self.INPUT_RATE
        out_cost = (self.output_tokens / 1_000_000) * self.OUTPUT_RATE
        return in_cost + out_cost


GLOBAL_METER = OpenAITokenMeter()
# OPENAI_METER = TokenMeter()


# ---------- .env loader ----------
def _load_dotenv_into_environ():
    root = pathlib.Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    if "ACCESS_KEY_ID" in os.environ and "AWS_ACCESS_KEY_ID" not in os.environ:
        os.environ["AWS_ACCESS_KEY_ID"] = os.environ["ACCESS_KEY_ID"]
    if "SECRET_ACCESS_KEY" in os.environ and "AWS_SECRET_ACCESS_KEY" not in os.environ:
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["SECRET_ACCESS_KEY"]


_load_dotenv_into_environ()

# Project modules
import pydanticai2.psycomark_agents as agents_mod
import pydanticai2.prompt_builder as prompt_builder
from pydanticai2.prompt_loader import S1_PROMPTS, S2_PROMPTS
from pydanticai2.s1_graph import s1_graph
from pydanticai2.psycomark_graph import s2_graph, s2_parallel_graph

# --- LIVE LOGGING CONFIGURATION ---
logger.remove()
logger.add(
    lambda msg: tqdm.write(msg, end=""),
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="DEBUG",
)


# --- S1 Evaluator ---
class S1Evaluator:
    """
    Strict implementation of 'Macro Overlap F1-Score' for Conspiracy Marker Extraction.
    - Threshold: IoU >= 0.5
    - Macro Average: Fixed across 5 specific categories.
    """

    # Define the 5 immutable schema categories
    SCHEMA_LABELS = {"Actor", "Action", "Effect", "Evidence", "Victim"}

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        # Track TP/FP/FN for the fixed schema labels only
        self.tp = {k: 0 for k in self.SCHEMA_LABELS}
        self.fp = {k: 0 for k in self.SCHEMA_LABELS}
        self.fn = {k: 0 for k in self.SCHEMA_LABELS}

    def compute_iou(self, span_a: dict, span_b: dict) -> float:
        # (Same intersection logic as before)
        start_a, end_a = span_a["start"], span_a["end"]
        start_b, end_b = span_b["start"], span_b["end"]

        inter_s = max(start_a, start_b)
        inter_e = min(end_a, end_b)

        if inter_e <= inter_s:
            return 0.0

        inter = inter_e - inter_s
        union = (end_a - start_a) + (end_b - start_b) - inter
        return inter / union if union > 0 else 0.0

    def normalize_span(self, s: dict) -> Optional[dict]:
        # (Same normalization, but filters for valid schema labels)
        start = s.get("startIndex") or s.get("start")
        end = s.get("endIndex") or s.get("end")
        label = s.get("type") or s.get("label")

        if start is None or end is None or not label:
            return None

        # Capitalize label to match schema (e.g., "actor" -> "Actor")
        clean_label = str(label).strip().capitalize()
        if clean_label not in self.SCHEMA_LABELS:
            return None  # Ignore non-schema junk

        return {"start": int(start), "end": int(end), "type": clean_label}

    def update(self, predictions: List[Dict], ground_truth: List[Dict]):
        pred_map = defaultdict(list)
        gt_map = defaultdict(list)

        # 1. Normalize and Group
        for p in predictions:
            norm = self.normalize_span(p)
            if norm:
                pred_map[norm["type"]].append(norm)

        for g in ground_truth:
            norm = self.normalize_span(g)
            if norm:
                gt_map[norm["type"]].append(norm)

        # 2. Match per Category (Fixed 5)
        for label in self.SCHEMA_LABELS:
            preds = pred_map[label]
            golds = gt_map[label]

            matched_p_indices = set()

            for g in golds:
                best_iou = 0.0
                best_idx = -1

                for i, p in enumerate(preds):
                    if i in matched_p_indices:
                        continue

                    iou = self.compute_iou(p, g)
                    # [STRICT ALIGNMENT] Check IoU >= 0.5
                    if iou >= self.iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_idx = i

                if best_idx != -1:
                    self.tp[label] += 1
                    matched_p_indices.add(best_idx)
                else:
                    self.fn[label] += 1

            # False Positives = Unmatched Predictions
            self.fp[label] += len(preds) - len(matched_p_indices)

    def get_macro_f1(self) -> Dict[str, float]:
        f1_scores = []
        metrics = {}

        # Calculate F1 for every category in the fixed schema
        for label in self.SCHEMA_LABELS:
            tp = self.tp[label]
            fp = self.fp[label]
            fn = self.fn[label]

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            metrics[f"{label}_f1"] = f1
            f1_scores.append(f1)

        # Unweighted Average of exactly 5 scores
        metrics["macro_f1"] = sum(f1_scores) / 5.0
        return metrics


# --- I/O Utilities ---
def load_input(path: str, fmt: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        logger.error(f"Input not found: {path}")
        sys.exit(1)
    items = []
    if fmt == "jsonl":
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        row = json.loads(line)
                        doc_id = (
                            row.get("doc_id") or row.get("_id") or str(row.get("id"))
                        )
                        label = row.get("label") or row.get("conspiracy")
                        meta_dict = row.get("metadata", {})
                        gt_markers = (
                            row.get("markers")
                            or row.get("spans")
                            or meta_dict.get("markers")
                            or meta_dict.get("spans")
                            or []
                        )
                        items.append(
                            {
                                "id": doc_id,
                                "text": row.get("text", "").strip(),
                                "metadata": {
                                    "subreddit": row.get("subreddit"),
                                    "label": label,
                                    "s2_subtype": row.get("s2_subtype"),
                                    "gt_markers": gt_markers,
                                },
                            }
                        )
                    except Exception:
                        continue
    else:
        items.append(
            {
                "id": p.stem,
                "text": p.read_text(encoding="utf-8").strip(),
                "metadata": {},
            }
        )
    logger.info(f"Loaded {len(items)} documents from {path}")
    return items


def load_processed_ids(file_name: str) -> Set[str]:
    """Reads a JSONL file and returns a set of processed doc IDs."""
    if not os.path.exists(file_name):
        return set()
    ids = set()
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        row = json.loads(line)
                        ids.add(row.get("_id") or row.get("doc_id"))
                    except:
                        pass
    except Exception as e:
        logger.warning(f"Error reading {file_name}: {e}")
    return ids


def append_jsonl(file_name: str, item: Dict):
    with open(file_name, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def finalize_zip(jsonl_name: str, zip_name: str):
    if os.path.exists(jsonl_name):
        with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(jsonl_name)
        logger.info(f"Compressed {jsonl_name} -> {zip_name}")


# --- Core Pipeline Logic ---
async def process_document(
    row: Dict[str, Any], args: argparse.Namespace, rag_collections: Dict[str, Any]
) -> Tuple[Optional[Dict], Optional[Dict]]:
    doc_id = row["id"]
    text = row["text"]
    meta = row["metadata"]

    if len(text) < 10:
        logger.warning(f"[{doc_id}] Skipping text too short (<10 chars).")
        return None, None

    # --- Step 1: S1 Extraction ---
    s1_record = None
    s1_spans_for_s2 = []
    s1_dossier = ""

    if args.task in ("s1", "both"):
        try:
            s1_rag = rag_collections.get("s1")
            few_shots = []
            if s1_rag:
                if args.rerank:
                    few_shots = agents_mod.retrieve_stratified_s1_reranked(s1_rag, text)
                else:
                    few_shots = agents_mod.retrieve_stratified_s1(s1_rag, text)

            # logger.info(
            #    f"[{doc_id}] Retrieved {len(few_shots)} S1 few-shots with Reranking."
            # )
            # logger.info(f"[{doc_id}] Few-shots: {few_shots}")

            s1_initial_state = {
                "doc_id": doc_id,
                "text": text,
                "few_shots": few_shots,
                "metadata": meta,
                "k": args.s1_k,
                "raw_runs": [],
                "consensus_spans": [],
                "final_spans": [],
            }

            final_state = await asyncio.wait_for(
                s1_graph.ainvoke(s1_initial_state), timeout=200
            )

            if "token_usage" in final_state:
                GLOBAL_METER.add(final_state["token_usage"])

            verified_spans = final_state.get("final_spans", [])
            gen_out = final_state.get("generator_output") or final_state.get(
                "draft_output"
            )

            complexity = "Unknown"
            narrative = "Unknown"
            if gen_out:
                if hasattr(gen_out, "text_complexity"):
                    complexity = gen_out.text_complexity
                    narrative = gen_out.dominant_narrative
                elif isinstance(gen_out, dict):
                    complexity = gen_out.get("text_complexity", "Unknown")
                    narrative = gen_out.get("dominant_narrative", "Unknown")

            markers = []
            for s in verified_spans:
                markers.append(
                    {
                        "type": s["label"],
                        "startIndex": s["start"],
                        "endIndex": s["end"],
                        "text": s["text"],
                    }
                )

            s1_record = {"_id": doc_id, "markers": markers}
            s1_spans_for_s2 = markers
            s1_dossier = agents_mod.synthesize_dossier(
                markers, complexity=complexity, narrative=narrative
            )
            logger.info(f"[{doc_id}] S1 Complete: {len(markers)} markers found.")

        except asyncio.TimeoutError:
            logger.error(f"[{doc_id}] S1 Timed Out (200s)")
            s1_record = {"_id": doc_id, "markers": [], "error": "Timeout"}
        except Exception as e:
            logger.error(f"[{doc_id}] S1 Failed: {e}")
            s1_record = {"_id": doc_id, "markers": [], "error": str(e)}

    # --- Step 2: S2 Classification ---
    s2_record = None
    if args.task in ("s2", "both"):
        gt_raw = str(meta.get("label", "")).lower().strip()
        if gt_raw in ["cant_tell", "can't tell", "ambiguous"]:
            logger.info(f"[{doc_id}] Skipping S2 (Ground Truth is Ambiguous)")
            return s1_record, None

        try:
            rag_context = ""
            s2_rag = rag_collections.get("s2")
            logger.info(f"[{doc_id}] Preparing S2 RAG retrieval...")
            if s2_rag:
                if args.rerank:
                    precedents = agents_mod.retrieve_balanced_precedents(
                        s2_rag, text, k=4, overretrieve_factor=4
                    )
                else:
                    precedents = agents_mod.retrieve_fewshots(
                        s2_rag, text, k=4, filters={"is_hard_negative": True}
                    )
                logger.info(f"[{doc_id}] Retrieved {len(precedents)} S2 precedents.")
                if precedents:
                    rag_context = json.dumps(precedents, indent=2, ensure_ascii=False)

            # logger.info(
            #    f"[{doc_id}] S2| Retrieved {len(rag_context)} S1 few-shots with Reranking."
            # )
            # logger.info(f"[{doc_id}] S2| Few-shots: {rag_context}")

            s2_initial_state = {
                "doc_id": doc_id,
                "text": text,
                "s1_spans": s1_spans_for_s2,
                "marker_summary": s1_dossier or "No markers.",
                "rag_context": rag_context,
                "juror_temperature": args.s2_temp,
                "metadata": meta,
                "parallel_council_result": None,
                "calibrated_output": None,
                "council_result": None,
                "final_output": None,
            }

            graph_to_use = s2_parallel_graph if args.s2_mode == "parallel" else s2_graph
            final_state = await asyncio.wait_for(
                graph_to_use.ainvoke(s2_initial_state), timeout=200
            )

            s2_result = (
                final_state.get("calibrated_output")
                if args.s2_mode == "parallel"
                else final_state.get("final_output")
            )

            if "token_usage" in final_state:
                GLOBAL_METER.add(final_state["token_usage"])

            if s2_result:
                s2_record = {
                    "_id": doc_id,
                    "conspiracy": (
                        "Yes" if s2_result.label.lower() == "conspiracy" else "No"
                    ),
                    "confidence": s2_result.confidence,
                    "rationale": s2_result.rationale,
                }
                gt_norm = "yes" if gt_raw in ["conspiracy", "yes"] else "no"
                pred_norm = "yes" if s2_result.label.lower() == "conspiracy" else "no"
                if pred_norm == gt_norm:
                    logger.success(
                        f"[{doc_id}] ✅ CORRECT (Pred: {pred_norm.upper()} | GT: {gt_norm.upper()}) [Conf: {s2_result.confidence:.2f}]"
                    )
                else:
                    logger.error(
                        f"[{doc_id}] ❌ FAILURE (Pred: {pred_norm.upper()} | GT: {gt_norm.upper()}) [Conf: {s2_result.confidence:.2f}]"
                    )
            else:
                s2_record = {"_id": doc_id, "conspiracy": "No", "error": "None result"}

        except asyncio.TimeoutError:
            logger.error(f"[{doc_id}] S2 Timed Out (200s)")
            s2_record = {"_id": doc_id, "conspiracy": "No", "error": "Timeout"}
        except Exception as e:
            logger.error(f"[{doc_id}] S2 Failed: {e}")
            s2_record = {"_id": doc_id, "conspiracy": "No", "error": str(e)}

    return s1_record, s2_record


# --- Main Entry Point ---
async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--format", default="jsonl")
    parser.add_argument("--rag-dir", default="data/rag_online_v3")
    parser.add_argument("--task", default="both")
    parser.add_argument("--s1-k", type=int, default=2)
    parser.add_argument("--s2-temp", type=float, default=0.4)
    parser.add_argument("--s2-mode", default="parallel")
    parser.add_argument("--rerank", action="store_true", default=True)
    parser.add_argument("--no-rerank", action="store_false", dest="rerank")
    parser.add_argument(
        "--concurrency", type=int, default=1, help="Max parallel documents"
    )
    parser.add_argument("--experiment-name", default="PsyCoMark_SOTA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from existing output files",
    )
    args = parser.parse_args()

    s1_evaluator = S1Evaluator()
    random.seed(args.seed)

    # 1. Load All Rows
    all_rows = load_input(args.input, args.format)

    # 2. Check for Resume
    processed_s1 = set()
    processed_s2 = set()
    if args.resume:
        processed_s1 = load_processed_ids("submission_s1.jsonl")
        processed_s2 = load_processed_ids("submission_s2.jsonl")
        if processed_s1 or processed_s2:
            logger.info(
                f"Resuming: Found {len(processed_s1)} S1 docs and {len(processed_s2)} S2 docs."
            )

    # 3. Filter Rows
    rows_to_process = []
    skipped_count = 0

    # Pre-load S1 data for evaluator if resuming
    if args.resume and processed_s1:
        # We need to feed the evaluator the *already processed* data so the F1 score starts correct
        with open("submission_s1.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                try:
                    s1_res = json.loads(line)
                    doc_id = s1_res.get("_id")
                    # Find ground truth
                    orig = next((r for r in all_rows if r["id"] == doc_id), None)
                    if orig and orig["metadata"].get("gt_markers"):
                        s1_evaluator.update(
                            s1_res.get("markers", []), orig["metadata"]["gt_markers"]
                        )
                except:
                    pass

    for row in all_rows:
        doc_id = row["id"]
        # Logic: If task is S1, check S1 done. If S2, check S2 done. If Both, check BOTH done.
        s1_done = doc_id in processed_s1
        s2_done = doc_id in processed_s2

        should_run = True
        if args.task == "s1" and s1_done:
            should_run = False
        elif args.task == "s2" and s2_done:
            should_run = False
        elif args.task == "both":
            # If S2 is done, we are fully done.
            # If S1 is done but S2 isn't, we effectively need to re-run S1 to pass state to S2 (unless we load state, which is complex)
            # Simplification: If S2 is done, skip.
            # If S2 isn't done, but label is 'cant_tell', check if S1 is done.
            gt_label = str(row["metadata"].get("label", "")).lower()
            is_ambiguous = gt_label in ["cant_tell", "can't tell", "ambiguous"]

            if s2_done:
                should_run = False
            elif is_ambiguous and s1_done:
                should_run = False  # S2 not needed for ambiguous, S1 is done

        if should_run:
            rows_to_process.append(row)
        else:
            skipped_count += 1

    logger.info(f"Skipping {skipped_count} documents. Queueing {len(rows_to_process)}.")

    if not rows_to_process:
        logger.success("All documents processed! Nothing to do.")
        return

    # 4. Setup Logistics
    log_file = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"  # Don't delete if resuming!
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="DEBUG",
    )

    rag_collections = {"s1": None, "s2": None}
    if args.rag_dir:
        try:
            logger.info(f"Loading RAG collections from {args.rag_dir}...")
            rag_collections["s1"] = agents_mod.get_rag_collection(
                args.rag_dir, "s1_markers"
            )
            rag_collections["s2"] = agents_mod.get_rag_collection(
                args.rag_dir, "s2_examples"
            )
            logger.info("RAG Collections Loaded.")
        except Exception as e:
            logger.warning(f"RAG Load Failed: {e}")

    mlflow.set_experiment(args.experiment_name)
    sem = asyncio.Semaphore(args.concurrency)

    async def worker(row):
        async with sem:
            with logger.contextualize(doc_id=row["id"]):
                logger.info("STARTING PROCESSING")
                return await process_document(row, args, rag_collections)

    tasks = [worker(row) for row in rows_to_process]

    # Initialize counters (Metrics need to account for what we loaded from resume)
    # Note: Accuracy is tricky to "resume" without reading all S2 outputs.
    # For now, we calculate accuracy on *this session's run* + *what we can reconstruct*.
    correct_s2 = 0
    total_eval_s2 = 0

    # Initialize history for F1 Calculation
    s2_y_true = []
    s2_y_pred = []

    # Pre-calc S2 accuracy from file if resuming (Optional but good for progress bar)
    if args.resume and processed_s2:
        with open("submission_s2.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                try:
                    res = json.loads(line)
                    if "conspiracy" in res:
                        orig = next(
                            (r for r in all_rows if r["id"] == res.get("_id")), None
                        )
                        if orig:
                            # Normalize GT
                            gt_raw = str(orig["metadata"].get("label", "")).lower()
                            # Skip ambiguous for binary F1 calculation
                            if gt_raw not in ["cant_tell", "ambiguous", "can't tell"]:
                                gt_norm = normalize_label(gt_raw)
                                pred_norm = normalize_label(res["conspiracy"])

                                s2_y_true.append(gt_norm)
                                s2_y_pred.append(pred_norm)
                except:
                    pass

    logger.info(
        f"Starting pipeline with {len(rows_to_process)} docs (Concurrency={args.concurrency})"
    )

    with mlflow.start_run():
        with tqdm(total=len(rows_to_process), desc="Pipeline Progress") as pbar:
            for future in asyncio.as_completed(tasks):
                s1_res, s2_res = await future

                if s1_res:
                    append_jsonl("submission_s1.jsonl", s1_res)
                    doc_id = s1_res["_id"]
                    orig = next((r for r in all_rows if r["id"] == doc_id), None)
                    if orig and orig["metadata"].get("gt_markers"):
                        s1_evaluator.update(
                            s1_res.get("markers", []), orig["metadata"]["gt_markers"]
                        )

                if s2_res:
                    append_jsonl("submission_s2.jsonl", s2_res)
                    if "conspiracy" in s2_res:
                        orig = next(
                            (r for r in all_rows if r["id"] == s2_res["_id"]), None
                        )
                        if orig:
                            # [FIX] Robust Label Extraction (Root vs Metadata)
                            gt_raw = orig.get("label") or orig.get("metadata", {}).get(
                                "label"
                            )
                            gt_raw = str(gt_raw).lower().strip()

                            # Skip ambiguous for binary F1
                            if gt_raw and gt_raw not in [
                                "cant_tell",
                                "ambiguous",
                                "can't tell",
                                "none",
                            ]:
                                gt_norm = normalize_label(gt_raw)
                                pred_norm = normalize_label(s2_res["conspiracy"])

                                s2_y_true.append(gt_norm)
                                s2_y_pred.append(pred_norm)

                # 1. S1 Macro F1
                s1_f1 = s1_evaluator.get_macro_f1()["macro_f1"]

                # 2. S2 Weighted F1
                s2_f1_val = 0.0
                if len(s2_y_true) > 0:
                    # 'weighted' accounts for class imbalance
                    s2_f1_val = f1_score(
                        s2_y_true,
                        s2_y_pred,
                        average="weighted",
                        labels=["conspiracy", "non"],
                        zero_division=0,
                    )
                # 3. Update Bar
                pbar.set_postfix_str(
                    f"S1 F1: {s1_f1:.1%} | S2 F1(w): {s2_f1_val:.1%} | Cost: ${GLOBAL_METER.total_cost():.2f}"
                )
                pbar.update(1)

        if total_eval_s2 > 0:
            final_acc = correct_s2 / total_eval_s2
            mlflow.log_metric("s2_accuracy", final_acc)
            logger.success(f"FINAL S2 Accuracy: {final_acc:.2%}")

        s1_metrics = s1_evaluator.get_macro_f1()
        logger.success(f"FINAL S1 F1: {s1_metrics['macro_f1']:.2%}")
        mlflow.log_metric("s1_macro_f1", s1_metrics["macro_f1"])

        for f in ["submission_s1.jsonl", "submission_s2.jsonl"]:
            if os.path.exists(f):
                finalize_zip(f, f.replace(".jsonl", ".zip"))
                mlflow.log_artifact(f.replace(".jsonl", ".zip"))
        if os.path.exists(log_file):
            mlflow.log_artifact(log_file)


if __name__ == "__main__":
    asyncio.run(main_async())
