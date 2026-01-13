#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prompt_runner.py — The Competition Execution Engine.

This script orchestrates the PsyCoMark pipeline:
1. S1: Runs the Parallel Consensus Graph (Ensemble -> Vote -> Verify).
2. S2: Runs the Council of Jurors (Multi-Persona Classification).
3. Logging: Tracks experiment metrics and artifacts via MLflow.

Usage:
  python prompt_runner.py data/test.jsonl --task both --s1-k 3
"""

import os
import sys
import json
import asyncio
import argparse
import random
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import pathlib

# Third-party
import mlflow
from tqdm.asyncio import tqdm
from loguru import logger

# -----------------------
# CLI & Environment setup
# -----------------------

# --- Make repo root importable ---
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Global buffer: Key = doc_id, Value = list of formatted log strings
LOG_BUFFERS = defaultdict(list)


# --- Cost Estimation Helper ---
class TokenMeter:
    """Tracks token usage and estimates cost (based on Claude 3.5 Sonnet rates)."""

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        # Rates per 1M tokens (Approximate for Sonnet)
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

    def __str__(self):
        return (
            f"Tokens: {self.input_tokens + self.output_tokens:,} "
            f"(In: {self.input_tokens:,}, Out: {self.output_tokens:,}) "
            f"| Est. Cost: ${self.total_cost():.4f}"
        )


# Global Meter
GLOBAL_METER = TokenMeter()


# ---------- .env loader (no deps) ----------
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
    # Map non-standard names to AWS_* so boto3 sees them
    if "ACCESS_KEY_ID" in os.environ and "AWS_ACCESS_KEY_ID" not in os.environ:
        os.environ["AWS_ACCESS_KEY_ID"] = os.environ["ACCESS_KEY_ID"]
    if "SECRET_ACCESS_KEY" in os.environ and "AWS_SECRET_ACCESS_KEY" not in os.environ:
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ["SECRET_ACCESS_KEY"]


_load_dotenv_into_environ()


# Project modules
import pydanticai2.psycomark_agents as agents_mod
import pydanticai2.prompt_builder as prompt_builder
from pydanticai2.prompt_loader import S1_PROMPTS, S2_PROMPTS


# Import the S1 Graph (DD-CoT: Generator → Critic → Refiner)
from pydanticai2.s1_graph import s1_graph

# Import S2 Graphs - Legacy (Sequential Debate) and Parallel (Anti-Echo Chamber)
from pydanticai2.psycomark_graph import (
    s2_graph,  # Legacy: Sequential Debate
    s2_parallel_graph,  # New: Anti-Echo Chamber (Parallel Voting)
)

# --- Logging Configuration ---
LOG_BUFFERS: Dict[str, List[str]] = {}


def buffer_sink(message):
    """Captures logs per-doc to flush them atomically later."""
    rec = message.record
    doc_id = rec["extra"].get("doc_id")
    if doc_id:
        if doc_id not in LOG_BUFFERS:
            LOG_BUFFERS[doc_id] = []
        LOG_BUFFERS[doc_id].append(message)
    else:
        # Global logs go straight to stderr
        sys.stderr.write(message)


# Reset logger
logger.remove()
logger.add(
    buffer_sink,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="DEBUG",
)


# --- I/O Utilities ---


def load_input(path: str, fmt: str) -> List[Dict[str, Any]]:
    """Loads input data and normalizes metadata."""
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
                        # Normalize ID
                        doc_id = (
                            row.get("doc_id") or row.get("_id") or str(row.get("id"))
                        )

                        # Capture Ground Truth (if training/dev set)
                        # 'conspiracy' might be "Yes"/"No" or "conspiracy"/"non"
                        label = row.get("label") or row.get("conspiracy")

                        items.append(
                            {
                                "id": doc_id,
                                "text": row.get("text", "").strip(),
                                "metadata": {
                                    "subreddit": row.get("subreddit"),
                                    "label": label,  # Critical for Live Eval
                                    "s2_subtype": row.get("s2_subtype"),
                                },
                            }
                        )
                    except Exception:
                        continue
    else:
        # Raw text file
        items.append(
            {
                "id": p.stem,
                "text": p.read_text(encoding="utf-8").strip(),
                "metadata": {},
            }
        )

    logger.info(f"Loaded {len(items)} documents from {path}")
    return items


def append_jsonl(file_name: str, item: Dict):
    """Crash-proof incremental saving."""
    with open(file_name, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def finalize_zip(jsonl_name: str, zip_name: str):
    """Compresses the final output."""
    if os.path.exists(jsonl_name):
        with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(jsonl_name)
        logger.info(f"Compressed {jsonl_name} -> {zip_name}")


def save_jsonl(file_name: str, zip_name: str, items: List[Dict]):
    """Saves output to JSONL and immediately zips it for leaderboard submission."""
    with open(file_name, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(file_name)

    # Cleanup raw file to save space? Optional.
    # os.remove(file_name)


# --- Core Pipeline Logic ---


async def process_document(
    row: Dict[str, Any], args: argparse.Namespace, rag_collections: Dict[str, Any]
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Runs the full pipeline (S1 Graph -> S2 Graph) for a single document.
    """
    doc_id = row["id"]
    text = row["text"]
    meta = row["metadata"]

    if len(text) < 10:
        logger.warning(f"[{doc_id}] Skipping text too short (<10 chars).")
        return None, None

    # --- Step 1: S1 Extraction (Consensus Graph) ---
    s1_record = None
    s1_spans_for_s2 = []  # Raw list of dicts for S2
    s1_dossier = ""  # Narrative summary for S2

    if args.task in ("s1", "both"):
        try:
            # 1. Prepare S1 State
            # Note: We fetch stratified RAG examples *before* the graph if needed,
            # or rely on the graph/agent to handle empty few_shots.
            # Here we fetch them to be safe:
            s1_rag = rag_collections.get("s1")
            few_shots = []
            if s1_rag:
                if args.rerank:
                    few_shots = agents_mod.retrieve_stratified_s1_reranked(s1_rag, text)
                else:
                    few_shots = agents_mod.retrieve_stratified_s1(s1_rag, text)

            s1_initial_state = {
                "doc_id": doc_id,
                "text": text,
                "few_shots": few_shots,
                "k": args.s1_k,  # <--- Pass CLI arg here
                "raw_runs": [],
                "consensus_spans": [],
                "final_spans": [],
            }

            # 2. Invoke S1 Graph
            final_state = await s1_graph.ainvoke(s1_initial_state)

            # [Optimization] Track S1 Tokens
            if "token_usage" in final_state:
                GLOBAL_METER.add(final_state["token_usage"])

            # 3. Extract Results
            verified_spans = final_state.get("final_spans", [])

            # --- [NEW] Extract Dynamic Metadata from S1 Generator ---
            # We look for the raw DDCoTExtraction object to get complexity/narrative
            complexity = "Unknown"
            narrative = "Unknown"

            # Try to find the generator output in the graph state
            # Adjust 'generator_output' key based on your s1_graph.py implementation
            gen_out = final_state.get("generator_output") or final_state.get(
                "draft_output"
            )

            if gen_out:
                # If it's a Pydantic object (DDCoTExtraction)
                if hasattr(gen_out, "text_complexity"):
                    complexity = gen_out.text_complexity
                    narrative = gen_out.dominant_narrative
                # If it's a dict (serialized state)
                elif isinstance(gen_out, dict):
                    complexity = gen_out.get("text_complexity", "Unknown")
                    narrative = gen_out.get("dominant_narrative", "Unknown")

            # 4. Format for Submission
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

            # 5. Prepare Evidence for S2
            s1_spans_for_s2 = markers
            s1_dossier = agents_mod.synthesize_dossier(
                markers, complexity=complexity, narrative=narrative
            )

            logger.info(
                f"[{doc_id}] S1 Complete: Found {len(markers)} consensus markers."
            )

        except Exception as e:
            logger.error(f"[{doc_id}] S1 Graph Failed: {e}")
            s1_record = {"_id": doc_id, "markers": [], "error": str(e)}

    # --- Step 2: S2 Classification (Council Graph) ---
    s2_record = None

    if args.task in ("s2", "both"):
        try:
            # 1. Retrieve RAG Precedents (Hard Negatives) for the Judge
            rag_context = ""
            s2_rag = rag_collections.get("s2")
            if s2_rag:
                # Retrieve specific hard negatives to help the Judge distinguish reporting vs endorsing
                if args.rerank:
                    precedents = agents_mod.retrieve_hard_negatives_reranked(
                        s2_rag, text, k=4, overretrieve_factor=4
                    )
                else:
                    precedents = agents_mod.retrieve_fewshots(
                        s2_rag, text, k=4, filters={"is_hard_negative": True}
                    )
                if precedents:
                    rag_context = json.dumps(precedents, indent=2, ensure_ascii=False)

            # 2. Prepare S2 State based on mode
            if args.s2_mode == "parallel":
                # Anti-Echo Chamber: Parallel Voting State
                s2_initial_state = {
                    "doc_id": doc_id,
                    "text": text,
                    "s1_spans": s1_spans_for_s2,
                    "marker_summary": s1_dossier or "No markers extracted.",
                    "rag_context": rag_context,
                    "juror_temperature": args.s2_temp,
                    "metadata": meta,
                    # Parallel-specific fields
                    "parallel_council_result": None,
                    "calibrated_output": None,
                }
                # Use Parallel Graph (Anti-Echo Chamber)
                final_state = await s2_parallel_graph.ainvoke(s2_initial_state)
                s2_result = final_state.get("calibrated_output")
            else:
                # Legacy: Sequential Debate State
                s2_initial_state = {
                    "doc_id": doc_id,
                    "text": text,
                    "s1_spans": s1_spans_for_s2,
                    "marker_summary": s1_dossier or "No markers extracted.",
                    "rag_context": rag_context,
                    "juror_temperature": args.s2_temp,
                    "metadata": meta,
                    "council_result": None,
                    "final_output": None,
                }
                # Use Legacy Graph (Sequential Debate)
                final_state = await s2_graph.ainvoke(s2_initial_state)
                s2_result = final_state.get("final_output")

            # [Optimization] Track S2 Tokens
            if "token_usage" in final_state:
                GLOBAL_METER.add(final_state["token_usage"])

            if s2_result:
                # 4. Format for Submission
                s2_record = {
                    "_id": doc_id,
                    "conspiracy": (
                        "Yes" if s2_result.label.lower() == "conspiracy" else "No"
                    ),
                    # Debug fields
                    "confidence": s2_result.confidence,
                    "rationale": s2_result.rationale,
                }
            else:
                s2_record = {
                    "_id": doc_id,
                    "conspiracy": "No",
                    "error": "Graph returned None",
                }

        except Exception as e:
            logger.error(f"[{doc_id}] S2 Graph Failed: {e}")
            s2_record = {"_id": doc_id, "conspiracy": "No", "error": str(e)}

    return s1_record, s2_record


# --- Main Entry Point ---


async def main_async():
    parser = argparse.ArgumentParser(description="PsyCoMark Competition Runner")

    # Inputs
    parser.add_argument("input", help="Path to input .jsonl or .txt")
    parser.add_argument("--format", default="jsonl", choices=["jsonl", "txt"])
    parser.add_argument("--rag-dir", help="Path to ChromaDB directory for RAG")

    # Task Control
    parser.add_argument("--task", default="both", choices=["s1", "s2", "both"])

    # Config (S1) - DD-CoT Architecture
    parser.add_argument(
        "--s1-k", type=int, default=3, help="Ensemble size (internal to graph)"
    )
    parser.add_argument(
        "--s2-temp", type=float, default=0.4, help="Temperature for S2 Jurors"
    )

    # Config (S2) - Anti-Echo Chamber Architecture
    parser.add_argument(
        "--s2-mode",
        choices=["legacy", "parallel"],
        default="parallel",
        help="S2 mode: 'legacy' (Sequential Debate) or 'parallel' (Anti-Echo Chamber, recommended)",
    )

    # RAG Enhancement
    parser.add_argument(
        "--rerank",
        action="store_true",
        default=True,
        help="Enable cross-encoder reranking for better RAG retrieval quality (default: enabled)",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_false",
        dest="rerank",
        help="Disable cross-encoder reranking (faster but lower quality)",
    )

    # General Config
    parser.add_argument("--experiment-name", default="PsyCoMark_Dev")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # 1. Setup Environment
    random.seed(args.seed)

    # 2. Load Data
    rows = load_input(args.input, args.format)

    # [NEW] Setup File Logging for MLflow Artifact
    log_file = "run.log"
    if os.path.exists(log_file):
        os.remove(log_file)
    # Add file sink to capture everything
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="DEBUG",
    )

    # 3. Init RAG
    rag_collections = {"s1": None, "s2": None}
    if args.rag_dir:
        try:
            rag_collections["s1"] = agents_mod.get_rag_collection(
                args.rag_dir, "s1_markers"
            )
            rag_collections["s2"] = agents_mod.get_rag_collection(
                args.rag_dir, "s2_examples"
            )
        except Exception as e:
            logger.warning(f"RAG Load Failed: {e}. Proceeding without retrieval.")

    # 4. MLflow Experiment Start
    mlflow.set_experiment(args.experiment_name)

    # Log architecture configuration
    logger.info("=" * 60)
    logger.info("🚀 PsyCoMark Pipeline Configuration")
    logger.info("=" * 60)
    logger.info(f"   Task: {args.task.upper()}")
    if args.task in ("s1", "both"):
        logger.info("   S1 Architecture: DD-CoT (Generator → Critic → Refiner)")
    if args.task in ("s2", "both"):
        if args.s2_mode == "parallel":
            logger.info("   S2 Architecture: Anti-Echo Chamber (Parallel Voting)")
            logger.info("     → All jurors vote independently and simultaneously")
            logger.info("     → Calibrated Judge weighs dissent")
        else:
            logger.info("   S2 Architecture: Legacy (Sequential Debate)")
            logger.info("     → Jurors debate in sequence")
            logger.info("     → Standard Judge review")
    logger.info("=" * 60)

    with mlflow.start_run():
        mlflow.log_params(vars(args))

        logger.info(f"Starting Run: {len(rows)} docs | Mode: {args.s2_mode.upper()}")

        # S1 Prompts
        if args.task in ("s1", "both"):
            # --- 4. DD-CoT Generator (Optimal) ---
            s1_ddcot_gen_sys = (
                getattr(S1_PROMPTS, "ddcot_gen_system", None)
                or prompt_builder.build_s1_ddcot_system()
            )
            s1_ddcot_gen_usr = (
                getattr(S1_PROMPTS, "ddcot_gen_user_template", None)
                or prompt_builder.build_s1_ddcot_user_template()
            )
            mlflow.log_text(
                s1_ddcot_gen_sys, "prompts/s1/s1_ddcot_generator_optimized.txt"
            )
            mlflow.log_text(s1_ddcot_gen_usr, "prompts/s1/s1_ddcot_user_optimized.txt")

            # --- 5. DD-CoT Critic (Optimal) ---
            s1_ddcot_crit_sys = (
                getattr(S1_PROMPTS, "ddcot_critic_system", None)
                or prompt_builder.build_s1_ddcot_critic_system()
            )
            s1_ddcot_crit_usr = (
                getattr(S1_PROMPTS, "ddcot_critic_user_template", None)
                or prompt_builder.build_s1_ddcot_critic_user_template()
            )
            mlflow.log_text(
                s1_ddcot_crit_sys, "prompts/s1/s1_ddcot_critic_optimized.txt"
            )
            mlflow.log_text(
                s1_ddcot_crit_usr, "prompts/s1/s1_ddcot_critic_user_optimized.txt"
            )

            # --- 6. DD-CoT Refiner (Optimal) ---
            s1_ddcot_ref_sys = (
                getattr(S1_PROMPTS, "ddcot_refiner_system", None)
                or prompt_builder.build_s1_ddcot_refiner_system()
            )
            s1_ddcot_ref_usr = (
                getattr(S1_PROMPTS, "ddcot_refiner_user_template", None)
                or prompt_builder.build_s1_ddcot_refiner_user_template()
            )
            mlflow.log_text(
                s1_ddcot_ref_sys, "prompts/s1/s1_ddcot_refiner_optimized.txt"
            )
            mlflow.log_text(
                s1_ddcot_ref_usr, "prompts/s1/s1_ddcot_refiner_user_optimized.txt"
            )

        # S2 Prompts
        if args.task in ("s2", "both"):
            # Helper to safely get prompt content
            def get_s2_p(attr, fallback_func):
                return (
                    getattr(S2_PROMPTS, attr)
                    if S2_PROMPTS and hasattr(S2_PROMPTS, attr)
                    else fallback_func()
                )

            if args.s2_mode == "parallel":
                # === PARALLEL MODE (Anti-Echo Chamber) ===
                logger.info("Logging S2 Parallel (Anti-Echo Chamber) Prompts...")

                # Log Parallel Juror System Prompts
                mlflow.log_text(
                    get_s2_p(
                        "parallel_pros_sys",
                        prompt_builder.build_s2_parallel_prosecutor_system,
                    ),
                    "prompts/s2/parallel_prosecutor_sys.txt",
                )
                mlflow.log_text(
                    get_s2_p(
                        "parallel_def_sys",
                        prompt_builder.build_s2_parallel_defense_system,
                    ),
                    "prompts/s2/parallel_defense_sys.txt",
                )
                mlflow.log_text(
                    get_s2_p(
                        "parallel_lit_sys",
                        prompt_builder.build_s2_parallel_literalist_system,
                    ),
                    "prompts/s2/parallel_literalist_sys.txt",
                )
                mlflow.log_text(
                    get_s2_p(
                        "parallel_prof_sys",
                        prompt_builder.build_s2_parallel_profiler_system,
                    ),
                    "prompts/s2/parallel_profiler_sys.txt",
                )

                # Log Shared User Template (used by all parallel jurors)
                mlflow.log_text(
                    get_s2_p(
                        "parallel_user", prompt_builder.build_s2_parallel_user_template
                    ),
                    "prompts/s2/parallel_user.txt",
                )

                # Log Calibrated Judge Prompts
                mlflow.log_text(
                    get_s2_p(
                        "calibrated_judge_sys",
                        prompt_builder.build_s2_calibrated_judge_system,
                    ),
                    "prompts/s2/calibrated_judge_sys.txt",
                )
                mlflow.log_text(
                    get_s2_p(
                        "calibrated_judge_user",
                        prompt_builder.build_s2_calibrated_judge_user_template,
                    ),
                    "prompts/s2/calibrated_judge_user.txt",
                )
            else:
                # === LEGACY MODE (Sequential Debate) ===
                logger.info("📦 Logging S2 Legacy (Sequential Debate) Prompts...")

                # Log System Personas
                mlflow.log_text(
                    get_s2_p("pros_sys", prompt_builder.build_s2_prosecutor_system),
                    "prompts/s2/sys_prosecutor.txt",
                )
                mlflow.log_text(
                    get_s2_p("def_sys", prompt_builder.build_s2_defense_system),
                    "prompts/s2/sys_defense.txt",
                )
                mlflow.log_text(
                    get_s2_p("lit_sys", prompt_builder.build_s2_literalist_system),
                    "prompts/s2/sys_literalist.txt",
                )
                mlflow.log_text(
                    get_s2_p("prof_sys", prompt_builder.build_s2_profiler_system),
                    "prompts/s2/sys_profiler.txt",
                )
                mlflow.log_text(
                    get_s2_p("judge_sys", prompt_builder.build_s2_judge_system),
                    "prompts/s2/sys_judge.txt",
                )

                # Log User Templates
                mlflow.log_text(
                    get_s2_p(
                        "pros_user", prompt_builder.build_s2_prosecutor_user_template
                    ),
                    "prompts/s2/user_prosecutor.txt",
                )
                mlflow.log_text(
                    get_s2_p("def_user", prompt_builder.build_s2_defense_user_template),
                    "prompts/s2/user_defense.txt",
                )
                mlflow.log_text(
                    get_s2_p("judge_user", prompt_builder.build_s2_judge_user_template),
                    "prompts/s2/user_judge.txt",
                )

        # 5. Execution Loop

        # Eval Metrics
        correct_s2 = 0
        total_eval_s2 = 0

        # Semaphore for Global Doc Concurrency
        sem = asyncio.Semaphore(1)

        async def worker(row):
            async with sem:
                with logger.contextualize(doc_id=row["id"]):
                    return await process_document(row, args, rag_collections)

        tasks = [worker(row) for row in rows]

        # Metrics
        correct_s2 = 0
        total_eval_s2 = 0

        with tqdm(total=len(rows), desc="Running Pipeline") as pbar:
            for future in asyncio.as_completed(tasks):
                s1_res, s2_res = await future

                # [Optimization] Incremental Save
                if s1_res:
                    append_jsonl("submission_s1.jsonl", s1_res)
                if s2_res:
                    append_jsonl("submission_s2.jsonl", s2_res)

                # Live Accuracy Update
                if s2_res and "conspiracy" in s2_res:
                    row_id = s2_res["_id"]
                    orig = next((r for r in rows if r["id"] == row_id), None)
                    if orig and orig["metadata"].get("label"):
                        gt = str(orig["metadata"]["label"]).lower()
                        gt_norm = "yes" if gt in ["yes", "conspiracy"] else "no"
                        if s2_res["conspiracy"].lower() == gt_norm:
                            correct_s2 += 1
                        total_eval_s2 += 1

                # Flush Logs
                doc_id = (
                    s1_res["_id"]
                    if s1_res
                    else (s2_res["_id"] if s2_res else "unknown")
                )
                if doc_id in LOG_BUFFERS:
                    tqdm.write(f"\n--- {doc_id} ---")
                    for msg in LOG_BUFFERS[doc_id]:
                        tqdm.write(msg.record["message"])
                        pass  # Squelch for clean progress bar, or uncomment to see
                    del LOG_BUFFERS[doc_id]

                acc_str = (
                    f"Acc: {correct_s2/total_eval_s2:.1%}"
                    if total_eval_s2 > 0
                    else "Acc: N/A"
                )
                cost_str = f"${GLOBAL_METER.total_cost():.2f}"
                pbar.set_postfix_str(f"{acc_str} | {cost_str}")
                pbar.update(1)

        # Finalize
        logger.info("=" * 40)
        logger.info(f" FINAL COST: {GLOBAL_METER}")
        logger.info("=" * 40)

        # 6. Save & Finish
        if total_eval_s2 > 0:
            final_acc = correct_s2 / total_eval_s2
            mlflow.log_metric("s2_accuracy", final_acc)
            logger.success(f"Final Accuracy: {final_acc:.2%}")

        # Compress & Log Artifacts
        if os.path.exists("submission_s1.jsonl"):
            finalize_zip("submission_s1.jsonl", "submission_s1.zip")
            mlflow.log_artifact("submission_s1.zip")

        if os.path.exists("submission_s2.jsonl"):
            finalize_zip("submission_s2.jsonl", "submission_s2.zip")
            mlflow.log_artifact("submission_s2.zip")

        if os.path.exists(log_file):
            mlflow.log_artifact(log_file)


if __name__ == "__main__":
    asyncio.run(main_async())
