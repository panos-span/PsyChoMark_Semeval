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

# Import the new S1 Graph (The "Consensus Engine")
from pydanticai2.s1_graph import s1_graph
from pydanticai2.psycomark_graph import s2_graph

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

            # 3. Extract Results
            verified_spans = final_state.get("final_spans", [])

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
            s1_dossier = agents_mod.synthesize_dossier(markers)

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
                precedents = agents_mod.retrieve_fewshots(
                    s2_rag, text, k=4, filters={"is_hard_negative": True}
                )
                if precedents:
                    rag_context = json.dumps(precedents, indent=2, ensure_ascii=False)

            # 2. Prepare S2 State
            s2_initial_state = {
                "doc_id": doc_id,
                "text": text,
                "s1_spans": s1_spans_for_s2,
                "marker_summary": s1_dossier or "No markers extracted.",
                "rag_context": rag_context,
                "juror_temperature": args.s2_temp,  # <--- Pass CLI arg here
                "metadata": meta,
                "council_result": None,
                "final_output": None,
            }

            # 3. Invoke S2 Graph (Council -> Judge)
            final_state = await s2_graph.ainvoke(s2_initial_state)
            s2_result = final_state.get("final_output")

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

    # Config (S1)
    parser.add_argument(
        "--s1-k", type=int, default=3, help="Ensemble size (internal to graph)"
    )
    parser.add_argument(
        "--s2-temp", type=float, default=0.4, help="Temperature for S2 Jurors"
    )

    # Config (S2)
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

    with mlflow.start_run():
        mlflow.log_params(vars(args))

        # Log active templates
        if args.task in ("s1", "both"):
            # Import private builder to snapshot the exact system prompt
            from prompt_builder import build_s1_discriminative_system

            mlflow.log_text(
                build_s1_discriminative_system(),
                "prompts/s1/system_discriminative.txt",
            )
        # --- 2. Log S2 Prompts (Judge & The Full Council) ---
        if args.task in ("s2", "both"):
            # A. The Judge (ReX Logic)
            mlflow.log_text(
                prompt_builder.build_s2_judge_system(), "prompts/s2/system_judge.txt"
            )

            # B. Juror 1: The Literalist (Grammar/Attribution)
            mlflow.log_text(
                prompt_builder.build_s2_triage_system(),
                "prompts/s2/juror_literalist.txt",
            )

            # C. Juror 2: The Believer (Prosecutor/High Recall)
            mlflow.log_text(
                prompt_builder.build_s2_system(include_cot=False),
                "prompts/s2/juror_believer.txt",
            )

            # D. Juror 3: The Profiler (Tone/Vibe)
            mlflow.log_text(
                prompt_builder.build_s2_profiler_system(),
                "prompts/s2/juror_profiler.txt",
            )

            # E. Juror 4: The Defense (Hanlon's Razor)
            mlflow.log_text(
                prompt_builder.build_s2_defense_system(), "prompts/s2/juror_defense.txt"
            )

            # F. User Prompts (Templates)
            # Log the templates used for the Council and Judge interactions
            council_template = """
<case_file>
  <evidence_text>{text}</evidence_text>
  <forensic_markers>{marker_summary}</forensic_markers>
  <instruction>Review the evidence above according to your System Role. Render your Verdict.</instruction>
</case_file>
            """
            mlflow.log_text(council_template, "prompts/s2/user_template_council.txt")

        # 5. Execution Loop
        s1_results = []
        s2_results = []

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

        with tqdm(total=len(rows), desc="Running Pipeline") as pbar:
            for future in asyncio.as_completed(tasks):
                s1_res, s2_res = await future

                if s1_res:
                    s1_results.append(s1_res)
                if s2_res:
                    s2_results.append(s2_res)
                    # Live Accuracy
                    row_id = s2_res["_id"]
                    orig_row = next((r for r in rows if r["id"] == row_id), None)
                    if orig_row and orig_row["metadata"]["label"]:
                        gt = str(orig_row["metadata"]["label"]).lower()
                        gt_norm = "yes" if gt in ["yes", "conspiracy"] else "no"
                        pred = s2_res["conspiracy"].lower()
                        if pred == gt_norm:
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
                pbar.set_postfix_str(acc_str)
                pbar.update(1)

        # 6. Save & Finish
        if s1_results:
            save_jsonl("submission.jsonl", "submission_s1.zip", s1_results)
            mlflow.log_artifact("submission_s1.zip")

        if s2_results:
            save_jsonl("submission.jsonl", "submission_s2.zip", s2_results)
            mlflow.log_artifact("submission_s2.zip")
            if total_eval_s2 > 0:
                final_acc = correct_s2 / total_eval_s2
                mlflow.log_metric("s2_accuracy", final_acc)
                logger.success(f"Final S2 Accuracy: {final_acc:.2%}")

        # [NEW] Upload the full run log as an artifact
        if os.path.exists(log_file):
            mlflow.log_artifact(log_file)


if __name__ == "__main__":
    asyncio.run(main_async())
