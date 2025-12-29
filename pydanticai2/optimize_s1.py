#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimize_s1.py — GEPA Optimization Engine.

Fixes:
- Uses mlflow.log_text() for prompts (preventing 'Invalid value for metric' errors).
- Implements Rich Feedback Scorer for GEPA.
- Harmonized with prompt_runner.py logging standards.
"""

import os
import sys
import json
import asyncio
import argparse
import pathlib
from typing import List, Dict, Any

# Third-party
import mlflow
import mlflow.genai
from mlflow.genai import scorer
from mlflow.genai.optimize import GepaPromptOptimizer
from mlflow.entities import Feedback
from loguru import logger

# --- Make repo root importable ---
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


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

# Project Modules
from pydanticai2.psycomark_agents import (
    run_s1_discriminative,
    get_rag_collection,
    retrieve_stratified_s1,
)
from pydanticai2.prompt_builder import (
    build_s1_discriminative_system,
    build_s1_critic_system,
    build_s1_refiner_system,
    build_s1_user_template,
    build_s1_critic_user_template,
    build_s1_refiner_user_template,
)

# -----------------------------------------------------------------------------
# 1. Setup & Data Loading
# -----------------------------------------------------------------------------


def load_eval_data(path: str, limit: int = 20) -> List[Dict]:
    """Loads and slices the Gold Standard dataset."""
    dataset = []
    p = pathlib.Path(path)
    if not p.exists():
        logger.error(f"Dataset not found: {path}")
        sys.exit(1)

    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                # Map to MLflow GenAI format
                dataset.append(
                    {
                        "inputs": {"text": row["text"]},
                        "outputs": {"gold_spans": row["spans"]},
                    }
                )
            except Exception as e:
                logger.warning(f"Skipping bad line in eval data: {e}")

    logger.info(f"Loaded {len(dataset)} examples. Slicing to top {limit}...")
    return dataset[:limit]


# -----------------------------------------------------------------------------
# 2. Rich Feedback Scorer (CRITICAL for GEPA)
# -----------------------------------------------------------------------------


@scorer
def s1_rich_scorer(outputs, expectations):
    """
    Calculates F1 and generates Text Rationale for GEPA.
    Returns a Feedback object with a numeric score AND text rationale.
    """
    pred_spans = outputs.get("final_spans", [])
    gold_spans = expectations.get("gold_spans", [])

    pred_texts = {s["text"].lower().strip() for s in pred_spans}
    gold_texts = {s["text"].lower().strip() for s in gold_spans}

    # [FIX] Fail immediately on empty prediction (Crash detection)
    if not pred_spans and gold_spans:
        logger.debug("Empty prediction (Model Crash or Silence)")
        return Feedback(value=0.0, rationale="FAILED: Model returned no spans.")

    tp = 0
    fn_items = []
    fp_items = []

    # Fuzzy Intersection (Substring match)
    for g in gold_texts:
        if any(g in p or p in g for p in pred_texts):
            tp += 1
        else:
            fn_items.append(g)

    for p in pred_texts:
        if not any(p in g or g in p for g in gold_texts):
            fp_items.append(p)

    precision = tp / len(pred_texts) if pred_texts else 1.0
    recall = tp / len(gold_texts) if gold_texts else 1.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )  # Ensure float

    # Generate Feedback Rationale
    rationale_parts = []
    if fn_items:
        rationale_parts.append(f"MISSED (Recall): {', '.join(fn_items[:3])}")
    if fp_items:
        rationale_parts.append(f"HALLUCINATED (Precision): {', '.join(fp_items[:3])}")
    if not rationale_parts:
        rationale_parts.append("Perfect extraction.")

    # Logging detail to file (not metric)
    logger.debug(f"F1: {f1:.2f} | {rationale_parts[0]}")

    return Feedback(value=float(f1), rationale=" | ".join(rationale_parts))


# -----------------------------------------------------------------------------
# 3. Prediction Wrapper
# -----------------------------------------------------------------------------


GEN_SYS_URI, GEN_USER_URI = None, None
CRITIC_SYS_URI, CRITIC_USER_URI = None, None
REFINER_SYS_URI, REFINER_USER_URI = None, None
S1_RAG_COLLECTION = None  # <--- [NEW] Global RAG Collection


def predict_wrapper(text: str):
    # Load ALL 6 candidates
    # System
    g_sys = mlflow.genai.load_prompt(GEN_SYS_URI).format() if GEN_SYS_URI else None
    c_sys = (
        mlflow.genai.load_prompt(CRITIC_SYS_URI).format() if CRITIC_SYS_URI else None
    )
    r_sys = (
        mlflow.genai.load_prompt(REFINER_SYS_URI).format() if REFINER_SYS_URI else None
    )

    # User Templates (Raw string, no format)
    g_usr = mlflow.genai.load_prompt(GEN_USER_URI).template if GEN_USER_URI else None
    c_usr = (
        mlflow.genai.load_prompt(CRITIC_USER_URI).template if CRITIC_USER_URI else None
    )
    r_usr = (
        mlflow.genai.load_prompt(REFINER_USER_URI).template
        if REFINER_USER_URI
        else None
    )

    try:
        spans = asyncio.run(
            run_s1_discriminative(
                text,
                gen_prompt_override=g_sys,
                user_prompt_template_override=g_usr,
                critic_prompt_override=c_sys,
                critic_user_template_override=c_usr,
                refiner_prompt_override=r_sys,
                refiner_user_template_override=r_usr,
            )
        )
        return {"final_spans": [s.model_dump() for s in spans]}
    except:
        return {"final_spans": []}


# -----------------------------------------------------------------------------
# 4. Main Execution
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="GEPA Prompt Optimization Runner")
    parser.add_argument(
        "--data", default="data/gold/optimization_set.jsonl", help="Path to eval data"
    )
    parser.add_argument(
        "--rag-dir", default="data/rag_online_v3", help="Path to ChromaDB RAG"
    )  # <--- [NEW]
    parser.add_argument("--limit", type=int, default=20, help="Examples to use")
    parser.add_argument("--budget", type=int, default=60, help="Max metric calls")
    parser.add_argument(
        "--experiment", default="GEPA_S1_Optimization", help="MLflow Experiment Name"
    )
    # Using Sonnet 4.5 via Bedrock
    parser.add_argument(
        "--model-reflector",
        default="bedrock:/eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
        help="Model used for reflection",
    )
    args = parser.parse_args()

    # 1. Init Global RAG
    global S1_RAG_COLLECTION
    if args.rag_dir and os.path.exists(args.rag_dir):
        logger.info(f"Initializing RAG from {args.rag_dir}...")
        try:
            S1_RAG_COLLECTION = get_rag_collection(args.rag_dir, "s1_markers")
            logger.success("RAG Collection Loaded successfully.")
        except Exception as e:
            logger.warning(
                f"RAG Load Failed: {e}. Optimization will proceed without context."
            )
    else:
        logger.warning(
            "No RAG directory provided or found. Optimization will be Zero-Shot."
        )

    # 1. Logger Setup (File + Console)
    log_file = "optimization.log"
    if os.path.exists(log_file):
        os.remove(log_file)
    logger.remove()
    logger.add(sys.stderr, level="INFO")  # Clean console
    logger.add(log_file, level="DEBUG")  # Detailed file log

    # 2. Load Data
    eval_dataset = load_eval_data(args.data, args.limit)

    # 3. MLflow Setup
    mlflow.set_experiment(args.experiment)
    logger.info(f"Starting Run: {args.experiment}")

    global GEN_SYS_URI, GEN_USER_URI
    global CRITIC_SYS_URI, CRITIC_USER_URI
    global REFINER_SYS_URI, REFINER_USER_URI

    with mlflow.start_run():
        mlflow.log_params(vars(args))

        # A. Safely Log Baseline Prompts as ARTIFACTS (Text), NOT Metrics
        logger.info("Snapshotting Baseline Prompts...")

        gen_template = build_s1_discriminative_system()
        critic_template = build_s1_critic_system()
        refiner_template = build_s1_refiner_system()

        mlflow.log_text(gen_template, "baseline_prompts/s1_generator.txt")
        mlflow.log_text(critic_template, "baseline_prompts/s1_critic.txt")
        mlflow.log_text(refiner_template, "baseline_prompts/s1_refiner.txt")

        # 1. Register Baselines
        logger.info("Registering Prompts...")
        # Generator
        gen_tmpl = build_s1_discriminative_system()
        mlflow.log_text(gen_tmpl, "baseline_prompts/s1_generator.txt")
        gen_info = mlflow.genai.register_prompt(name="s1_gen_sys", template=gen_tmpl)
        GEN_SYS_URI = gen_info.uri

        gen_user_tmpl = build_s1_user_template()
        mlflow.log_text(gen_user_tmpl, "baseline_prompts/s1_gen_user.txt")
        gen_user_info = mlflow.genai.register_prompt(
            name="s1_gen_user", template=gen_user_tmpl
        )
        GEN_USER_URI = gen_user_info.uri

        # Critic
        critic_tmpl = build_s1_critic_system()
        mlflow.log_text(critic_tmpl, "baseline_prompts/s1_critic.txt")
        critic_info = mlflow.genai.register_prompt(
            name="s1_critic", template=critic_tmpl
        )
        CRITIC_SYS_URI = critic_info.uri

        critic_user_tmpl = build_s1_critic_user_template()
        mlflow.log_text(critic_user_tmpl, "baseline_prompts/s1_critic_user.txt")
        critic_user_info = mlflow.genai.register_prompt(
            name="s1_critic_user", template=critic_user_tmpl
        )
        CRITIC_USER_URI = critic_user_info.uri

        # Refiner [NEW]
        refiner_tmpl = build_s1_refiner_system()
        mlflow.log_text(refiner_tmpl, "baseline_prompts/s1_refiner.txt")
        refiner_info = mlflow.genai.register_prompt(
            name="s1_refiner", template=refiner_tmpl
        )
        REFINER_SYS_URI = refiner_info.uri

        refiner_user_tmpl = build_s1_refiner_user_template()
        mlflow.log_text(refiner_user_tmpl, "baseline_prompts/s1_refiner_user.txt")
        refiner_user_info = mlflow.genai.register_prompt(
            name="s1_refiner_user", template=refiner_user_tmpl
        )
        REFINER_USER_URI = refiner_user_info.uri
        logger.success("Baseline Prompts Registered.")

        # C. Configure Optimizer
        optimizer_config = GepaPromptOptimizer(
            reflection_model=args.model_reflector,
            max_metric_calls=args.budget,
            display_progress_bar=True,
        )

        # D. Run Optimization
        logger.info("🚀 Launching GEPA...")
        results = mlflow.genai.optimize_prompts(
            predict_fn=predict_wrapper,
            train_data=eval_dataset,
            prompt_uris=[
                GEN_SYS_URI,
                CRITIC_SYS_URI,
                REFINER_SYS_URI,
                GEN_USER_URI,
                CRITIC_USER_URI,
                REFINER_USER_URI,
            ],
            optimizer=optimizer_config,
            scorers=[s1_rich_scorer],
        )

        logger.success("Optimization Complete!")

    # Inspect results for best prompts
    print(results)

    # Check 1: Dictionary map (Multi-prompt optimization standard)
    output_dir = pathlib.Path("prompts/optimized")
    output_dir.mkdir(parents=True, exist_ok=True)
    # 1. Extract the list of optimized PromptVersion objects
    optimized_list = []
    if hasattr(results, "optimized_prompts"):
        optimized_list = results.optimized_prompts
    elif hasattr(results, "best_prompts"):
        # If it's a dict, get values; if list, use as is
        optimized_list = (
            list(results.best_prompts.values())
            if isinstance(results.best_prompts, dict)
            else results.best_prompts
        )

    if not optimized_list:
        logger.error("Results object contained no optimized prompts.")
        return
    # 2. Map Prompt Names to Target Filenames
    # We use the NAME (e.g., "s1_gen_sys") because version numbers change
    name_to_filename = {
        "s1_gen_sys": "s1_generator_optimized.txt",
        "s1_gen_user": "s1_user_optimized.txt",
        "s1_critic_sys": "s1_critic_optimized.txt",  # Note: Ensure this matches your registration name
        "s1_critic_user": "s1_critic_user_optimized.txt",
        "s1_refiner_sys": "s1_refiner_optimized.txt",
        "s1_refiner_user": "s1_refiner_user_optimized.txt",
        # Fallbacks for alternative naming conventions you might have used
        "s1_generator": "s1_generator_optimized.txt",
        "s1_critic": "s1_critic_optimized.txt",
        "s1_refiner": "s1_refiner_optimized.txt",
    }
    found_count = 0
    for prompt_obj in optimized_list:
        # Check if this prompt's name is in our mapping
        if prompt_obj.name in name_to_filename:
            fname = name_to_filename[prompt_obj.name]
            file_path = output_dir / fname

            # Write to file
            file_path.write_text(prompt_obj.template, encoding="utf-8")

            # Log artifact
            mlflow.log_artifact(str(file_path))
            logger.success(f"Saved optimized prompt '{prompt_obj.name}' to {fname}")
            found_count += 1
        else:
            logger.warning(
                f"Optimized prompt '{prompt_obj.name}' found but no filename mapping defined."
            )
    if found_count == 0:
        logger.warning(
            "No optimized prompts matched the expected filenames. Check your registration names."
        )
        # Debug: print what we actually got
        for p in optimized_list:
            logger.info(f"Available Result: Name='{p.name}', Version='{p.version}'")


if __name__ == "__main__":
    main()
