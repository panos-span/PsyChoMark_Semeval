#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimize_s2.py — S2 Council Optimization Engine.
"""

import os
import sys
import json
import asyncio
import argparse
import pathlib
from typing import List, Dict
import pandas as pd  # <--- NEW
from datetime import datetime  # <--- NEW

# Global Feedback Collector for S2
S2_FEEDBACK_LOG = []  # <--- NEW

import litellm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from litellm.exceptions import RateLimitError, ServiceUnavailableError

# =============================================================================
# [CRITICAL FIX] Monkey-Patch litellm to force retries for GEPA
# =============================================================================

# 1. Save the original function
_original_litellm_completion = litellm.completion


# 2. Define the retry logic
@retry(
    retry=retry_if_exception_type((RateLimitError, ServiceUnavailableError)),
    stop=stop_after_attempt(20),  # Try 20 times
    wait=wait_exponential(multiplier=2, min=5, max=60),  # Wait 5s, 10s... up to 60s
    reraise=True,
)
def _robust_completion(*args, **kwargs):
    """Wrapper that forces retries on RateLimitErrors"""
    return _original_litellm_completion(*args, **kwargs)


# 3. Apply the patch
# Now, when 'gepa' calls litellm.completion, it hits our retry loop first.
litellm.completion = _robust_completion

# Optional: Set global timeouts just in case
os.environ["LITELLM_REQUEST_TIMEOUT"] = "120"
# Third-party

import mlflow
import mlflow.genai
from mlflow.genai import scorer
from mlflow.genai.optimize import GepaPromptOptimizer
from mlflow.entities import Feedback
from loguru import logger

# Import prompt loader
from pydanticai2.prompt_loader import S2_PROMPTS


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
    # Legacy S2 (Sequential Debate)
    run_s2_sequential_debate,
    run_s2_judge_review,
    # Anti-Echo Chamber S2 (Parallel Voting) - NEW
    run_s2_parallel_council,
    run_s2_calibrated_judge,
    # Utilities
    get_rag_collection,
    retrieve_hard_negatives_reranked,
    format_s2_rag_to_xml,
)
from pydanticai2.prompt_builder import (
    # Legacy prompts
    build_s2_prosecutor_system,
    build_s2_defense_system,
    build_s2_literalist_system,
    build_s2_profiler_system,
    build_s2_prosecutor_user_template,
    build_s2_defense_user_template,
    build_s2_literalist_user_template,
    build_s2_profiler_user_template,
    build_s2_judge_system,
    build_s2_judge_user_template,
    # Parallel prompts (Anti-Echo Chamber) - NEW
    build_s2_parallel_prosecutor_system,
    build_s2_parallel_defense_system,
    build_s2_parallel_literalist_system,
    build_s2_parallel_profiler_system,
    build_s2_parallel_user_template,
    build_s2_calibrated_judge_system,
    build_s2_calibrated_judge_user_template,
)


# -----------------------------------------------------------------------------
# 1. Setup
# -----------------------------------------------------------------------------

# Legacy URIs
P_SYS, D_SYS, L_SYS, PR_SYS = None, None, None, None
P_USER, D_USER, L_USER, PR_USER = None, None, None, None
JUDGE_SYS, JUDGE_USER = None, None

# Parallel URIs (Anti-Echo Chamber) - NEW
PARALLEL_P_SYS, PARALLEL_D_SYS, PARALLEL_L_SYS, PARALLEL_PR_SYS = None, None, None, None
PARALLEL_USER = None  # Shared user template for all parallel jurors
CALIBRATED_JUDGE_SYS, CALIBRATED_JUDGE_USER = None, None

# Mode flag
USE_PARALLEL_MODE = True

# RAG Collection (Global)
S2_RAG_COLLECTION = None


def load_classification_data(path: str, limit: int = 20) -> List[Dict]:
    """
    Loads dataset with [CRITICAL] Truncation to avoid ContextWindowExceededError.
    """
    dataset = []
    p = pathlib.Path(path)
    if not p.exists():
        logger.error(f"Dataset not found: {path}")
        return []

    # [SAFETY] Truncation thresholds
    MAX_TEXT_CHARS = 4000  # ~1000 tokens
    MAX_MARKER_CHARS = 1000  # ~250 tokens

    with open(p, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                raw_label = row.get("label")
                if not raw_label:
                    raw_label = row.get("metadata", {}).get("label", "non")

                # Normalize Label
                label_str = str(raw_label).lower().strip()
                if label_str in ["yes", "true", "1", "conspiracy"]:
                    label = "conspiracy"
                elif label_str in ["no", "false", "0", "non"]:
                    label = "non"
                else:
                    # Skip 'cant_tell' or 'ambiguous' for binary optimization
                    # unless you want to specifically train the 'ambiguous' flag
                    continue

                # [SAFETY] Truncate Main Text
                text = row.get("text", "")
                if len(text) > MAX_TEXT_CHARS:
                    text = text[:MAX_TEXT_CHARS] + "...[TRUNCATED]"

                doc_id = row.get("doc_id", f"line_{line_num}")

                s1_spans = row.get("s1_spans", [])
                if not s1_spans and "spans" in row:
                    s1_spans = row["spans"]

                if s1_spans:
                    marker_summary = "\n".join(
                        [f"- [{s['label'].upper()}]: {s['text']}" for s in s1_spans]
                    )
                else:
                    marker_summary = "No specific forensic markers identified."

                # [SAFETY] Truncate Marker Summary
                if len(marker_summary) > MAX_MARKER_CHARS:
                    marker_summary = (
                        marker_summary[:MAX_MARKER_CHARS] + "...[TRUNCATED]"
                    )

                payload = {
                    "gold_label": label,
                    "doc_id": doc_id,
                    "s2_subtype": row.get("s2_subtype", "unknown"),
                    "is_hard_negative": row.get("is_hard_negative", False),
                }
                payload_str = json.dumps(payload)

                dataset.append(
                    {
                        "inputs": {
                            "text": text,
                            "s1_spans": s1_spans,
                            "marker_summary": marker_summary,
                            "passthrough_gold": payload_str,
                        },
                        "outputs": {"dummy_target": "ignore_me"},
                    }
                )
            except Exception as e:
                logger.warning(f"Skipping bad line {line_num}: {e}")

    dataset = dataset[:limit]

    conspiracy_count = sum(
        1
        for d in dataset
        if json.loads(d["inputs"]["passthrough_gold"])["gold_label"] == "conspiracy"
    )
    logger.success(f"DATASET LOADED: {len(dataset)} items (Truncation Active).")
    logger.info(
        f" > Conspiracy: {conspiracy_count} | Non: {len(dataset) - conspiracy_count}"
    )

    return dataset


# -----------------------------------------------------------------------------
# 2. Classification Scorer (Reads from OUTPUTS via Trojan Horse)
# -----------------------------------------------------------------------------


def generate_s2_actionable_feedback(
    gold_label: str,
    pred_label: str,
    council_tally: dict,
    council_votes: list,
    confidence: float,
    doc_id: str,
    s2_subtype: str,
    is_hard_negative: bool,
) -> str:
    """
    Generate prioritized, concise feedback.
    PREVENTS OVERFLOW by capping length and focusing on the biggest error.
    """
    # Normalize labels
    pred_label = str(pred_label).lower().strip()
    gold_label = str(gold_label).lower().strip()

    consp_votes = council_tally.get("conspiracy", 0)
    non_votes = council_tally.get("non", 0) + council_tally.get("non-conspiracy", 0)
    council_consensus = "conspiracy" if consp_votes > non_votes else "non"

    # =========================================================
    # 1. POSITIVE ANCHORING (Top Priority: Lock in wins)
    # =========================================================
    if pred_label == gold_label:
        if is_hard_negative:
            return f"KEEP: Correctly identified Hard Negative ({s2_subtype})."
        if council_consensus != gold_label:
            return f"GREAT SAVE: Judge correctly overruled wrong Council ({council_consensus})."
        return f"PERFECT: Correct Verdict ({gold_label})."

    # =========================================================
    # 2. NEGATIVE FEEDBACK (Hierarchy of Severity)
    # =========================================================

    # PRIORITY 1: Calibration (Major Logic Fail)
    # If the model is wrong but highly confident, this breaks the entire system.
    if confidence > 0.85:
        return f"CRITICAL: OVERCONFIDENT ({confidence:.2f}). You are wrong but certain. Reduce certainty on ambiguous texts."

    # PRIORITY 2: Hard Negative Traps (Specific Domain Fail)
    # These are the "trick questions" the model must learn to recognize.
    if is_hard_negative:
        trap_hint = (
            "Reporting vs Endorsing"
            if "report" in s2_subtype
            else "Debunking vs Conspiring"
        )
        return f"FAILURE: Fell for Hard Negative Trap ({s2_subtype}). Remember: {trap_hint}."

    # PRIORITY 3: Dissent Ignored (The "Hidden Gem" signal)
    # If a Juror got it right but the Judge ignored them, point that out specifically.
    if council_consensus != gold_label:
        # Find the first juror who got it right
        hero_vote = next(
            (v for v in council_votes if str(v.get("verdict")).lower() == gold_label),
            None,
        )
        if hero_vote:
            juror_name = hero_vote.get("juror", "?")
            # TRUNCATE rationale to 100 chars to save tokens
            reason = hero_vote.get("rationale", "")[:100]
            return f"MISSED SIGNAL: The {juror_name} was right! Reason: '{reason}...'"

    # PRIORITY 4: Judge Failure (Judge broke the consensus)
    if council_consensus == gold_label and pred_label != gold_label:
        return f"JUDGE ERROR: The Council was right ({council_consensus}), but Judge overruled incorrectly."

    # PRIORITY 5: Generic Failure (Last Resort)
    return f"WRONG: Pred {pred_label} != Gold {gold_label}. Votes: {consp_votes} vs {non_votes}. Re-evaluate evidence."


@scorer
def s2_rich_scorer(outputs, expectations):
    """
    Gradient Scorer: Rewards partial consensus to guide the optimizer up the hill.
    """
    # 1. Unpack Data (Trojan Horse)
    gold_label = "non"
    doc_id = "unknown"
    s2_subtype = "unknown"
    is_hard_negative = False

    try:
        raw_payload = outputs.get("passthrough_gold_ref")
        if raw_payload:
            data = json.loads(raw_payload)
            gold_label = str(data.get("gold_label", "non")).lower().strip()
            doc_id = data.get("doc_id", "unknown")
            s2_subtype = data.get("s2_subtype", "unknown")
            is_hard_negative = data.get("is_hard_negative", False)
    except Exception as e:
        logger.error(f"Scorer Unpack Failed: {e}")

    # 2. Extract Predictions
    pred_label = str(outputs.get("final_label", "non")).lower().strip()
    pred_conf = outputs.get(
        "final_confidence", 0.0
    )  # Need to ensure wrapper returns this
    council_tally = outputs.get("council_tally", {})

    # Calculate Vote Ratios for Gradient Scoring
    total_votes = sum(council_tally.values()) if council_tally else 1
    consp_votes = council_tally.get("conspiracy", 0)
    non_votes = council_tally.get("non", 0) + council_tally.get("non-conspiracy", 0)

    consp_ratio = consp_votes / total_votes
    non_ratio = non_votes / total_votes

    # 3. CALCULATE GRADIENT SCORE
    # We reward the model for getting CLOSER to the truth (more votes), even if it fails.
    score = 0.0

    if gold_label == "conspiracy":
        # Target: Maximize Conspiracy Votes
        score = consp_ratio
        # Bonus: If Judge correctly ruled Conspiracy despite split, boost to 1.0
        if pred_label == "conspiracy":
            score = 1.0
    else:  # gold_label == "non"
        # Target: Maximize Non Votes
        score = non_ratio
        if pred_label == "non":
            score = 1.0

    # 4. Generate Feedback (The "Why")
    rationale = generate_s2_actionable_feedback(
        gold_label,
        pred_label,
        council_tally,
        outputs.get("council_votes", []),
        pred_conf,
        doc_id,
        s2_subtype,
        is_hard_negative,
    )

    council_consensus = "conspiracy" if consp_votes > non_votes else "non"

    # ---------------------------------------------------------
    # [NEW] Collect Feedback for Analysis
    # ---------------------------------------------------------
    S2_FEEDBACK_LOG.append(
        {
            "timestamp": datetime.now().isoformat(),
            "doc_id": doc_id,
            "score": float(score),
            "gold_label": gold_label,
            "pred_label": pred_label,
            "consensus": council_consensus,
            "tally": str(council_tally),  # Log the raw votes too
            "rationale": rationale,
            "subtype": s2_subtype,  # Helpful for hard negative analysis
        }
    )

    # 5. Log & Return
    logger.info(
        f"[{doc_id}] Grade: {score:.2f} (Gold: {gold_label} | Votes: {consp_votes}-{non_votes})"
    )

    return Feedback(value=float(score), rationale=rationale)


# -----------------------------------------------------------------------------
# 3. Prediction Wrapper (The Tunnel)
# -----------------------------------------------------------------------------

# Helper to log prompts only once
_LOGGED_ONCE = False


def predict_wrapper(
    text: str,
    s1_spans: List[dict],
    marker_summary: str,
    passthrough_gold: str = "{}",
    # --- NEW: GEPA INJECTION POINTS ---
    # These optional args catch the mutated strings from the optimizer
    prosecutor_sys_opt: str = None,
    defense_sys_opt: str = None,
    literalist_sys_opt: str = None,
    profiler_sys_opt: str = None,
    judge_sys_opt: str = None,
    judge_user_opt: str = None,
):
    """
    Prediction wrapper that accepts GEPA-mutated prompts.
    Prioritizes passed arguments (Optimized) over Global URIs (Static).
    """
    global _LOGGED_ONCE

    # 1. RAG Context Logic
    MAX_RAG_CHARS = 1500
    rag_context = ""
    if S2_RAG_COLLECTION:
        try:
            precedents = retrieve_hard_negatives_reranked(S2_RAG_COLLECTION, text, k=2)
            if precedents:
                rag_raw = json.dumps(precedents, indent=2, ensure_ascii=False)
                if len(rag_raw) > MAX_RAG_CHARS:
                    rag_raw = rag_raw[:MAX_RAG_CHARS] + "\n...[TRUNCATED]"
                rag_context = rag_raw
        except Exception as e:
            logger.warning(f"RAG Failed: {e}")

    md_rag = f"## Legal Precedents (RAG Context)\n{rag_context}"
    rag_str = md_rag

    # 2. Helper to Resolve Prompts (Opt String > Static URI)
    def resolve_prompt(opt_string, static_uri):
        if opt_string:
            # GEPA passed a mutated string. Inject RAG manually.
            return opt_string.replace("{{rag_context}}", rag_str)
        elif static_uri:
            # Fallback to static prompt from MLflow
            try:
                tmpl = mlflow.genai.load_prompt(static_uri).template
                return tmpl.replace("{{rag_context}}", rag_str)
            except:
                return None
        return None

    # 3. Resolve all prompts
    # Note: We map the generic 'prosecutor_sys_opt' to either Legacy or Parallel globals based on mode

    if USE_PARALLEL_MODE:
        p_s = resolve_prompt(prosecutor_sys_opt, PARALLEL_P_SYS)
        d_s = resolve_prompt(defense_sys_opt, PARALLEL_D_SYS)
        l_s = resolve_prompt(literalist_sys_opt, PARALLEL_L_SYS)
        pr_s = resolve_prompt(profiler_sys_opt, PARALLEL_PR_SYS)
        # For user template, we assume it's static unless optimized
        par_u = resolve_prompt(None, PARALLEL_USER)

        j_s = resolve_prompt(judge_sys_opt, CALIBRATED_JUDGE_SYS)
        j_u = resolve_prompt(judge_user_opt, CALIBRATED_JUDGE_USER)
    else:
        # Legacy Mode
        p_s = resolve_prompt(prosecutor_sys_opt, P_SYS)
        d_s = resolve_prompt(defense_sys_opt, D_SYS)
        l_s = resolve_prompt(literalist_sys_opt, L_SYS)
        pr_s = resolve_prompt(profiler_sys_opt, PR_SYS)
        # Load legacy user templates (static)
        p_u = resolve_prompt(None, P_USER)
        d_u = resolve_prompt(None, D_USER)
        l_u = resolve_prompt(None, L_USER)
        pr_u = resolve_prompt(None, PR_USER)

        j_s = resolve_prompt(judge_sys_opt, JUDGE_SYS)
        j_u = resolve_prompt(judge_user_opt, JUDGE_USER)

    # 4. Parse Doc ID
    doc_id = "unknown"
    try:
        doc_id = json.loads(passthrough_gold).get("doc_id", "unknown")
    except:
        pass

    try:
        # =====================================================================
        # PARALLEL MODE
        # =====================================================================
        if USE_PARALLEL_MODE:
            logger.debug(f"[{doc_id}] Running Parallel Council (Optimized)...")

            council_res = asyncio.run(
                run_s2_parallel_council(
                    text=text,
                    s1_spans=s1_spans,
                    marker_summary=marker_summary,
                    rag_context="",  # Injected above
                    prosecutor_sys_override=p_s,
                    defense_sys_override=d_s,
                    literalist_sys_override=l_s,
                    profiler_sys_override=pr_s,
                    parallel_user_template_override=par_u,
                )
            )

            judge_res = asyncio.run(
                run_s2_calibrated_judge(
                    text=text,
                    council_result=council_res,
                    rag_context=rag_context,
                    judge_sys_override=j_s,
                    judge_user_template_override=j_u,
                )
            )

            # Formatting output (same as before)
            council_votes = []
            for vote in council_res.votes:
                v = vote.model_dump() if hasattr(vote, "model_dump") else vote
                if hasattr(v.get("juror"), "value"):
                    v["juror"] = v["juror"].value
                council_votes.append(v)

            return {
                "final_label": judge_res.label,
                "final_confidence": judge_res.confidence,
                "council_tally": council_res.tally,
                "council_votes": council_votes,
                "passthrough_gold_ref": passthrough_gold,
            }

        # =====================================================================
        # LEGACY MODE
        # =====================================================================
        else:
            logger.debug(f"[{doc_id}] Running Legacy Debate (Optimized)...")
            council_res = asyncio.run(
                run_s2_sequential_debate(
                    text=text,
                    s1_spans=s1_spans,
                    marker_summary=marker_summary,
                    rag_context="",
                    prosecutor_sys_override=p_s,
                    defense_sys_override=d_s,
                    literalist_sys_override=l_s,
                    profiler_sys_override=pr_s,
                    prosecutor_user_template_override=p_u,
                    defense_user_template_override=d_u,
                    literalist_user_template_override=l_u,
                    profiler_user_template_override=pr_u,
                )
            )

            judge_res = asyncio.run(
                run_s2_judge_review(
                    text=text,
                    council_result=council_res,
                    rag_context=rag_context,
                    judge_sys_override=j_s,
                    judge_user_template_override=j_u,
                )
            )

            return {
                "final_label": judge_res.label,
                "final_confidence": judge_res.confidence,
                "council_tally": council_res.tally,
                "passthrough_gold_ref": passthrough_gold,
            }

    except Exception as e:
        logger.error(f"[{doc_id}] Prediction Failed: {e}", exc_info=True)
        return {
            "final_label": "non",
            "final_confidence": 0.0,
            "passthrough_gold_ref": passthrough_gold,
        }


# -----------------------------------------------------------------------------
# 4. Main Execution
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="GEPA S2 Council Optimization Engine")
    parser.add_argument("--data", default="data/raw/dev_ready_for_pipeline.jsonl")
    parser.add_argument("--limit", type=int, default=10, help="Examples to use")
    parser.add_argument("--budget", type=int, default=30, help="Max metric calls")
    parser.add_argument(
        "--rag-dir", default="data/rag_openai_contrastive", help="Path to ChromaDB RAG"
    )
    parser.add_argument("--experiment", default="GEPA_S2_Optimization")
    parser.add_argument(
        "--model-reflector",
        default="openai:/gpt-5.2",
    )
    parser.add_argument(
        "--phase",
        choices=[
            # Legacy phases (Sequential Debate)
            "all",
            "judge",
            "council",
            "core",
            # Parallel phases (Anti-Echo Chamber) - RECOMMENDED
            "parallel-all",
            "parallel-judge",
            "parallel-council",
            "parallel-core",
            # [NEW] Isolated Juror Phases
            "parallel-prosecutor",
            "parallel-defense",
            "parallel-literalist",
            "parallel-profiler",
        ],
        default="parallel-core",
        help="""Optimization phase:
            === LEGACY (Sequential Debate) ===
            'all'             = Optimize all 10 prompts (not recommended - huge search space)
            'judge'           = Optimize only Judge (2 prompts - most impactful)
            'council'         = Optimize only Council members (8 prompts - requires optimized Judge)
            'core'            = Optimize only system prompts, skip user templates (5 prompts)
            
            === PARALLEL (Anti-Echo Chamber) - RECOMMENDED ===
            'parallel-all'    = Optimize all 8 parallel prompts (4 jurors + shared user + calibrated judge)
            'parallel-judge'  = Optimize only Calibrated Judge (2 prompts - most impactful)
            'parallel-council'= Optimize only parallel jurors (5 prompts - 4 sys + 1 shared user)
            'parallel-core'   = Optimize only system prompts (5 prompts - recommended)
            
            === ISOLATED JURORS (Fine-Tuning) ===
            'parallel-prosecutor' = Optimize ONLY Prosecutor System Prompt
            'parallel-defense'    = Optimize ONLY Defense System Prompt
            'parallel-literalist' = Optimize ONLY Literalist System Prompt
            'parallel-profiler'   = Optimize ONLY Profiler System Prompt
        """,
    )
    args = parser.parse_args()

    # Logger Setup: DEBUG to file, INFO to console
    log_file = f"optimization_s2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    if os.path.exists(log_file):
        os.remove(log_file)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_file, level="DEBUG", rotation="10 MB")

    global P_SYS, D_SYS, L_SYS, PR_SYS
    global P_USER, D_USER, L_USER, PR_USER
    global JUDGE_SYS, JUDGE_USER
    global PARALLEL_P_SYS, PARALLEL_D_SYS, PARALLEL_L_SYS, PARALLEL_PR_SYS
    global PARALLEL_USER, CALIBRATED_JUDGE_SYS, CALIBRATED_JUDGE_USER
    global USE_PARALLEL_MODE
    global S2_RAG_COLLECTION

    # Initialize RAG
    if args.rag_dir and os.path.exists(args.rag_dir):
        logger.info(f"Initializing S2 RAG from {args.rag_dir}...")
        try:
            S2_RAG_COLLECTION = get_rag_collection(args.rag_dir, "s2_precedents")
            logger.success("S2 RAG Collection Loaded.")
        except Exception as e:
            logger.warning(
                f"RAG Load Failed: {e}. Optimization proceeds without dynamic precedents."
            )
    else:
        logger.warning(
            "No RAG directory provided. Using dataset-embedded context only."
        )

    # Load Data
    eval_dataset = load_classification_data(args.data, args.limit)
    if not eval_dataset:
        logger.error("No data loaded. Exiting.")
        sys.exit(1)

    # MLflow Setup
    mlflow.set_experiment(args.experiment)
    logger.info(f"Starting Run: {args.experiment}")

    with mlflow.start_run():
        mlflow.log_params(vars(args))
        logger.info("Registering S2 Prompts & Logging Baselines...")

        # 1. Load prompts using S2_PROMPTS singleton
        s2_prompts = S2_PROMPTS

        # 2. Log baselines for comparison
        mlflow.log_text(s2_prompts.pros_sys, "baseline_prompts/s2_prosecutor.txt")
        mlflow.log_text(s2_prompts.def_sys, "baseline_prompts/s2_defense.txt")
        mlflow.log_text(s2_prompts.lit_sys, "baseline_prompts/s2_literalist.txt")
        mlflow.log_text(s2_prompts.prof_sys, "baseline_prompts/s2_profiler.txt")
        mlflow.log_text(s2_prompts.judge_sys, "baseline_prompts/s2_judge.txt")

        # 4. Register Prompts
        P_SYS = mlflow.genai.register_prompt("s2_pros_sys", s2_prompts.pros_sys).uri
        D_SYS = mlflow.genai.register_prompt("s2_def_sys", s2_prompts.def_sys).uri
        L_SYS = mlflow.genai.register_prompt("s2_lit_sys", s2_prompts.lit_sys).uri
        PR_SYS = mlflow.genai.register_prompt("s2_prof_sys", s2_prompts.prof_sys).uri

        P_USER = mlflow.genai.register_prompt("s2_pros_user", s2_prompts.pros_user).uri
        D_USER = mlflow.genai.register_prompt("s2_def_user", s2_prompts.def_user).uri
        L_USER = mlflow.genai.register_prompt("s2_lit_user", s2_prompts.lit_user).uri
        PR_USER = mlflow.genai.register_prompt("s2_prof_user", s2_prompts.prof_user).uri

        JUDGE_SYS = mlflow.genai.register_prompt(
            "s2_judge_sys", s2_prompts.judge_sys
        ).uri
        JUDGE_USER = mlflow.genai.register_prompt(
            "s2_judge_user", s2_prompts.judge_user
        ).uri

        logger.success("Legacy Baseline Prompts Registered.")

        # 4b. Register Parallel (Anti-Echo Chamber) Prompts
        logger.info("Registering Parallel Anti-Echo Chamber Prompts...")

        mlflow.log_text(
            s2_prompts.parallel_pros_sys, "baseline_prompts/s2_parallel_prosecutor.txt"
        )
        mlflow.log_text(
            s2_prompts.parallel_def_sys, "baseline_prompts/s2_parallel_defense.txt"
        )
        mlflow.log_text(
            s2_prompts.parallel_lit_sys, "baseline_prompts/s2_parallel_literalist.txt"
        )
        mlflow.log_text(
            s2_prompts.parallel_prof_sys, "baseline_prompts/s2_parallel_profiler.txt"
        )
        mlflow.log_text(
            s2_prompts.parallel_user, "baseline_prompts/s2_parallel_user.txt"
        )
        mlflow.log_text(
            s2_prompts.calibrated_judge_sys, "baseline_prompts/s2_calibrated_judge.txt"
        )
        mlflow.log_text(
            s2_prompts.calibrated_judge_user,
            "baseline_prompts/s2_calibrated_judge_user.txt",
        )

        PARALLEL_P_SYS = mlflow.genai.register_prompt(
            "s2_parallel_pros_sys", s2_prompts.parallel_pros_sys
        ).uri
        PARALLEL_D_SYS = mlflow.genai.register_prompt(
            "s2_parallel_def_sys", s2_prompts.parallel_def_sys
        ).uri
        PARALLEL_L_SYS = mlflow.genai.register_prompt(
            "s2_parallel_lit_sys", s2_prompts.parallel_lit_sys
        ).uri
        PARALLEL_PR_SYS = mlflow.genai.register_prompt(
            "s2_parallel_prof_sys", s2_prompts.parallel_prof_sys
        ).uri
        PARALLEL_USER = mlflow.genai.register_prompt(
            "s2_parallel_user", s2_prompts.parallel_user
        ).uri
        CALIBRATED_JUDGE_SYS = mlflow.genai.register_prompt(
            "s2_calibrated_judge_sys", s2_prompts.calibrated_judge_sys
        ).uri
        CALIBRATED_JUDGE_USER = mlflow.genai.register_prompt(
            "s2_calibrated_judge_user", s2_prompts.calibrated_judge_user
        ).uri

        logger.success("Parallel Anti-Echo Chamber Prompts Registered.")

        # 5. Build Prompt URI List Based on Phase
        # Determine if parallel mode is requested
        is_parallel_phase = args.phase.startswith("parallel-")
        USE_PARALLEL_MODE = is_parallel_phase

        if args.phase == "judge":
            # Legacy: Optimize only Judge (most impactful, final decision maker)
            prompt_uris_to_optimize = [JUDGE_SYS, JUDGE_USER]
            logger.info("📌 Phase: JUDGE (Legacy) - Optimizing 2 prompts (Judge only)")
        elif args.phase == "council":
            # Legacy: Optimize Council members (requires optimized Judge)
            prompt_uris_to_optimize = [
                P_SYS,
                P_USER,
                D_SYS,
                D_USER,
                L_SYS,
                L_USER,
                PR_SYS,
                PR_USER,
            ]
            logger.info(
                "📌 Phase: COUNCIL (Legacy) - Optimizing 8 prompts (All jurors)"
            )
        elif args.phase == "core":
            # Legacy: Optimize only system prompts
            prompt_uris_to_optimize = [P_SYS, D_SYS, L_SYS, PR_SYS, JUDGE_SYS]
            logger.info("📌 Phase: CORE (Legacy) - Optimizing 5 system prompts")
        elif args.phase == "all":
            # Legacy: Full optimization (not recommended)
            prompt_uris_to_optimize = [
                P_SYS,
                P_USER,
                D_SYS,
                D_USER,
                L_SYS,
                L_USER,
                PR_SYS,
                PR_USER,
                JUDGE_SYS,
                JUDGE_USER,
            ]
            logger.warning(
                "⚠️ Phase: ALL (Legacy) - Optimizing all 10 prompts (large search space!)"
            )
        # ===== PARALLEL PHASES (Anti-Echo Chamber) =====
        elif args.phase == "parallel-judge":
            # Parallel: Optimize only Calibrated Judge
            prompt_uris_to_optimize = [CALIBRATED_JUDGE_SYS, CALIBRATED_JUDGE_USER]
            logger.info(
                "📌 Phase: PARALLEL-JUDGE - Optimizing 2 prompts (Calibrated Judge)"
            )
        elif args.phase == "parallel-council":
            # Parallel: Optimize parallel jurors + shared user template
            prompt_uris_to_optimize = [
                PARALLEL_P_SYS,
                PARALLEL_D_SYS,
                PARALLEL_L_SYS,
                PARALLEL_PR_SYS,
                PARALLEL_USER,  # Shared user template
            ]
            logger.info(
                "📌 Phase: PARALLEL-COUNCIL - Optimizing 5 prompts (4 juror sys + 1 shared user)"
            )
        elif args.phase == "parallel-core":
            # Parallel Recommended: Only system prompts (no user templates)
            prompt_uris_to_optimize = [
                PARALLEL_P_SYS,
                PARALLEL_D_SYS,
                PARALLEL_L_SYS,
                PARALLEL_PR_SYS,
                CALIBRATED_JUDGE_SYS,
            ]
            logger.info(
                "🚀 Phase: PARALLEL-CORE - Optimizing 5 system prompts (RECOMMENDED)"
            )
        elif args.phase == "parallel-all":
            # Parallel: Full optimization
            prompt_uris_to_optimize = [
                PARALLEL_P_SYS,
                PARALLEL_D_SYS,
                PARALLEL_L_SYS,
                PARALLEL_PR_SYS,
                PARALLEL_USER,
                CALIBRATED_JUDGE_SYS,
                CALIBRATED_JUDGE_USER,
            ]
            logger.warning("⚠️ Phase: PARALLEL-ALL - Optimizing all 7 parallel prompts")
        # [NEW] Isolated Juror Phases
        # We optimize ONLY the System Prompt to avoid breaking the shared User Template for others.
        elif args.phase == "parallel-prosecutor":
            prompt_uris_to_optimize = [PARALLEL_P_SYS]
            logger.info(
                "🎯 Phase: PARALLEL-PROSECUTOR - Optimizing 1 prompt (Prosecutor Sys)"
            )

        elif args.phase == "parallel-defense":
            prompt_uris_to_optimize = [PARALLEL_D_SYS]
            logger.info(
                "🎯 Phase: PARALLEL-DEFENSE - Optimizing 1 prompt (Defense Sys)"
            )

        elif args.phase == "parallel-literalist":
            prompt_uris_to_optimize = [PARALLEL_L_SYS]
            logger.info(
                "🎯 Phase: PARALLEL-LITERALIST - Optimizing 1 prompt (Literalist Sys)"
            )

        elif args.phase == "parallel-profiler":
            prompt_uris_to_optimize = [PARALLEL_PR_SYS]
            logger.info(
                "🎯 Phase: PARALLEL-PROFILER - Optimizing 1 prompt (Profiler Sys)"
            )
        else:
            logger.error(f"Unknown phase: {args.phase}")
            sys.exit(1)

        if USE_PARALLEL_MODE:
            logger.info(
                "🔄 Mode: PARALLEL (Anti-Echo Chamber) - Independent voting enabled"
            )
        else:
            logger.info(
                "📜 Mode: LEGACY (Sequential Debate) - Traditional council debate"
            )

        logger.info(
            f"🎯 Optimizing {len(prompt_uris_to_optimize)} prompts with budget={args.budget}"
        )

        # 6. Run Optimization
        logger.info("🚀 Launching S2 GEPA...")
        optimizer_config = GepaPromptOptimizer(
            reflection_model=args.model_reflector,
            max_metric_calls=args.budget,
            display_progress_bar=True,
        )

        results = mlflow.genai.optimize_prompts(
            predict_fn=predict_wrapper,
            train_data=eval_dataset,
            prompt_uris=prompt_uris_to_optimize,
            optimizer=optimizer_config,
            scorers=[s2_rich_scorer],
        )

        if S2_FEEDBACK_LOG:
            csv_path = "s2_feedback_history.csv"
            # Create DataFrame
            df = pd.DataFrame(S2_FEEDBACK_LOG)
            # Save to local CSV
            df.to_csv(csv_path, index=False)
            # Log as MLflow Artifact
            mlflow.log_artifact(csv_path)

            # Optional: Calculate quick stats for the logs
            avg_score = df["score"].mean()
            logger.success(
                f"Feedback Analysis saved to {csv_path} ({len(df)} records). Avg Score: {avg_score:.2f}"
            )

        logger.success("Optimization Complete!")
        if os.path.exists(log_file):
            mlflow.log_artifact(log_file)

    # 6. Save Results
    output_dir = pathlib.Path("prompts/openai")
    output_dir.mkdir(parents=True, exist_ok=True)

    optimized_list = []
    if hasattr(results, "best_prompts"):
        optimized_list = (
            list(results.best_prompts.values())
            if isinstance(results.best_prompts, dict)
            else results.best_prompts
        )
    elif hasattr(results, "optimized_prompts"):
        optimized_list = results.optimized_prompts

    name_map = {
        # Parallel prompts (Anti-Echo Chamber)
        "s2_parallel_pros_sys": "s2_parallel_prosecutor.txt",
        "s2_parallel_def_sys": "s2_parallel_defense.txt",
        "s2_parallel_lit_sys": "s2_parallel_literalist.txt",
        "s2_parallel_prof_sys": "s2_parallel_profiler.txt",
        "s2_parallel_user": "s2_parallel_user.txt",
        "s2_calibrated_judge_sys": "s2_calibrated_judge_eda.txt",
        "s2_calibrated_judge_user": "s2_calibrated_judge_user.txt",
    }

    saved_count = 0
    for prompt_obj in optimized_list:
        if prompt_obj.name in name_map:
            fname = name_map[prompt_obj.name]
            (output_dir / fname).write_text(prompt_obj.template, encoding="utf-8")
            mlflow.log_artifact(str(output_dir / fname))
            logger.success(f"Saved optimized prompt '{prompt_obj.name}' to {fname}")
            saved_count += 1

    if saved_count == 0:
        logger.error("No S2 prompts returned from optimization.")
    else:
        logger.success(f"Saved {saved_count} optimized S2 prompts to {output_dir}")


if __name__ == "__main__":
    main()
