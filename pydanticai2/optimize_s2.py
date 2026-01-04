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

# --- Make repo root importable FIRST ---
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# Third-party
import mlflow
import mlflow.genai
from mlflow.genai import scorer
from mlflow.genai.optimize import GepaPromptOptimizer
from mlflow.entities import Feedback
from loguru import logger


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
    run_s2_sequential_debate,
    run_s2_judge_review,
    get_rag_collection,
    retrieve_fewshots,
    format_s2_rag_to_xml,
)
from pydanticai2.prompt_builder import (
    build_s2_prosecutor_system,  # For Prosecutor
    build_s2_defense_system,  # For Defense
    build_s2_literalist_system,  # For Literalist
    build_s2_profiler_system,  # For Profiler
    build_s2_prosecutor_user_template,
    build_s2_defense_user_template,
    build_s2_literalist_user_template,
    build_s2_profiler_user_template,
    build_s2_judge_system,
    build_s2_judge_user_template,
)


# -----------------------------------------------------------------------------
# 1. Setup
# -----------------------------------------------------------------------------

# Globals for URIs
P_SYS, D_SYS, L_SYS, PR_SYS = None, None, None, None
P_USER, D_USER, L_USER, PR_USER = None, None, None, None
JUDGE_SYS, JUDGE_USER = None, None
# RAG Collection (Global)
S2_RAG_COLLECTION = None


def load_classification_data(path: str, limit: int = 20) -> List[Dict]:
    """
    Loads and slices the Gold Standard dataset using the Trojan Horse pattern.
    Gold labels are injected into INPUTS to bypass MLflow's target sanitization.
    """
    dataset = []
    p = pathlib.Path(path)
    if not p.exists():
        logger.error(f"Dataset not found: {path}")
        return []

    with open(p, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                label = str(row.get("label", "non")).lower().strip()
                text = row.get("text")
                doc_id = row.get("doc_id", f"line_{line_num}")

                # [CRITICAL] Retrieve Pre-computed S1 Spans
                s1_spans = row.get("s1_spans", [])
                if not s1_spans and "spans" in row:
                    s1_spans = row["spans"]  # Fallback to 'spans' key

                # Generate the Marker Summary string needed by the prompts
                if s1_spans:
                    marker_summary = "\n".join(
                        [f"- [{s['label'].upper()}]: {s['text']}" for s in s1_spans]
                    )
                else:
                    marker_summary = "No specific forensic markers identified."

                # --- THE TROJAN HORSE ---
                # We inject gold data into INPUTS so it goes to the wrapper
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
                            # --- THE TROJAN HORSE ---
                            "passthrough_gold": payload_str,
                        },
                        # MLflow requires an output col, we give it a dummy
                        "outputs": {"dummy_target": "ignore_me"},
                    }
                )
            except Exception as e:
                logger.warning(f"Skipping bad line {line_num}: {e}")

    dataset = dataset[:limit]

    # Sanity Check
    conspiracy_count = sum(
        1
        for d in dataset
        if json.loads(d["inputs"]["passthrough_gold"])["gold_label"] == "conspiracy"
    )
    hard_neg_count = sum(
        1
        for d in dataset
        if json.loads(d["inputs"]["passthrough_gold"]).get("is_hard_negative", False)
    )
    logger.success(f"DATASET LOADED: {len(dataset)} items.")
    logger.info(
        f" > Conspiracy: {conspiracy_count} | Non: {len(dataset) - conspiracy_count}"
    )
    logger.info(f" > Hard Negatives: {hard_neg_count}")

    return dataset


# -----------------------------------------------------------------------------
# 2. Classification Scorer (Reads from OUTPUTS via Trojan Horse)
# -----------------------------------------------------------------------------


def generate_s2_actionable_feedback(
    gold_label: str,
    pred_label: str,
    council_tally: dict,
    council_votes: list,  # List of vote dicts with juror, verdict, rationale
    doc_id: str,
    s2_subtype: str,
    is_hard_negative: bool,
) -> str:
    """
    Generate specific, actionable feedback for the S2 optimizer.
    Focuses on WHAT went wrong and WHO needs to improve.

    council_votes is a list of dicts: [{"juror": "...", "verdict": "...", "rationale": "..."}, ...]
    """
    feedback_parts = []

    consp_votes = council_tally.get("conspiracy", 0)
    non_votes = council_tally.get("non", 0) + council_tally.get("non-conspiracy", 0)
    council_consensus = "conspiracy" if consp_votes > non_votes else "non"

    # Tag hard negatives for special attention
    if is_hard_negative:
        feedback_parts.append(f"[HARD_NEGATIVE:{s2_subtype}]")

    # Case A: Judge Fault (Judge overruled correct Council)
    if council_consensus == gold_label and pred_label != gold_label:
        feedback_parts.append(
            f"JUDGE_FAILURE: Council was correct ({council_consensus}), but Judge overruled to {pred_label}."
        )
        feedback_parts.append(
            "FIX: Judge should trust Council consensus on this type of text."
        )

    # Case B: Council Fault (Council was wrong)
    elif council_consensus != gold_label:
        feedback_parts.append(
            f"COUNCIL_FAILURE: Debate reached wrong consensus ({council_consensus}). Votes: {council_tally}."
        )

        if gold_label == "non":
            # False Positive - Council incorrectly flagged as conspiracy
            feedback_parts.append(
                "FALSE_POSITIVE: Defense failed to argue for acquittal."
            )
            feedback_parts.append(
                "FIX: Defense should emphasize: no hidden plot, just normal criticism/reporting."
            )
        else:
            # False Negative - Council missed the conspiracy
            feedback_parts.append(
                "FALSE_NEGATIVE: Prosecutor failed to identify the conspiracy."
            )
            feedback_parts.append(
                "FIX: Prosecutor should look for: hidden actors, secret plots, cover-up language."
            )

        # Identify which council member(s) voted wrong
        wrong_voters = []
        for vote in council_votes:
            if isinstance(vote, dict):
                juror_name = vote.get("juror", "unknown")
                vote_verdict = str(vote.get("verdict", "")).lower()
            else:
                # Handle object attributes if not serialized
                juror_name = getattr(vote, "juror", "unknown")
                vote_verdict = str(getattr(vote, "verdict", "")).lower()

            if "conspiracy" in vote_verdict and gold_label == "non":
                wrong_voters.append(f"{juror_name}(voted conspiracy)")
            elif "non" in vote_verdict and gold_label == "conspiracy":
                wrong_voters.append(f"{juror_name}(voted non)")

        if wrong_voters:
            feedback_parts.append(f"WRONG_VOTERS: {', '.join(wrong_voters[:2])}")

    # Case C: Judge agreed with wrong Council (total failure)
    else:
        feedback_parts.append(
            f"TOTAL_SYSTEM_FAILURE: Everyone got it wrong. Votes: {council_tally}"
        )

    return " | ".join(feedback_parts)


@scorer
def s2_rich_scorer(outputs, expectations):
    """
    Enhanced Diagnostic Scorer with:
    1. Trojan Horse pattern - reads gold from outputs
    2. Detailed failure analysis by component
    3. Actionable feedback for optimization
    """
    # 1. Extract prediction
    pred_label = str(outputs.get("final_label", "non")).lower().strip()
    council_tally = outputs.get("council_tally", {})
    council_votes = outputs.get("council_votes", [])  # List of vote dicts

    # --- UNPACK FROM WRAPPER OUTPUT (Trojan Horse) ---
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
        else:
            logger.warning("Scorer: Missing 'passthrough_gold_ref' in model outputs.")
            # Fallback to expectations if available
            gold_label = str(expectations.get("gold_label", "non")).lower().strip()
    except Exception as e:
        logger.error(f"Scorer Unpack Failed: {e}")
        gold_label = str(expectations.get("gold_label", "non")).lower().strip()

    # Calculate Consensus
    consp_votes = council_tally.get("conspiracy", 0)
    non_votes = council_tally.get("non", 0) + council_tally.get("non-conspiracy", 0)
    council_consensus = "conspiracy" if consp_votes > non_votes else "non"

    # Log scoring details
    logger.info(
        f"[{doc_id}] SCORING: pred={pred_label}, gold={gold_label}, "
        f"consensus={council_consensus}, tally={council_tally}"
    )

    # 2. Success Case
    if pred_label == gold_label:
        # Bonus feedback for correct predictions
        if is_hard_negative:
            rationale = f"CORRECT [HARD_NEGATIVE:{s2_subtype}]: Verdict={pred_label}, Consensus={council_consensus}, Tally={council_tally}"
        else:
            rationale = f"CORRECT: Verdict={pred_label}, Consensus={council_consensus}, Tally={council_tally}"

        logger.info(f"[{doc_id}] ✅ Score: 1.0 | {rationale}")
        return Feedback(value=1.0, rationale=rationale)

    # 3. Failure Analysis with Actionable Feedback
    rationale = generate_s2_actionable_feedback(
        gold_label=gold_label,
        pred_label=pred_label,
        council_tally=council_tally,
        council_votes=council_votes,  # Pass list of vote dicts
        doc_id=doc_id,
        s2_subtype=s2_subtype,
        is_hard_negative=is_hard_negative,
    )

    logger.warning(
        f"[{doc_id}] ❌ Score: 0.0 | Consensus={council_consensus} | {rationale}"
    )
    return Feedback(value=0.0, rationale=rationale)


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
):
    """
    Prediction wrapper with Trojan Horse pattern.

    Args:
        text: The input text for classification.
        s1_spans: Pre-computed S1 spans.
        marker_summary: Formatted string of S1 markers.
        passthrough_gold: The hidden JSON payload containing gold labels.
    """
    global _LOGGED_ONCE

    # 1. Helper to safely inject variables into prompts
    def safe_fmt(uri, **kwargs):
        if not uri:
            return None
        try:
            prompt_obj = mlflow.genai.load_prompt(uri)
            template = prompt_obj.template
            # Replace placeholders
            for key, value in kwargs.items():
                template = template.replace("{{" + key + "}}", str(value))
            return template
        except Exception:
            try:
                return mlflow.genai.load_prompt(uri).format()
            except Exception:
                return None

    # 2. Build RAG context dynamically
    rag_context = ""
    if S2_RAG_COLLECTION:
        try:
            precedents = retrieve_fewshots(
                S2_RAG_COLLECTION,
                text,
                k=4,
                filters={"is_hard_negative": True},
            )
            if precedents:
                rag_context = json.dumps(precedents, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"S2 Dynamic RAG Retrieval Failed: {e}")

    rag_str = format_s2_rag_to_xml(rag_context)

    # 3. Load System Prompts with RAG context injection
    p_s = safe_fmt(P_SYS, rag_context=rag_str)
    d_s = safe_fmt(D_SYS, rag_context=rag_str)
    l_s = safe_fmt(L_SYS, rag_context=rag_str)
    pr_s = safe_fmt(PR_SYS, rag_context=rag_str)
    j_s = safe_fmt(JUDGE_SYS, rag_context=rag_str)

    # 4. Load User Templates with RAG context
    p_u = safe_fmt(P_USER, rag_context=rag_str)
    d_u = safe_fmt(D_USER, rag_context=rag_str)
    l_u = safe_fmt(L_USER, rag_context=rag_str)
    pr_u = safe_fmt(PR_USER, rag_context=rag_str)
    j_u = safe_fmt(JUDGE_USER, rag_context=rag_str)

    # [DEBUG] Log the actual prompts being used (Once per run)
    if not _LOGGED_ONCE:
        logger.info("--- [SNAPSHOT] Active System Prompts (First 200 chars) ---")
        logger.info(f"Prosecutor: {str(p_s)[:200] if p_s else 'NOT LOADED'}...")
        logger.info(f"Defense:    {str(d_s)[:200] if d_s else 'NOT LOADED'}...")
        logger.info(f"Literalist: {str(l_s)[:200] if l_s else 'NOT LOADED'}...")
        logger.info(f"Profiler:   {str(pr_s)[:200] if pr_s else 'NOT LOADED'}...")
        logger.info(f"Judge:      {str(j_s)[:200] if j_s else 'NOT LOADED'}...")
        _LOGGED_ONCE = True

    # Parse doc_id for logging
    doc_id = "unknown"
    try:
        payload = json.loads(passthrough_gold)
        doc_id = payload.get("doc_id", "unknown")
    except Exception:
        pass

    try:
        # 5. Run Council Debate
        logger.debug(f"[{doc_id}] Running S2 Council Debate...")
        council_res = asyncio.run(
            run_s2_sequential_debate(
                text=text,
                s1_spans=s1_spans,
                marker_summary=marker_summary,
                prosecutor_sys_override=p_s,
                rag_context="",  # Already injected into prompts
                defense_sys_override=d_s,
                literalist_sys_override=l_s,
                profiler_sys_override=pr_s,
                prosecutor_user_template_override=p_u,
                defense_user_template_override=d_u,
                literalist_user_template_override=l_u,
                profiler_user_template_override=pr_u,
            )
        )

        logger.debug(f"[{doc_id}] Council Tally: {council_res.tally}")

        # 6. Run Judge Review
        logger.debug(f"[{doc_id}] Running S2 Judge Review...")
        judge_res = asyncio.run(
            run_s2_judge_review(
                text=text,
                council_result=council_res,
                rag_context=rag_context,
                judge_sys_override=j_s,
                judge_user_template_override=j_u,
            )
        )

        logger.debug(f"[{doc_id}] Judge Verdict: {judge_res.label}")

        # 7. Serialize council votes to list of dicts for scorer
        council_votes = []
        if hasattr(council_res, "votes") and council_res.votes:
            for vote in council_res.votes:
                if hasattr(vote, "model_dump"):
                    # Pydantic model - serialize properly
                    vote_dict = vote.model_dump()
                    # Convert enum to string
                    if hasattr(vote_dict.get("juror"), "value"):
                        vote_dict["juror"] = vote_dict["juror"].value
                    elif isinstance(vote_dict.get("juror"), str):
                        pass  # Already string
                    council_votes.append(vote_dict)
                elif isinstance(vote, dict):
                    council_votes.append(vote)
                else:
                    # Fallback: extract attributes
                    council_votes.append(
                        {
                            "juror": str(getattr(vote, "juror", "unknown")),
                            "verdict": str(getattr(vote, "verdict", "unknown")),
                            "rationale": str(getattr(vote, "rationale", "")),
                        }
                    )

        return {
            "final_label": judge_res.label,
            "final_rationale": judge_res.rationale,
            "council_tally": council_res.tally,
            "council_votes": council_votes,  # List of dicts
            # --- THE TUNNEL EXIT ---
            # We explicitly return the gold data so the scorer can see it in 'outputs'
            "passthrough_gold_ref": passthrough_gold,
        }

    except Exception as e:
        logger.error(f"[{doc_id}] Prediction Wrapper Failed: {e}", exc_info=True)
        return {
            "final_label": "non",  # Safe default
            "final_rationale": f"SYSTEM_ERROR: {str(e)[:100]}",
            "council_tally": {},
            "council_votes": [],  # Empty list
            "passthrough_gold_ref": passthrough_gold,
        }


# -----------------------------------------------------------------------------
# 4. Main Execution
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="GEPA S2 Council Optimization Engine")
    parser.add_argument("--data", default="data/gold/optimization_set.jsonl")
    parser.add_argument("--limit", type=int, default=20, help="Examples to use")
    parser.add_argument("--budget", type=int, default=60, help="Max metric calls")
    parser.add_argument(
        "--rag-dir", default="data/rag_online_v3", help="Path to ChromaDB RAG"
    )
    parser.add_argument("--experiment", default="GEPA_S2_Optimization")
    parser.add_argument(
        "--model-reflector",
        default="bedrock:/eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    parser.add_argument(
        "--phase",
        choices=["all", "judge", "council", "core"],
        default="core",
        help="""Optimization phase:
            'all'     = Optimize all 10 prompts (not recommended - huge search space)
            'judge'   = Optimize only Judge (2 prompts - most impactful)
            'council' = Optimize only Council members (8 prompts - requires optimized Judge)
            'core'    = Optimize only system prompts, skip user templates (5 prompts - recommended)
        """,
    )
    args = parser.parse_args()

    # Logger Setup: DEBUG to file, INFO to console
    log_file = "optimization_s2.log"
    if os.path.exists(log_file):
        os.remove(log_file)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_file, level="DEBUG", rotation="10 MB")

    global P_SYS, D_SYS, L_SYS, PR_SYS
    global P_USER, D_USER, L_USER, PR_USER
    global JUDGE_SYS, JUDGE_USER
    global S2_RAG_COLLECTION

    # Initialize RAG
    if args.rag_dir and os.path.exists(args.rag_dir):
        logger.info(f"Initializing S2 RAG from {args.rag_dir}...")
        try:
            S2_RAG_COLLECTION = get_rag_collection(args.rag_dir, "s2_examples")
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

        # 1. Generate Prompt Strings
        pros_sys_str = build_s2_prosecutor_system()
        def_sys_str = build_s2_defense_system()
        lit_sys_str = build_s2_literalist_system()
        prof_sys_str = build_s2_profiler_system()
        judge_sys_str = build_s2_judge_system()

        # 2. Log baselines for comparison
        mlflow.log_text(pros_sys_str, "baseline_prompts/s2_prosecutor.txt")
        mlflow.log_text(def_sys_str, "baseline_prompts/s2_defense.txt")
        mlflow.log_text(lit_sys_str, "baseline_prompts/s2_literalist.txt")
        mlflow.log_text(prof_sys_str, "baseline_prompts/s2_profiler.txt")
        mlflow.log_text(judge_sys_str, "baseline_prompts/s2_judge.txt")

        # 3. Log prompts to file for verification
        logger.debug(
            f"=== PROSECUTOR (Warm Start) ===\n{pros_sys_str[:500]}\n==============================="
        )
        logger.debug(
            f"=== DEFENSE (Warm Start) ===\n{def_sys_str[:500]}\n==============================="
        )

        # 4. Register Prompts
        P_SYS = mlflow.genai.register_prompt("s2_pros_sys", pros_sys_str).uri
        D_SYS = mlflow.genai.register_prompt("s2_def_sys", def_sys_str).uri
        L_SYS = mlflow.genai.register_prompt("s2_lit_sys", lit_sys_str).uri
        PR_SYS = mlflow.genai.register_prompt("s2_prof_sys", prof_sys_str).uri

        P_USER = mlflow.genai.register_prompt(
            "s2_pros_user", build_s2_prosecutor_user_template()
        ).uri
        D_USER = mlflow.genai.register_prompt(
            "s2_def_user", build_s2_defense_user_template()
        ).uri
        L_USER = mlflow.genai.register_prompt(
            "s2_lit_user", build_s2_literalist_user_template()
        ).uri
        PR_USER = mlflow.genai.register_prompt(
            "s2_prof_user", build_s2_profiler_user_template()
        ).uri

        JUDGE_SYS = mlflow.genai.register_prompt("s2_judge_sys", judge_sys_str).uri
        JUDGE_USER = mlflow.genai.register_prompt(
            "s2_judge_user", build_s2_judge_user_template()
        ).uri

        logger.success("Baseline Prompts Registered.")

        # 5. Build Prompt URI List Based on Phase
        if args.phase == "judge":
            # Phase 1: Optimize only Judge (most impactful, final decision maker)
            prompt_uris_to_optimize = [JUDGE_SYS, JUDGE_USER]
            logger.info("📌 Phase: JUDGE - Optimizing 2 prompts (Judge only)")
        elif args.phase == "council":
            # Phase 2: Optimize Council members (requires optimized Judge)
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
            logger.info("📌 Phase: COUNCIL - Optimizing 8 prompts (All jurors)")
        elif args.phase == "core":
            # Recommended: Optimize only system prompts (user templates are mostly data injection)
            prompt_uris_to_optimize = [P_SYS, D_SYS, L_SYS, PR_SYS, JUDGE_SYS]
            logger.info("📌 Phase: CORE - Optimizing 5 system prompts (recommended)")
        else:  # "all"
            # Full optimization (not recommended - huge search space)
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
                "⚠️ Phase: ALL - Optimizing all 10 prompts (large search space!)"
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

        logger.success("Optimization Complete!")
        if os.path.exists(log_file):
            mlflow.log_artifact(log_file)

    # 6. Save Results
    output_dir = pathlib.Path("prompts/optimized_s2")
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
        "s2_pros_sys": "s2_prosecutor_optimized.txt",
        "s2_pros_user": "s2_prosecutor_user_optimized.txt",
        "s2_def_sys": "s2_defense_optimized.txt",
        "s2_def_user": "s2_defense_user_optimized.txt",
        "s2_lit_sys": "s2_literalist_optimized.txt",
        "s2_lit_user": "s2_literalist_user_optimized.txt",
        "s2_prof_sys": "s2_profiler_optimized.txt",
        "s2_prof_user": "s2_profiler_user_optimized.txt",
        "s2_judge_sys": "s2_judge_optimized.txt",
        "s2_judge_user": "s2_judge_user_optimized.txt",
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
