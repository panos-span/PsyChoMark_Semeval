#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimize_s1.py — GEPA Optimization Engine.

STRATEGY: "Trojan Horse" Passthrough.
We pass the Gold Labels into the INPUTS, creating a tunnel through the Predict Wrapper
directly to the Scorer. This bypasses MLflow's target sanitization logic.
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
from difflib import SequenceMatcher
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
    find_best_span,  # For span localization
    format_s1_fewshots_to_xml,  # For few-shot formatting
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
# 1. Setup & Data Loading (The Injection)
# -----------------------------------------------------------------------------


def load_eval_data(path: str, limit: int = 20) -> List[Dict]:
    """Loads and slices the Gold Standard dataset."""
    dataset = []
    p = pathlib.Path(path)
    if not p.exists():
        logger.error(f"Dataset not found: {path}")
        sys.exit(1)

    with open(p, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue
            try:
                row = json.loads(line)

                # 1. Get Spans
                gold_spans = row.get("spans", [])
                if not isinstance(gold_spans, list):
                    gold_spans = []

                # 2. Serialize Payload
                payload = {
                    "gold_spans": gold_spans,
                    "doc_id": row.get("doc_id", f"line_{line_num}"),
                }
                payload_str = json.dumps(payload)

                dataset.append(
                    {
                        "inputs": {
                            "text": row["text"],
                            # --- THE TROJAN HORSE ---
                            # We inject gold data into INPUTS so it goes to the wrapper
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
    valid_count = sum(
        1 for d in dataset if json.loads(d["inputs"]["passthrough_gold"])["gold_spans"]
    )
    logger.success(f"DATASET LOADED: {len(dataset)} items.")
    logger.info(f" > Items with Spans: {valid_count}")

    return dataset


# -----------------------------------------------------------------------------
# 2. Rich Feedback Scorer (Reads from OUTPUTS, not EXPECTATIONS)
# -----------------------------------------------------------------------------


def compute_overlap_score(a: str, b: str, a_pos=None, b_pos=None) -> float:
    """
    Enhanced similarity score that handles:
    1. Position-based overlap (IoU) when indices are available
    2. Exact/substring matches
    3. Split entities (e.g., "NASA/the government" vs "NASA" or "the government")
    4. Fuzzy word overlap

    Args:
        a: First span text
        b: Second span text
        a_pos: Optional (start, end) tuple for first span
        b_pos: Optional (start, end) tuple for second span
    """
    # --- Position-based scoring (most accurate) ---
    if a_pos and b_pos:
        a_start, a_end = a_pos
        b_start, b_end = b_pos

        # Check for valid positions
        if a_start >= 0 and a_end > a_start and b_start >= 0 and b_end > b_start:
            # Calculate Intersection over Union (IoU)
            intersection_start = max(a_start, b_start)
            intersection_end = min(a_end, b_end)
            intersection = max(0, intersection_end - intersection_start)

            union = (a_end - a_start) + (b_end - b_start) - intersection

            if union > 0:
                iou = intersection / union
                if iou > 0.3:  # Significant overlap
                    return iou

    # --- Text-based scoring (fallback) ---
    a_norm = a.lower().strip()
    b_norm = b.lower().strip()

    if not a_norm or not b_norm:
        return 0.0

    # Exact match
    if a_norm == b_norm:
        return 1.0

    # Handle compound entities split by "/" or "and" BEFORE substring check
    # This gives higher score for matching part of a compound entity
    for sep in ["/", " and ", ", "]:
        if sep in b_norm:
            parts = [p.strip() for p in b_norm.split(sep)]
            for part in parts:
                if a_norm == part:
                    return 0.9  # Exact match of a compound part
                if a_norm in part or part in a_norm:
                    return 0.8
        if sep in a_norm:
            parts = [p.strip() for p in a_norm.split(sep)]
            for part in parts:
                if b_norm == part:
                    return 0.9  # Exact match of a compound part
                if b_norm in part or part in b_norm:
                    return 0.8

    # Substring containment (one contains the other)
    if a_norm in b_norm or b_norm in a_norm:
        # Score based on coverage ratio
        shorter = min(len(a_norm), len(b_norm))
        longer = max(len(a_norm), len(b_norm))
        return shorter / longer if longer > 0 else 0.0

    # Character-level similarity
    char_score = SequenceMatcher(None, a_norm, b_norm).ratio()

    # Word-level Jaccard
    set_a = set(a_norm.split())
    set_b = set(b_norm.split())
    union = len(set_a.union(set_b))
    intersection = len(set_a.intersection(set_b))
    jaccard_score = intersection / union if union > 0 else 0.0

    return max(char_score, jaccard_score)


def normalize_label(label) -> str:
    """Normalize label to a simple string for comparison."""
    label_str = str(label)
    # Handle enum representations like "S1Label.Actor" -> "Actor"
    if "." in label_str:
        label_str = label_str.split(".")[-1]
    return label_str.lower().strip()


def generate_actionable_feedback(
    gold_spans: list,
    pred_spans: list,
    gold_matched: set,
    pred_matched: set,
    label_errors: list,
    doc_id: str,
) -> str:
    """
    Generate specific, actionable feedback for the optimizer.
    Focuses on WHAT to fix and HOW.
    """
    feedback_parts = []

    # 1. Label errors - be specific about what the correct label should be
    if label_errors:
        feedback_parts.append(f"FIX LABELS: {'; '.join(label_errors[:3])}")

    # 2. Missed spans - show what should have been extracted WITH labels
    missed_indices = [i for i in range(len(gold_spans)) if i not in gold_matched]
    if missed_indices:
        missed_details = []
        for idx in missed_indices[:3]:
            g = gold_spans[idx]
            g_text = g.get("text", "")[:50]  # Truncate for readability
            g_label = g.get("label", "Unknown")
            missed_details.append(f"{g_label}:'{g_text}'")
        feedback_parts.append(f"EXTRACT MISSING: {'; '.join(missed_details)}")

    # 3. Hallucinations - only flag if they don't partially match anything
    halluc_indices = [i for i in range(len(pred_spans)) if i not in pred_matched]
    if halluc_indices:
        # Filter out very short hallucinations that might be annotation noise
        significant_halluc = []
        for idx in halluc_indices[:3]:
            p = pred_spans[idx]
            p_text = p.get("text", "")
            # Only report if it's substantial (not just "it", "he", etc.)
            if len(p_text) > 3:
                significant_halluc.append(f"'{p_text[:40]}'")

        if significant_halluc:
            feedback_parts.append(f"REMOVE: {'; '.join(significant_halluc)}")

    if not feedback_parts:
        return "PERFECT: All spans correctly extracted and labeled."

    return " | ".join(feedback_parts)


@scorer
def s1_rich_scorer(outputs, expectations):
    """
    Enhanced Diagnostic Scorer with:
    1. Better overlap matching for split entities
    2. Partial credit for near-matches
    3. Actionable feedback for optimization
    4. Lenient handling of annotation ambiguity
    """
    pred_spans = outputs.get("final_spans", [])

    # Debug: Log what we received from the model
    logger.debug(f"Scorer received outputs keys: {list(outputs.keys())}")
    logger.debug(f"Scorer pred_spans count: {len(pred_spans)}")
    if pred_spans:
        logger.debug(f"Scorer pred_spans sample: {pred_spans[:2]}")

    # --- UNPACK FROM WRAPPER OUTPUT ---
    gold_spans = []
    doc_id = "unknown"

    try:
        raw_payload = outputs.get("passthrough_gold_ref")
        if raw_payload:
            data = json.loads(raw_payload)
            gold_spans = data.get("gold_spans", [])
            doc_id = data.get("doc_id", "unknown")
        else:
            logger.warning("Scorer: Missing 'passthrough_gold_ref' in model outputs.")
    except Exception as e:
        logger.error(f"Scorer Unpack Failed: {e}")

    # ---------------------------------------------------------
    # EDGE CASES
    # ---------------------------------------------------------

    if not gold_spans and not pred_spans:
        logger.debug(
            f"[{doc_id}] S1 Score: 1.0 | Correct: No spans expected, none extracted."
        )
        return Feedback(
            value=1.0,
            rationale="PERFECT: Correctly identified no conspiracy markers in this text.",
        )

    if gold_spans and not pred_spans:
        # Format expected spans with labels for better feedback
        expected = [
            f"{g.get('label', '?')}:'{g.get('text', '')[:30]}'" for g in gold_spans[:3]
        ]
        logger.debug(f"[{doc_id}] S1 Score: 0.0 | FAILED (Recall). Missed: {expected}")
        return Feedback(
            value=0.0,
            rationale=f"EXTRACT MISSING: {'; '.join(expected)}. The model found nothing but these markers exist.",
        )

    if pred_spans and not gold_spans:
        halluc = [f"'{p.get('text', '')[:30]}'" for p in pred_spans[:3]]
        logger.debug(
            f"[{doc_id}] S1 Score: 0.0 | FAILED (Precision). Hallucinated: {halluc}"
        )
        return Feedback(
            value=0.0,
            rationale=f"REMOVE ALL: {'; '.join(halluc)}. This text has NO conspiracy markers - it's a negative example.",
        )

    # ---------------------------------------------------------
    # MATCHING ALGORITHM (Enhanced with Position-Based Scoring)
    # ---------------------------------------------------------

    gold_matched = set()
    pred_matched = set()

    # Track different types of matches
    exact_matches = 0  # Correct text AND label
    partial_matches = 0  # Correct text, wrong label (partial credit)
    label_errors = []

    # Filter out hallucinated spans (start == -1 means not found in text)
    valid_pred_spans = []
    hallucinated_not_in_text = []
    for p in pred_spans:
        if p.get("start", 0) == -1:
            hallucinated_not_in_text.append(p.get("text", "")[:30])
        else:
            valid_pred_spans.append(p)

    # Match predictions to gold spans
    for p_idx, p in enumerate(valid_pred_spans):
        p_text = p.get("text", "")
        p_lbl = normalize_label(p.get("label", ""))
        p_start = p.get("start")
        p_end = p.get("end")
        p_pos = (p_start, p_end) if p_start is not None and p_end is not None else None

        best_score = 0.0
        best_g_idx = -1

        for g_idx, g in enumerate(gold_spans):
            if g_idx in gold_matched:
                continue
            g_text = g.get("text", "")
            g_start = g.get("start")
            g_end = g.get("end")
            g_pos = (
                (g_start, g_end) if g_start is not None and g_end is not None else None
            )

            # Use position-based matching when available
            score = compute_overlap_score(p_text, g_text, p_pos, g_pos)
            if score > best_score:
                best_score = score
                best_g_idx = g_idx

        # Match threshold: 0.4 for partial credit, allows for split entities
        if best_score >= 0.4 and best_g_idx >= 0:
            g_target = gold_spans[best_g_idx]
            g_lbl = normalize_label(g_target.get("label", ""))

            gold_matched.add(best_g_idx)
            pred_matched.add(p_idx)

            if p_lbl == g_lbl:
                # Full match: correct text AND label
                exact_matches += 1
            else:
                # Partial match: found the span but wrong label
                partial_matches += 1
                label_errors.append(f"'{p_text[:30]}' → change {p_lbl} to {g_lbl}")

    # ---------------------------------------------------------
    # SCORING (Weighted F-beta with partial credit)
    # ---------------------------------------------------------

    fn = len(gold_spans) - len(gold_matched)  # Missed gold spans
    # Hallucinated = unmatched valid predictions + spans not found in text
    fp = (len(valid_pred_spans) - len(pred_matched)) + len(hallucinated_not_in_text)

    # Give partial credit for correct span with wrong label
    tp_effective = exact_matches + (partial_matches * 0.5)

    total_positive = tp_effective + fp + (partial_matches * 0.5)
    precision = tp_effective / total_positive if total_positive > 0 else 0

    total_relevant = tp_effective + fn + (partial_matches * 0.5)
    recall = tp_effective / total_relevant if total_relevant > 0 else 0

    # F2 score (recall-weighted)
    beta = 2.0
    f_beta = (
        (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
        if (precision + recall) > 0
        else 0
    )

    # ---------------------------------------------------------
    # ACTIONABLE FEEDBACK
    # ---------------------------------------------------------

    rationale = generate_actionable_feedback(
        gold_spans, valid_pred_spans, gold_matched, pred_matched, label_errors, doc_id
    )

    # Add info about spans not found in text
    if hallucinated_not_in_text:
        rationale += f" | NOT_IN_TEXT: {'; '.join(hallucinated_not_in_text[:2])}"

    logger.debug(f"[{doc_id}] S1 Score: {f_beta:.2f} | {rationale}")
    return Feedback(value=float(f_beta), rationale=rationale)


# -----------------------------------------------------------------------------
# 3. Prediction Wrapper (The Tunnel)
# -----------------------------------------------------------------------------

GEN_SYS_URI, GEN_USER_URI = None, None
CRITIC_SYS_URI, CRITIC_USER_URI = None, None
REFINER_SYS_URI, REFINER_USER_URI = None, None
S1_RAG_COLLECTION = None


def predict_wrapper(text: str, passthrough_gold: str = "{}"):
    """
    Args:
        text: The input text for the model.
        passthrough_gold: The hidden JSON payload containing gold labels.
    """
    # 1. Retrieve & Format RAG
    few_shots = []
    if S1_RAG_COLLECTION:
        few_shots = retrieve_stratified_s1(S1_RAG_COLLECTION, text, k_total=6)
    few_shots_str = format_s1_fewshots_to_xml(few_shots)

    # 2. Helper to safely inject variables into prompts
    def load_with_context(uri):
        if not uri:
            return None
        try:
            prompt_obj = mlflow.genai.load_prompt(uri)
            template = prompt_obj.template
            # Replace double-brace placeholder used in our templates
            if "{{few_shot_examples}}" in template:
                template = template.replace("{{few_shot_examples}}", few_shots_str)
            return template
        except Exception as e:
            logger.warning(f"Failed to load prompt from {uri}: {e}")
            return None

    # 3. Load Prompts (system prompts with few-shot injection)
    g_sys = load_with_context(GEN_SYS_URI)
    c_sys = load_with_context(CRITIC_SYS_URI)
    r_sys = load_with_context(REFINER_SYS_URI)

    # Load User Templates (no few-shot injection needed)
    g_usr = mlflow.genai.load_prompt(GEN_USER_URI).template if GEN_USER_URI else None
    c_usr = (
        mlflow.genai.load_prompt(CRITIC_USER_URI).template if CRITIC_USER_URI else None
    )
    r_usr = (
        mlflow.genai.load_prompt(REFINER_USER_URI).template
        if REFINER_USER_URI
        else None
    )

    # Debug: Log prompts being used
    logger.debug(
        f"Using gen_sys prompt (first 200 chars): {g_sys[:200] if g_sys else 'None'}..."
    )
    logger.debug(f"Using g_usr template: {g_usr[:100] if g_usr else 'None'}...")
    logger.debug(f"Few shots count: {len(few_shots)}")

    try:
        # Note: few_shots are already injected into g_sys via load_with_context,
        # so we pass empty list to avoid double-injection in assemble_s1_system_prompt
        spans = asyncio.run(
            run_s1_discriminative(
                text,
                gen_prompt_override=g_sys,
                few_shots=[],  # Already injected into g_sys
                user_prompt_template_override=g_usr,
                critic_prompt_override=c_sys,
                critic_user_template_override=c_usr,
                refiner_prompt_override=r_sys,
                refiner_user_template_override=r_usr,
            )
        )

        # Debug: Log what the model returned
        logger.debug(f"Model returned {len(spans)} spans")
        if spans:
            logger.debug(f"Sample span: {spans[0]}")

        # Convert spans to dicts and LOCALIZE (calculate start/end indices)
        final_spans = []
        assigned_count = {}  # Track occurrences for duplicate spans

        for s in spans:
            if hasattr(s, "model_dump"):
                span_dict = s.model_dump()
            elif hasattr(s, "dict"):
                span_dict = s.dict()
            elif isinstance(s, dict):
                span_dict = s
            else:
                span_dict = {"text": str(s), "label": "Unknown"}

            # Normalize the label in the output
            if "label" in span_dict:
                span_dict["label"] = normalize_label(span_dict["label"])

            # --- LOCALIZE SPAN: Calculate start/end indices ---
            span_text = span_dict.get("text", "")
            if span_text and (
                span_dict.get("start") is None or span_dict.get("end") is None
            ):
                # Track which occurrence we're looking for
                key = (span_dict.get("label", ""), span_text.strip())
                nth = assigned_count.get(key, 0)

                # Find the span in the original text
                start, end = find_best_span(text, span_text, nth=nth)

                if start != -1:
                    span_dict["start"] = start
                    span_dict["end"] = end
                    # Use the actual text from the document (verbatim)
                    span_dict["text"] = text[start:end]
                    assigned_count[key] = nth + 1
                else:
                    # Span not found in text - mark as potential hallucination
                    logger.warning(f"Span not found in text: '{span_text[:50]}...'")
                    span_dict["start"] = -1
                    span_dict["end"] = -1

            final_spans.append(span_dict)

        logger.debug(
            f"Final spans to return: {final_spans[:2] if final_spans else 'empty'}"
        )

        return {
            "final_spans": final_spans,
            # --- THE TUNNEL EXIT ---
            # We explicitly return the gold data so the scorer can see it in 'outputs'
            "passthrough_gold_ref": passthrough_gold,
        }

    except Exception as e:
        logger.error(f"Prediction Wrapper Failed: {e}", exc_info=True)
        return {
            "final_spans": [{"text": "SYSTEM_CRASH", "label": "Error"}],
            "passthrough_gold_ref": passthrough_gold,
        }


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
    )
    parser.add_argument("--limit", type=int, default=20, help="Examples to use")
    parser.add_argument("--budget", type=int, default=100, help="Max metric calls")
    parser.add_argument(
        "--experiment", default="GEPA_S1_Optimization_V2", help="MLflow Experiment Name"
    )
    parser.add_argument(
        "--model-reflector",
        default="bedrock:/eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    parser.add_argument(
        "--phase",
        choices=[
            "all",
            "generator",
            "critic",
            "refiner",
            "gen-sys",
            "critic-sys",
            "refiner-sys",
        ],
        default="all",
        help="""Optimization phase - optimize ONE prompt at a time:
            'all'         = Optimize all 6 prompts (large search space)
            'generator'   = Optimize Generator system + user (2 prompts)
            'critic'      = Optimize Critic system + user (2 prompts)
            'refiner'     = Optimize Refiner system + user (2 prompts)
            'gen-sys'     = Optimize ONLY Generator system (1 prompt - recommended)
            'critic-sys'  = Optimize ONLY Critic system (1 prompt)
            'refiner-sys' = Optimize ONLY Refiner system (1 prompt)
        """,
    )
    args = parser.parse_args()

    # Logger Setup
    log_file = "optimization_s1.log"
    if os.path.exists(log_file):
        os.remove(log_file)
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    logger.add(log_file, level="DEBUG")

    # Init Global RAG
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

    # Load Data
    eval_dataset = load_eval_data(args.data, args.limit)

    # MLflow Setup
    mlflow.set_experiment(args.experiment)
    logger.info(f"Starting Run: {args.experiment}")

    global GEN_SYS_URI, GEN_USER_URI
    global CRITIC_SYS_URI, CRITIC_USER_URI
    global REFINER_SYS_URI, REFINER_USER_URI

    with mlflow.start_run():
        mlflow.log_params(vars(args))

        # Snapshot Baselines
        gen_template = build_s1_discriminative_system()
        critic_template = build_s1_critic_system()
        refiner_template = build_s1_refiner_system()
        mlflow.log_text(gen_template, "baseline_prompts/s1_generator.txt")
        mlflow.log_text(critic_template, "baseline_prompts/s1_critic.txt")
        mlflow.log_text(refiner_template, "baseline_prompts/s1_refiner.txt")

        # Register Prompts
        logger.info("Registering Prompts...")
        GEN_SYS_URI = mlflow.genai.register_prompt("s1_gen_sys", gen_template).uri
        GEN_USER_URI = mlflow.genai.register_prompt(
            "s1_gen_user", build_s1_user_template()
        ).uri
        CRITIC_SYS_URI = mlflow.genai.register_prompt(
            "s1_critic_sys", critic_template
        ).uri
        CRITIC_USER_URI = mlflow.genai.register_prompt(
            "s1_critic_user", build_s1_critic_user_template()
        ).uri
        REFINER_SYS_URI = mlflow.genai.register_prompt(
            "s1_refiner_sys", refiner_template
        ).uri
        REFINER_USER_URI = mlflow.genai.register_prompt(
            "s1_refiner_user", build_s1_refiner_user_template()
        ).uri
        logger.success("Baseline Prompts Registered.")

        # Build Prompt URI List Based on Phase
        if args.phase == "generator":
            prompt_uris_to_optimize = [GEN_SYS_URI, GEN_USER_URI]
            logger.info(
                "📌 Phase: GENERATOR - Optimizing 2 prompts (Generator sys + user)"
            )
        elif args.phase == "critic":
            prompt_uris_to_optimize = [CRITIC_SYS_URI, CRITIC_USER_URI]
            logger.info("📌 Phase: CRITIC - Optimizing 2 prompts (Critic sys + user)")
        elif args.phase == "refiner":
            prompt_uris_to_optimize = [REFINER_SYS_URI, REFINER_USER_URI]
            logger.info("📌 Phase: REFINER - Optimizing 2 prompts (Refiner sys + user)")
        elif args.phase == "gen-sys":
            prompt_uris_to_optimize = [GEN_SYS_URI]
            logger.info(
                "📌 Phase: GEN-SYS - Optimizing 1 prompt (Generator system only)"
            )
        elif args.phase == "critic-sys":
            prompt_uris_to_optimize = [CRITIC_SYS_URI]
            logger.info(
                "📌 Phase: CRITIC-SYS - Optimizing 1 prompt (Critic system only)"
            )
        elif args.phase == "refiner-sys":
            prompt_uris_to_optimize = [REFINER_SYS_URI]
            logger.info(
                "📌 Phase: REFINER-SYS - Optimizing 1 prompt (Refiner system only)"
            )
        else:  # "all"
            prompt_uris_to_optimize = [
                GEN_SYS_URI,
                GEN_USER_URI,
                CRITIC_SYS_URI,
                CRITIC_USER_URI,
                REFINER_SYS_URI,
                REFINER_USER_URI,
            ]
            logger.info("📌 Phase: ALL - Optimizing all 6 prompts (large search space)")

        logger.info(
            f"🎯 Optimizing {len(prompt_uris_to_optimize)} prompts with budget={args.budget}"
        )

        # Run Optimization
        logger.info("🚀 Launching GEPA...")
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
            scorers=[s1_rich_scorer],
        )

        logger.success("Optimization Complete!")
        if os.path.exists(log_file):
            mlflow.log_artifact(log_file)

    # Save Results
    output_dir = pathlib.Path("prompts/optimized_s1")
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

    name_to_filename = {
        "s1_gen_sys": "s1_generator_optimized.txt",
        "s1_gen_user": "s1_user_optimized.txt",
        "s1_critic_sys": "s1_critic_optimized.txt",
        "s1_critic_user": "s1_critic_user_optimized.txt",
        "s1_refiner_sys": "s1_refiner_optimized.txt",
        "s1_refiner_user": "s1_refiner_user_optimized.txt",
    }

    for prompt_obj in optimized_list:
        if prompt_obj.name in name_to_filename:
            fname = name_to_filename[prompt_obj.name]
            (output_dir / fname).write_text(prompt_obj.template, encoding="utf-8")
            logger.success(f"Saved optimized prompt '{prompt_obj.name}' to {fname}")


if __name__ == "__main__":
    main()
