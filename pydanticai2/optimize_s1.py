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

from anthropic import APIConnectionError
import litellm
import pandas as pd  # <--- NEW
from datetime import datetime

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from litellm.exceptions import RateLimitError, ServiceUnavailableError

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

# Import prompt loader
from pydanticai2.prompt_loader import S1_PROMPTS

# 1. Save the original function
_original_litellm_completion = litellm.completion


# 2. Define the retry logic (Expanded to catch Connection Errors)
@retry(
    retry=retry_if_exception_type(
        (
            RateLimitError,
            ServiceUnavailableError,
            APIConnectionError,  # <--- Added this for your specific error
            ConnectionError,
        )
    ),
    stop=stop_after_attempt(20),  # Try 20 times
    wait=wait_exponential(multiplier=2, min=5, max=60),  # Wait up to 60s
    reraise=True,
)
def _robust_completion(*args, **kwargs):
    """Wrapper that forces retries on RateLimit & Connection Errors"""
    return _original_litellm_completion(*args, **kwargs)


# 3. Apply the patch
litellm.completion = _robust_completion

# 4. Increase Timeout
os.environ["LITELLM_REQUEST_TIMEOUT"] = "120"


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
    run_s1_ddcot,  # NEW: DD-CoT pipeline
    get_rag_collection,
    retrieve_stratified_s1_reranked,
    find_best_span,  # For span localization
    format_s1_fewshots_to_markdown,
    run_s1_pattern_recognition,  # For few-shot formatting
    S1Deps,
)
from pydanticai2.prompt_builder import (
    # Legacy prompts
    build_s1_discriminative_system,
    build_s1_critic_system,
    build_s1_refiner_system,
    build_s1_user_template,
    build_s1_critic_user_template,
    build_s1_refiner_user_template,
    # DD-CoT prompts (NEW)
    build_s1_ddcot_system,
    build_s1_ddcot_user_template,
    build_s1_ddcot_critic_system,
    build_s1_ddcot_critic_user_template,
    build_s1_ddcot_refiner_system,
    build_s1_ddcot_refiner_user_template,
)

# -----------------------------------------------------------------------------
# 1. Setup & Data Loading (The Injection)
# -----------------------------------------------------------------------------

# Global Feedback Collector
FEEDBACK_LOG = []  # <--- NEW


def load_eval_data(path: str, limit: int = 20) -> List[Dict]:
    """
    Loads dataset, handling nested metadata structure.
    """
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

                # --- [FIX] ROBUST SPAN EXTRACTION ---
                # Check root first, then metadata
                gold_spans = row.get("spans")
                if not gold_spans:
                    gold_spans = row.get("markers")
                if not gold_spans:
                    gold_spans = row.get("metadata", {}).get("markers")
                if not gold_spans:
                    gold_spans = row.get("metadata", {}).get("spans")

                # Default to empty if nothing found
                if not isinstance(gold_spans, list):
                    gold_spans = []

                # --- FILTER LOGIC (Keep Positives) ---
                if len(gold_spans) == 0:
                    continue

                # Normalize keys (some data has 'startIndex', some has 'start')
                normalized_spans = []
                for s in gold_spans:
                    n = s.copy()
                    # Fix start/end keys
                    if "startIndex" in n:
                        n["start"] = n.pop("startIndex")
                    if "endIndex" in n:
                        n["end"] = n.pop("endIndex")
                    if "type" in n:
                        n["label"] = n.pop("type")
                    normalized_spans.append(n)

                payload = {
                    "gold_spans": normalized_spans,
                    "doc_id": row.get("doc_id", f"line_{line_num}"),
                }
                payload_str = json.dumps(payload)

                dataset.append(
                    {
                        "inputs": {
                            "text": row["text"],
                            "passthrough_gold": payload_str,
                        },
                        "outputs": {"dummy_target": "ignore_me"},
                    }
                )

                if len(dataset) >= limit:
                    break

            except Exception as e:
                logger.warning(f"Skipping bad line {line_num}: {e}")

    logger.success(f"DATASET LOADED: {len(dataset)} valid examples.")
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
    boundary_warnings: list,
    hallucinated_not_in_text: list,  # Added to integrate into structure
    doc_id: str,
) -> str:
    """
    Generate structured, numbered feedback with positive reinforcement.
    Format:
    1. SUCCESS (Positive Reinforcement)
    2. CRITICAL (Logic Errors)
    3. REFINEMENT (Boundaries)
    4. RECALL (Missed)
    5. NOISE (Hallucinations)
    """
    sections = []
    idx = 1

    # --- 1. Positive Reinforcement (Anchor) ---
    # Crucial for GEPA: Tells the optimizer what NOT to change
    if gold_matched:
        good_examples = []
        for i in list(gold_matched):
            g = gold_spans[i]
            good_examples.append(f"'{g.get('text', '')}'")

        msg = f"KEEP DOING: Correctly extracted {len(gold_matched)}/{len(gold_spans)} spans (e.g., {', '.join(good_examples)})."
        sections.append(f"{idx}. {msg}")
        idx += 1

    # --- 2. Logic Errors (High Impact) ---
    if label_errors:
        sections.append(f"{idx}. FIX LABELS: {'; '.join(label_errors)}")
        idx += 1

    # --- 3. Boundary Issues (Precision) ---
    if boundary_warnings:
        sections.append(f"{idx}. TIGHTEN BOUNDARIES: {'; '.join(boundary_warnings)}")
        idx += 1

    # --- 4. Missed Spans (Recall) ---
    missed_indices = [i for i in range(len(gold_spans)) if i not in gold_matched]
    if missed_indices:
        missed_details = []
        for i in missed_indices:
            g = gold_spans[i]
            g_text = g.get("text", "")
            g_label = g.get("label", "?")
            missed_details.append(f"{g_label}:'{g_text}'")
        sections.append(f"{idx}. EXTRACT MISSING: {'; '.join(missed_details)}")
        idx += 1

    # --- 5. Hallucinations (Noise) ---
    halluc_indices = [i for i in range(len(pred_spans)) if i not in pred_matched]
    noise_msgs = []

    # A. Spans not in text (Fabrication)
    if hallucinated_not_in_text:
        noise_msgs.append(f"NOT IN TEXT: {'; '.join(hallucinated_not_in_text)}")

    # B. Spans in text but irrelevant (False Positive)
    if halluc_indices:
        fps = []
        for i in halluc_indices:
            p = pred_spans[i]
            if len(p.get("text", "")) > 3:
                fps.append(f"'{p.get('text', '')}'")
        if fps:
            noise_msgs.append(f"IRRELEVANT: {', '.join(fps)}")

    if noise_msgs:
        sections.append(f"{idx}. REMOVE NOISE: {' | '.join(noise_msgs)}")

    if not sections:
        return "PERFECT: All spans correctly extracted and labeled."

    return "\n".join(sections)


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
        # CASE: The model stayed silent, but shouldn't have.
        # This is the specific failure mode for "Professional" texts.
        snippet = outputs.get("passthrough_gold_ref", "")[:100]  # Context hint

        # [FIX] Now actually using the snippet in the log
        logger.warning(
            f"[{doc_id}] ⚠️ SILENT FAILURE (Recall 0.0) | "
            f"Gold had {len(gold_spans)} markers, Model found 0.\n"
            f"Context Snippet: '{snippet}...'\n"
            f"Check if Guardrails are too strict."
        )

        expected = [f"{g.get('label', '?')}:'{g.get('text', '')}'" for g in gold_spans]
        return Feedback(
            value=0.0,
            rationale=f"EXTRACT MISSING: {'; '.join(expected)}. The model found nothing.",
        )

    if pred_spans and not gold_spans:
        logger.warning(
            f"[{doc_id}] ⚠️ HALLUCINATION (Precision 0.0) | "
            f"Model found {len(pred_spans)} markers in a clean text."
        )
        halluc = [f"'{p.get('text', '')}'" for p in pred_spans]
        return Feedback(value=0.0, rationale=f"REMOVE ALL: {'; '.join(halluc)}.")

    # ---------------------------------------------------------
    # MATCHING ALGORITHM (Enhanced with Position-Based Scoring)
    # ---------------------------------------------------------

    gold_matched = set()
    pred_matched = set()

    exact_matches = 0
    partial_score_sum = 0.0
    label_errors = []
    boundary_warnings = []

    valid_pred_spans = [p for p in pred_spans if p.get("start", 0) != -1]
    # Collect these for the feedback generator
    hallucinated_not_in_text = [
        p.get("text", "") for p in pred_spans if p.get("start", 0) == -1
    ]

    for p_idx, p in enumerate(valid_pred_spans):
        p_text = p.get("text", "")
        p_lbl = normalize_label(p.get("label", ""))
        p_pos = (p.get("start"), p.get("end")) if p.get("start") is not None else None

        best_score = 0.0
        best_g_idx = -1

        for g_idx, g in enumerate(gold_spans):
            if g_idx in gold_matched:
                continue
            g_text = g.get("text", "")
            g_pos = (
                (g.get("start"), g.get("end")) if g.get("start") is not None else None
            )

            score = compute_overlap_score(p_text, g_text, p_pos, g_pos)
            if score > best_score:
                best_score = score
                best_g_idx = g_idx

        # Threshold: 0.4 for any credit
        if best_score >= 0.4 and best_g_idx >= 0:
            g_target = gold_spans[best_g_idx]
            g_lbl = normalize_label(g_target.get("label", ""))

            gold_matched.add(best_g_idx)
            pred_matched.add(p_idx)

            if p_lbl == g_lbl:
                if best_score > 0.9:
                    exact_matches += 1
                else:
                    # Boundary Error (Partial Credit)
                    partial_score_sum += best_score
                    boundary_warnings.append(
                        f"'{p_text}' should be '{g_target.get('text','')[:20]}'"
                    )
            else:
                # Label Error (Penalized Credit)
                partial_score_sum += best_score * 0.5
                label_errors.append(f"'{p_text}' (Is: {p_lbl}, Should be: {g_lbl})")

    # ---------------------------------------------------------
    # SCORING (Weighted F-beta)
    # ---------------------------------------------------------

    fn = len(gold_spans) - len(gold_matched)
    fp = (len(valid_pred_spans) - len(pred_matched)) + len(hallucinated_not_in_text)

    tp_effective = exact_matches + partial_score_sum

    total_positive = tp_effective + fp
    precision = tp_effective / total_positive if total_positive > 0 else 0

    total_relevant = tp_effective + fn
    recall = tp_effective / total_relevant if total_relevant > 0 else 0

    beta = 2.0
    f_beta = (
        (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
        if (precision + recall) > 0
        else 0
    )

    # ---------------------------------------------------------
    # STRUCTURED FEEDBACK GENERATION
    # ---------------------------------------------------------

    rationale = generate_actionable_feedback(
        gold_spans,
        valid_pred_spans,
        gold_matched,
        pred_matched,
        label_errors,
        boundary_warnings,
        hallucinated_not_in_text,  # Pass this explicitly now
        doc_id,
    )

    # [NEW] Collect Feedback for Analysis
    FEEDBACK_LOG.append(
        {
            "timestamp": datetime.now().isoformat(),
            "doc_id": doc_id,
            "score": float(f_beta),
            "rationale": rationale,
            "n_gold": len(gold_spans),
            "n_pred": len(pred_spans),
        }
    )

    logger.info(f"[{doc_id}] S1 Score: {f_beta:.2f} |\n{rationale}")
    return Feedback(value=float(f_beta), rationale=rationale)


# -----------------------------------------------------------------------------
# 3. Prediction Wrapper (The Tunnel)
# -----------------------------------------------------------------------------

# Legacy prompt URIs
GEN_SYS_URI, GEN_USER_URI = None, None
CRITIC_SYS_URI, CRITIC_USER_URI = None, None
REFINER_SYS_URI, REFINER_USER_URI = None, None

# DD-CoT prompt URIs (NEW)
PAT_SYS_URI, PAT_USER_URI = None, None
DDCOT_CRITIC_SYS_URI, DDCOT_CRITIC_USER_URI = None, None
DDCOT_REFINER_SYS_URI, DDCOT_REFINER_USER_URI = None, None

S1_RAG_COLLECTION = None
USE_DDCOT_MODE = True  # Flag to switch between legacy and DD-CoT


def predict_wrapper(
    text: str,
    passthrough_gold: str = "{}",
    # Pattern Key
    s1_pattern_sys: str = None,
    s1_pattern_user: str = None,
):
    """
    Unified prediction wrapper that accepts GEPA-mutated prompts.
    """
    # 1. Retrieve RAG
    few_shots = []
    if S1_RAG_COLLECTION:
        few_shots = retrieve_stratified_s1_reranked(S1_RAG_COLLECTION, text, k_total=3)
    few_shots_str = format_s1_fewshots_to_markdown(few_shots)

    # 2. Helper to Hydrate Prompts (Opt String > Static URI)
    def resolve(opt_str, static_uri):
        if opt_str:
            # GEPA passed a mutated string. Inject variables manually.
            return opt_str.replace("{{few_shot_examples}}", few_shots_str)
        elif static_uri:
            # Fallback to static prompt from MLflow
            try:
                tmpl = mlflow.genai.load_prompt(static_uri).template
                return tmpl.replace("{{few_shot_examples}}", few_shots_str)
            except:
                return None
        return None

    # 3. Setup Dependencies
    deps = S1Deps(raw_text=text, few_shots=few_shots, doc_id="eval_sample")

    try:
        # Resolve Pattern Prompt (Primary Target)
        pat_sys_prompt = resolve(s1_pattern_sys, PAT_SYS_URI)

        pat_user_prompt = resolve(s1_pattern_user, PAT_USER_URI)

        # Run Pattern Recognition Agent
        # Note: We prioritize the Pattern Agent if PAT_SYS_URI is active or passed
        if pat_sys_prompt or s1_pattern_sys:
            logger.debug("[S1 Pattern] Running Optimized Pattern Agent...")
            extraction_result = asyncio.run(
                run_s1_pattern_recognition(
                    text=text,
                    deps=deps,
                    temperature=0.0,
                    system_prompt_override=pat_sys_prompt,  # INJECTED HERE
                    user_prompt_template_override=pat_user_prompt,  # INJECTED HERE
                )
            )

            if isinstance(extraction_result, tuple):
                extraction_result = extraction_result[0]

            raw_spans = extraction_result.extractions

        else:
            # Run Legacy DD-CoT Pipeline with Overrides
            raw_spans = asyncio.run(
                run_s1_ddcot(
                    text,
                    few_shots=[],  # Already injected
                    gen_prompt_override=g_sys,
                    gen_user_template_override=g_usr,
                    critic_prompt_override=c_sys,
                    critic_user_template_override=c_usr,
                    refiner_prompt_override=r_sys,
                    refiner_user_template_override=r_usr,
                )
            )

        # Handle tuple return if usage is enabled in agent
        if isinstance(extraction_result, tuple):
            extraction_result = extraction_result[0]

        # 4. Extract Spans
        raw_spans = extraction_result.extractions

        # 5. Deterministic Verification (The "Map-Reduce" Engine)
        # This calculates exact start/end indices and drops hallucinations
        final_spans = []
        assigned_count = {}  # Track nth occurrence for duplicates

        for span in raw_spans:
            # Normalize label
            label = normalize_label(span.label)
            span_text = span.text.strip()

            if not span_text:
                continue

            # Track occurrences
            key = (label, span_text)
            nth = assigned_count.get(key, 0)

            # Find best span logic
            start, end = find_best_span(text, span_text, nth=nth)

            if start != -1:
                final_spans.append(
                    {
                        "label": label,
                        "text": text[start:end],  # Use verbatim source text
                        "start": start,
                        "end": end,
                        "why": span.why_this_label,  # Preserve reasoning
                    }
                )
                assigned_count[key] = nth + 1
            else:
                logger.warning(f"Span not found in text: '{span_text}'")

        logger.debug(f"S1 Extracted {len(final_spans)} valid markers.")

        return {
            "final_spans": final_spans,
            # --- THE TUNNEL EXIT ---
            # We explicitly return the gold data so the scorer can see it in 'outputs'
            "passthrough_gold_ref": passthrough_gold,
        }

    except Exception as e:
        logger.error(f"Prediction Wrapper Failed: {e}", exc_info=True)
        return {
            "final_spans": [],  # Fail safe: Return empty list, not crash
            "passthrough_gold_ref": passthrough_gold,
        }


# -----------------------------------------------------------------------------
# 4. Main Execution
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="GEPA Prompt Optimization Runner")
    parser.add_argument(
        "--data",
        default="data/raw/dev_ready_for_pipeline.jsonl",
        help="Path to eval data",
    )
    parser.add_argument(
        "--rag-dir", default="data/rag_openai_contrastive", help="Path to ChromaDB RAG"
    )
    parser.add_argument("--limit", type=int, default=10, help="Examples to use")
    parser.add_argument("--budget", type=int, default=30, help="Max metric calls")
    parser.add_argument(
        "--experiment",
        default="GEPA_S1_Optimization_V2_OPENAI",
        help="MLflow Experiment Name",
    )
    parser.add_argument(
        "--model-reflector",
        default="openai:/gpt-5.2",
    )
    parser.add_argument(
        "--phase",
        choices=[
            # ===== LEGACY PHASES =====
            "all",
            "generator",
            "critic",
            "refiner",
            "gen-sys",
            "critic-sys",
            "refiner-sys",
            # ===== DD-CoT PHASES (Optimal Architecture) =====
            "ddcot-all",
            "ddcot-gen",
            "ddcot-critic",
            "pattern-sys",
            "ddcot-refiner",
            "pattern-user",
            "ddcot-gen-sys",
            "ddcot-critic-sys",
            "ddcot-refiner-sys",
        ],
        default="ddcot-gen-sys",
        help="""Optimization phase - optimize ONE prompt at a time:
            === LEGACY PHASES ===
            'all'             = Optimize all 6 legacy prompts
            'generator'       = Optimize Legacy Generator (sys + user)
            'critic'          = Optimize Legacy Critic (sys + user)
            'refiner'         = Optimize Legacy Refiner (sys + user)
            'gen-sys'         = Optimize ONLY Legacy Generator system
            'critic-sys'      = Optimize ONLY Legacy Critic system
            'refiner-sys'     = Optimize ONLY Legacy Refiner system
            
            === DD-CoT PHASES (Recommended) ===
            'ddcot-all'       = Optimize all 6 DD-CoT prompts
            'ddcot-gen'       = Optimize DD-CoT Generator (sys + user)
            'ddcot-critic'    = Optimize DD-CoT Critic (sys + user)
            'ddcot-refiner'   = Optimize DD-CoT Refiner (sys + user)
            'pattern-sys'     = Optimize ONLY Pattern system (RECOMMENDED)
            'pattern-user'    = Optimize ONLY Pattern user template
            'ddcot-critic-sys'= Optimize ONLY DD-CoT Critic system
            'ddcot-refiner-sys'= Optimize ONLY DD-CoT Refiner system
        """,
    )
    args = parser.parse_args()

    # Logger Setup
    log_file = f"optimization_s1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
            S1_RAG_COLLECTION = get_rag_collection(args.rag_dir, "s1_patterns")
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
    global PAT_USER_URI, PAT_SYS_URI
    global DDCOT_CRITIC_SYS_URI, DDCOT_CRITIC_USER_URI
    global DDCOT_REFINER_SYS_URI, DDCOT_REFINER_USER_URI
    global USE_DDCOT_MODE

    # Determine mode based on phase
    USE_DDCOT_MODE = args.phase.startswith("ddcot")
    USE_PATTERN_MODE = args.phase == "pattern-sys"

    with mlflow.start_run():
        mlflow.log_params(vars(args))

        # Snapshot Baselines using S1Prompts
        s1_prompts = S1_PROMPTS
        mlflow.log_text(s1_prompts.gen_system, "baseline_prompts/s1_generator.txt")
        mlflow.log_text(s1_prompts.critic_system, "baseline_prompts/s1_critic.txt")
        mlflow.log_text(s1_prompts.refiner_system, "baseline_prompts/s1_refiner.txt")

        # Register Prompts
        logger.info("Registering Prompts...")
        # GEN_SYS_URI = mlflow.genai.register_prompt(
        #    "s1_gen_sys", s1_prompts.gen_system
        # ).uri
        # GEN_USER_URI = mlflow.genai.register_prompt(
        #    "s1_gen_user", s1_prompts.gen_user_template
        # ).uri
        # CRITIC_SYS_URI = mlflow.genai.register_prompt(
        #    "s1_critic_sys", s1_prompts.critic_system
        # ).uri
        # CRITIC_USER_URI = mlflow.genai.register_prompt(
        #    "s1_critic_user", s1_prompts.critic_user_template
        # ).uri
        # REFINER_SYS_URI = mlflow.genai.register_prompt(
        #    "s1_refiner_sys", s1_prompts.refiner_system
        # ).uri
        # REFINER_USER_URI = mlflow.genai.register_prompt(
        #    "s1_refiner_user", s1_prompts.refiner_user_template
        # ).uri
        # logger.success("Legacy Baseline Prompts Registered.")

        # ===== DD-CoT Prompts Registration =====
        logger.info("Registering DD-CoT Prompts...")
        mlflow.log_text(
            s1_prompts.ddcot_gen_system, "baseline_prompts/s1_ddcot_generator.txt"
        )
        mlflow.log_text(
            s1_prompts.ddcot_critic_system, "baseline_prompts/s1_ddcot_critic.txt"
        )
        mlflow.log_text(
            s1_prompts.ddcot_refiner_system, "baseline_prompts/s1_ddcot_refiner.txt"
        )

        # DDCOT_GEN_SYS_URI = mlflow.genai.register_prompt(
        #    "s1_ddcot_gen_sys", s1_prompts.ddcot_gen_system
        # ).uri
        # DDCOT_GEN_USER_URI = mlflow.genai.register_prompt(
        #    "s1_ddcot_gen_user", s1_prompts.ddcot_gen_user_template
        # ).uri
        # DDCOT_CRITIC_SYS_URI = mlflow.genai.register_prompt(
        #    "s1_ddcot_critic_sys", s1_prompts.ddcot_critic_system
        # ).uri
        # DDCOT_CRITIC_USER_URI = mlflow.genai.register_prompt(
        #    "s1_ddcot_critic_user", s1_prompts.ddcot_critic_user_template
        # ).uri
        # DDCOT_REFINER_SYS_URI = mlflow.genai.register_prompt(
        #    "s1_ddcot_refiner_sys", s1_prompts.ddcot_refiner_system
        # ).uri
        # DDCOT_REFINER_USER_URI = mlflow.genai.register_prompt(
        #    "s1_ddcot_refiner_user", s1_prompts.ddcot_refiner_user_template
        # ).uri
        # logger.success("DD-CoT Prompts Registered.")

        PAT_SYS_URI = mlflow.genai.register_prompt(
            "s1_pat_sys", s1_prompts.pat_system
        ).uri
        PAT_USER_URI = mlflow.genai.register_prompt(
            "s1_pat_user", s1_prompts.pat_user_template
        ).uri

        # Build Prompt URI List Based on Phase
        # ===== LEGACY PHASES =====
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
        # ===== DD-CoT PHASES (Optimal Architecture) =====
        elif args.phase == "ddcot-all":
            prompt_uris_to_optimize = [
                DDCOT_CRITIC_SYS_URI,
                DDCOT_CRITIC_USER_URI,
                DDCOT_REFINER_SYS_URI,
                DDCOT_REFINER_USER_URI,
            ]
            logger.info("🎯 Phase: DDCOT-ALL - Optimizing all 6 DD-CoT prompts")
        elif args.phase == "ddcot-critic":
            prompt_uris_to_optimize = [DDCOT_CRITIC_SYS_URI, DDCOT_CRITIC_USER_URI]
            logger.info(
                "🎯 Phase: DDCOT-CRITIC - Optimizing DD-CoT Critic (sys + user)"
            )
        elif args.phase == "ddcot-refiner":
            prompt_uris_to_optimize = [DDCOT_REFINER_SYS_URI, DDCOT_REFINER_USER_URI]
            logger.info(
                "🎯 Phase: DDCOT-REFINER - Optimizing DD-CoT Refiner (sys + user)"
            )
        elif args.phase == "pattern-sys":
            prompt_uris_to_optimize = [PAT_SYS_URI]
            logger.info(
                "🎯 Phase: PATTERN-SYS - Optimizing Pattern system (RECOMMENDED)"
            )
        elif args.phase == "pattern-user":
            prompt_uris_to_optimize = [PAT_USER_URI]
            logger.info("🎯 Phase: PATTERN-USER - Optimizing Pattern user template")
        elif args.phase == "ddcot-critic-sys":
            prompt_uris_to_optimize = [DDCOT_CRITIC_SYS_URI]
            logger.info("🎯 Phase: DDCOT-CRITIC-SYS - Optimizing DD-CoT Critic system")
        elif args.phase == "ddcot-refiner-sys":
            prompt_uris_to_optimize = [DDCOT_REFINER_SYS_URI]
            logger.info(
                "🎯 Phase: DDCOT-REFINER-SYS - Optimizing DD-CoT Refiner system"
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

        # [NEW] Save Automated Feedback Analysis
        if FEEDBACK_LOG:
            csv_path = "feedback_history.csv"
            pd.DataFrame(FEEDBACK_LOG).to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path)
            logger.success(
                f"Feedback Analysis saved to {csv_path} ({len(FEEDBACK_LOG)} records)"
            )

        logger.success("Optimization Complete!")
        if os.path.exists(log_file):
            mlflow.log_artifact(log_file)

    # Save Results
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

    name_to_filename = {
        "s1_pat_sys": "s1_pattern_generator.txt",
        "s1_pat_user": "s1_pattern_user.txt",
    }

    for prompt_obj in optimized_list:
        if prompt_obj.name in name_to_filename:
            fname = name_to_filename[prompt_obj.name]
            (output_dir / fname).write_text(prompt_obj.template, encoding="utf-8")
            logger.success(f"Saved optimized prompt '{prompt_obj.name}' to {fname}")


if __name__ == "__main__":
    main()
