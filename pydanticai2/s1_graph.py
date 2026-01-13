#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s1_graph.py — LangGraph Workflows for S1 Span Extraction.

This module provides TWO graph architectures:
1. LEGACY: Self-Consistency Ensemble (k=3) → Voting → Verifier
2. OPTIMAL: DD-CoT Generator → Critic → Refiner → Verifier

Use `s1_ddcot_graph` for optimal performance (recommended).
Use `s1_ensemble_graph` for legacy compatibility.
"""

import asyncio
import json
from collections import Counter, defaultdict
from typing import TypedDict, List, Dict, Optional
from loguru import logger
from langgraph.graph import StateGraph, END, START
from typing import Annotated  # Import Annotated
import operator


# Import agents and utilities
from psycomark_agents import (
    run_s1_discriminative,
    run_s1_ddcot,
    S1Span,
    DDCoTSpan,
    DDCoTExtraction,
    EnhancedS1Critique,
    DDCoTRefinement,
    find_best_span,
    deduplicate_overlapping_spans,
    verify_span_boundaries,
    get_s1_ddcot_critic,
    get_s1_ddcot_refiner,
    S1Deps,
    safe_agent_run,
)


# ===========================================================================
# SHARED STATE DEFINITIONS
# ===========================================================================


# 1. Define a reducer to sum dictionaries
def aggregate_usage(left: Dict[str, int], right: Dict[str, int]) -> Dict[str, int]:
    return {
        "input_tokens": left.get("input_tokens", 0) + right.get("input_tokens", 0),
        "output_tokens": left.get("output_tokens", 0) + right.get("output_tokens", 0),
        "total_tokens": left.get("total_tokens", 0) + right.get("total_tokens", 0),
    }


class S1GraphState(TypedDict):
    """State for legacy ensemble graph."""

    doc_id: str
    text: str  # The raw document
    few_shots: List[dict]  # Context for the agent
    k: int  # Ensemble Size (default 3)
    raw_runs: List[List[S1Span]]  # Output from k agents
    consensus_spans: List[S1Span]  # Spans that passed the vote
    final_spans: List[Dict]  # Final spans with start/end indices

    token_usage: Annotated[Dict[str, int], aggregate_usage]


class S1DDCoTGraphState(TypedDict):
    """State for DD-CoT optimal graph."""

    doc_id: str
    text: str  # The raw document
    few_shots: List[dict]  # Dynamic few-shot examples from RAG

    # DD-CoT Generator output
    text_complexity: str  # "simple" | "moderate" | "complex"
    dominant_narrative: str  # "conspiracy" | "neutral" | "debunking" | "mixed"
    draft_extractions: List[DDCoTSpan]  # Initial extraction with reasoning

    # Enhanced Critic output
    critique: Optional[EnhancedS1Critique]  # Structured feedback
    requires_refinement: bool

    # Refiner output
    refined_extractions: List[DDCoTSpan]  # After refinement

    # Final output
    final_spans: List[Dict]  # Final spans with start/end indices

    token_usage: Annotated[Dict[str, int], aggregate_usage]


# ===========================================================================
# LEGACY ENSEMBLE GRAPH (for backward compatibility)
# ===========================================================================


async def s1_ensemble_node(state: S1GraphState) -> Dict:
    """Runs the discriminative agent k=3 times in parallel."""
    text = state["text"]
    k = state.get("k", 3)

    logger.info(f"[{state['doc_id']}] Starting S1 Ensemble (k={k})...")

    tasks = [run_s1_ddcot(text, state.get("few_shots", [])) for _ in range(k)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    valid_runs = []
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            logger.error(f"[{state['doc_id']}] Run {i} failed: {res}")
            valid_runs.append([])
        else:
            valid_runs.append(res)

    return {"raw_runs": valid_runs}


def _normalize_key(text: str) -> str:
    """Normalizes text for voting purposes."""
    t = text.lower().strip()
    for prefix in ["the ", "a ", "an "]:
        if t.startswith(prefix):
            return t[len(prefix) :]
    return t


def s1_consensus_node(state: S1GraphState) -> Dict:
    """Filters spans that didn't appear in at least 2 runs."""
    runs = state["raw_runs"]
    if not runs:
        return {"consensus_spans": []}

    vote_counter = Counter()
    best_span_map = {}

    for run in runs:
        seen_in_run = set()
        for span in run:
            norm_text = _normalize_key(span.text)
            key = (span.label, norm_text)

            if key not in seen_in_run:
                vote_counter[key] += 1
                seen_in_run.add(key)

                if key not in best_span_map or len(span.text) > len(
                    best_span_map[key].text
                ):
                    best_span_map[key] = span

    k_size = len(runs)
    threshold = 2 if k_size >= 3 else 1

    passed_spans = [
        best_span_map[key] for key, count in vote_counter.items() if count >= threshold
    ]

    logger.info(
        f"[{state['doc_id']}] Consensus: {len(passed_spans)} spans passed out of {len(vote_counter)} candidates."
    )
    return {"consensus_spans": passed_spans}


def s1_structure_verifier_node(state: S1GraphState) -> Dict:
    """
    Maps text strings back to (start, end) indices in the raw text.

    Enhanced with:
    - Overlap deduplication (removes subset spans)
    - Boundary verification (ensures word-aligned spans)
    - 5-strategy span location
    """
    raw_text = state["text"]
    candidates = state.get("consensus_spans", [])
    doc_id = state.get("doc_id", "unknown")

    located_spans = []
    assigned_count = defaultdict(int)

    for span in candidates:
        # Handle both Pydantic models and dicts
        if hasattr(span, "text"):
            snippet = span.text
            label = span.label
            why = getattr(span, "why", None)
        elif isinstance(span, dict):
            # Dict fallback
            snippet = span.get("text", "")
            label = span.get("label", "Unknown")
            why = span.get("why", None)
        else:
            # Fallback for unknown types
            snippet = str(span)
            label = "Unknown"
            why = None

        key = (label, snippet.strip())
        nth = assigned_count[key]

        start, end = find_best_span(raw_text, snippet, nth=nth)

        if start == -1:
            start, end = find_best_span(raw_text, snippet, nth=0)

        if start != -1:
            located_spans.append(
                {
                    "label": label.value if hasattr(label, "value") else label,
                    "text": raw_text[start:end],
                    "start": start,
                    "end": end,
                    "why": why,
                }
            )
            assigned_count[key] += 1
        else:
            logger.warning(f"[{doc_id}] Dropped phantom span: '{snippet}'")

    # Apply post-processing improvements
    verified = verify_span_boundaries(located_spans, raw_text)
    deduped = deduplicate_overlapping_spans(verified, same_label_only=True)
    final_output = sorted(deduped, key=lambda x: x["start"])

    logger.info(
        f"[{doc_id}] Verifier: {len(candidates)} -> {len(final_output)} spans (after dedup)"
    )
    return {"final_spans": final_output}


# ===========================================================================
# DD-CoT OPTIMAL GRAPH
# ===========================================================================


async def s1_ddcot_generator_node(state: S1DDCoTGraphState) -> Dict:
    """
    DD-CoT Generator: Extracts spans with discriminative reasoning.
    """
    text = state["text"]
    few_shots = state.get("few_shots", [])
    doc_id = state.get("doc_id", "unknown")

    logger.info(f"[{doc_id}] Starting DD-CoT Generator...")

    try:
        # Run DD-CoT generator (skip critic for this node - we do it separately)
        result, usage_dict = await run_s1_ddcot(
            text=text,
            few_shots=few_shots,
            skip_critic=True,  # Only run generator
            return_full_extraction=True,
            return_usage=True,  # <--- Request usage stats
        )

        if isinstance(result, DDCoTExtraction):
            logger.info(
                f"[{doc_id}] DD-CoT Generator: {len(result.extractions)} spans, "
                f"complexity={result.text_complexity}, narrative={result.dominant_narrative}"
            )
            return {
                "text_complexity": result.text_complexity,
                "dominant_narrative": result.dominant_narrative,
                "draft_extractions": result.extractions,
                "token_usage": usage_dict,  # <--- Update state
            }
        else:
            # Fallback if result is a list
            logger.warning(
                f"[{doc_id}] DD-CoT returned list instead of DDCoTExtraction"
            )
            return {
                "text_complexity": "unknown",
                "dominant_narrative": "unknown",
                "draft_extractions": [],
                "token_usage": usage_dict,
            }

    except Exception as e:
        logger.error(f"[{doc_id}] DD-CoT Generator failed: {e}")
        import traceback

        traceback.print_exc()
        return {
            "text_complexity": "error",
            "dominant_narrative": "error",
            "draft_extractions": [],
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }


async def s1_ddcot_critic_node(state: S1DDCoTGraphState) -> Dict:
    """
    Enhanced Critic: Checks for verbatim, granularity, label, and exhaustiveness errors.
    """
    from pydanticai2.prompt_builder import build_s1_ddcot_critic_user_template

    text = state["text"]
    draft_extractions = state.get("draft_extractions", [])
    doc_id = state.get("doc_id", "unknown")

    if not draft_extractions:
        logger.warning(f"[{doc_id}] No draft extractions to critique")
        return {"critique": None, "requires_refinement": False}

    logger.info(f"[{doc_id}] Running Enhanced Critic...")

    try:
        critic_agent = get_s1_ddcot_critic()
        deps = S1Deps(raw_text=text, few_shots=state.get("few_shots", []))

        draft_json_str = json.dumps(
            [s.model_dump() for s in draft_extractions], indent=2
        )

        c_tmpl = build_s1_ddcot_critic_user_template()
        critique_user_msg = (
            c_tmpl.replace("{{text}}", text)
            .replace("{{draft_json}}", draft_json_str)
            .replace("{{complexity}}", state.get("text_complexity", "unknown"))
            .replace("{{narrative}}", state.get("dominant_narrative", "unknown"))
        )

        critique_res = await safe_agent_run(critic_agent, critique_user_msg, deps)
        critique: EnhancedS1Critique = critique_res.output

        # [FIX] Capture Usage
        usage = critique_res.usage()
        usage_dict = {
            "input_tokens": usage.request_tokens or 0,
            "output_tokens": usage.response_tokens or 0,
            "total_tokens": usage.total_tokens or 0,
        }

        n_errors = (
            len(critique.verbatim_errors)
            + len(critique.granularity_errors)
            + len(critique.label_errors)
            + len(critique.missed_spans)
        )
        logger.info(
            f"[{doc_id}] Critic: {n_errors} issues, requires_refinement={critique.requires_refinement}"
        )

        return {
            "critique": critique,
            "requires_refinement": critique.requires_refinement,
            "token_usage": usage_dict,  # <--- Accumulates usage
        }

    except Exception as e:
        logger.error(f"[{doc_id}] Critic failed: {e}")
        import traceback

        traceback.print_exc()
        return {"critique": None, "requires_refinement": False}


async def s1_ddcot_refiner_node(state: S1DDCoTGraphState) -> Dict:
    """
    DD-CoT Refiner: Applies critique feedback while maintaining discriminative reasoning.
    """
    from pydanticai2.prompt_builder import build_s1_ddcot_refiner_user_template

    text = state["text"]
    draft_extractions = state.get("draft_extractions", [])
    critique = state.get("critique")
    doc_id = state.get("doc_id", "unknown")

    # If no refinement needed, pass through
    if not state.get("requires_refinement", False) or critique is None:
        logger.info(f"[{doc_id}] No refinement needed, passing through draft")
        return {"refined_extractions": draft_extractions}

    logger.info(f"[{doc_id}] Running DD-CoT Refiner...")

    try:
        refiner_agent = get_s1_ddcot_refiner()
        deps = S1Deps(raw_text=text, few_shots=state.get("few_shots", []))

        draft_json_str = json.dumps(
            [s.model_dump() for s in draft_extractions], indent=2
        )
        critique_json_str = json.dumps(critique.model_dump(), indent=2)

        r_tmpl = build_s1_ddcot_refiner_user_template()
        refine_user_msg = (
            r_tmpl.replace("{{text}}", text)
            .replace("{{draft_json}}", draft_json_str)
            .replace("{{critique_json}}", critique_json_str)
        )

        refine_res = await safe_agent_run(refiner_agent, refine_user_msg, deps)
        refinement: DDCoTRefinement = refine_res.output

        # [FIX] Capture Usage
        usage = refine_res.usage()
        usage_dict = {
            "input_tokens": usage.request_tokens or 0,
            "output_tokens": usage.response_tokens or 0,
            "total_tokens": usage.total_tokens or 0,
        }

        logger.info(
            f"[{doc_id}] Refiner: {len(refinement.refined_extractions)} spans, "
            f"{len(refinement.fixes_applied)} fixes applied"
        )

        return {
            "refined_extractions": refinement.refined_extractions,
            "token_usage": usage_dict,
        }

    except Exception as e:
        logger.error(f"[{doc_id}] Refiner failed: {e}")
        import traceback

        traceback.print_exc()
        return {"refined_extractions": draft_extractions}


def s1_ddcot_verifier_node(state: S1DDCoTGraphState) -> Dict:
    """
    Structure Verifier for DD-CoT: Maps spans to (start, end) indices.

    Enhanced with:
    - Overlap deduplication (removes subset spans)
    - Boundary verification (ensures word-aligned spans)
    - 5-strategy span location (exact -> case-insensitive -> normalized -> fuzzy -> alignment)
    """
    raw_text = state["text"]
    doc_id = state.get("doc_id", "unknown")

    # Use refined if available, else draft
    candidates = state.get("refined_extractions") or state.get("draft_extractions", [])

    located_spans = []
    assigned_count = defaultdict(int)

    for span in candidates:
        snippet = span.text
        label = span.label
        why = getattr(span, "why_this_label", None)

        key = (label, snippet.strip())
        nth = assigned_count[key]

        start, end = find_best_span(raw_text, snippet, nth=nth)
        if start == -1:
            start, end = find_best_span(raw_text, snippet, nth=0)

        if start != -1:
            located_spans.append(
                {
                    "label": label.value if hasattr(label, "value") else label,
                    "text": raw_text[start:end],
                    "start": start,
                    "end": end,
                    "why": why,
                }
            )
            assigned_count[key] += 1
        else:
            logger.warning(f"[{doc_id}] Dropped phantom span: '{snippet}'")

    # Apply post-processing improvements
    # 1. Verify and fix boundaries
    verified = verify_span_boundaries(located_spans, raw_text)

    # 2. Remove overlapping/subset spans with same label
    deduped = deduplicate_overlapping_spans(verified, same_label_only=True)

    # 3. Sort by document position
    final_output = sorted(deduped, key=lambda x: x["start"])

    logger.info(
        f"[{doc_id}] Verifier: {len(candidates)} candidates -> "
        f"{len(located_spans)} located -> {len(final_output)} final (after dedup)"
    )
    return {"final_spans": final_output}


# ===========================================================================
# GRAPH COMPILATION
# ===========================================================================


def build_s1_ensemble_graph():
    """Build the legacy ensemble graph."""
    workflow = StateGraph(S1GraphState)

    workflow.add_node("ensemble", s1_ensemble_node)
    workflow.add_node("consensus", s1_consensus_node)
    workflow.add_node("verifier", s1_structure_verifier_node)

    workflow.add_edge(START, "ensemble")
    workflow.add_edge("ensemble", "consensus")
    workflow.add_edge("consensus", "verifier")
    workflow.add_edge("verifier", END)

    return workflow.compile()


def build_s1_ddcot_graph():
    """Build the optimal DD-CoT graph."""
    workflow = StateGraph(S1DDCoTGraphState)

    workflow.add_node("ddcot_generator", s1_ddcot_generator_node)
    workflow.add_node("enhanced_critic", s1_ddcot_critic_node)
    workflow.add_node("refiner", s1_ddcot_refiner_node)
    workflow.add_node("verifier", s1_ddcot_verifier_node)

    workflow.add_edge(START, "ddcot_generator")
    workflow.add_edge("ddcot_generator", "enhanced_critic")
    workflow.add_edge("enhanced_critic", "refiner")
    workflow.add_edge("refiner", "verifier")
    workflow.add_edge("verifier", END)

    return workflow.compile()


# ===========================================================================
# EXPORTED GRAPHS
# ===========================================================================

# Legacy graph (backward compatibility)
s1_ensemble_graph = build_s1_ensemble_graph()
s1_graph = s1_ensemble_graph  # Alias for backward compatibility

# Optimal DD-CoT graph (RECOMMENDED)
s1_ddcot_graph = build_s1_ddcot_graph()

# [FIX] Alias s1_graph to the NEW DD-CoT graph
s1_graph = s1_ddcot_graph


# ===========================================================================
# GRAPH VISUALIZATION (Optional)
# ===========================================================================

if __name__ == "__main__":
    try:
        # Generate PNG for DD-CoT graph
        png_bytes = s1_ddcot_graph.get_graph().draw_mermaid_png()
        with open("s1_ddcot_graph.png", "wb") as f:
            f.write(png_bytes)
        print("Saved s1_ddcot_graph.png")

        # Generate PNG for legacy ensemble graph
        png_bytes = s1_ensemble_graph.get_graph().draw_mermaid_png()
        with open("s1_ensemble_graph.png", "wb") as f:
            f.write(png_bytes)
        print("Saved s1_ensemble_graph.png")

    except Exception as e:
        print(f"Could not generate graph images: {e}")
