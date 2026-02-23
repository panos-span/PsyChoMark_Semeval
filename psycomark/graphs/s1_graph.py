"""
psycomark.graphs.s1_graph — S1 DD-CoT LangGraph Workflow.

Pipeline topology (linear):

    Generator ─→ Critic ─→ Refiner ─→ Verifier

Features:
    - **Soft Gate**: Prevents the Critic from wiping all spans on subtle texts
    - **Context Injection**: Refiner receives narrative + complexity for boundary tuning
    - **Aggressive Dedup**: Verifier resolves exact overlaps across labels
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Dict, List, Optional, TypedDict

import operator
from langgraph.graph import END, START, StateGraph
from loguru import logger

from psycomark.config import safe_agent_run
from psycomark.schemas.s1 import (
    DDCoTExtraction,
    DDCoTRefinement,
    DDCoTSpan,
    EnhancedS1Critique,
    S1Deps,
    S1Span,
)
from psycomark.agents.s1_agents import (
    run_s1_ddcot,
    get_s1_ddcot_critic,
    get_s1_ddcot_refiner,
)
from psycomark.agents.span_utils import (
    deduplicate_overlapping_spans,
    find_best_span,
    find_span_with_context,
    verify_span_boundaries,
)


# ---------------------------------------------------------------------------
# Usage Aggregator
# ---------------------------------------------------------------------------


def _aggregate_usage(left: Dict[str, int], right: Dict[str, int]) -> Dict[str, int]:
    return {
        "input_tokens": left.get("input_tokens", 0) + right.get("input_tokens", 0),
        "output_tokens": left.get("output_tokens", 0) + right.get("output_tokens", 0),
        "total_tokens": left.get("total_tokens", 0) + right.get("total_tokens", 0),
    }


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class S1DDCoTGraphState(TypedDict):
    doc_id: str
    text: str
    few_shots: List[dict]
    metadata: Dict[str, Any]

    # DD-CoT Generator output
    text_complexity: str
    dominant_narrative: str
    draft_extractions: List[DDCoTSpan]

    # Enhanced Critic output
    critique: Optional[EnhancedS1Critique]
    requires_refinement: bool

    # Refiner output
    refined_extractions: List[DDCoTSpan]

    # Final output
    final_spans: List[Dict]

    token_usage: Annotated[Dict[str, int], _aggregate_usage]


# ---------------------------------------------------------------------------
# Node: DD-CoT Generator
# ---------------------------------------------------------------------------


async def _generator_node(state: S1DDCoTGraphState) -> Dict:
    text = state["text"]
    few_shots = state.get("few_shots", [])
    doc_id = state.get("doc_id", "unknown")
    metadata = state.get("metadata", {})

    logger.info(f"[{doc_id}] Starting DD-CoT Generator…")

    try:
        result, usage_dict = await run_s1_ddcot(
            text=text,
            few_shots=few_shots,
            skip_critic=True,
            return_full_extraction=True,
            metadata=metadata,
            return_usage=True,
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
                "token_usage": usage_dict,
            }
        return {
            "text_complexity": "unknown",
            "dominant_narrative": "unknown",
            "draft_extractions": [],
            "token_usage": usage_dict,
        }

    except Exception as e:
        logger.error(f"[{doc_id}] DD-CoT Generator failed: {e}")
        return {
            "text_complexity": "error",
            "dominant_narrative": "error",
            "draft_extractions": [],
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }


# ---------------------------------------------------------------------------
# Node: Enhanced Critic  (with Soft Gate)
# ---------------------------------------------------------------------------


async def _critic_node(state: S1DDCoTGraphState) -> Dict:
    from psycomark.prompts.builder import build_s1_ddcot_critic_user_template

    text = state["text"]
    draft_extractions = state.get("draft_extractions", [])
    doc_id = state.get("doc_id", "unknown")

    if not draft_extractions:
        return {"critique": None, "requires_refinement": False}

    logger.info(f"[{doc_id}] Running Enhanced Critic…")

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
        usage = critique_res.usage()

        # ----- Soft Gate Logic (Anti-False Negative) -----
        nuclear_triggers = [
            "REMOVE ALL",
            "NEGATIVE EXAMPLE",
            "ZERO EXTRACTIONS",
            "NO CONSPIRACY",
        ]

        wants_to_nuke = any(
            any(trigger in str(err).upper() for trigger in nuclear_triggers)
            for err in critique.granularity_errors
        )

        if wants_to_nuke:
            n_actors = sum(1 for s in draft_extractions if s.label == "Actor")
            n_actions = sum(1 for s in draft_extractions if s.label == "Action")
            is_significant = (n_actors >= 1 and n_actions >= 1) or len(
                draft_extractions
            ) >= 3

            if is_significant:
                logger.warning(
                    f"[{doc_id}] SOFT GATE: Intercepted Critic's 'REMOVE ALL'. "
                    f"Draft has {n_actors} Actors / {n_actions} Actions."
                )
                critique.granularity_errors = [
                    e
                    for e in critique.granularity_errors
                    if not any(t in str(e).upper() for t in nuclear_triggers)
                ]
                critique.granularity_errors.append(
                    "CHECK_STRICTLY: Verify spans exist, but do not auto-delete."
                )
                critique.requires_refinement = True

        usage_dict = {
            "input_tokens": getattr(usage, "request_tokens", 0) or 0,
            "output_tokens": getattr(usage, "response_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }

        n_errors = (
            len(critique.verbatim_errors)
            + len(critique.granularity_errors)
            + len(critique.label_errors)
            + len(critique.missed_spans)
        )
        logger.info(f"[{doc_id}] Critic: {n_errors} issues found")

        return {
            "critique": critique,
            "requires_refinement": critique.requires_refinement,
            "token_usage": usage_dict,
        }

    except Exception as e:
        logger.error(f"[{doc_id}] Critic failed: {e}")
        return {"critique": None, "requires_refinement": False}


# ---------------------------------------------------------------------------
# Node: DD-CoT Refiner
# ---------------------------------------------------------------------------


async def _refiner_node(state: S1DDCoTGraphState) -> Dict:
    from psycomark.prompts.builder import build_s1_ddcot_refiner_user_template

    text = state["text"]
    draft_extractions = state.get("draft_extractions", [])
    critique = state.get("critique")
    doc_id = state.get("doc_id", "unknown")

    if not state.get("requires_refinement", False) or critique is None:
        logger.info(f"[{doc_id}] No refinement needed, passing through draft")
        return {"refined_extractions": draft_extractions}

    logger.info(f"[{doc_id}] Running DD-CoT Refiner…")

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
            .replace("{{narrative}}", state.get("dominant_narrative", "unknown"))
            .replace("{{complexity}}", state.get("text_complexity", "unknown"))
        )

        refine_res = await safe_agent_run(refiner_agent, refine_user_msg, deps)
        refinement: DDCoTRefinement = refine_res.output
        usage = refine_res.usage()

        usage_dict = {
            "input_tokens": getattr(usage, "request_tokens", 0) or 0,
            "output_tokens": getattr(usage, "response_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }

        boundary_fixes = sum(
            1
            for f in refinement.fixes_applied
            if "EXPAND" in f.upper()
            or "BOUNDARY" in f.upper()
            or "TELESCOP" in f.upper()
        )
        logger.info(
            f"[{doc_id}] Refiner: {len(refinement.refined_extractions)} spans. "
            f"Fixes: {len(refinement.fixes_applied)} (Boundary Expansions: {boundary_fixes})"
        )

        return {
            "refined_extractions": refinement.refined_extractions,
            "token_usage": usage_dict,
        }

    except Exception as e:
        logger.error(f"[{doc_id}] Refiner failed: {e}")
        return {"refined_extractions": draft_extractions}


# ---------------------------------------------------------------------------
# Node: Structure Verifier
# ---------------------------------------------------------------------------


def _verifier_node(state: S1DDCoTGraphState) -> Dict:
    raw_text = state["text"]
    doc_id = state.get("doc_id", "unknown")

    candidates = state.get("refined_extractions") or state.get("draft_extractions", [])
    located_spans = []
    stats = {"context_hits": 0, "fallback_hits": 0, "misses": 0}

    for span in candidates:
        snippet = " ".join(span.text.split())
        label = span.label
        why = getattr(span, "why_this_label", None)

        left_ctx = getattr(span, "preceding_context", "") or ""
        right_ctx = getattr(span, "following_context", "") or ""

        start, end = -1, -1
        method = "none"

        if left_ctx or right_ctx:
            start, end = find_span_with_context(
                raw_text, snippet, left_ctx, right_ctx, nth=0
            )
            if start != -1:
                method = "context"
                stats["context_hits"] += 1

        if start == -1:
            start, end = find_best_span(raw_text, snippet, nth=0)
            if start != -1:
                method = "fallback"
                stats["fallback_hits"] += 1

        if start != -1:
            located_spans.append(
                {
                    "label": label.value if hasattr(label, "value") else label,
                    "text": raw_text[start:end],
                    "start": start,
                    "end": end,
                    "why": why,
                    "method": method,
                }
            )
        else:
            stats["misses"] += 1

    verified = verify_span_boundaries(located_spans, raw_text)
    deduped = deduplicate_overlapping_spans(verified, same_label_only=False)
    final_output = sorted(deduped, key=lambda x: x["start"])

    logger.info(
        f"[{doc_id}] Verifier: {len(candidates)} -> {len(final_output)} spans. "
        f"(Hits: {stats['context_hits']}+{stats['fallback_hits']} | Misses: {stats['misses']})"
    )

    return {"final_spans": final_output}


# ---------------------------------------------------------------------------
# Graph Compilation
# ---------------------------------------------------------------------------


def build_s1_ddcot_graph():
    wf = StateGraph(S1DDCoTGraphState)
    wf.add_node("ddcot_generator", _generator_node)
    wf.add_node("enhanced_critic", _critic_node)
    wf.add_node("refiner", _refiner_node)
    wf.add_node("verifier", _verifier_node)

    wf.add_edge(START, "ddcot_generator")
    wf.add_edge("ddcot_generator", "enhanced_critic")
    wf.add_edge("enhanced_critic", "refiner")
    wf.add_edge("refiner", "verifier")
    wf.add_edge("verifier", END)
    return wf.compile()


s1_graph = build_s1_ddcot_graph()
