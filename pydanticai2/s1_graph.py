#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s1_graph_optimized.py — Gated Pattern Recognition Workflow.

Architecture:
1. GATE: Binary check (Is this worth processing?) -> Exits early if Neutral.
2. GENERATOR: Extracts patterns using the "Contrastive" Prompt.
3. FILTER (Ex-Critic): Semantic check. DELETES false positives (Natural Forces, Reporters).
   * Note: We REMOVED the "Refiner" node. It was causing grammar hallucinations.
4. VERIFIER: Deterministic grounding. Drops non-verbatim matches.
"""

from typing import TypedDict, List, Annotated
from loguru import logger
from langgraph.graph import StateGraph, END, START

# Import your updated agents
from psycomark_agents import (
    run_s1_pattern_recognition,  # The new function we created
    S1PatternExtraction,  # The new Pydantic model
    S1Deps,
    find_best_span,
)

# ===========================================================================
# 1. STATE DEFINITION
# ===========================================================================


class S1State(TypedDict):
    doc_id: str
    text: str
    # The extraction object from Pydantic
    extraction: Annotated[S1PatternExtraction, "replace"]
    # Final verified list
    final_spans: List[dict]
    # Metrics
    steps_taken: List[str]


# ===========================================================================
# 2. NODE DEFINITIONS
# ===========================================================================


async def gate_node(state: S1State):
    """
    The Bouncer. Checks if text is a Hard Negative (Tutorial, Science, neutral news).
    We use a lightweight prompt here or heuristics.
    """
    text = state["text"]

    # 1. Heuristic Gating (Fast & Free)
    # If text is very short or clearly irrelevant, skip.
    if len(text) < 50:
        logger.info(f"[{state['doc_id']}] Gate: Text too short. Skipping.")
        return {"extraction": None, "steps_taken": ["Gate (Short)"]}

    # 2. Keyword "Safe List" (Optional but effective)
    # If it's a tutorial context, we can skip.
    safe_words = ["tutorial", "how to make", "blender", "rendering"]
    if any(w in text.lower() for w in safe_words) and "conspiracy" not in text.lower():
        logger.info(f"[{state['doc_id']}] Gate: Safe keyword detected. Skipping.")
        return {"extraction": None, "steps_taken": ["Gate (Keyword)"]}

    # If it passes heuristics, we let the Generator proceed.
    return {"steps_taken": ["Gate (Pass)"]}


async def generator_node(state: S1State):
    """
    The Pattern Extractor. Uses the S1PatternExtraction model.
    """
    text = state["text"]

    # [FIX] Extract RAG examples from state
    few_shots = state.get("few_shots", [])

    # [FIX] Pass them into S1Deps
    deps = S1Deps(
        raw_text=text,
        doc_id=state["doc_id"],
        few_shots=few_shots,  # <--- CRITICAL BRIDGE
    )

    # Run the optimized agent
    result = await run_s1_pattern_recognition(text, deps, temperature=0.0)

    return {"extraction": result, "steps_taken": state["steps_taken"] + ["Generator"]}


async def semantic_filter_node(state: S1State):
    """
    The Skeptical Auditor.
    Instead of asking an LLM to 'Critique', we iterate through the extraction
    and applying the "Guardrails" programmatically or via a fast check.
    """
    raw_extraction = state["extraction"]
    if not raw_extraction or not raw_extraction.extractions:
        return {"extraction": raw_extraction}  # Nothing to filter

    filtered_spans = []

    # We apply the "Delete, Don't Rewrite" philosophy here.
    for span in raw_extraction.extractions:
        keep = True

        # Guardrail 1: Reporter Trap Check
        # If the label is "Actor" but the text is "Reuters", kill it.
        bad_actors = ["reuters", "the author", "op", "users", "critics"]
        if span.label == "Actor" and span.text.lower() in bad_actors:
            logger.info(f"[{state['doc_id']}] Filter: Deleted Reporter '{span.text}'")
            keep = False

        # Guardrail 2: Natural Force Check
        # If the prompt missed it, we catch it here.
        natural_forces = ["virus", "covid", "inflation", "market"]
        if span.label == "Actor" and any(
            nf in span.text.lower() for nf in natural_forces
        ):
            # Only keep if 'why' explains engineering
            if "engineer" not in span.why_this_label.lower():
                logger.info(
                    f"[{state['doc_id']}] Filter: Deleted Natural Force '{span.text}'"
                )
                keep = False

        if keep:
            filtered_spans.append(span)

    # Update state with filtered list
    raw_extraction.extractions = filtered_spans
    return {
        "extraction": raw_extraction,
        "steps_taken": state["steps_taken"] + ["Filter"],
    }


async def verifier_node(state: S1State):
    """
    The Grounding Truth.
    Locates exact indices. If not found verbatim, DROPS the span.
    """
    extraction = state["extraction"]
    if not extraction or not extraction.extractions:
        return {"final_spans": []}

    final_output = []

    for span in extraction.extractions:
        # Strict exact match search
        start, end = find_best_span(state["text"], span.text)

        if start != -1:
            final_output.append(
                {
                    "label": span.label,
                    "text": state["text"][start:end],  # Exact substring
                    "start": start,
                    "end": end,
                    "why": span.why_this_label,
                }
            )
        else:
            # Strict Penalty: If verbatim search fails, assume Hallucination.
            logger.warning(
                f"[{state['doc_id']}] Verifier: Dropped non-verbatim '{span.text}'"
            )

    return {
        "final_spans": final_output,
        "steps_taken": state["steps_taken"] + ["Verifier"],
    }


# ===========================================================================
# 3. CONDITIONAL EDGES
# ===========================================================================


def check_gate(state: S1State):
    """Decides where to go after Gate."""
    # If heuristics killed it (extraction is None), end.
    if state.get("extraction") is None and "Gate (Pass)" not in state["steps_taken"]:
        return "end"
    return "generate"


def check_empty(state: S1State):
    """If Generator found nothing, skip Filter/Verifier."""
    if not state["extraction"].extractions:
        return "end"
    return "filter"


# ===========================================================================
# 4. GRAPH COMPILATION
# ===========================================================================


def build_s1_pattern_graph():
    workflow = StateGraph(S1State)

    workflow.add_node("gate", gate_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("filter", semantic_filter_node)
    workflow.add_node("verifier", verifier_node)

    # Flow
    workflow.add_edge(START, "gate")

    workflow.add_conditional_edges(
        "gate", check_gate, {"generate": "generator", "end": END}
    )

    workflow.add_conditional_edges(
        "generator", check_empty, {"filter": "filter", "end": END}
    )

    workflow.add_edge("filter", "verifier")
    workflow.add_edge("verifier", END)

    return workflow.compile()


# Export for use
s1_graph = build_s1_pattern_graph()
