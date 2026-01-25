from typing import Any, Dict, List, Optional, TypedDict, Annotated

from langgraph.graph import END, START, StateGraph
from loguru import logger
from psycomark_agents import (
    # Legacy S2
    S2CouncilOutput,
    S2Output,
    run_s2_judge_review,
    run_s2_sequential_debate,
    # Anti-Echo Chamber S2 (NEW)
    ParallelCouncilOutput,
    CalibratedJudgeOutput,
    run_s2_parallel_council,
    run_s2_calibrated_judge,
)


def aggregate_usage(left: Dict[str, int], right: Dict[str, int]) -> Dict[str, int]:
    """Reducer to sum token usage across graph nodes."""
    return {
        "input_tokens": left.get("input_tokens", 0) + right.get("input_tokens", 0),
        "output_tokens": left.get("output_tokens", 0) + right.get("output_tokens", 0),
        "total_tokens": left.get("total_tokens", 0) + right.get("total_tokens", 0),
    }


# ===========================================================================
# LEGACY S2 GRAPH (Sequential Debate)
# ===========================================================================


class S2GraphState(TypedDict):
    doc_id: str
    text: str
    s1_spans: List[dict]
    marker_summary: str
    rag_context: str  # "Case Law" (Hard Negatives) from Retrieval
    metadata: Dict[str, Any]
    juror_temperature: float

    # Internal / Outputs
    council_result: Optional[S2CouncilOutput]
    final_output: Optional[S2Output]

    token_usage: Annotated[Dict[str, int], aggregate_usage]


# --- Node 1: The Council (Legacy Sequential) ---
async def s2_council_node(state: S2GraphState):
    doc_id = state["doc_id"]
    temp = state.get("juror_temperature", 0.4)
    rag = state.get("rag_context", "")
    logger.info(f"[{doc_id}] Convening Sequential Debate...")

    result = await run_s2_sequential_debate(
        text=state["text"],
        s1_spans=state["s1_spans"],
        marker_summary=state["marker_summary"],
        rag_context=rag,
        temperature=temp,
    )

    logger.debug(f"[{state['doc_id']}] Council Votes: {result.tally}")
    return {"council_result": result}


# --- Node 2: The Judge (Legacy) ---
async def s2_judge_node(state: S2GraphState):
    """Legacy Judge node - uses sequential debate output."""
    council = state["council_result"]
    text = state["text"]
    rag = state.get("rag_context", "")

    if not council or not council.votes:
        logger.error(f"[{state['doc_id']}] Council failed (0 votes). Skipping Judge.")
        return {
            "final_output": S2Output(
                label="non",
                rationale="Mistrial: Council failed to convene.",
                confidence=0.0,
                key_evidence=[],
            )
        }

    result = await run_s2_judge_review(
        text=text,
        council_result=council,
        doc_id=state["doc_id"],
        rag_context=rag,
    )

    return {"final_output": result}


# --- Legacy Graph Wiring ---
workflow = StateGraph(S2GraphState)
workflow.add_node("council", s2_council_node)
workflow.add_node("judge", s2_judge_node)

workflow.add_edge(START, "council")
workflow.add_edge("council", "judge")
workflow.add_edge("judge", END)

s2_graph = workflow.compile()


# ===========================================================================
# ANTI-ECHO CHAMBER S2 GRAPH (Parallel Voting + Calibrated Judge)
# ===========================================================================


class S2ParallelGraphState(TypedDict):
    """State for Anti-Echo Chamber S2 pipeline."""

    doc_id: str
    text: str
    s1_spans: List[dict]
    marker_summary: str
    rag_context: str
    metadata: Dict[str, Any]
    juror_temperature: float

    # Parallel Council Output (enhanced)
    parallel_council_result: Optional[ParallelCouncilOutput]

    # Calibrated Judge Output (enhanced)
    calibrated_output: Optional[CalibratedJudgeOutput]

    # For backward compatibility - convert to S2Output
    final_output: Optional[S2Output]

    token_usage: Annotated[Dict[str, int], aggregate_usage]


# --- Node 1: Parallel Council (Anti-Echo Chamber) ---
async def s2_parallel_council_node(state: S2ParallelGraphState):
    """
    All jurors vote INDEPENDENTLY and SIMULTANEOUSLY.
    No echo chamber - each juror only sees the evidence, not other votes.
    """
    doc_id = state["doc_id"]
    temp = state.get("juror_temperature", 0.4)
    rag = state.get("rag_context", "")

    logger.info(f"[{doc_id}] Convening PARALLEL Council (Anti-Echo Chamber)...")

    logger.info(f"[{doc_id}] RAG Context {rag[:100]}...")  # Log snippet of RAG context

    result, usage = await run_s2_parallel_council(
        text=state["text"],
        s1_spans=state["s1_spans"],
        marker_summary=state["marker_summary"],
        rag_context=rag,
        temperature=temp,
        return_usage=True,  # <--- Request Usage
    )

    logger.info(
        f"[{doc_id}] Parallel Council: {result.tally}, "
        f"Consensus: {result.consensus_level}, Dissent: {result.dissent_strength:.2f}"
    )

    return {"parallel_council_result": result, "token_usage": usage}


# --- Node 2: Calibrated Judge (Dissent-Aware) ---
async def s2_calibrated_judge_node(state: S2ParallelGraphState):
    """
    Calibrated Judge: Weighs dissent, lowers confidence on splits,
    can override council if evidence warrants.
    """
    council = state["parallel_council_result"]
    text = state["text"]
    rag = state.get("rag_context", "")
    doc_id = state["doc_id"]
    metadata = state.get("metadata", {})  # <--- Extract Metadata

    if not council or not council.votes:
        logger.error(f"[{doc_id}] Parallel Council failed (0 votes).")
        fallback = CalibratedJudgeOutput(
            label="non",
            confidence=0.0,
            rationale="Mistrial: Parallel council failed to convene.",
            dissent_considered=False,
            key_evidence=[],
            council_override=False,
            borderline_flag=True,
        )
        # Also create S2Output for backward compatibility
        legacy_output = S2Output(
            label="non",
            rationale="Mistrial: Parallel council failed to convene.",
            confidence=0.0,
            key_evidence=[],
        )
        return {"calibrated_output": fallback, "final_output": legacy_output}

    logger.info(f"[{doc_id}] Running Calibrated Judge...")
    logger.info(f"[{doc_id}] RAG Context {rag[:100]}...")  # Log snippet of RAG context

    result, usage = await run_s2_calibrated_judge(
        text=text,
        council_result=council,
        doc_id=doc_id,
        rag_context=rag,
        return_usage=True,
        metadata=metadata,  # <--- Pass Metadata for Contextual Priors
    )

    # Convert to legacy S2Output for backward compatibility
    legacy_output = S2Output(
        label=result.label,
        rationale=result.rationale,
        confidence=result.confidence,
        key_evidence=result.key_evidence,
    )

    logger.info(
        f"[{doc_id}] Calibrated Judge: {result.label} "
        f"(conf={result.confidence:.2f}, override={result.council_override}, borderline={result.borderline_flag})"
    )

    return {
        "calibrated_output": result,
        "final_output": legacy_output,
        "token_usage": usage,
    }


# --- Anti-Echo Chamber Graph Wiring ---
parallel_workflow = StateGraph(S2ParallelGraphState)
parallel_workflow.add_node("parallel_council", s2_parallel_council_node)
parallel_workflow.add_node("calibrated_judge", s2_calibrated_judge_node)

parallel_workflow.add_edge(START, "parallel_council")
parallel_workflow.add_edge("parallel_council", "calibrated_judge")
parallel_workflow.add_edge("calibrated_judge", END)

s2_parallel_graph = parallel_workflow.compile()


# ===========================================================================
# EXPORTS
# ===========================================================================

# Legacy (backward compatible)
# s2_graph - Sequential debate graph

# Anti-Echo Chamber (recommended)
# s2_parallel_graph - Parallel voting graph

# Alias for default (can switch to parallel when ready)
s2_default_graph = s2_parallel_graph  # Change to s2_parallel_graph when validated


# --- Optional: Generate graph visualizations ---
try:
    png_bytes = s2_graph.get_graph().draw_mermaid_png()
    with open("s2_graph.png", "wb") as f:
        f.write(png_bytes)

    parallel_png = s2_parallel_graph.get_graph().draw_mermaid_png()
    with open("s2_parallel_graph.png", "wb") as f:
        f.write(parallel_png)
except Exception as e:
    logger.debug(f"Could not generate graph PNGs: {e}")
