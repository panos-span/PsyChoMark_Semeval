from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from loguru import logger
from psycomark_agents import (
    S2CouncilOutput,
    S2Output,
    run_s2_judge_review,
    run_s2_sequential_debate,
)


# --- State ---
class S2GraphState(TypedDict):
    doc_id: str
    text: str
    s1_spans: List[dict]
    marker_summary: str
    rag_context: str  # "Case Law" (Hard Negatives) from Retrieval
    metadata: Dict[str, Any]
    juror_temperature: float  # <--- NEW

    # Internal / Outputs
    council_result: Optional[S2CouncilOutput]
    final_output: Optional[S2Output]


# --- Node 1: The Council ---
async def s2_council_node(state: S2GraphState):
    doc_id = state["doc_id"]
    temp = state.get("juror_temperature", 0.4)
    rag = state.get("rag_context", "")  # <--- Retrieve context
    logger.info(f"[{doc_id}] Convening Sequential Debate...")

    # Use the new Sequential Debate Runner
    result = await run_s2_sequential_debate(
        text=state["text"],
        s1_spans=state["s1_spans"],
        marker_summary=state["marker_summary"],
        rag_context=rag,
        temperature=temp,
    )

    logger.debug(f"[{state['doc_id']}] Council Votes: {result.tally}")
    return {"council_result": result}


# --- Node 2: The Judge ---
async def s2_judge_node(state: S2GraphState):
    """
    The Chief Justice Node.
    Delegates to the optimized run_s2_judge_review function.
    """
    council = state["council_result"]
    text = state["text"]
    rag = state.get("rag_context", "")

    # [CRITICAL GUARD]
    # If the Council produced no votes (e.g. all agents crashed),
    # we must short-circuit or provide a fallback to prevent Judge hallucination.
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

    # Call the shared, optimized function
    result = await run_s2_judge_review(
        text=text,
        council_result=council,
        doc_id=state["doc_id"],
        rag_context=rag,
    )

    return {"final_output": result}


# --- Graph Wiring ---
workflow = StateGraph(S2GraphState)
workflow.add_node("council", s2_council_node)
workflow.add_node("judge", s2_judge_node)

workflow.add_edge(START, "council")
workflow.add_edge("council", "judge")
workflow.add_edge("judge", END)

s2_graph = workflow.compile()

# 2. Generate the PNG bytes
png_bytes = s2_graph.get_graph().draw_mermaid_png()

# 3. Save to a file
with open("s2_graph.png", "wb") as f:
    f.write(png_bytes)
