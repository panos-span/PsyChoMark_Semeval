import asyncio
from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END, START
from loguru import logger

from psycomark_agents import (
    run_s2_sequential_debate,
    S2Juror,
    S2CouncilOutput,
    S2Output,
    LLM,
)
from prompt_builder import build_s2_judge_system
from pydantic_ai import Agent


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
    logger.info(f"[{doc_id}] Convening Sequential Debate...")

    # Use the new Sequential Debate Runner
    result = await run_s2_sequential_debate(
        text=state["text"],
        s1_spans=state["s1_spans"],
        marker_summary=state["marker_summary"],
        temperature=temp,
    )

    logger.debug(f"[{state['doc_id']}] Council Votes: {result.tally}")
    return {"council_result": result}


# --- Node 2: The Judge ---
async def s2_judge_node(state: S2GraphState):
    """
    The Chief Justice Node (Updated for Sequential Debate).
    Synthesizes the 'Court Transcript' where Defense specifically refutes Prosecution.
    """
    doc_id = state["doc_id"]
    council = state["council_result"]
    votes = council.votes

    # 1. Sort Votes for the Transcript (Prosecutor -> Defense -> Witnesses)
    # This ensures the Judge reads the flow of argument logically.
    order_map = {
        S2Juror.BELIEVER: 1,  # The Prosecutor
        S2Juror.DEFENSE: 2,  # The Defense Attorney
        S2Juror.LITERALIST: 3,  # Expert Witness 1
        S2Juror.PROFILER: 4,  # Expert Witness 2
    }
    # Sort votes based on the map; unknowns go last
    votes.sort(key=lambda x: order_map.get(x.juror, 99))

    # 2. Construct the "Court Transcript"
    # We apply specific labels so the Judge understands the role of each argument.
    transcript_lines = []
    for v in votes:
        role_name = v.juror.value.upper()

        if v.juror == S2Juror.BELIEVER:
            transcript_lines.append(
                f"PROSECUTION ({role_name}):\nArgues: {v.verdict.upper()}\n"
                f'Indictment: "{v.rationale}"\n'
            )
        elif v.juror == S2Juror.DEFENSE:
            transcript_lines.append(
                f"DEFENSE ({role_name}):\nArgues: {v.verdict.upper()}\n"
                f'Rebuttal: "{v.rationale}"\n'
            )
        else:
            transcript_lines.append(
                f"WITNESS ({role_name}):\nTestimony: {v.verdict.upper()}\n"
                f'Statement: "{v.rationale}"\n'
            )

    transcript = "\n".join(transcript_lines)

    # 3. Prepare the Judge's System Prompt
    rag_txt = state.get("rag_context", "") or "No specific precedents available."
    system_prompt = build_s2_judge_system(rag_context=rag_txt)

    # 4. Create Stateless Agent
    judge_agent = Agent(
        LLM, output_type=S2Output, system_prompt=system_prompt, retries=2
    )

    # 5. Construct the Case File (Transcript Mode)
    user_prompt = f"""
<court_transcript id="{doc_id}">
  <evidence_exhibit_A>
{state['text']}
  </evidence_exhibit_A>

  <debate_transcript>
{transcript}
  </debate_transcript>

  <instruction>
    You have heard the Prosecution and the Defense.
    
    1. **Evaluation:** Did the Defense successfully refute the Prosecutor's specific point using Hanlon's Razor or the 'Librarian Defense'?
    2. **Check:** If the Prosecutor relies on "Implicit Support" but the Defense proves "Attribution/Reporting", you MUST Acquit.
    3. **Verdict:** Render the Final Decision based on the ReX Protocol.
  </instruction>
</court_transcript>
"""

    try:
        # Run the Judge
        result = await judge_agent.run(user_prompt)

        # Log Overrules
        majority = max(council.tally, key=council.tally.get) if council.tally else "non"
        if result.output.label != majority:
            logger.warning(
                f"[{doc_id}] JUDGE OVERRULED COUNCIL! (Council: {majority} -> Judge: {result.output.label})"
            )

        return {"final_output": result.output}

    except Exception as e:
        logger.error(f"[{doc_id}] Judge Logic Failed: {e}")

        # Fail-safe: Fallback to simple majority vote
        fallback_label = (
            "conspiracy"
            if council.tally.get("conspiracy", 0) > council.tally.get("non", 0)
            else "non"
        )

        fallback_output = S2Output(
            label=fallback_label,
            rationale=f"Judge agent failed ({str(e)}). Defaulted to Council majority.",
            confidence=0.5,
            key_evidence=[],
        )
        return {"final_output": fallback_output}


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
