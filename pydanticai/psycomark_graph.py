import asyncio
import operator
from typing import TypedDict, List, Optional, Any, Dict, Annotated
from langgraph.graph import StateGraph, END, START

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelSettings
from loguru import logger

# Import shared resources
from psycomark_agents import LLM, S2Output, S2Deps


# --- 1. State Definition ---
# Helper reducer: simply overwrites the old value with the new one
def overwrite(old, new):
    return new


class ClassificationState(TypedDict):
    # Inputs
    target_text: str
    few_shot_str: str

    # Internal Reasoning - Annotated to allow updates without "Multiple values" errors
    generated_principles: Annotated[str, overwrite]
    skeptic_argument: Annotated[str, overwrite]

    # Outputs
    final_output: Optional[S2Output]


# --- 2. Node Agents (Optimized System Prompts) ---


# Node A: Legislator
# Goal: Induce principles of STANCE (Endorsement vs Attribution)
class LegislatorOutput(BaseModel):
    principles: str = Field(
        ...,
        description="3 distinct forensic principles distinguishing Endorsement from Reporting.",
    )


legislator_agent = Agent(
    LLM,
    output_type=LegislatorOutput,
    system_prompt="You are an expert Forensic Linguist specializing in Stance Detection. Your goal is to identify the subtle linguistic markers that distinguish genuine belief in a conspiracy from mere reporting or analysis.",
    model_settings=ModelSettings(temperature=0.5),
)


# Node B: Skeptic
# Goal: High-recall detection of Reporting/Satire markers
class SkepticOutput(BaseModel):
    argument: str = Field(
        ...,
        description="A rigorous argument for why this text is Non-Conspiracy (Reporting, Satire, or Neutral).",
    )


skeptic_agent = Agent(
    LLM,
    output_type=SkepticOutput,
    system_prompt="You are a rigorous Fact-Checker and Media Analyst. Your job is to prevent False Positives by identifying reporting verbs, attribution, satire, or neutral context.",
    model_settings=ModelSettings(temperature=0.3),
)

# Node C: Judge
# Goal: Weigh evidence with a presumption of innocence (Non-Conspiracy)
judge_agent = Agent(
    LLM,
    output_type=S2Output,
    system_prompt="You are the Chief Justice of the Stance Detection Court. You evaluate texts to determine if the author GENUINELY ENDORSES a conspiracy theory.",
    model_settings=ModelSettings(temperature=0.0),
)

# --- 3. Node Functions (Optimized User Prompts) ---


async def legislator_node(state: ClassificationState) -> Dict[str, Any]:
    examples = state["few_shot_str"]
    if not examples:
        return {
            "generated_principles": "1. Unattributed Assertion (Factuality). 2. Urgent Call to Action. 3. Rejection of Official Epistemology."
        }

    prompt = f"""
    <task>
    Analyze the provided examples of 'Conspiracy' (Endorsement) vs 'Non-Conspiracy' (Reporting/Debunking).
    
    Formulate 3 distinct **Stance Principles** that separate these classes. 
    Focus on:
    1. **Attribution vs. Assertion**: How do conspiracy texts assert plots as absolute fact, while non-conspiracy texts attribute them to others?
    2. **Epistemics**: How do authors signal "forbidden knowledge" or "waking up"?
    3. **Tone**: How does the urgency/anger of a believer differ from the neutrality/mockery of a reporter?
    </task>

    <examples>
    {examples}
    </examples>
    
    Output ONLY the 3 principles.
    """
    try:
        result = await legislator_agent.run(prompt)
        return {"generated_principles": result.output.principles}
    except Exception as e:
        logger.error(f"Legislator failed: {e}")
        return {"generated_principles": "Error generating principles."}


async def skeptic_node(state: ClassificationState) -> Dict[str, Any]:
    text = state["target_text"]
    prompt = f"""
    <text_to_analyze>
    {text}
    </text_to_analyze>

    <mission>
    Play the role of a Skeptic. Vigorously argue why this text is **Non-Conspiracy** (Label: non).
    </mission>

    <checklist>
    1. **Attribution Check**: Does the text use reporting verbs ("claimed", "said", "users posted") to distance the author from the plot?
    2. **Satire Check**: Is the text mocking the conspiracy (e.g., scare quotes, sarcasm, exaggeration)?
    3. **Questioning**: Is the author just asking questions or analyzing the theory without endorsing it?
    4. **Incoherence**: Is the text too fragmented to be a coherent endorsement?
    </checklist>

    Generate a specific defense argument citing exact words from the text.
    """
    try:
        result = await skeptic_agent.run(prompt)
        return {"skeptic_argument": result.output.argument}
    except Exception as e:
        logger.error(f"Skeptic failed: {e}")
        return {"skeptic_argument": "No counter-argument generated."}


async def judge_node(state: ClassificationState) -> Dict[str, Any]:
    text = state["target_text"]
    principles = state["generated_principles"]
    skeptic_view = state["skeptic_argument"]

    prompt = f"""
    <case_file>
    <target_text>
    {text}
    </target_text>
    
    <legal_principles>
    {principles}
    </legal_principles>
    
    <defense_argument_from_skeptic>
    {skeptic_view}
    </defense_argument_from_skeptic>
    </case_file>
    
    <mandate>
    Render a final verdict. 
    
    **Burden of Proof**: 
    The text is presumed 'non' (Reporting/Analysis) UNLESS there is clear evidence of **Endorsement**.
    
    1. Does the text explicitly assert the conspiracy as fact (Principle Check)?
    2. Does the Skeptic successfully highlight attribution or distancing markers (e.g., "They say...")?
    3. If the text is ambiguous, rule 'non'.
    
    Output the final label and a rationale explaining the Stance.
    </mandate>
    """
    try:
        result = await judge_agent.run(prompt)
        return {"final_output": result.output}
    except Exception as e:
        logger.error(f"Judge failed: {e}")
        return {"final_output": S2Output(label="non", rationale="Judge failed.")}


# --- 4. Graph Construction ---


def build_pbp_graph():
    workflow = StateGraph(ClassificationState)

    # Add nodes
    workflow.add_node("legislator", legislator_node)
    workflow.add_node("skeptic", skeptic_node)
    workflow.add_node("judge", judge_node)

    # Parallel Fan-Out from START
    workflow.add_edge(START, "legislator")
    workflow.add_edge(START, "skeptic")

    # Join at Judge
    workflow.add_edge("legislator", "judge")
    workflow.add_edge("skeptic", "judge")

    workflow.add_edge("judge", END)

    return workflow.compile()


# Instantiate the compiled graph once
PBP_APP = build_pbp_graph()

# --- 5. Runner Entry Point ---


async def run_s2_pbp(doc_id: str, text: str, fewshots: List[dict]) -> S2Output:
    """
    Executes the Principle-Based Prompting (PBP) Graph for S2.
    """
    # 1. Format few-shots
    few_shot_str = ""
    for ex in fewshots:
        label = ex.get("label", "unknown")
        txt = ex.get("text", "") or ex.get("doc_text", "")
        rationale = ex.get("rationale", "")
        few_shot_str += (
            f"Example (Label: {label}):\nText: {txt}\nRationale: {rationale}\n---\n"
        )

    # 2. Prepare Input State
    inputs = {
        "target_text": text,
        "few_shot_str": few_shot_str,
        "generated_principles": "",
        "skeptic_argument": "",
        "final_output": None,
    }

    logger.info(f"[{doc_id}] PBP-Graph: Invoking Legislator & Skeptic...")

    # 3. Invoke Graph
    try:
        result_state = await PBP_APP.ainvoke(inputs)
        final_out = result_state.get("final_output")

        if not final_out:
            return S2Output(label="non", rationale="Graph returned no output.")

        logger.success(f"[{doc_id}] Judge Verdict: {final_out.label}")
        return final_out
    except Exception as e:
        logger.error(f"[{doc_id}] Graph Execution Failed: {e}")
        return S2Output(label="non", rationale=f"Graph error: {e}")
