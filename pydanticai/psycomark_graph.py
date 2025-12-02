import asyncio
import random
from typing import TypedDict, List, Optional, Any, Dict, Annotated
from langgraph.graph import StateGraph, END, START

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelSettings
from loguru import logger

# Import shared resources
from psycomark_agents import LLM, S2Output


# --- 0. Throttling Handler ---
async def safe_agent_run(agent, prompt, deps=None):
    """Executes agent.run with exponential backoff for AWS Throttling."""
    max_retries = 8
    base_delay = 2.0
    for attempt in range(max_retries):
        try:
            if deps:
                return await agent.run(prompt, deps=deps)
            return await agent.run(prompt)
        except Exception as e:
            if "ThrottlingException" in str(e) or "Too many tokens" in str(e):
                if attempt == max_retries - 1:
                    raise e
                delay = (base_delay * (2**attempt)) + random.uniform(0.5, 2.0)
                logger.warning(f" AWS Throttling. Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)
            else:
                raise e


# --- 1. State Definition ---
def overwrite(old, new):
    return new


class ReXState(TypedDict):
    # Inputs
    target_text: str
    marker_summary_str: str  # Narrative summary from S1

    # Internal Debate (Annotated to allow updates)
    defense_argument: Annotated[str, overwrite]  # Why it is NOT Conspiracy
    prosecution_argument: Annotated[str, overwrite]  # Why it is NOT Reporting

    # Output
    final_output: Optional[S2Output]


# --- 2. Node Agents ---


# Node A: The Defense (Reporting/Satire Analyst)
# Mission: Prove 'Non-Conspiracy' by excluding 'Endorsement'.
class DefenseOutput(BaseModel):
    argument: str = Field(
        ...,
        description="Argument for why this text is Reporting/Satire and NOT Endorsement.",
    )


defense_agent = Agent(
    LLM,
    output_type=DefenseOutput,
    system_prompt="You are a Skeptical Media Analyst. Your goal is to prove that a text is merely REPORTING on a conspiracy, or Mocking it, rather than endorsing it.",
    model_settings=ModelSettings(temperature=0.3),
)


# Node B: The Prosecutor (Conspiracy Analyst)
# Mission: Prove 'Endorsement' by excluding 'Reporting'.
class ProsecutionOutput(BaseModel):
    argument: str = Field(
        ...,
        description="Argument for why this text is Genuine Endorsement and NOT just Reporting.",
    )


prosecutor_agent = Agent(
    LLM,
    output_type=ProsecutionOutput,
    system_prompt="You are a Forensic Investigator. Your goal is to identify linguistic proof of GENUINE ENDORSEMENT, ruling out neutral reporting.",
    model_settings=ModelSettings(temperature=0.3),
)

# Node C: The Judge (ReX Evaluator)
# Mission: Decide which exclusion failed.
judge_agent = Agent(
    LLM,
    output_type=S2Output,
    system_prompt="You are a Supreme Court Judge using Reverse Exclusion Logic. You must determine if the 'Non-Conspiracy' explanation can be definitively ruled out.",
    model_settings=ModelSettings(temperature=0.0),
)

# --- 3. Node Functions ---


async def defense_node(state: ReXState) -> Dict[str, Any]:
    prompt = f"""
    <text_to_analyze>
    {state['target_text']}
    </text_to_analyze>
    
    <summary_context>
    {state['marker_summary_str']}
    </summary_context>

    <task>
    Construct a defense argument for why this text is **Non-Conspiracy** (Label: non).
    
    You must argue why the text **IS NOT** genuine endorsement:
    1. Cite reporting verbs ("they said", "claimed") that distance the author.
    2. Cite satire, mockery, or neutral analysis.
    3. Explain why the extracted markers are just context, not belief.
    </task>
    """
    try:
        res = await safe_agent_run(defense_agent, prompt)
        return {"defense_argument": res.output.argument}
    except Exception as e:
        logger.error(f"Defense failed: {e}")
        return {"defense_argument": "Failed to generate defense."}


async def prosecution_node(state: ReXState) -> Dict[str, Any]:
    prompt = f"""
    <text_to_analyze>
    {state['target_text']}
    </text_to_analyze>

    <task>
    Construct a prosecution argument for why this text is **Conspiracy Endorsement** (Label: conspiracy).
    
    You must argue why the text **IS NOT** merely reporting:
    1. Identify assertions of fact without attribution ("The cabal IS controlling us").
    2. Identify calls to action or urgent warnings ("Wake up!").
    3. Identify "truth-telling" vocabulary ("The real truth", "Mainstream lies").
    </task>
    """
    try:
        res = await safe_agent_run(prosecutor_agent, prompt)
        return {"prosecution_argument": res.output.argument}
    except Exception as e:
        logger.error(f"Prosecutor failed: {e}")
        return {"prosecution_argument": "Failed to generate prosecution."}


async def judge_node(state: ReXState) -> Dict[str, Any]:
    prompt = f"""
    <case_file>
    <text_evidence>
    {state['target_text']}
    </text_evidence>
    
    <defense_motion_to_dismiss>
    (Argument for 'Non-Conspiracy' / Reporting):
    {state['defense_argument']}
    </defense_motion_to_dismiss>
    
    <prosecution_charges>
    (Argument for 'Conspiracy Endorsement'):
    {state['prosecution_argument']}
    </prosecution_charges>
    </case_file>
    
    <judicial_instruction>
    You are the Judge. You must render a verdict based on **Stance Detection**.
    
    **The Law (Definitions):**
    1. **Reporting (Non):** The author attributes the claims to someone else ("He said...", "Users claim...").
    2. **Endorsement (Conspiracy):** The author asserts the claims as absolute fact in their own voice.
    
    **Decision Protocol:**
    1. **Evaluate the Defense:** Does the Defense successfully point to *attribution verbs* or *distancing language* in the text? 
       - IF YES -> The text is Reporting. Verdict: **non**.
       
    2. **Evaluate the Prosecution:** Does the Prosecution identify *unattributed assertions* of conspiracy facts?
       - IF YES AND Defense is weak -> Verdict: **conspiracy**.
       
    **Tie-Breaker:** If the text is ambiguous or mixes both, rule **non** (Presumption of Innocence).
    </judicial_instruction>
    
    Return the final label and a summary of which argument prevailed.
    """
    try:
        res = await safe_agent_run(judge_agent, prompt)
        return {"final_output": res.output}
    except Exception as e:
        logger.error(f"Judge failed: {e}")
        return {"final_output": S2Output(label="non", rationale="Graph failed.")}


# --- 4. Graph Construction ---


def build_rex_graph():
    workflow = StateGraph(ReXState)
    workflow.add_node("defense", defense_node)
    workflow.add_node("prosecutor", prosecution_node)
    workflow.add_node("judge", judge_node)

    # Parallel debate
    workflow.add_edge(START, "defense")
    workflow.add_edge(START, "prosecutor")

    # Converge at Judge
    workflow.add_edge("defense", "judge")
    workflow.add_edge("prosecutor", "judge")
    workflow.add_edge("judge", END)

    return workflow.compile()


REX_APP = build_rex_graph()

# --- 5. Runner Entry Point ---


async def run_s2_graph(
    doc_id: str, text: str, marker_summary: Optional[Dict[str, Any]] = None
) -> S2Output:
    """
    Executes the ReX-GoT (Reverse Exclusion Graph).
    """
    import json

    summary_str = (
        json.dumps(marker_summary, ensure_ascii=False)
        if marker_summary
        else "No summary provided."
    )

    inputs = {
        "target_text": text,
        "marker_summary_str": summary_str,
        "defense_argument": "",
        "prosecution_argument": "",
        "final_output": None,
    }

    logger.info(f"[{doc_id}] ReX-Graph: Defense vs Prosecutor Debate...")

    try:
        result = await REX_APP.ainvoke(inputs)
        logger.info(f"[{doc_id}] ReX-Graph Result: {result.get("final_output")}")
        return result.get("final_output") or S2Output(
            label="non", rationale="No output."
        )
    except Exception as e:
        logger.error(f"[{doc_id}] Graph Error: {e}")
        return S2Output(label="non", rationale=f"Graph Error: {e}")
