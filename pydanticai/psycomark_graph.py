import asyncio
import random
from typing import TypedDict, Optional, Any, Dict, Annotated
from langgraph.graph import StateGraph, END, START

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelSettings
from loguru import logger

# Import shared resources
from psycomark_agents import LLM, S2Output

# [NEW] Import rich context from prompt_builder
from prompt_builder import psycho_theory_preamble, playbook_block, data_profile_block


# --- 0. Shared Context Builder ---
def build_agent_context() -> str:
    """
    Constructs the shared 'Constitution' for all graph agents.
    Combines the Psycholinguistic role, the Playbook definitions, and Data Profile.
    """
    return (
        psycho_theory_preamble() + "\n" + playbook_block() + "\n" + data_profile_block()
    )


# --- 1. Throttling Handler ---
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


# --- 2. State Definition ---
def overwrite(old, new):
    return new


class ReXState(TypedDict):
    # Inputs
    target_text: str
    marker_summary_str: str  # Narrative summary from S1

    # Internal Debate (Annotated to allow updates)
    defense_argument: Annotated[str, overwrite]  # Argument for Reporting/Satire
    prosecution_argument: Annotated[str, overwrite]  # Argument for Endorsement

    # Output
    final_output: Optional[S2Output]


# --- 3. Node Agents (Enriched) ---

SHARED_CONTEXT = build_agent_context()


# Node A: The Defense (Reporting/Satire Analyst)
class DefenseOutput(BaseModel):
    argument: str = Field(
        ...,
        description="Argument proving the text is merely Reporting, Summarizing, or Mocking.",
    )


defense_agent = Agent(
    LLM,
    output_type=DefenseOutput,
    # [NEW] Inject shared context before specific instructions
    system_prompt=f"""
{SHARED_CONTEXT}

<role_specific>
You are a Skeptical Media Analyst. Your goal is to prove that the text is attributed to a THIRD PARTY (Reporting) or is MOCKING the claim (Satire).
Use the definitions in <psycomark_playbook> to distinguish between mentioning a cue (e.g., "they said the elites...") vs. using a cue (e.g., "the elites are...").
</role_specific>
""".strip(),
    model_settings=ModelSettings(temperature=0.3),
)


# Node B: The Prosecutor (Conspiracy Analyst)
class ProsecutionOutput(BaseModel):
    argument: str = Field(
        ...,
        description="Argument proving the author explicitly ENDORSES the conspiracy as fact.",
    )


prosecutor_agent = Agent(
    LLM,
    output_type=ProsecutionOutput,
    # [NEW] Inject shared context
    system_prompt=f"""
{SHARED_CONTEXT}

<role_specific>
You are a Forensic Investigator. Your goal is to find 'Stance Leakage'.
Refer to the <cues_epistemics> in the playbook: Look for self-sealing logic or direct commands ("do your research") that prove endorsement.
</role_specific>
""".strip(),
    model_settings=ModelSettings(temperature=0.3),
)


# Node C: The Judge (ReX Evaluator)
judge_agent = Agent(
    LLM,
    output_type=S2Output,
    # [NEW] Inject shared context
    system_prompt=f"""
{SHARED_CONTEXT}

<role_specific>
You are a Supreme Court Judge using the 'Attribution Firewall' protocol.
Use the definitions of Actor/Action/Effect from the preamble to evaluate the arguments.
</role_specific>
""".strip(),
    model_settings=ModelSettings(temperature=0.0),
)


# --- 4. Node Functions ---


async def defense_node(state: ReXState) -> Dict[str, Any]:
    prompt = f"""
    <evidence_text>
    {state['target_text']}
    </evidence_text>
    
    <s1_context>
    {state['marker_summary_str']}
    </s1_context>

    <task>
    Build the **Defense Case** for Label: `non`.
    
    1. Check for **Attribution**: Does the text credit the "Action" (from definitions) to a third party?
    2. Check for **Distancing**: Are there reporting verbs?
    3. Check for **Mockery**: Is the "Effect" (e.g., mind control) treated as absurd?
    
    Refer to the <data_profile>: remember [URL] often indicates a link submission summary.
    </task>
    """
    try:
        res = await safe_agent_run(defense_agent, prompt)
        return {"defense_argument": res.output.argument}
    except Exception as e:
        logger.error(f"Defense failed: {e}")
        return {"defense_argument": "Defense failed."}


async def prosecution_node(state: ReXState) -> Dict[str, Any]:
    prompt = f"""
    <evidence_text>
    {state['target_text']}
    </evidence_text>

    <task>
    Build the **Prosecution Case** for Label: `conspiracy`.
    
    1. **Stance Leakage**: Where does the author drop the reporter mask?
    2. **Epistemics**: Use <cues_epistemics> (e.g., "wake up") to prove intent.
    3. **First-Person**: Does the author include themselves in the "Victim" group?
    </task>
    """
    try:
        res = await safe_agent_run(prosecutor_agent, prompt)
        return {"prosecution_argument": res.output.argument}
    except Exception as e:
        logger.error(f"Prosecutor failed: {e}")
        return {"prosecution_argument": "Prosecution failed."}


async def judge_node(state: ReXState) -> Dict[str, Any]:
    prompt = f"""
    <case_file>
    <text_evidence>
    {state['target_text']}
    </text_evidence>
    
    <narrative_context>
    {state['marker_summary_str']}
    </narrative_context>
    
    <defense_motion>
    {state['defense_argument']}
    </defense_motion>
    
    <prosecution_charges>
    {state['prosecution_argument']}
    </prosecution_charges>
    </case_file>
    
    <judicial_instruction>
    **Protocol: The Attribution Firewall**
    
    1. **Presumption of Innocence (Non):** If the text fits the <data_profile> of a submission statement (summarizing a link), default to `non`.
    2. **Burden of Proof:** The Prosecutor must prove the author *endorses* the <Action> described.
    
    Compare the arguments against the <psycholinguistic_preamble>. 
    - If the "Actor" is attributed to someone else -> `non`.
    - If the author asserts the "Actor" is real -> `conspiracy`.
    </judicial_instruction>
    """
    try:
        res = await safe_agent_run(judge_agent, prompt)
        return {"final_output": res.output}
    except Exception as e:
        logger.error(f"Judge failed: {e}")
        return {"final_output": S2Output(label="non", rationale="Graph failed.")}


# --- 5. Graph Construction ---


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

# --- 6. Runner Entry Point ---


async def run_s2_graph(
    doc_id: str, text: str, marker_summary: Optional[Dict[str, Any]] = None
) -> S2Output:
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

    logger.info(f"[{doc_id}] ReX-Graph (Enriched): Defense vs Prosecutor Debate...")

    try:
        result = await REX_APP.ainvoke(inputs)
        final = result.get("final_output")
        if not final:
            final = S2Output(label="non", rationale="Graph produced no output.")

        logger.info(f"[{doc_id}] ReX Result: {final.label} | Why: {final.rationale}")
        return final

    except Exception as e:
        logger.error(f"[{doc_id}] Graph Error: {e}")
        return S2Output(label="non", rationale=f"Graph Error: {e}")
