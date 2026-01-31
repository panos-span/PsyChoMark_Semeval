#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s2_graph.py — Anti-Echo Chamber Adjudication Pipeline.

Architecture:
1. PROFILER (Python): Calculates Forensic Priors (Uncertainty, JAQ-ratio).
2. COUNCIL (Parallel): 4 Independent Agents vote.
3. JUDGE (Calibrated): Synthesizes votes + RAG + Forensic Stats.
"""

import re
from typing import Any, Dict, List, Optional, TypedDict, Annotated
from langgraph.graph import END, START, StateGraph
from loguru import logger

from psycomark_agents import (
    ParallelCouncilOutput,
    CalibratedJudgeOutput,
    run_s2_parallel_council,
    run_s2_calibrated_judge,
    S2Output,
)


def aggregate_usage(left: Dict[str, int], right: Dict[str, int]) -> Dict[str, int]:
    return {
        "input_tokens": left.get("input_tokens", 0) + right.get("input_tokens", 0),
        "output_tokens": left.get("output_tokens", 0) + right.get("output_tokens", 0),
        "total_tokens": left.get("total_tokens", 0) + right.get("total_tokens", 0),
    }


# ===========================================================================
# 1. STATE DEFINITION
# ===========================================================================


class ForensicStats(TypedDict):
    # Basic
    uncertainty_ratio: float
    question_density: float
    is_jaqing: bool

    # Advanced (Forensic 2.0)
    agency_gap: float  # High = "They" did it (Conspiracy)
    epistemic_intensity: float  # High = "Wake Up/Truth" (Conspiracy)
    shouting_score: float  # High = CAPS/!!! (Conspiracy)


class S2ParallelGraphState(TypedDict):
    doc_id: str
    text: str
    s1_spans: List[dict]
    marker_summary: str
    rag_context: str
    metadata: Dict[str, Any]
    juror_temperature: float

    # [NEW] Forensic Statistics
    forensic_stats: ForensicStats

    # Outputs
    parallel_council_result: Optional[ParallelCouncilOutput]
    calibrated_output: Optional[CalibratedJudgeOutput]
    final_output: Optional[S2Output]
    token_usage: Annotated[Dict[str, int], aggregate_usage]
    appeal_count: int


# ===========================================================================
# 2. NODE: FORENSIC PROFILER (Python Logic)
# ===========================================================================


# ===========================================================================
# NEW NODE: THE APPEAL COURT
# ===========================================================================
async def appeal_court_node(state: S2ParallelGraphState):
    """
    Triggered when Confidence is low or Council is split.
    Acts as a 'Tie-Breaker' by forcing a specific focus based on Forensic Data.
    """
    logger.info(f"[{state['doc_id']}] Entering Appeal Court (Low Confidence)...")

    council = state["parallel_council_result"]
    judge_prev = state["calibrated_output"]
    text = state["text"]
    rag = state.get("rag_context", "")
    metadata = state.get("metadata", {})
    stats = state.get("forensic_stats", {})

    # Current Verdict to Flip
    current_verdict = judge_prev.label
    target_verdict = "non" if current_verdict == "conspiracy" else "conspiracy"

    # --- FORENSIC DIAGNOSIS LOGIC ---
    # We choose a specific argument based on the data profile.

    diagnosis_msg = ""

    # Case 1: False Negative Risk (Verdict: Non, but signs of insinuation)
    if current_verdict == "non":
        if stats.get("is_jaqing"):
            diagnosis_msg = "FORENSIC ALERT: The text uses the 'Just Asking Questions' tactic. Re-evaluate if these questions presuppose a cover-up."
        elif stats.get("epistemic_intensity", 0) > 1.5:
            diagnosis_msg = "FORENSIC ALERT: High 'Truth/Wake Up' terminology detected. This suggests dog-whistling despite the neutral tone."
        else:
            diagnosis_msg = "ADVOCATE TASK: The previous court missed the subtle 'Structural Assertion'. Look for claims of engineered harm."

    # Case 2: False Positive Risk (Verdict: Conspiracy, but maybe just angry/vague)
    elif current_verdict == "conspiracy":
        if stats.get("shouting_score", 0) > 3.0:
            diagnosis_msg = "FORENSIC ALERT: High emotional intensity (Shouting) detected. Distinguish between 'Anger at Incompetence' (Non) and 'Belief in Plot' (Conspiracy)."
        elif stats.get("agency_gap", 0) < 0.2:
            diagnosis_msg = "FORENSIC ALERT: The writer targets specific named entities, not vague 'They/Them'. Re-evaluate if this is standard political critique."
        else:
            diagnosis_msg = "ADVOCATE TASK: Apply 'Hanlon's Razor'. Can this be explained by stupidity or greed rather than coordinated malice?"

    # --- ADVERSARIAL PROMPT ---
    appeal_instruction = f"""
    *** MANDATORY ADVERSARIAL REVIEW ***
    The previous court ruled '{current_verdict.upper()}' but with LOW CONFIDENCE.
    
    You are the {target_verdict.upper()} ADVOCATE.
    
    {diagnosis_msg}
    
    Your goal is to build the strongest possible case for {target_verdict.upper()}.
    If the evidence for '{target_verdict.upper()}' is solid, OVERRIDE the previous verdict with HIGH CONFIDENCE.
    """

    # 2. Re-Run Judge with Override Prompt
    # We append the instruction to the RAG context to force attention
    enhanced_rag = f"{rag}\n\n[BINDING APPEAL INSTRUCTION]:\n{appeal_instruction}"

    result, usage = await run_s2_calibrated_judge(
        text=text,
        council_result=council,
        doc_id=state["doc_id"],
        rag_context=enhanced_rag,  # Instruction injected here
        return_usage=True,
        metadata=metadata,
    )

    logger.info(
        f"[{state['doc_id']}] Appeal Verdict: {result.label} (Conf: {result.confidence:.2f})"
    )

    # Update State
    return {
        "calibrated_output": result,
        "token_usage": usage,
        "appeal_count": state.get("appeal_count", 0) + 1,
    }


# ===========================================================================
# CONDITIONAL EDGES
# ===========================================================================
def check_verdict_quality(state: S2ParallelGraphState):
    """
    Decides whether to accept the verdict or appeal.
    """
    judge = state["calibrated_output"]
    attempts = state.get("appeal_count", 0)

    # Stop conditions
    if attempts >= 1:  # Max 1 retry to save cost/time
        return END

    # Quality conditions
    is_high_conf = judge.confidence > 0.80
    is_unanimous = state["parallel_council_result"].consensus_level == "unanimous"

    if is_high_conf or is_unanimous:
        return END

    return "appeal"


def calculate_forensic_stats(text: str) -> ForensicStats:
    """
    Forensic Profiling 2.0: Calculates linguistic markers for the Judge.
    Fully compliant with ForensicStats TypedDict.
    """
    # 1. Pre-compute lower case for matching (Optimization)
    text_lower = text.lower()
    words_raw = text.split()  # Keep case for Shouting check
    words_lower = text_lower.split()  # For lexical matching
    total_words = len(words_raw) if words_raw else 1

    # --- 1. Uncertainty (Speculation) ---
    hedges = {
        "maybe",
        "might",
        "possibly",
        "seems",
        "appears",
        "could",
        "unsure",
        "allegedly",
        "claims",
        "purportedly",
        "rumored",
        "potential",
        "think",
        "believe",
    }
    hedge_count = sum(1 for w in words_lower if w in hedges)
    # Normalized by 5% of text length to avoid punishing long texts
    uncertainty_ratio = hedge_count / max(1, total_words * 0.05)

    # --- 2. Epistemic Intensity (Dog Whistles) ---
    truth_lexicon = {
        "truth",
        "lie",
        "lies",
        "wake",
        "awake",
        "sheep",
        "realize",
        "proven",
        "undeniable",
        "exposed",
        "reveal",
        "hidden",
        "agenda",
        "narrative",
        "mainstream",
        "msm",
        "psyop",
        "shill",
        "bot",
    }
    epistemic_count = sum(1 for w in words_lower if w in truth_lexicon)
    epistemic_intensity = (epistemic_count / total_words) * 100

    # --- 3. Agency Gap (Vague vs Specific) ---
    vague_agents = {"they", "them", "their", "elites", "globalists", "cabal", "powers"}
    vague_count = sum(1 for w in words_lower if w in vague_agents)

    # Proxy for Specific Entities: Count common named entities in this domain
    # (A full NER model is too slow here, so we use a keyword proxy)
    specific_keywords = {
        "biden",
        "trump",
        "cdc",
        "fbi",
        "cia",
        "fda",
        "who",
        "pfizer",
        "moderna",
        "putin",
        "zelensky",
        "nato",
        "eu",
        "china",
        "russia",
    }
    specific_count = sum(1 for w in words_lower if w in specific_keywords)

    # High Ratio (>0.5) = Conspiratorial (Blaming "Them" vs. "Biden")
    agency_gap = vague_count / (specific_count + 1)

    # --- 4. Question Density & JAQing ---
    question_mark_count = text.count("?")
    # Estimate sentences by punctuation
    sentence_count = max(1, text.count(".") + text.count("!") + question_mark_count)
    question_density = question_mark_count / sentence_count

    # JAQing Detection (Leading Questions)
    leading_markers = [
        "why is",
        "why are",
        "coincidence",
        "strange",
        "curious",
        "how come",
        "why won't",
    ]
    # Use text_lower to split, ensuring we catch "Why" and "why"
    questions_segments = [q.strip() for q in text_lower.split("?") if q.strip()]
    is_jaqing = any(any(m in q for m in leading_markers) for q in questions_segments)

    # --- 5. Shouting Score (Urgency) ---
    # Count ALL CAPS words (excluding 'I', 'A')
    shouting_count = sum(1 for w in words_raw if w.isupper() and len(w) > 1)
    shouting_score = (shouting_count / total_words) * 100

    return {
        "uncertainty_ratio": round(uncertainty_ratio, 2),
        "question_density": round(question_density, 2),
        "is_jaqing": is_jaqing,
        "agency_gap": round(agency_gap, 2),
        "epistemic_intensity": round(epistemic_intensity, 2),
        "shouting_score": round(shouting_score, 2),
    }


async def forensic_profiler_node(state: S2ParallelGraphState):
    """
    Computes statistical priors before the LLMs run.
    """
    stats = calculate_forensic_stats(state["text"])
    logger.info(f"[{state['doc_id']}] Forensic Stats: {stats}")
    return {"forensic_stats": stats}


# ===========================================================================
# 3. NODE: PARALLEL COUNCIL
# ===========================================================================


async def s2_parallel_council_node(state: S2ParallelGraphState):
    """
    Votes with awareness of Forensic Stats.
    """
    doc_id = state["doc_id"]
    temp = state.get("juror_temperature", 0.4)
    rag = state.get("rag_context", "")
    metadata = state.get("metadata", {})  # Extract metadata
    stats = state.get("forensic_stats", {})

    # Inject stats into the marker summary so jurors see it
    enhanced_summary = (
        f"{state['marker_summary']}\n"
        f"[FORENSIC DATA]: Uncertainty={stats.get('uncertainty_ratio')}, "
        f"JAQing={stats.get('is_jaqing')}"
    )

    result, usage = await run_s2_parallel_council(
        text=state["text"],
        s1_spans=state["s1_spans"],
        marker_summary=enhanced_summary,  # Pass enriched context
        rag_context=rag,
        temperature=temp,
        metadata=metadata,
        return_usage=True,
    )

    logger.info(
        f"[{doc_id}] Council: {result.tally}, " f"Consensus: {result.consensus_level}"
    )

    return {"parallel_council_result": result, "token_usage": usage}


# ===========================================================================
# 4. NODE: CALIBRATED JUDGE
# ===========================================================================


async def s2_calibrated_judge_node(state: S2ParallelGraphState):
    """
    Judge uses Forensic Stats to break ties.
    """
    council = state["parallel_council_result"]
    text = state["text"]
    rag = state.get("rag_context", "")
    doc_id = state["doc_id"]
    metadata = state.get("metadata", {})
    stats = state.get("forensic_stats", {})

    # Merge stats into metadata for the Judge Agent
    metadata.update(stats)

    if not council or not council.votes:
        logger.error(f"[{doc_id}] Council failed.")
        # [Fail safe code omitted for brevity, same as before]
        return {}

    result, usage = await run_s2_calibrated_judge(
        text=text,
        council_result=council,
        doc_id=doc_id,
        rag_context=rag,
        return_usage=True,
        metadata=metadata,  # Judge sees exact stats
    )

    # Legacy Adapter
    legacy_output = S2Output(
        label=result.label,
        rationale=result.rationale,
        confidence=result.confidence,
        key_evidence=result.key_evidence,
    )

    logger.info(f"[{doc_id}] Judge: {result.label} (Conf: {result.confidence:.2f})")

    return {
        "calibrated_output": result,
        "final_output": legacy_output,
        "token_usage": usage,
    }


# ===========================================================================
# 5. GRAPH WIRING
# ===========================================================================

parallel_workflow = StateGraph(S2ParallelGraphState)

# Add Nodes
# 1. Define Nodes FIRST
parallel_workflow.add_node("profiler", forensic_profiler_node)
parallel_workflow.add_node("parallel_council", s2_parallel_council_node)
parallel_workflow.add_node("calibrated_judge", s2_calibrated_judge_node)
parallel_workflow.add_node(
    "appeal_court", appeal_court_node
)  # <--- Define BEFORE using in edges

# 2. Define Edges SECOND
parallel_workflow.add_edge(START, "profiler")
parallel_workflow.add_edge("profiler", "parallel_council")
parallel_workflow.add_edge("parallel_council", "calibrated_judge")

# 3. Conditional Logic
parallel_workflow.add_conditional_edges(
    "calibrated_judge",
    check_verdict_quality,
    {END: END, "appeal": "appeal_court"},  # Valid because 'appeal_court' node exists
)

# 4. Final Edge
parallel_workflow.add_edge(
    "appeal_court", END
)  # Valid because 'appeal_court' node exists

s2_parallel_graph = parallel_workflow.compile()
