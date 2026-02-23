"""
psycomark.graphs.s2_graph — S2 Anti-Echo Chamber LangGraph Workflow.

Pipeline topology (linear):

    Forensic Profiler ─→ Parallel Council ─→ Calibrated Judge

Features:
    - **Forensic Stats**: Static linguistic metrics (attribution density,
      JAQ detection, epistemic intensity, agency gap, shouting score)
    - **Anti-Echo Chamber**: Four jurors vote *independently*
    - **Confidence Damping**: Split council caps judge confidence
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from loguru import logger

from psycomark.schemas.s2 import (
    CalibratedJudgeOutput,
    ParallelCouncilOutput,
    S2Output,
)
from psycomark.agents.s2_agents import (
    run_s2_calibrated_judge,
    run_s2_parallel_council,
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
# Forensic Stats Type
# ---------------------------------------------------------------------------


class ForensicStats(TypedDict):
    uncertainty_ratio: float
    question_density: float
    is_jaqing: bool
    agency_gap: float
    epistemic_intensity: float
    shouting_score: float
    attribution_density: float


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class S2ParallelGraphState(TypedDict):
    doc_id: str
    text: str

    s1_markers: Optional[List[dict]]
    s1_spans: Optional[List[dict]]

    marker_summary: str
    rag_context: str
    metadata: Dict[str, Any]
    juror_temperature: float

    forensic_stats: ForensicStats
    council_output: Optional[ParallelCouncilOutput]
    calibrated_output: Optional[CalibratedJudgeOutput]
    final_output: Optional[S2Output]

    token_usage: Annotated[Dict[str, int], _aggregate_usage]


# ---------------------------------------------------------------------------
# Node: Forensic Profiler (static — no LLM call)
# ---------------------------------------------------------------------------


def _calculate_forensic_stats(text: str) -> ForensicStats:
    raw_words = re.findall(r"\b[A-Za-z]+\b", text)
    total_words = len(raw_words) if raw_words else 1
    text_lower = text.lower()
    words_lower = re.findall(r"\w+", text_lower)

    # Attribution density (reporter detector)
    attribution_verbs = {
        "said",
        "says",
        "stated",
        "claimed",
        "claims",
        "reported",
        "reporting",
        "according",
        "cited",
        "quoted",
        "tweeted",
        "announced",
        "sources",
    }
    attribution_count = sum(1 for w in words_lower if w in attribution_verbs)
    attribution_density = (attribution_count / total_words) * 100

    # Uncertainty / hedging
    hedging = {
        "maybe",
        "perhaps",
        "possibly",
        "could",
        "might",
        "wonder",
        "unsure",
        "question",
        "curious",
        "seem",
        "appears",
        "allegedly",
        "unknown",
    }
    hedging_count = sum(1 for w in words_lower if w in hedging)
    uncertainty_ratio = hedging_count / total_words

    # Epistemic intensity
    truth_terms = {
        "proof",
        "proven",
        "truth",
        "fact",
        "undeniable",
        "obvious",
        "clear",
        "expose",
        "revealed",
        "woke",
        "awakened",
        "reality",
        "lying",
    }
    intensity_count = sum(1 for w in words_lower if w in truth_terms)
    epistemic_intensity = intensity_count / total_words

    # Agency gap (passive voice)
    passive = {"been", "being", "was", "were", "by"}
    agency_gap = sum(1 for w in words_lower if w in passive) / total_words

    # JAQ detection
    question_marks = text.count("?")
    sentence_count = len(re.split(r"[.!?]+", text)) or 1
    question_density = question_marks / sentence_count
    is_jaqing = question_density > 0.35 and uncertainty_ratio > 0.05

    # Shouting score (CAPS > 1 char, excludes I/A)
    shouting_words = [w for w in raw_words if w.isupper() and len(w) > 1]
    shouting_score = len(shouting_words) / total_words

    return {
        "uncertainty_ratio": round(uncertainty_ratio, 3),
        "epistemic_intensity": round(epistemic_intensity, 3),
        "agency_gap": round(agency_gap, 3),
        "is_jaqing": is_jaqing,
        "shouting_score": round(shouting_score, 3),
        "attribution_density": round(attribution_density, 2),
        "question_density": round(question_density, 2),
    }


def _profiler_node(state: S2ParallelGraphState) -> Dict:
    text = state["text"]
    stats = _calculate_forensic_stats(text)

    warnings: list[str] = []
    if stats["attribution_density"] > 3.5:
        warnings.append(f"HIGH ATTRIBUTION ({stats['attribution_density']}%)")
    if stats["is_jaqing"]:
        warnings.append("JAQ PATTERN (High Uncertainty + Questions)")
    if stats["shouting_score"] > 0.10:
        warnings.append(f"HIGH EMOTION (Caps: {stats['shouting_score']*100:.1f}%)")
    if stats["agency_gap"] > 0.06:
        warnings.append(f"HIGH PASSIVE VOICE (Hidden Hands: {stats['agency_gap']:.2f})")

    stats_block = (
        f"[LINGUISTIC METRICS]\n"
        f"- Attribution Density: {stats['attribution_density']}%\n"
        f"- Uncertainty Ratio: {stats['uncertainty_ratio']:.3f}\n"
        f"- Shouting Score: {stats['shouting_score']:.3f}\n"
        f"- Agency Gap: {stats['agency_gap']:.3f}\n\n"
        f"[ACTIVE WARNINGS]\n" + ("\n".join(warnings) if warnings else "None detected.")
    )

    new_meta = state.get("metadata", {}).copy()
    new_meta["forensic_stats"] = stats_block
    new_meta.update(stats)

    return {"forensic_stats": stats, "metadata": new_meta}


# ---------------------------------------------------------------------------
# Node: Parallel Council
# ---------------------------------------------------------------------------


async def _council_node(state: S2ParallelGraphState):
    doc_id = state["doc_id"]
    temp = state.get("juror_temperature", 0.4)
    rag = state.get("rag_context", "")
    metadata = state.get("metadata", {})

    markers = state.get("s1_markers") or state.get("s1_spans") or []

    summary_base = state.get("marker_summary", "No markers found.")
    if not markers and "No markers" in summary_base:
        summary_base += (
            "\n[NOTE]: No structural markers found. Rely on TONE and INSINUATION."
        )

    forensic_ctx = ""
    stats = state.get("forensic_stats", {})
    if stats.get("attribution_density", 0) > 3.5:
        forensic_ctx += (
            "\n[CONTEXT]: High Attribution. Watch for Reporting vs Endorsement."
        )
    if stats.get("is_jaqing"):
        forensic_ctx += "\n[CONTEXT]: High Question Density (JAQing pattern)."

    enhanced_summary = f"{summary_base}\n{forensic_ctx}"

    logger.info(f"[{doc_id}] S2 Council Session Started…")

    result, usage = await run_s2_parallel_council(
        text=state["text"],
        s1_spans=markers,
        marker_summary=enhanced_summary,
        rag_context=rag,
        temperature=temp,
        metadata=metadata,
        return_usage=True,
    )

    logger.info(
        f"[{doc_id}] Council Verdict: {result.tally} ({result.consensus_level})"
    )
    return {"council_output": result, "token_usage": usage}


# ---------------------------------------------------------------------------
# Node: Calibrated Judge (with Programmatic Damping)
# ---------------------------------------------------------------------------


async def _judge_node(state: S2ParallelGraphState):
    council = state["council_output"]
    text = state["text"]
    rag = state.get("rag_context", "")
    doc_id = state["doc_id"]
    metadata = state.get("metadata", {})

    result, usage = await run_s2_calibrated_judge(
        text=text,
        council_result=council,
        doc_id=doc_id,
        rag_context=rag,
        return_usage=True,
        metadata=metadata,
    )

    final_conf = result.confidence
    if council and council.consensus_level in ["split", "chaotic"]:
        if final_conf > 0.75:
            logger.warning(
                f"[{doc_id}] Damping Confidence {final_conf:.2f} -> 0.75 (Split Council)"
            )
            final_conf = 0.75

    final = S2Output(
        label=result.label,
        rationale=result.rationale,
        confidence=final_conf,
        key_evidence=result.key_evidence,
    )

    logger.info(f"[{doc_id}] Verdict: {final.label} (Conf: {final.confidence:.2f})")
    return {"calibrated_output": result, "final_output": final, "token_usage": usage}


# ---------------------------------------------------------------------------
# Graph Compilation
# ---------------------------------------------------------------------------

_wf = StateGraph(S2ParallelGraphState)
_wf.add_node("profiler", _profiler_node)
_wf.add_node("parallel_council", _council_node)
_wf.add_node("calibrated_judge", _judge_node)

_wf.add_edge(START, "profiler")
_wf.add_edge("profiler", "parallel_council")
_wf.add_edge("parallel_council", "calibrated_judge")
_wf.add_edge("calibrated_judge", END)

s2_graph = _wf.compile()
