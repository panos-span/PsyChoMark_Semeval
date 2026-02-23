"""
psycomark.agents.s2_agents_lite — Lite S2 Agent for Local Models.

Simplified 2-juror council + lite judge for endorsement classification.
Uses LiteVote/LiteJudgeOutput schemas — designed for small models (e.g. Qwen3-8B).
"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from loguru import logger
from pydantic_ai import Agent, ModelSettings

from psycomark.config import LLM, AGENT_RETRIES, OPENAI_SEMAPHORE, safe_agent_run
from psycomark.schemas.s2 import S2Deps, S2Juror
from psycomark.schemas.s2_lite import LiteJudgeOutput, LiteVote


# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

LITE_PROSECUTOR_SYS = """\
You are a PROSECUTOR juror analyzing social media text for conspiracy endorsement.

Your job is to determine whether the author ENDORSES a conspiracy theory.

- "conspiracy" = the author promotes, supports, or sincerely believes a conspiratorial claim.
- "non" = the author is reporting, debunking, being sarcastic, or neutrally discussing.

Be alert to implicit endorsement: rhetorical questions that assume a conspiracy,
"just asking questions" patterns, and us-vs-them framing.

Respond with your verdict, confidence (0.0-1.0), and a one-sentence rationale.
"""

LITE_DEFENSE_SYS = """\
You are a DEFENSE juror analyzing social media text for conspiracy endorsement.

Your job is to determine whether the author ENDORSES a conspiracy theory.
You should apply Hanlon's Razor — prefer non-conspiratorial explanations when possible.

- "conspiracy" = the author promotes, supports, or sincerely believes a conspiratorial claim.
- "non" = the author is reporting, debunking, being sarcastic, or neutrally discussing.

Be skeptical of surface-level conspiracy language — it may be satire, reporting, or criticism.

Respond with your verdict, confidence (0.0-1.0), and a one-sentence rationale.
"""

LITE_JUDGE_SYS = """\
You are a JUDGE making the final classification decision.

You will receive:
1. The original text
2. A summary of two jurors' votes (Prosecutor and Defense)

Rules:
- If both jurors agree: follow their verdict.
- If they disagree: carefully weigh the evidence and make your own call.
- Default to "non" in ambiguous cases.

Respond with your final label, confidence (0.0-1.0), and rationale.
"""


# ---------------------------------------------------------------------------
# Lite Juror Factory
# ---------------------------------------------------------------------------


def create_lite_juror_agent(
    system_prompt: str,
    temperature: float = 0.4,
) -> Agent[S2Deps, LiteVote]:
    """Create a lite juror agent."""
    return Agent(
        LLM,
        output_type=LiteVote,
        deps_type=S2Deps,
        system_prompt=system_prompt,
        model_settings=ModelSettings(temperature=temperature),
        retries=AGENT_RETRIES,
    )


# ---------------------------------------------------------------------------
# Lite 2-Juror Council
# ---------------------------------------------------------------------------


async def run_s2_lite_council(
    text: str,
    s1_spans: List[dict],
    marker_summary: str,
    rag_context: str = "",
    temperature: float = 0.4,
    metadata: Dict = {},
    return_usage: bool = False,
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, int]]]:
    """
    Lite 2-juror council: Prosecutor + Defense vote independently.

    Returns a dict with votes, tally, and consensus info.
    """
    deps = S2Deps(
        raw_text=text,
        s1_markers=s1_spans,
        marker_summary=marker_summary,
        rag_context=rag_context,
    )

    subreddit = metadata.get("subreddit", "Unknown")
    user_msg = (
        f"Classify this text as 'conspiracy' or 'non'.\n\n"
        f"Source: r/{subreddit}\n"
        f"Marker Summary: {marker_summary}\n\n"
        f"Text:\n{text}"
    )

    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    juror_configs = [
        (S2Juror.BELIEVER, LITE_PROSECUTOR_SYS),
        (S2Juror.DEFENSE, LITE_DEFENSE_SYS),
    ]

    valid_votes: list[dict] = []

    async with OPENAI_SEMAPHORE:
        for role, sys_prompt in juror_configs:
            try:
                agent = create_lite_juror_agent(sys_prompt, temperature)
                res = await safe_agent_run(agent, user_msg, deps)

                if hasattr(res, "usage"):
                    u = res.usage()
                    total_usage["input_tokens"] += getattr(u, "request_tokens", 0) or 0
                    total_usage["output_tokens"] += getattr(u, "response_tokens", 0) or 0
                    total_usage["total_tokens"] += getattr(u, "total_tokens", 0) or 0

                vote = res.output
                valid_votes.append({
                    "juror": role.value,
                    "verdict": vote.verdict,
                    "confidence": vote.confidence,
                    "rationale": vote.rationale,
                })

            except Exception as e:
                logger.warning(f"[Lite Council] {role.value} failed: {e}")

    tally = Counter(v["verdict"] for v in valid_votes)
    total = len(valid_votes)

    if total == 0:
        consensus = "chaotic"
    elif len(tally) == 1:
        consensus = "unanimous"
    else:
        consensus = "split"

    result = {
        "votes": valid_votes,
        "tally": dict(tally),
        "consensus_level": consensus,
    }

    return (result, total_usage) if return_usage else result


# ---------------------------------------------------------------------------
# Lite Judge
# ---------------------------------------------------------------------------


async def run_s2_lite_judge(
    text: str,
    council_result: Dict[str, Any],
    doc_id: str = "unknown",
    rag_context: str = "",
    return_usage: bool = False,
    metadata: Dict[str, Any] = {},
) -> Union[LiteJudgeOutput, Tuple[LiteJudgeOutput, Dict[str, int]]]:
    """
    Lite judge: makes final classification based on council votes.
    """
    deps = S2Deps(raw_text=text, doc_id=doc_id, rag_context=rag_context)

    # Build vote transcript
    vote_lines = []
    for v in council_result.get("votes", []):
        vote_lines.append(
            f"- {v['juror']}: {v['verdict'].upper()} "
            f"(confidence: {v['confidence']:.2f}) — {v['rationale']}"
        )
    transcript = "\n".join(vote_lines) if vote_lines else "No votes received."

    user_msg = (
        f"Make the final classification for document {doc_id}.\n\n"
        f"## Council Votes\n{transcript}\n\n"
        f"## Tally: {council_result.get('tally', {})}\n"
        f"## Consensus: {council_result.get('consensus_level', 'unknown')}\n\n"
        f"## Original Text\n{text}"
    )

    judge_agent = Agent(
        LLM,
        output_type=LiteJudgeOutput,
        system_prompt=LITE_JUDGE_SYS,
        retries=AGENT_RETRIES,
    )

    usage_dict = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    try:
        async with OPENAI_SEMAPHORE:
            res = await safe_agent_run(judge_agent, user_msg, deps=deps)
            if hasattr(res, "usage"):
                u = res.usage()
                usage_dict["input_tokens"] = getattr(u, "request_tokens", 0) or 0
                usage_dict["output_tokens"] = getattr(u, "response_tokens", 0) or 0
                usage_dict["total_tokens"] = getattr(u, "total_tokens", 0) or 0

            output = res.output

            # Default split cases to "non"
            if council_result.get("consensus_level") == "split":
                if output.confidence < 0.65:
                    output.label = "non"

            return (output, usage_dict) if return_usage else output

    except Exception as e:
        logger.error(f"[Lite Judge] Failed: {e}")
        fallback = LiteJudgeOutput(label="non", confidence=0.0, rationale="Error")
        return (fallback, usage_dict) if return_usage else fallback
