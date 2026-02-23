"""
psycomark.agents.s2_agents — S2 Endorsement Classification Agents.

Implements the Anti-Echo Chamber pipeline:
    1. Parallel Council: Four adversarial personas vote independently
       (Prosecutor, Defense, Literalist, Profiler)
    2. Calibrated Judge: Dissent-aware final adjudication with
       confidence damping based on council consensus

Also provides:
    - ``synthesize_dossier``: Converts S1 markers into a forensic summary for S2
    - ``format_s2_rag_to_xml``: Formats RAG precedents for prompt injection
"""

from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from loguru import logger
from pydantic_ai import Agent, ModelSettings

from psycomark.config import LLM, OPENAI_SEMAPHORE, safe_agent_run
from psycomark.schemas.s2 import (
    CalibratedJudgeOutput,
    EnhancedS2Vote,
    ParallelCouncilOutput,
    S2Deps,
    S2Juror,
    S2Output,
)


# ---------------------------------------------------------------------------
# Prompt Assembly Utilities
# ---------------------------------------------------------------------------


def format_s2_rag_to_xml(rag_context: str) -> str:
    """Wrap raw RAG context in XML tags if not already wrapped."""
    if not rag_context:
        return "No relevant case law found."
    if "<legal_precedents_context>" in rag_context:
        return rag_context
    return f"<legal_precedents>\n{rag_context}\n</legal_precedents>"


def assemble_s2_system_prompt(
    base_prompt: str,
    rag_context: str,
    metadata: dict,
    use_markdown: bool = True,
) -> str:
    """
    Hydrate an S2 system prompt with RAG context and source metadata.

    Supports both Markdown (GPT-optimised) and XML (Claude legacy) formats.
    """
    prompt = base_prompt

    rag_content = (
        rag_context
        if rag_context and len(rag_context) > 10
        else "No relevant legal precedents found."
    )
    subreddit = metadata.get("subreddit") or metadata.get("source") or "Unknown"
    if subreddit.startswith("r/"):
        subreddit = subreddit[2:]
    prior_text = (
        "Conspiracy Hub"
        if subreddit.lower() in ("conspiracy", "highstrangeness")
        else "Mainstream/Neutral"
    )

    if use_markdown:
        md_rag = f"## Legal Precedents (RAG Context)\n{rag_content}"
        if "{{rag_context}}" in prompt:
            prompt = prompt.replace("{{rag_context}}", md_rag)
        else:
            prompt += f"\n\n{md_rag}"

        md_source = f"## Source Context\n**Source:** r/{subreddit}\n**Contextual Prior:** {prior_text}"
        if "{{source_context}}" in prompt:
            prompt = prompt.replace("{{source_context}}", md_source)
        else:
            prompt = f"{md_source}\n\n{prompt}"
    else:
        if "{{rag_context}}" in prompt:
            prompt = prompt.replace("{{rag_context}}", rag_content)
        xml_source = f"<source_context>\n  SOURCE: r/{subreddit}\n  (Contextual Prior: {prior_text})\n</source_context>"
        if "{{source_context}}" in prompt:
            prompt = prompt.replace("{{source_context}}", xml_source)

    return prompt


# ---------------------------------------------------------------------------
# Juror Agent Factory
# ---------------------------------------------------------------------------


def create_parallel_juror_agent(
    role: S2Juror,
    deps: S2Deps,
    system_prompt: str,
    temperature: float = 0.4,
) -> Agent[S2Deps, EnhancedS2Vote]:
    """Create an ephemeral juror agent for parallel council voting."""
    return Agent(
        LLM,
        output_type=EnhancedS2Vote,
        deps_type=S2Deps,
        system_prompt=system_prompt,
        model_settings=ModelSettings(temperature=temperature),
        retries=2,
    )


# ---------------------------------------------------------------------------
# Parallel Council Runner
# ---------------------------------------------------------------------------


async def run_s2_parallel_council(
    text: str,
    s1_spans: List[dict],
    marker_summary: str,
    rag_context: str = "",
    temperature: float = 0.4,
    metadata: Dict = {},
    # GEPA system prompt overrides
    prosecutor_sys_override: Optional[str] = None,
    defense_sys_override: Optional[str] = None,
    literalist_sys_override: Optional[str] = None,
    profiler_sys_override: Optional[str] = None,
    # GEPA user template override
    parallel_user_template_override: Optional[str] = None,
    return_usage: bool = False,
    active_jurors: Optional[List[S2Juror]] = None,
) -> Union[ParallelCouncilOutput, Tuple[ParallelCouncilOutput, Dict[str, int]]]:
    """
    Parallel council: all four jurors vote **independently**.

    Anti-echo-chamber guarantees:
        1. No juror sees another's vote
        2. Each juror must steelman the opposing view
        3. Uncertainty flags are collected for Judge calibration
    """
    from psycomark.prompts.loader import S2_PROMPTS
    from psycomark.prompts.builder import build_s2_parallel_user_template

    deps = S2Deps(
        raw_text=text,
        s1_markers=s1_spans,
        marker_summary=marker_summary,
        rag_context=rag_context,
    )

    # Shared user prompt (same evidence, no prior votes)
    user_template = (
        parallel_user_template_override
        or getattr(S2_PROMPTS, "parallel_user", None)
        or build_s2_parallel_user_template()
    )

    subreddit = metadata.get("subreddit", "Unknown Source")
    user_msg = (
        user_template.replace("{{text}}", text)
        .replace("{{marker_summary}}", marker_summary)
        .replace("{{rag_context}}", rag_context)
        .replace("{{subreddit}}", subreddit)
    )

    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def _get_hydrated_sys(role: S2Juror, override: Optional[str]) -> str:
        base = override
        if not base:
            attr_map = {
                S2Juror.BELIEVER: "parallel_pros_sys",
                S2Juror.DEFENSE: "parallel_def_sys",
                S2Juror.LITERALIST: "parallel_lit_sys",
                S2Juror.PROFILER: "parallel_prof_sys",
            }
            base = getattr(S2_PROMPTS, attr_map.get(role, ""), "")
        return assemble_s2_system_prompt(base, rag_context, metadata)

    all_configs = [
        (S2Juror.BELIEVER, prosecutor_sys_override),
        (S2Juror.DEFENSE, defense_sys_override),
        (S2Juror.LITERALIST, literalist_sys_override),
        (S2Juror.PROFILER, profiler_sys_override),
    ]

    if active_jurors is not None:
        juror_configs = [c for c in all_configs if c[0] in active_jurors]
    else:
        juror_configs = all_configs

    valid_votes: list[EnhancedS2Vote] = []

    # Serial execution with semaphore to respect rate limits
    async with OPENAI_SEMAPHORE:
        for role, sys_override in juror_configs:
            try:
                final_sys = _get_hydrated_sys(role, sys_override)
                agent = create_parallel_juror_agent(role, deps, final_sys, temperature)
                res = await safe_agent_run(agent, user_msg, deps)

                if hasattr(res, "usage"):
                    u = res.usage()
                    total_usage["input_tokens"] += u.request_tokens or 0
                    total_usage["output_tokens"] += u.response_tokens or 0
                    total_usage["total_tokens"] += u.total_tokens or 0

                if res:
                    vote = res.output
                    vote.juror = role
                    valid_votes.append(vote)

                await asyncio.sleep(1)  # rate-limit courtesy

            except Exception as e:
                logger.warning(f"[Parallel Council] {role.value} failed: {e}")

    # Compute aggregates
    tally = Counter(v.verdict for v in valid_votes)
    consp = [v for v in valid_votes if v.verdict == "conspiracy"]
    non = [v for v in valid_votes if v.verdict == "non"]

    consp_avg = sum(v.confidence for v in consp) / len(consp) if consp else 0.0
    non_avg = sum(v.confidence for v in non) / len(non) if non else 0.0
    weighted = sum(
        v.confidence if v.verdict == "conspiracy" else -v.confidence
        for v in valid_votes
    )

    total = len(valid_votes)
    majority = max(tally.values()) if tally else 0
    minority = min(tally.values()) if len(tally) > 1 else 0
    dissent = minority / total if total > 0 else 0.0

    if total == 0:
        consensus: Literal["unanimous", "strong", "split", "chaotic"] = "chaotic"
    elif majority == total:
        consensus = "unanimous"
    elif majority >= 3:
        consensus = "strong"
    else:
        consensus = "split"

    # Aggregate uncertainty flags
    all_flags: list[str] = []
    for v in valid_votes:
        all_flags.extend(v.uncertainty_flags)
    common = [f for f, c in Counter(all_flags).items() if c >= 2]

    output = ParallelCouncilOutput(
        votes=valid_votes,
        tally=dict(tally),
        conspiracy_confidence_avg=consp_avg,
        non_confidence_avg=non_avg,
        weighted_score=weighted,
        dissent_strength=dissent,
        consensus_level=consensus,
        common_uncertainty_flags=common,
    )

    return (output, total_usage) if return_usage else output


# ---------------------------------------------------------------------------
# Calibrated Judge
# ---------------------------------------------------------------------------


async def run_s2_calibrated_judge(
    text: str,
    council_result: ParallelCouncilOutput,
    doc_id: str = "unknown",
    rag_context: str = "",
    judge_sys_override: Optional[str] = None,
    judge_user_template_override: Optional[str] = None,
    return_usage: bool = False,
    metadata: Dict[str, Any] = {},
) -> Union[CalibratedJudgeOutput, Tuple[CalibratedJudgeOutput, Dict[str, int]]]:
    """
    Calibrated Judge: dissent-aware adjudication.

    Confidence damping:
        - Unanimous (4-0): 0.95
        - Strong (3-1): capped at 0.80
        - Split (2-2): capped at 0.65, defaults to 'non'
    """
    from psycomark.prompts.loader import S2_PROMPTS
    from psycomark.prompts.builder import (
        build_s2_calibrated_judge_system,
        build_s2_calibrated_judge_user_template,
    )

    subreddit = metadata.get("subreddit", "Unknown")
    kill_zones = {"conspiracy", "HighStrangeness", "Wuhan_Flu", "LockdownSkepticism"}
    safe_zones = {"news", "worldnews", "science", "skeptic"}

    if subreddit in kill_zones:
        ctx_note = f"> **CONTEXT ALERT:** r/{subreddit} (High Probability Conspiracy)"
    elif subreddit in safe_zones:
        ctx_note = f"> **CONTEXT ALERT:** r/{subreddit} (High Probability Reporting)"
    else:
        ctx_note = f"> **Source Context:** r/{subreddit}"

    # Forensic stats block
    stats_keys = [
        "uncertainty_ratio",
        "question_density",
        "is_jaqing",
        "agency_gap",
        "epistemic_intensity",
        "shouting_score",
    ]
    stats_lines = []
    for k in stats_keys:
        if k in metadata:
            val = metadata[k]
            val_str = (
                ("YES" if val else "NO")
                if isinstance(val, bool)
                else (f"{val:.2f}" if isinstance(val, float) else str(val))
            )
            stats_lines.append(f"- **{k.replace('_', ' ').title()}**: {val_str}")
    forensic_stats_str = (
        "\n".join(stats_lines) if stats_lines else "No forensic stats available."
    )

    deps = S2Deps(raw_text=text, doc_id=doc_id, rag_context=rag_context)

    # Build vote transcript
    vote_lines = []
    for v in council_result.votes:
        vote_lines.append(
            f"### Juror: {v.juror.value.upper()}\n"
            f"- **Verdict:** {v.verdict.upper()}\n"
            f"- **Confidence:** {v.confidence:.2f}\n"
            f"- **Rationale:** {v.rationale}\n"
            f"- **Key Signal:** {v.key_signal}\n"
            f"- **Steelman:** {v.steelman_opposing}\n"
            f"- **Flags:** {', '.join(v.uncertainty_flags) if v.uncertainty_flags else 'None'}\n"
        )
    transcript = "\n".join(vote_lines)

    council_analysis = (
        f"## Council Synthesis\n"
        f"- **Vote Tally:** {council_result.tally}\n"
        f"- **Weighted Score:** {council_result.weighted_score:.2f}\n"
        f"- **Consensus:** {council_result.consensus_level.upper()}\n"
        f"- **Dissent Strength:** {council_result.dissent_strength:.2f}\n"
        f"- **Avg Confidence (Conspiracy):** {council_result.conspiracy_confidence_avg:.2f}\n"
        f"- **Avg Confidence (Non):** {council_result.non_confidence_avg:.2f}\n"
        f"- **Common Flags:** {', '.join(council_result.common_uncertainty_flags) or 'None'}\n\n"
        f"{ctx_note}"
    )

    # Resolve prompts
    base_sys = (
        judge_sys_override
        or getattr(S2_PROMPTS, "calibrated_judge_sys", None)
        or build_s2_calibrated_judge_system()
    )
    full_sys = (
        f"{base_sys}\n\n<legal_precedents>\n{rag_context}\n</legal_precedents>"
        if rag_context
        else base_sys
    )

    usr_tmpl = (
        judge_user_template_override
        or getattr(S2_PROMPTS, "calibrated_judge_user", None)
        or build_s2_calibrated_judge_user_template()
    )

    user_prompt = (
        usr_tmpl.replace("{{text}}", text)
        .replace("{{transcript}}", transcript)
        .replace("{{council_analysis}}", council_analysis)
        .replace("{{id}}", doc_id)
        .replace("{{forensic_stats}}", forensic_stats_str)
    )

    judge_agent = Agent(
        LLM, output_type=CalibratedJudgeOutput, system_prompt=full_sys, retries=2
    )
    usage_dict = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    try:
        async with OPENAI_SEMAPHORE:
            res = await safe_agent_run(judge_agent, user_prompt, deps=deps)
            if hasattr(res, "usage"):
                u = res.usage()
                usage_dict["input_tokens"] = u.request_tokens or 0
                usage_dict["output_tokens"] = u.response_tokens or 0
                usage_dict["total_tokens"] = u.total_tokens or 0
            output = res.output

            if council_result.consensus_level in ("split", "chaotic"):
                output.borderline_flag = True

            majority_verdict = (
                max(council_result.tally.keys(), key=lambda k: council_result.tally[k])
                if council_result.tally
                else "non"
            )
            if (
                output.label != majority_verdict
                and council_result.consensus_level != "split"
            ):
                output.council_override = True
                logger.warning(
                    f"[Judge] OVERRIDE: {output.label} vs council {majority_verdict}"
                )

            return (output, usage_dict) if return_usage else output

    except Exception as e:
        logger.error(f"[Calibrated Judge] Failed: {e}")
        fallback = CalibratedJudgeOutput(
            label="non", confidence=0.0, rationale="Error", dissent_considered=False
        )
        return (fallback, usage_dict) if return_usage else fallback


# ---------------------------------------------------------------------------
# S1 → S2 Bridge Utilities
# ---------------------------------------------------------------------------


def synthesize_dossier(
    markers: List[Dict],
    complexity: str = "Unknown",
    narrative: str = "Unknown",
) -> str:
    """Transform S1 markers into a readable forensic summary for S2 input."""
    if not markers:
        return "No markers found."

    buckets: Dict[str, set] = defaultdict(set)
    for m in markers:
        txt = m.get("text") if isinstance(m, dict) else m.text
        lbl = m.get("type") if isinstance(m, dict) else m.label
        txt = " ".join(str(txt).split())
        if lbl is None:
            lbl = "Unknown"
        elif hasattr(lbl, "value"):
            lbl = lbl.value
        buckets[str(lbl).capitalize()].add(f'"{txt}"')

    lines = [
        f"DYNAMIC ASSESSMENT: Complexity={complexity.upper()} | Narrative={narrative.upper()}",
        "-" * 40,
    ]
    if buckets["Evidence"]:
        lines.append(f"EVIDENTIAL BASIS: {', '.join(buckets['Evidence'])}")
    else:
        lines.append("EVIDENTIAL BASIS: None (Assertion only).")
    for key, label in [
        ("Actor", "ALLEGED PERPETRATORS"),
        ("Action", "ALLEGED METHODS"),
        ("Effect", "ALLEGED OUTCOMES"),
        ("Victim", "ALLEGED VICTIMS"),
    ]:
        if buckets[key]:
            lines.append(f"{label} ({key}s): {', '.join(buckets[key])}")

    return "\n".join(lines)
