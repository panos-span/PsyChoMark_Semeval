"""
psycomark.agents.s1_agents — S1 Marker Span Extraction Agents.

Implements the DD-CoT Self-Refine pipeline:
    1. DD-CoT Generator: Extracts spans with discriminative reasoning
    2. Enhanced Critic: Multi-dimensional quality audit
    3. Refiner: Corrects extractions preserving valid spans
    4. Deterministic Verifier: Non-LLM span grounding (see span_utils)

Agent factories support dynamic prompt injection for GEPA optimization.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from pydantic_ai import Agent, ModelSettings, RunContext

from psycomark.agents.span_utils import find_best_span
from psycomark.config import LLM, OPENAI_SEMAPHORE, safe_agent_run
from psycomark.schemas.s1 import (DDCoTExtraction, DDCoTRefinement, DDCoTSpan,
                                  EnhancedS1Critique, S1Deps, S1Label, S1Span)

# ---------------------------------------------------------------------------
# Agent Factories (support GEPA prompt injection)
# ---------------------------------------------------------------------------


def get_s1_ddcot_generator(
    system_prompt_override: Optional[str] = None,
) -> Agent[S1Deps, DDCoTExtraction]:
    """
    Factory for DD-CoT Generator agent.

    Uses higher temperature (0.7) for exploratory extraction.
    """
    from psycomark.prompts.loader import S1_PROMPTS

    if system_prompt_override:
        sys_prompt = system_prompt_override
    elif hasattr(S1_PROMPTS, "ddcot_gen_system"):
        sys_prompt = S1_PROMPTS.ddcot_gen_system
    else:
        from psycomark.prompts.builder import build_s1_ddcot_system

        sys_prompt = build_s1_ddcot_system()

    return Agent(
        model=LLM,
        output_type=DDCoTExtraction,
        deps_type=S1Deps,
        system_prompt=sys_prompt,
        model_settings=ModelSettings(temperature=0.7),
    )


def get_s1_ddcot_critic(
    system_prompt_override: Optional[str] = None,
) -> Agent[S1Deps, EnhancedS1Critique]:
    """
    Factory for Enhanced Critic agent.

    Uses temperature=0.0 for deterministic auditing.
    """
    from psycomark.prompts.loader import S1_PROMPTS

    if system_prompt_override:
        sys_prompt = system_prompt_override
    elif hasattr(S1_PROMPTS, "ddcot_critic_system"):
        sys_prompt = S1_PROMPTS.ddcot_critic_system
    else:
        from psycomark.prompts.builder import build_s1_ddcot_critic_system

        sys_prompt = build_s1_ddcot_critic_system()

    return Agent(
        model=LLM,
        output_type=EnhancedS1Critique,
        deps_type=S1Deps,
        system_prompt=sys_prompt,
        model_settings=ModelSettings(temperature=0.0),
    )


def get_s1_ddcot_refiner(
    system_prompt_override: Optional[str] = None,
) -> Agent[S1Deps, DDCoTRefinement]:
    """
    Factory for DD-CoT Refiner agent.

    Uses temperature=0.0 for strict compliance.
    """
    from psycomark.prompts.loader import S1_PROMPTS

    if system_prompt_override:
        sys_prompt = system_prompt_override
    elif hasattr(S1_PROMPTS, "ddcot_refiner_system"):
        sys_prompt = S1_PROMPTS.ddcot_refiner_system
    else:
        from psycomark.prompts.builder import build_s1_ddcot_refiner_system

        sys_prompt = build_s1_ddcot_refiner_system()

    return Agent(
        model=LLM,
        output_type=DDCoTRefinement,
        deps_type=S1Deps,
        system_prompt=sys_prompt,
        model_settings=ModelSettings(temperature=0.0),
    )


# ---------------------------------------------------------------------------
# Conversion & Validation Utilities
# ---------------------------------------------------------------------------


def ddcot_span_to_s1_span(ddcot_span: DDCoTSpan) -> S1Span:
    """Convert ``DDCoTSpan`` to legacy ``S1Span``."""
    return S1Span(
        label=ddcot_span.label,
        text=ddcot_span.text,
        start=ddcot_span.start,
        end=ddcot_span.end,
        why=ddcot_span.why_this_label,
    )


def validate_ddcot_extraction(
    extraction: DDCoTExtraction, raw_text: str
) -> Tuple[DDCoTExtraction, List[str]]:
    """
    Post-extraction validator with boundary trimming.

    Grounds every extracted span in the source text, applying:
        1. Exact / best-match search
        2. Leading / trailing word trimming (±1 word tolerance)
        3. Whitespace normalisation fallback

    Returns ``(cleaned_extraction, issue_log)``.
    """
    valid_spans: list[DDCoTSpan] = []
    issues: list[str] = []

    for span in extraction.extractions:
        start, end = find_best_span(raw_text, span.text)

        # Boundary trimming (+/- 1 word)
        if start == -1:
            words = span.text.split()
            if len(words) > 1:
                # Try removing leading word
                trimmed = " ".join(words[1:])
                s, e = find_best_span(raw_text, trimmed)
                if s != -1:
                    issues.append(
                        f"[RECOVERY] Trimmed leading word: '{span.text}' → '{trimmed}'"
                    )
                    span.text, start, end = trimmed, s, e
                else:
                    # Try removing trailing word
                    trimmed = " ".join(words[:-1])
                    s, e = find_best_span(raw_text, trimmed)
                    if s != -1:
                        issues.append(
                            f"[RECOVERY] Trimmed trailing word: '{span.text}' → '{trimmed}'"
                        )
                        span.text, start, end = trimmed, s, e

        # Whitespace normalisation fallback
        if start == -1:
            norm_span = " ".join(span.text.split())
            if norm_span in " ".join(raw_text.split()):
                s, _ = find_best_span(raw_text, span.text.split()[0])
                if s != -1:
                    start, end = s, min(len(raw_text), s + len(span.text))
                    issues.append(
                        f"[FUZZY] Whitespace mismatch recovered: '{span.text[:30]}...'"
                    )

        if start != -1:
            span.start, span.end = start, end
            valid_spans.append(span)
        else:
            issues.append(
                f"[HALLUCINATION] Span not grounded — removed: '{span.text[:30]}...'"
            )

    cleaned = DDCoTExtraction(
        text_complexity=extraction.text_complexity,
        dominant_narrative=extraction.dominant_narrative,
        extractions=valid_spans,
    )
    return cleaned, issues


# ---------------------------------------------------------------------------
# Few-Shot Formatting
# ---------------------------------------------------------------------------


def format_s1_fewshots_to_markdown(few_shots: List[Dict]) -> str:
    """Format few-shot examples as Markdown (optimised for GPT models)."""
    if not few_shots:
        return ""

    blocks = ["# Reference Examples"]
    for idx, ex in enumerate(few_shots):
        spans = ex.get("spans", [])
        label_val = str(ex.get("label", "")).lower()
        ex_type = (
            "CONSPIRACY_TEXT"
            if label_val in ("conspiracy", "yes", "true")
            else "NEUTRAL_TEXT"
        )

        spans_json = [
            f'{{"label": "{s.get("label", "?")}", "text": "{s.get("text", "")}"}}'
            for s in spans
        ]
        spans_block = "[\n  " + ",\n  ".join(spans_json) + "\n]" if spans_json else "[]"

        note = ""
        if ex_type == "NEUTRAL_TEXT" and spans:
            note = "\n> **Note:** This NEUTRAL text still has structural markers — extract them!"

        text_preview = ex.get("text", "").strip()[:500]
        ellipsis = "..." if len(ex.get("text", "")) > 500 else ""
        blocks.append(
            f"## Example {idx + 1} ({ex_type})\n"
            f"**Input Text:**\n> {text_preview}{ellipsis}\n\n"
            f"**Expected Output:**\n```json\n{spans_block}\n```{note}"
        )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# DD-CoT Pipeline Runner
# ---------------------------------------------------------------------------


async def run_s1_ddcot(
    text: str,
    few_shots: Optional[List[Dict]] = None,
    metadata: Optional[Dict] = None,
    # GEPA prompt overrides
    gen_prompt_override: Optional[str] = None,
    gen_user_template_override: Optional[str] = None,
    critic_prompt_override: Optional[str] = None,
    critic_user_template_override: Optional[str] = None,
    refiner_prompt_override: Optional[str] = None,
    refiner_user_template_override: Optional[str] = None,
    # Control flags
    skip_critic: bool = False,
    return_full_extraction: bool = False,
    return_usage: bool = False,
) -> Union[List[S1Span], DDCoTExtraction, Tuple]:
    """
    DD-CoT Self-Refine pipeline: Generator → Critic → Refiner.

    Args:
        text: Source document text.
        few_shots: Retrieved few-shot examples from RAG.
        metadata: Document metadata (subreddit, etc.).
        skip_critic: If True, skip critic/refiner (ablation mode).
        return_full_extraction: Return ``DDCoTExtraction`` instead of ``List[S1Span]``.
        return_usage: Also return token usage dict.

    Returns:
        List of extracted spans (or full extraction if requested).
    """
    from psycomark.prompts.builder import (
        build_s1_ddcot_critic_user_template,
        build_s1_ddcot_refiner_user_template, build_s1_ddcot_user_template)
    from psycomark.prompts.loader import S1_PROMPTS

    deps = S1Deps(raw_text=text, few_shots=few_shots or [], metadata=metadata or {})
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def _track(result):
        if hasattr(result, "usage"):
            u = result.usage()
            total_usage["input_tokens"] += getattr(u, "request_tokens", 0) or 0
            total_usage["output_tokens"] += getattr(u, "response_tokens", 0) or 0
            total_usage["total_tokens"] += getattr(u, "total_tokens", 0) or 0

    try:
        # --- Step 1: DD-CoT Generator ---
        generator = get_s1_ddcot_generator(gen_prompt_override)

        user_template = (
            gen_user_template_override
            or getattr(S1_PROMPTS, "ddcot_gen_user_template", None)
            or build_s1_ddcot_user_template()
        )

        gen_msg = user_template.replace("{{text}}", text)

        async with OPENAI_SEMAPHORE:
            gen_result = await safe_agent_run(generator, gen_msg, deps)
        _track(gen_result)

        extraction = gen_result.output
        extraction, gen_issues = validate_ddcot_extraction(extraction, text)

        if gen_issues:
            logger.debug(f"[DD-CoT] Validation: {len(gen_issues)} issues resolved")

        if skip_critic:
            spans = [ddcot_span_to_s1_span(s) for s in extraction.extractions]
            result = extraction if return_full_extraction else spans
            return (result, total_usage) if return_usage else result

        # --- Step 2: Enhanced Critic ---
        critic = get_s1_ddcot_critic(critic_prompt_override)

        draft_json = json.dumps(
            [s.model_dump() for s in extraction.extractions], indent=2
        )

        critic_template = (
            critic_user_template_override
            or getattr(S1_PROMPTS, "ddcot_critic_user_template", None)
            or build_s1_ddcot_critic_user_template()
        )

        critic_msg = critic_template.replace("{{text}}", text).replace(
            "{{draft_json}}", draft_json
        )

        async with OPENAI_SEMAPHORE:
            critique_result = await safe_agent_run(critic, critic_msg, deps)
        _track(critique_result)

        critique = critique_result.output
        if not critique.requires_refinement:
            spans = [ddcot_span_to_s1_span(s) for s in extraction.extractions]
            result = extraction if return_full_extraction else spans
            return (result, total_usage) if return_usage else result

        # --- Step 3: Refiner ---
        refiner = get_s1_ddcot_refiner(refiner_prompt_override)

        refiner_template = (
            refiner_user_template_override
            or getattr(S1_PROMPTS, "ddcot_refiner_user_template", None)
            or build_s1_ddcot_refiner_user_template()
        )

        critique_json = json.dumps(critique.model_dump(), indent=2)
        refiner_msg = (
            refiner_template.replace("{{text}}", text)
            .replace("{{draft_json}}", draft_json)
            .replace("{{critique_json}}", critique_json)
        )

        async with OPENAI_SEMAPHORE:
            refiner_result = await safe_agent_run(refiner, refiner_msg, deps)
        _track(refiner_result)

        refined = refiner_result.output.refined_extractions
        spans = [ddcot_span_to_s1_span(s) for s in refined]

        if return_full_extraction:
            result = DDCoTExtraction(
                text_complexity=extraction.text_complexity,
                dominant_narrative=extraction.dominant_narrative,
                extractions=refined,
            )
        else:
            result = spans

        return (result, total_usage) if return_usage else result

    except Exception as e:
        logger.error(f"[DD-CoT Pipeline] Failed: {e}")
        fallback: Any = (
            []
            if not return_full_extraction
            else DDCoTExtraction(
                text_complexity="simple", dominant_narrative="neutral", extractions=[]
            )
        )
        return (fallback, total_usage) if return_usage else fallback
