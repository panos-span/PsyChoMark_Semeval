"""
psycomark.agents.s1_agents_lite — Lite S1 Agent for Local Models.

Single-pass span extraction using simplified LiteExtraction schema.
Skips critic/refiner pipeline — designed for small models (e.g. Qwen3-8B).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from pydantic_ai import Agent, ModelSettings

from psycomark.agents.span_utils import find_best_span
from psycomark.config import LLM, AGENT_RETRIES, OPENAI_SEMAPHORE, safe_agent_run
from psycomark.schemas.s1 import S1Deps, S1Label, S1Span
from psycomark.schemas.s1_lite import LiteExtraction, LiteSpan


# ---------------------------------------------------------------------------
# System Prompt (self-contained, no external prompt files needed)
# ---------------------------------------------------------------------------

LITE_S1_SYSTEM_PROMPT = """\
You are a conspiracy-rhetoric marker extractor.

Given a social media post, extract verbatim text spans that serve as
psycholinguistic markers of conspiracy rhetoric. Each span gets one label:

- **Actor**: The alleged conspirator (person, group, or institution).
- **Action**: The alleged malicious act or mechanism.
- **Effect**: The alleged harmful outcome or consequence.
- **Victim**: The alleged target of harm.
- **Evidence**: Epistemic support for the conspiracy claim (e.g. "exposed", "exposed documents").

Rules:
1. Extract ONLY verbatim substrings — copy-paste from the text.
2. If there are no conspiracy markers, return an empty list.
3. Keep spans atomic — one concept per span.
4. A span can overlap categories; prefer the most specific label.
"""


# ---------------------------------------------------------------------------
# Lite Generator Factory
# ---------------------------------------------------------------------------


def get_s1_lite_generator() -> Agent[S1Deps, LiteExtraction]:
    """Factory for the lite S1 generator agent."""
    return Agent(
        model=LLM,
        output_type=LiteExtraction,
        deps_type=S1Deps,
        system_prompt=LITE_S1_SYSTEM_PROMPT,
        model_settings=ModelSettings(temperature=0.3),
        retries=AGENT_RETRIES,
    )


# ---------------------------------------------------------------------------
# Conversion Utilities
# ---------------------------------------------------------------------------


def lite_span_to_s1_span(lite_span: LiteSpan) -> S1Span:
    """Convert ``LiteSpan`` to legacy ``S1Span``."""
    # Map string label to S1Label enum
    label_map = {
        "Actor": S1Label.Actor,
        "Action": S1Label.Action,
        "Effect": S1Label.Effect,
        "Victim": S1Label.Victim,
        "Evidence": S1Label.Evidence,
    }
    return S1Span(
        label=label_map.get(lite_span.label, S1Label.Evidence),
        text=lite_span.text,
        why=lite_span.why or None,
        start=lite_span.start,
        end=lite_span.end,
    )


def validate_lite_extraction(
    extraction: LiteExtraction, raw_text: str
) -> Tuple[LiteExtraction, List[str]]:
    """Ground extracted spans in the source text."""
    valid_spans: list[LiteSpan] = []
    issues: list[str] = []

    for span in extraction.extractions:
        start, end = find_best_span(raw_text, span.text)

        # Boundary trimming (+/- 1 word)
        if start == -1:
            words = span.text.split()
            if len(words) > 1:
                for trimmed in [" ".join(words[1:]), " ".join(words[:-1])]:
                    s, e = find_best_span(raw_text, trimmed)
                    if s != -1:
                        issues.append(f"[RECOVERY] Trimmed: '{span.text}' → '{trimmed}'")
                        span.text, start, end = trimmed, s, e
                        break

        if start != -1:
            span.start = start
            span.end = end
            valid_spans.append(span)
        else:
            issues.append(f"[HALLUCINATION] Removed: '{span.text[:40]}...'")

    cleaned = LiteExtraction(extractions=valid_spans)
    return cleaned, issues


# ---------------------------------------------------------------------------
# Lite S1 Pipeline Runner
# ---------------------------------------------------------------------------


async def run_s1_lite(
    text: str,
    few_shots: Optional[List[Dict]] = None,
    metadata: Optional[Dict] = None,
    return_usage: bool = False,
) -> Union[List[S1Span], Tuple[List[S1Span], Dict[str, int]]]:
    """
    Single-pass lite S1 extraction — no critic, no refiner.

    Returns:
        List of extracted S1Span objects.
    """
    deps = S1Deps(raw_text=text, few_shots=few_shots or [], metadata=metadata or {})
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    try:
        generator = get_s1_lite_generator()

        user_msg = f"Extract conspiracy markers from this text:\n\n{text}"

        async with OPENAI_SEMAPHORE:
            gen_result = await safe_agent_run(generator, user_msg, deps)

        if hasattr(gen_result, "usage"):
            u = gen_result.usage()
            total_usage["input_tokens"] += getattr(u, "request_tokens", 0) or 0
            total_usage["output_tokens"] += getattr(u, "response_tokens", 0) or 0
            total_usage["total_tokens"] += getattr(u, "total_tokens", 0) or 0

        extraction = gen_result.output
        extraction, issues = validate_lite_extraction(extraction, text)

        if issues:
            logger.debug(f"[Lite S1] Validation: {len(issues)} issues resolved")

        spans = [lite_span_to_s1_span(s) for s in extraction.extractions]
        return (spans, total_usage) if return_usage else spans

    except Exception as e:
        logger.error(f"[Lite S1 Pipeline] Failed: {e}")
        fallback: List[S1Span] = []
        return (fallback, total_usage) if return_usage else fallback
