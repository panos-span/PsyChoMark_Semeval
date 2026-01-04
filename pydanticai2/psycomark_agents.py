#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psycomark_agents.py — Pydantic-AI Native Agents for the ReX-GoT / Competition Architecture.

This module defines the "Brains" of the system:
1. S1 Discriminative Agent: Extracts forensic spans with a negative-constraint scratchpad.
2. S2 Council Agents: (Optional) Specialized personas for the classification ensemble.

Key Features:
- Dependency Injection: Text and Few-Shots are injected at runtime via S1Deps.
- Structured Output: Enforces Pydantic schemas for all LLM responses.
- Robust Bedrock Setup: Auto-configures for AWS Bedrock Anthropic models.

Author: ReX-GoT Team
"""

from __future__ import annotations

import os
import re
import json
import asyncio
import sys
import pathlib
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any, Literal
from collections import Counter, defaultdict

# --- Make repo root importable FIRST ---
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pydanticai2.prompt_loader import S2_PROMPTS

# Pydantic & Pydantic-AI
from pydantic import BaseModel, Field, ConfigDict
import boto3
from botocore.config import Config  # Import Config
from pydantic_ai import Agent, RunContext, ModelSettings, ModelRetry
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.bedrock import BedrockProvider
import chromadb
from chromadb import Collection
from chromadb.utils import embedding_functions
from pydanticai2.prompt_loader import S1_PROMPTS
from pydanticai2.prompt_builder import (
    build_s1_critic_user_template,
    build_s1_refiner_user_template,
    build_s2_prosecutor_system,
    build_s2_defense_system,
    build_s2_literalist_system,
    build_s2_profiler_system,
    build_s2_prosecutor_user_template,
    build_s2_literalist_user_template,
    build_s2_profiler_user_template,
    build_s2_defense_user_template,
    build_s2_judge_system,
    build_s2_judge_user_template,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from botocore.exceptions import ClientError
from loguru import logger

try:
    from fuzzysearch import find_near_matches
except ImportError:
    find_near_matches = None  # Graceful fallback if not installed

# ===========================================================================
# 1. Configuration & Model Wiring
# ===========================================================================

AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv(
    "MODEL_ID",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",  # Default to Sonnet 3.5
)


# Configure Retry Mode to 'adaptive' (handles throttling automatically)
# 1. Configure Retries & Timeouts
_boto_config = Config(
    read_timeout=300,  # 5 minutes for long CoT generation
    connect_timeout=10,
    retries={
        "max_attempts": 20,  # Aggressive retries for throttling
        "mode": "adaptive",  # 'adaptive' handles backoff automatically
    },
)

# 2. Instantiate the Client Explicitly
_bedrock_client = boto3.client(
    service_name="bedrock-runtime", region_name=AWS_REGION, config=_boto_config
)


_provider = BedrockProvider(region_name=AWS_REGION, bedrock_client=_bedrock_client)
LLM = BedrockConverseModel(BEDROCK_MODEL_ID, provider=_provider)


def is_throttling_error(exception):
    """Returns True if the exception is an AWS ThrottlingException."""
    if isinstance(exception, ClientError):
        code = exception.response.get("Error", {}).get("Code", "")
        return code == "ThrottlingException"
    return False


# Retry configuration: Exponential backoff (1s, 2s, 4s...) up to 60s, max 15 attempts.
@retry(
    retry=retry_if_exception_type(ClientError),
    stop=stop_after_attempt(15),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    reraise=True,
)
async def safe_agent_run(agent, message, deps):
    """Wraps PydanticAI agent.run with explicit Throttling retries."""
    return await agent.run(message, deps=deps)


class BedrockTitanEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """
    ChromaDB-compatible wrapper for Amazon Titan Text v2.
    """

    def __init__(self, region_name: str = AWS_REGION):
        import boto3
        from botocore.config import Config

        # INCREASE TIMEOUT to 300s (5 minutes) to prevent crashes on large docs
        config = Config(
            read_timeout=300, connect_timeout=10, retries={"max_attempts": 3}
        )
        self.bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name=region_name,
            config=config,  # <--- Apply Config
        )

    def __call__(self, input: List[str]) -> List[List[float]]:
        # Titan v2 supports batching, but let's loop to be safe/simple regarding limits
        embeddings = []
        for text in input:
            try:
                body = json.dumps(
                    {
                        "inputText": text[:8000],  # Titan limit
                        "dimensions": 1024,
                        "normalize": True,
                    }
                )
                response = self.bedrock.invoke_model(
                    body=body,
                    modelId="amazon.titan-embed-text-v2:0",
                    accept="application/json",
                    contentType="application/json",
                )
                response_body = json.loads(response.get("body").read())
                embeddings.append(response_body.get("embedding"))
            except Exception as e:
                logger.error(f"[Embedder] Error: {e}")
                embeddings.append([0.0] * 1024)  # Fallback zero vector
        return embeddings


def get_rag_collection(path: str, name: str) -> Collection:
    """Initializes Chroma client and returns the collection."""
    client = chromadb.PersistentClient(path=path)
    ef = BedrockTitanEmbeddingFunction()
    logger.info(
        f"  - Loading Index {name} ({client.get_collection(name=name, embedding_function=ef).count()} docs)"
    )
    return client.get_collection(name=name, embedding_function=ef)


# ===========================================================================
# 2. Shared Data Structures (S1)
# ===========================================================================


class S1Label(str, Enum):
    Actor = "Actor"
    Action = "Action"
    Effect = "Effect"
    Victim = "Victim"
    Evidence = "Evidence"


class S1Span(BaseModel):
    """
    The atomic unit of extraction.
    Note: 'start' and 'end' are calculated by the Graph Verifier, not the LLM.
    """

    label: S1Label
    text: str = Field(
        ..., description="The exact verbatim substring found in the text."
    )

    # Optional fields for downstream processing, not required from LLM
    start: Optional[int] = None
    end: Optional[int] = None
    why: Optional[str] = None  # Context/Rationale if needed


class S1Deps(BaseModel):
    """
    Runtime Dependencies for S1.
    Passes the raw text and dynamic few-shot examples to the agent.
    """

    model_config = ConfigDict(extra="ignore")

    text: str = Field(..., alias="raw_text")  # The document content
    doc_id: Optional[str] = None
    few_shots: List[Dict[str, Any]] = Field(default_factory=list)


# ===========================================================================
# 3. S1: The Discriminative Forensic Agent
# ===========================================================================


class S1Rejection(BaseModel):
    """
    Helper model for the Audit step.
    Forces the model to categorize the rejection reason.
    """

    text: str = Field(..., description="The candidate phrase that was rejected.")
    reason: str = Field(
        ...,
        description="The specific reason for rejection (e.g., 'Pronoun', 'Generic', 'Author Opinion').",
    )


class S1Reasoning(BaseModel):
    """
    Streamlined Chain-of-Thought Schema for span extraction.
    Reduced cognitive overhead while maintaining quality.
    """

    # Step 1: Quick Assessment
    text_type: str = Field(
        description="Brief classification: 'conspiracy_claim', 'neutral_report', 'opinion_piece', or 'mixed'. This helps calibrate extraction but does NOT prevent extraction."
    )

    # Step 2: Reasoning (lightweight)
    reasoning: str = Field(
        description="1-2 sentences explaining your extraction strategy for this text."
    )

    # Step 3: Final Output (the main deliverable)
    final_spans: List[S1Span] = Field(
        description="The list of extracted markers. Each must be a verbatim substring from the text."
    )


class S1Critique(BaseModel):
    critiques: List[str] = Field(
        description="List of specific errors found (e.g. 'Missed Actor: The CIA'). Empty if perfect."
    )
    requires_refinement: bool = Field(description="True if changes are needed.")


class S1Refinement(BaseModel):
    final_spans: List[S1Span] = Field(description="The corrected list of spans.")


s1_critic_agent = Agent(
    model=LLM,
    output_type=S1Critique,
    deps_type=S1Deps,
    system_prompt=S1_PROMPTS.critic_system,
    model_settings=ModelSettings(temperature=0.0),  # Critic should be deterministic
)

s1_refiner_agent = Agent(
    model=LLM,
    output_type=S1Refinement,
    deps_type=S1Deps,
    system_prompt=S1_PROMPTS.refiner_system,
    model_settings=ModelSettings(
        temperature=0.0
    ),  # Refiner should be strictly compliant
)

# --- Global Agent Definition ---
s1_discriminative_agent = Agent(
    model=LLM,
    output_type=S1Reasoning,
    deps_type=S1Deps,
    retries=2,  # Low retries; the Ensemble handles gaps
    system_prompt=S1_PROMPTS.gen_system,
    model_settings=ModelSettings(
        temperature=0.7  # High temperature to encourage diversity for the ensemble
    ),
)


@s1_discriminative_agent.output_validator
def validate_s1_result(ctx: RunContext[S1Deps], result: S1Reasoning) -> S1Reasoning:
    """
    Assert-and-Retry Guardrail:
    1. Verbatim Check: Ensures spans exist in source text (prevent hallucinations).

    If validation fails, raises ModelRetry to send the error back to the LLM.
    """
    raw_text = ctx.deps.text
    errors = []

    for span in result.final_spans:
        # --- Rule 1: Verbatim Constraint ---
        # We perform a robust check. If exact match fails, we try the robust finder
        start, end = find_best_span(raw_text, span.text)

        if start == -1:
            # Hallucination detected
            errors.append(
                f"Span '{span.text}' NOT found in source text. You must extract verbatim text only."
            )
            continue

        # --- Rule 2: Minimum Length for Actions ---
        # Actions should not be single common verbs
        word_count = len(span.text.split())
        if span.label == S1Label.Action and word_count < 2:
            errors.append(
                f"Action '{span.text}' is too short. Include the verb AND its object(s)."
            )

    if errors:
        # Combine errors into a single prompt for the retry
        error_msg = "\n- ".join(errors)
        raise ModelRetry(
            f"Validation Failed. Please fix the following errors and re-generate the JSON:\n- {error_msg}"
        )

    return result


def format_s1_fewshots_to_xml(few_shots: List[Dict]) -> str:
    """
    Converts list of few-shot dicts to the XML string expected by the prompt.
    Enhanced: Shows spans in a clearer format with label grouping.
    """
    if not few_shots:
        return ""

    examples_xml = ["<reference_examples>"]
    for idx, ex in enumerate(few_shots):
        spans_to_show = ex.get("spans", [])

        # Determine type label for context
        label_val = str(ex.get("label", "")).lower()
        ex_type = (
            "CONSPIRACY_TEXT"
            if label_val in ["conspiracy", "yes", "true"]
            else "NEUTRAL_TEXT"
        )

        # Format spans in a more readable way
        spans_formatted = []
        for span in spans_to_show:
            label = span.get("label", "Unknown")
            text = span.get("text", "")
            spans_formatted.append(f'{{"label": "{label}", "text": "{text}"}}')

        spans_str = ",\\n      ".join(spans_formatted) if spans_formatted else "[]"

        examples_xml.append(
            f"""
  <example id="{idx+1}" type="{ex_type}">
    <input_text>{ex.get('text', '').strip()[:500]}{"..." if len(ex.get('text', '')) > 500 else ""}</input_text>
    <expected_output>[
      {spans_str}
    ]</expected_output>
    <note>{"This NEUTRAL text still has structural markers - extract them!" if ex_type == "NEUTRAL_TEXT" and spans_to_show else ""}</note>
  </example>"""
        )
    examples_xml.append("</reference_examples>")
    return "\n".join(examples_xml)


# --- 1. Shared Prompt Assembler ---
def assemble_s1_system_prompt(base_instruction: str, few_shots: List[Dict]) -> str:
    """
    Smart Assembler:
    1. Formats the few-shots into XML.
    2. If {{few_shot_examples}} exists in template -> Replaces it.
    3. If NOT -> Appends to end (Legacy Fallback).
    """
    xml_str = format_s1_fewshots_to_xml(few_shots)

    if "{{few_shot_examples}}" in base_instruction:
        return base_instruction.replace("{{few_shot_examples}}", xml_str)

    # Fallback: Append if variable is missing (and we have content)
    if xml_str:
        return f"{base_instruction}\n\n{xml_str}"

    return base_instruction


@s1_discriminative_agent.system_prompt
def generate_s1_system_prompt(ctx: RunContext[S1Deps]) -> str:
    """
    Dynamically builds the XML-structured prompt.
    """
    # 1. Base System Instruction
    # Load the optimized text file (or fallback to default)
    optimized_base = S1_PROMPTS.gen_system

    # Combine with context
    return assemble_s1_system_prompt(optimized_base, ctx.deps.few_shots)


# In psycomark_agents.py


async def run_s1_discriminative(
    text: str,
    few_shots: Optional[List[Dict]] = None,
    # Overrides for GEPA Optimization
    gen_prompt_override: Optional[str] = None,
    user_prompt_template_override: Optional[str] = None,
    critic_prompt_override: Optional[str] = None,
    critic_user_template_override: Optional[str] = None,  # <--- NEW
    refiner_prompt_override: Optional[str] = None,
    refiner_user_template_override: Optional[str] = None,  # <--- NEW
) -> List[S1Span]:

    deps = S1Deps(raw_text=text, few_shots=few_shots or [])

    # [FIX] Local Semaphore to avoid event loop issues
    sem = asyncio.Semaphore(1)

    # ---------------------------------------------------------
    # 1. SETUP GENERATOR (DRAFT)
    # ---------------------------------------------------------

    # A. System Prompt (The Brain)
    if gen_prompt_override:
        # OPTIMIZATION MODE: Manually assemble to include RAG few-shots
        effective_system_prompt = assemble_s1_system_prompt(
            gen_prompt_override, few_shots or []
        )
        active_gen_agent = Agent(
            LLM,
            output_type=S1Reasoning,
            deps_type=S1Deps,
            system_prompt=effective_system_prompt,
            model_settings=ModelSettings(temperature=0.7),
        )
    else:
        # PRODUCTION MODE: Use global agent (has @system_prompt decorator)
        active_gen_agent = s1_discriminative_agent

    # B. User Prompt (The Trigger)
    if user_prompt_template_override:
        # Override provided by MLflow
        gen_user_msg = user_prompt_template_override.replace("{{text}}", text)
    elif (
        hasattr(S1_PROMPTS, "gen_user_template")
        and "{{text}}" in S1_PROMPTS.gen_user_template
    ):
        # Optimized production template
        gen_user_msg = S1_PROMPTS.gen_user_template.replace("{{text}}", text)
    else:
        # Hardcoded fallback
        gen_user_msg = f"<analysis_target>\n{text}\n</analysis_target>"

    async with sem:
        try:
            # --- STEP 1: DRAFT ---
            # draft_result = await active_gen_agent.run(gen_user_msg, deps=deps)
            draft_result = await safe_agent_run(active_gen_agent, gen_user_msg, deps)
            draft_spans = draft_result.output.final_spans

            # ---------------------------------------------------------
            # 2. SETUP CRITIC (AUDITOR)
            # ---------------------------------------------------------

            # A. System Prompt
            if critic_prompt_override:
                active_critic_agent = Agent(
                    LLM,
                    output_type=S1Critique,
                    deps_type=S1Deps,
                    system_prompt=critic_prompt_override,
                    model_settings=ModelSettings(temperature=0.0),
                )
            else:
                active_critic_agent = s1_critic_agent

            # B. User Prompt (Dynamic Injection)
            draft_json_str = json.dumps([s.model_dump() for s in draft_spans], indent=2)

            if critic_user_template_override:
                c_tmpl = critic_user_template_override
            elif (
                hasattr(S1_PROMPTS, "critic_user_template")
                and "{{text}}" in S1_PROMPTS.critic_user_template
            ):
                c_tmpl = S1_PROMPTS.critic_user_template
            else:
                c_tmpl = build_s1_critic_user_template()  # Fallback function

            # Inject Variables
            critique_user_msg = c_tmpl.replace("{{text}}", text).replace(
                "{{draft_json}}", draft_json_str
            )

            # Run Critic
            # critique_res = await active_critic_agent.run(critique_user_msg, deps=deps)
            critique_res = await safe_agent_run(
                active_critic_agent, critique_user_msg, deps=deps
            )

            # Optimization: Short-circuit if perfect
            if (
                not critique_res.output.requires_refinement
                or not critique_res.output.critiques
            ):
                return draft_spans

            # ---------------------------------------------------------
            # 3. SETUP REFINER (EDITOR)
            # ---------------------------------------------------------

            # A. System Prompt
            if refiner_prompt_override:
                active_refiner_agent = Agent(
                    LLM,
                    output_type=S1Refinement,
                    deps_type=S1Deps,
                    system_prompt=refiner_prompt_override,
                    model_settings=ModelSettings(temperature=0.0),
                )
            else:
                active_refiner_agent = s1_refiner_agent

            # B. User Prompt (Dynamic Injection)
            critique_json_str = json.dumps(critique_res.output.critiques, indent=2)

            if refiner_user_template_override:
                r_tmpl = refiner_user_template_override
            elif (
                hasattr(S1_PROMPTS, "refiner_user_template")
                and "{{text}}" in S1_PROMPTS.refiner_user_template
            ):
                r_tmpl = S1_PROMPTS.refiner_user_template
            else:
                r_tmpl = build_s1_refiner_user_template()  # Fallback function

            # Inject Variables
            refine_user_msg = (
                r_tmpl.replace("{{text}}", text)
                .replace("{{draft_json}}", draft_json_str)
                .replace("{{critique_json}}", critique_json_str)
            )

            # Run Refiner
            # refine_res = await active_refiner_agent.run(refine_user_msg, deps=deps)
            refine_res = await safe_agent_run(
                active_refiner_agent, refine_user_msg, deps=deps
            )
            return refine_res.output.final_spans

        except Exception as e:
            logger.warning(f"S1 Chain Failed: {e}")
            return []


# ===========================================================================
# 4. Utilities: Search & Verification (The "Map-Reduce" Engine)
# ===========================================================================

# Smart Quote Normalization Map
_SMART_TO_STRAIGHT = {
    ord("“"): ord('"'),
    ord("”"): ord('"'),
    ord("‘"): ord("'"),
    ord("’"): ord("'"),
    ord("–"): ord("-"),
    ord("—"): ord("-"),
}
# Matches any whitespace, NBSP, thin space, etc.
_NORMALIZE_SPACE_RE = re.compile(r"[\s\u00A0\u2000-\u200B\u202F]+")


def _normalize_for_match(s: str) -> Tuple[str, List[int]]:
    """
    Normalizes text (smart quotes -> straight, whitespace -> single space, lowercase)
    AND returns a mapping from normalized indices back to original indices.

    Returns:
        (norm_text, idx_map)
        where idx_map[i] is the index in 's' corresponding to norm_text[i].
    """
    if not s:
        return "", []

    # 1. Fast path: If simple, just return lower (optimization)
    # But we need the map, so we must iterate if we want robustness.

    t = s.translate(_SMART_TO_STRAIGHT)

    norm_chars = []
    idx_map = []

    i = 0
    n = len(t)
    while i < n:
        char = t[i]

        # Check for whitespace run
        if char.isspace() or _NORMALIZE_SPACE_RE.match(char):
            # Found start of whitespace run
            # Scan until end of whitespace
            j = i + 1
            while j < n and (t[j].isspace() or _NORMALIZE_SPACE_RE.match(t[j])):
                j += 1

            # Collapse entire run to single space
            norm_chars.append(" ")
            idx_map.append(
                i
            )  # Map the space to the start of the original whitespace run

            i = j  # Skip to end of run
        else:
            # Standard character
            norm_chars.append(char.lower())
            idx_map.append(i)
            i += 1

    return "".join(norm_chars), idx_map


def find_best_span(raw_text: str, snippet: str, nth: int = 0) -> Tuple[int, int]:
    """
    Robustly locates the `nth` occurrence of `snippet` in `raw_text`.
    Returns (start, end) in raw_text, or (-1, -1).
    """
    if not snippet or not raw_text:
        return -1, -1

    # --- Strategy 1: Exact Match (Fastest) ---
    start = -1
    for _ in range(nth + 1):
        start = raw_text.find(snippet, start + 1)
        if start == -1:
            break
    if start != -1:
        return start, start + len(snippet)

    # --- Strategy 2: Case-Insensitive (Fast) ---
    raw_lower = raw_text.lower()
    snip_lower = snippet.lower()
    start = -1
    for _ in range(nth + 1):
        start = raw_lower.find(snip_lower, start + 1)
        if start == -1:
            break
    if start != -1:
        # Check if length matches (Unicode edge cases exists, but rare for English)
        return start, start + len(snippet)

    # --- Strategy 3: Normalized Match (Robust) ---
    # Handles "Smart Quotes" vs "Straight Quotes" and "Multiple   Spaces" vs "Single Space"
    raw_norm, raw_map = _normalize_for_match(raw_text)
    snip_norm, _ = _normalize_for_match(snippet)  # We don't need snippet map, just text

    if snip_norm in raw_norm:
        start_norm = -1
        # Find nth occurrence in normalized string
        for _ in range(nth + 1):
            start_norm = raw_norm.find(snip_norm, start_norm + 1)
            if start_norm == -1:
                break

        if start_norm != -1:
            # Found it! Now project back using the map.

            # Start Index projection
            orig_start = raw_map[start_norm]

            # End Index projection
            # The match in `raw_norm` ends at `start_norm + len(snip_norm) - 1`
            end_norm_idx = start_norm + len(snip_norm) - 1

            if end_norm_idx < len(raw_map):
                # 1. Start is simply the mapped index of the first char
                orig_start = raw_map[start_norm]

                # 2. End is more complex.
                # Ideally, the end of this character is the start of the NEXT character.
                # Check if there is a next character in the normalized map.
                if end_norm_idx + 1 < len(raw_map):
                    # We use the start of the next normalized char as our exclusive end
                    # This automatically accounts for collapsed whitespace runs.
                    orig_end = raw_map[end_norm_idx + 1]
                else:
                    # Edge Case: We are at the very end of the string.
                    # We can't look ahead.
                    # We must assume the rest of the string belongs to this char
                    # (or just +1 if it's a standard char, but calculating diff is safer).

                    # Fallback: Just take the remaining length of raw_text?
                    # Or simple +1 if we assume no trailing collapsed whitespace at EOF.

                    # Safe logic: If the normalized char was a space, extend to end of raw string
                    # if raw string ends with space. Otherwise +1.

                    # For safety in competition, +1 is usually acceptable at EOF
                    # unless your text ends with specific whitespace padding.
                    orig_last_char_idx = raw_map[end_norm_idx]
                    orig_end = orig_last_char_idx + 1

                    # Optional: Extend if it was a collapsed space at EOF
                    while (
                        orig_end < len(raw_text)
                        and raw_text[orig_last_char_idx].isspace()
                        and raw_text[orig_end].isspace()
                    ):
                        orig_end += 1

                return orig_start, orig_end

    # --- Strategy 4: Fuzzy Match (Fallback) ---
    if find_near_matches:
        # Allow ~15% edits, minimum 1 if length > 4
        max_dist = max(1, int(len(snippet) * 0.15)) if len(snippet) > 4 else 0

        # Prevent fuzzy matching on very short words (too many false positives)
        if len(snippet) < 3:
            max_dist = 0

        if max_dist > 0:
            matches = find_near_matches(snippet, raw_text, max_l_dist=max_dist)
            if len(matches) > nth:
                m = matches[nth]
                return m.start, m.end

    return -1, -1


# Wrapper for compatibility
def locate_span_in_text(full_text: str, substring: str) -> Tuple[int, int]:
    return find_best_span(full_text, substring, nth=0)


def retrieve_fewshots(
    collection: Collection, query_text: str, k: int = 8, filters: Optional[dict] = None
) -> List[dict]:
    try:
        results = collection.query(query_texts=[query_text], n_results=k, where=filters)
        examples = []
        if results["documents"] and results["metadatas"]:
            for i in range(len(results["documents"][0])):
                metadata = results["metadatas"][0][i] if results["metadatas"][0] else {}
                ex = {"text": results["documents"][0][i], **metadata}
                if "spans_json" in ex:
                    ex["spans"] = json.loads(ex.pop("spans_json"))
                examples.append(ex)
        return examples
    except Exception as e:
        logger.error(f"[RAG] Retrieval failed: {e}")
        return []


def retrieve_stratified_s1(
    collection: Collection, query_text: str, k_total: int = 6
) -> List[Dict]:
    """Retrieves balanced Conspiracy AND Non-Conspiracy examples."""
    if not collection:
        return []
    half = k_total // 2
    pos = retrieve_fewshots(
        collection, query_text, k=half, filters={"label": "conspiracy"}
    )
    neg = retrieve_fewshots(collection, query_text, k=half, filters={"label": "non"})
    stratified = []
    for p, n in zip(pos, neg):
        stratified.extend([p, n])
    if len(pos) > len(neg):
        stratified.extend(pos[len(neg) :])
    elif len(neg) > len(pos):
        stratified.extend(neg[len(pos) :])
    return stratified


# ===========================================================================
# S2: The Council of Rivals
# ===========================================================================


class S2Deps(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # --- Core Input ---
    raw_text: str
    doc_id: Optional[str] = None

    # --- Forensic Evidence (From S1) ---
    s1_markers: List[Dict[str, Any]] = Field(default_factory=list)
    # The "Forensic Dossier" string generated by synthesize_dossier
    marker_summary: Optional[str] = None

    # --- [NEW] Contextual Signals ---
    # Essential for the "Profiler" (e.g., {"subreddit": "conspiracy"})
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # --- [NEW] Legal Precedents ---
    # Essential for the "Judge" (contains text of similar Hard Negatives)
    rag_context: str = Field(default="")


class S2Output(BaseModel):
    """
    Final Verdict Schema.
    Used by the 'Judge' node to render the binding decision.
    """

    label: Literal["conspiracy", "non"] = Field(
        ...,
        description="The final classification. 'conspiracy' = Endorsement/Promotion. 'non' = Reporting/Debunking/Mocking.",
    )

    rationale: str = Field(
        ...,
        description="Explains the 'State of Mind': Why is this an endorsement vs. just a summary? Reference specific tone cues.",
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Certainty score. A value between 0.0 (no confidence) and 1.0 (absolute confidence).",
    )

    key_evidence: List[str] = Field(
        default_factory=list,
        description="List of 1-3 verbatim substrings that prove the author's stance (e.g., 'Finally the truth', 'Wake up').",
    )

    # Note: We removed the custom validator because Literal["conspiracy", "non"]
    # handles validation natively and faster in Pydantic.


class S2Juror(str, Enum):
    LITERALIST = "Literalist"  # Strict "Burden of Proof" (Acquits Hearsay)
    BELIEVER = "Believer"  # High Recall (Flags Implicit Support)
    PROFILER = "Profiler"  # Psycholinguistic (Flags "Us vs Them" tone)
    DEFENSE = "Defense"  # Hanlon's Razor (Flags Incompetence vs Malice)


class S2Vote(BaseModel):
    juror: S2Juror
    verdict: Literal["conspiracy", "non"]
    confidence: float
    rationale: str = Field(..., description="One sentence explanation.")


class S2CouncilOutput(BaseModel):
    """Aggregated output from the Council phase"""

    votes: List[S2Vote]
    tally: Dict[str, int]  # e.g. {"conspiracy": 3, "non": 2}
    weighted_score: float = Field(
        default=0.0,
        description="Confidence-weighted score: positive = conspiracy, negative = non",
    )
    debate_summary: str = Field(
        default="", description="Summary of prosecutor/defense arguments for context"
    )


# --- Juror Agent Factory ---
# We use a factory because each Juror needs a different System Prompt
# --- Corrected Factory ---
# --- Helper: Assembler for S2 ---
def format_s2_rag_to_xml(rag_context: str) -> str:
    """
    Wraps the raw RAG context (often a JSON string of precedents) in XML tags
    if not already wrapped.
    """
    if not rag_context:
        return "No relevant case law found."

    # If the rag_context is already formatted by format_s2_rag_context (which adds tags), return it.
    if "<legal_precedents_context>" in rag_context:
        return rag_context

    return f"<legal_precedents>\n{rag_context}\n</legal_precedents>"


# --- Updated Assembler S2 ---
def assemble_s2_system_prompt(base_instruction: str, rag_context: str) -> str:
    """
    Injects RAG context into variable or appends.
    """
    # 1. Prepare content (ensure it's not empty if we are going to replace)
    # Note: If rag_context is empty, we replace the variable with an empty string/note.
    content = format_s2_rag_to_xml(rag_context) if rag_context else ""

    if "{{rag_context}}" in base_instruction:
        return base_instruction.replace("{{rag_context}}", content)

    # Fallback
    if content:
        return f"{base_instruction}\n\n{content}"

    return base_instruction


# --- Helper: Factory for Jurors ---
# --- 2. Optimized Juror Factory ---
def create_juror_agent_optimized(
    role: S2Juror,
    deps: S2Deps,
    override_sys: Optional[str] = None,
    temperature: float = 0.4,
) -> Agent[S2Deps, S2Vote]:
    """
    Creates an ephemeral agent with:
    1. Base Prompt (Override > Loaded > Specific Default)
    2. Dynamic RAG Context injected
    """

    # 1. Determine Base System Prompt
    base = ""

    if override_sys:
        # Priority 1: GEPA Optimization Override
        base = override_sys

    elif S2_PROMPTS:
        # Priority 2: Optimized Artifacts (Production)
        if role == S2Juror.BELIEVER:
            base = S2_PROMPTS.pros_sys
        elif role == S2Juror.DEFENSE:
            base = S2_PROMPTS.def_sys
        elif role == S2Juror.LITERALIST:
            base = S2_PROMPTS.lit_sys
        elif role == S2Juror.PROFILER:
            base = S2_PROMPTS.prof_sys

    else:
        # Priority 3: Builder Fallbacks (No Optimization / First Run)
        # [FIX] Use specific builders, avoiding the generic build_s2_system
        if role == S2Juror.BELIEVER:
            base = build_s2_prosecutor_system()
        elif role == S2Juror.DEFENSE:
            base = build_s2_defense_system()
        elif role == S2Juror.LITERALIST:
            base = build_s2_literalist_system()
        elif role == S2Juror.PROFILER:
            base = build_s2_profiler_system()
        else:
            # Absolute fallback if a new enum is added but not handled
            base = build_s2_prosecutor_system()

    # 2. Assemble with Dynamic RAG
    # We use deps.rag_context which holds retrieved precedents
    # (Ensure assemble_s2_system_prompt is defined above this function)
    full_sys = assemble_s2_system_prompt(base, deps.rag_context)

    # 3. Return Typed Agent
    return Agent(
        LLM,
        output_type=S2Vote,
        deps_type=S2Deps,
        system_prompt=full_sys,
        model_settings=ModelSettings(temperature=temperature),
        retries=2,
    )


# --- System Prompt Selector ---
# def get_juror_system_prompt(role: S2Juror) -> str:
#    from prompt_builder import (
#        build_s2_triage_system,  # Literalist
#        build_s2_profiler_system,  # Profiler
#        build_s2_defense_system,  # Defense
#        build_s2_system,  # Believer (Standard)
#    )
#
#    if role == S2Juror.LITERALIST:
#        return build_s2_triage_system()
#    elif role == S2Juror.PROFILER:
#        return build_s2_profiler_system()
#    elif role == S2Juror.DEFENSE:
#        return build_s2_defense_system()
#    else:  # Believer/Standard
#        return build_s2_system(include_cot=False)  # Faster, no CoT needed for voting


# --- The Council Runner (Parallel) ---
# async def run_s2_council(
#    text: str,
#    s1_spans: List[dict],
#    marker_summary: str,
#    active_jurors: List[S2Juror] = [
#        S2Juror.LITERALIST,
#        S2Juror.BELIEVER,
#        S2Juror.PROFILER,
#    ],
#    temperature: float = 0.4,  # <--- NEW PARAMETER
# ) -> S2CouncilOutput:
#
#    deps = S2Deps(
#        raw_text=text,
#        s1_markers=s1_spans,
#        marker_summary=marker_summary,  # Pass the string directly
#    )
#
#    # User Prompt is shared (The Evidence)
#    # We use a simplified prompt for the jurors to keep it fast
#    user_prompt = f"""
# <case_file>
#  <evidence_text>
# {text}
#  </evidence_text>
#
#  <forensic_markers>
# {marker_summary}
#  </forensic_markers>
#
#  <instruction>
#    Review the evidence above according to your System Role.
#    Render your Verdict.
#  </instruction>
# </case_file>
# """
#
#    # [LOGGING] Print S2 User Prompt ONCE
#    if "s2_user" not in _PROMPT_LOG_FLAGS:
#        logger.info(
#            f"\n{'='*40}\n[DEBUG] S2 USER PROMPT (First Run)\n{'='*40}\n{user_prompt}\n{'='*40}"
#        )
#        _PROMPT_LOG_FLAGS.add("s2_user")
#
#    async def _run_juror(role: S2Juror):
#        # Create the specialized agent
#        agent = create_juror_agent(role, temperature=temperature)
#
#        try:
#            # [CRITICAL FIX] Wrap the execution in the semaphore
#            # This forces the 4 jurors to queue up if the API is busy
#            async with _SC_SEMAPHORE:
#                res = await agent.run(user_prompt, deps=deps)
#
#            # Stamp the vote with the juror's ID (Agent returns S2Vote, we ensure .juror is set)
#            vote = res.output
#            vote.juror = role
#            return vote
#        except Exception as e:
#            logger.warning(f"Juror {role} failed: {e}")
#            return None
#
#    # Run in Parallel
#    tasks = [_run_juror(role) for role in active_jurors]
#    results = await asyncio.gather(*tasks)
#    valid_votes = [r for r in results if r is not None]
#
#    # Tally
#    counts = Counter(v.verdict for v in valid_votes)
#
#    return S2CouncilOutput(votes=valid_votes, tally=dict(counts))


# --- Helper: Judge System Prompt Assembler ---
def assemble_s2_judge_system(base_sys: str, rag_context: str) -> str:
    if not rag_context:
        return base_sys
    return f"{base_sys}\n\n<legal_precedents>\n{rag_context}\n</legal_precedents>"


# ===========================================================================
# DEBUGGING UTILS
# ===========================================================================
def log_agent_execution(role: str, sys_prompt: str, user_prompt: str):
    """
    Logs the full prompt context for debugging optimization.
    """
    logger.debug(f"\n{'='*20} [S2 EXECUTION: {role.upper()}] {'='*20}")
    logger.debug(f">>> SYSTEM PROMPT :\n{sys_prompt}...")
    logger.debug(f">>> USER PROMPT (Full):\n{user_prompt}")
    logger.debug(f"{'='*60}\n")


# ===========================================================================
# S2 COUNCIL RUNNER (INSTRUMENTED)
# ===========================================================================


async def run_s2_sequential_debate(
    text: str,
    s1_spans: List[dict],
    marker_summary: str,
    rag_context: str = "",
    temperature: float = 0.4,
    # GEPA System Overrides
    prosecutor_sys_override: Optional[str] = None,
    defense_sys_override: Optional[str] = None,
    literalist_sys_override: Optional[str] = None,
    profiler_sys_override: Optional[str] = None,
    # GEPA User Template Overrides
    prosecutor_user_template_override: Optional[str] = None,
    defense_user_template_override: Optional[str] = None,
    literalist_user_template_override: Optional[str] = None,
    profiler_user_template_override: Optional[str] = None,
) -> S2CouncilOutput:

    deps = S2Deps(
        raw_text=text,
        s1_markers=s1_spans,
        marker_summary=marker_summary,
        rag_context=rag_context,
    )
    votes = []

    def resolve_template(override, loader_attr, default_func):
        if override:
            return override
        if S2_PROMPTS and hasattr(S2_PROMPTS, loader_attr):
            return getattr(S2_PROMPTS, loader_attr)
        return default_func()

    # ---------------------------------------------------
    # PHASE 1: PROSECUTOR
    # ---------------------------------------------------
    try:
        p_tmpl = resolve_template(
            prosecutor_user_template_override,
            "pros_user",
            build_s2_prosecutor_user_template,
        )
        # [FIX] Changed {{markers}} to {{marker_summary}}
        p_msg = p_tmpl.replace("{{text}}", text).replace(
            "{{marker_summary}}", marker_summary
        )

        p_agent = create_juror_agent_optimized(
            S2Juror.BELIEVER, deps, prosecutor_sys_override, temperature
        )
        p_res = await safe_agent_run(p_agent, p_msg, deps)

        if p_res:
            prosecutor_vote = p_res.output
            prosecutor_vote.juror = S2Juror.BELIEVER
            votes.append(prosecutor_vote)
    except Exception as e:
        logger.warning(f"Prosecutor Failed: {e}")
        prosecutor_vote = None

    # ---------------------------------------------------
    # PHASE 2: DEFENSE
    # ---------------------------------------------------
    d_vote = None  # Initialize for scope visibility
    try:
        d_tmpl = resolve_template(
            defense_user_template_override, "def_user", build_s2_defense_user_template
        )
        p_arg = prosecutor_vote.rationale if prosecutor_vote else "No indictment."

        # [FIX] Changed {{markers}} to {{marker_summary}}
        d_msg = (
            d_tmpl.replace("{{text}}", text)
            .replace("{{marker_summary}}", marker_summary)
            .replace("{{prosecution_arg}}", p_arg)
        )

        d_agent = create_juror_agent_optimized(
            S2Juror.DEFENSE, deps, defense_sys_override, temperature
        )
        d_res = await safe_agent_run(d_agent, d_msg, deps=deps)

        if d_res:
            d_vote = d_res.output
            d_vote.juror = S2Juror.DEFENSE
            votes.append(d_vote)
    except Exception:
        pass

    # ---------------------------------------------------
    # PHASE 3: WITNESSES (With Debate Context)
    # ---------------------------------------------------
    # Build debate summary for witness context
    pros_arg = (
        prosecutor_vote.rationale if prosecutor_vote else "No prosecution argument."
    )
    def_arg = d_vote.rationale if d_vote else "No defense argument."
    debate_summary = f"""<debate_summary>
<prosecution_argument>{pros_arg}</prosecution_argument>
<defense_argument>{def_arg}</defense_argument>
</debate_summary>"""

    async def _run_witness(
        role, sys_override, usr_override, loader_attr, def_func, debate_ctx
    ):
        try:
            w_tmpl = resolve_template(usr_override, loader_attr, def_func)
            # [FIX] Changed {{markers}} to {{marker_summary}}
            # [ENHANCEMENT] Add debate context for witnesses to evaluate both arguments
            w_msg = (
                w_tmpl.replace("{{text}}", text)
                .replace("{{marker_summary}}", marker_summary)
                .replace("{{debate_summary}}", debate_ctx)
            )
            # If template doesn't have {{debate_summary}}, append it
            if "{{debate_summary}}" not in w_tmpl:
                w_msg = f"{w_msg}\n\n{debate_ctx}"

            w_agent = create_juror_agent_optimized(
                role, deps, sys_override, temperature
            )
            res = await safe_agent_run(w_agent, w_msg, deps=deps)
            if res:
                v = res.output
                v.juror = role
                return v
        except Exception:
            return None

    tasks = [
        _run_witness(
            S2Juror.LITERALIST,
            literalist_sys_override,
            literalist_user_template_override,
            "lit_user",
            build_s2_literalist_user_template,
            debate_summary,  # Pass debate context
        ),
        _run_witness(
            S2Juror.PROFILER,
            profiler_sys_override,
            profiler_user_template_override,
            "prof_user",
            build_s2_profiler_user_template,
            debate_summary,  # Pass debate context
        ),
    ]
    w_results = await asyncio.gather(*tasks)
    votes.extend([w for w in w_results if w])

    counts = Counter(v.verdict for v in votes)

    # Confidence-Weighted Voting: positive = conspiracy, negative = non
    weighted_score = sum(
        v.confidence if v.verdict == "conspiracy" else -v.confidence for v in votes
    )

    # Build debate summary for downstream use
    pros_summary = prosecutor_vote.rationale if prosecutor_vote else "No prosecution."
    def_summary = d_vote.rationale if d_vote else "No defense."
    full_debate_summary = f"Prosecution: {pros_summary}\nDefense: {def_summary}"

    return S2CouncilOutput(
        votes=votes,
        tally=dict(counts),
        weighted_score=weighted_score,
        debate_summary=full_debate_summary,
    )


# ===========================================================================
# S2 JUDGE RUNNER (INSTRUMENTED)
# ===========================================================================


async def run_s2_judge_review(
    text: str,
    council_result: S2CouncilOutput,
    doc_id: str = "opt_sample",
    rag_context: str = "",
    judge_sys_override: Optional[str] = None,
    judge_user_template_override: Optional[str] = None,
) -> S2Output:

    deps = S2Deps(raw_text=text, doc_id=doc_id, rag_context=rag_context)

    # 1. Sort Votes & Build Transcript
    order_map = {
        S2Juror.BELIEVER: 1,
        S2Juror.DEFENSE: 2,
        S2Juror.LITERALIST: 3,
        S2Juror.PROFILER: 4,
    }
    votes = sorted(council_result.votes, key=lambda x: order_map.get(x.juror, 99))

    lines = []
    for v in votes:
        role = v.juror.value.upper()
        prefix = (
            "PROSECUTION"
            if v.juror == S2Juror.BELIEVER
            else "DEFENSE" if v.juror == S2Juror.DEFENSE else "WITNESS"
        )
        lines.append(
            f'{prefix} ({role}):\nArgues: {v.verdict.upper()}\nReasoning: "{v.rationale}"'
        )

    transcript = "\n\n".join(lines)

    # Add weighted score info for the judge
    weighted_info = f"""
<voting_analysis>
  <weighted_score>{council_result.weighted_score:.2f}</weighted_score>
  <interpretation>{"Leans CONSPIRACY" if council_result.weighted_score > 0 else "Leans NON-CONSPIRACY" if council_result.weighted_score < 0 else "TIED"}</interpretation>
  <vote_tally>{council_result.tally}</vote_tally>
</voting_analysis>"""

    # 3. Determine System Prompt
    if judge_sys_override:
        base_sys = judge_sys_override
    elif S2_PROMPTS and hasattr(S2_PROMPTS, "judge_sys"):
        base_sys = S2_PROMPTS.judge_sys
    else:
        base_sys = build_s2_judge_system()

    full_sys = assemble_s2_judge_system(base_sys, rag_context)

    # 4. Determine User Template
    if judge_user_template_override:
        usr_tmpl = judge_user_template_override
    elif S2_PROMPTS and hasattr(S2_PROMPTS, "judge_user"):
        usr_tmpl = S2_PROMPTS.judge_user
    else:
        usr_tmpl = build_s2_judge_user_template()

    # 5. Inject Data (include weighted voting analysis)
    user_prompt = (
        usr_tmpl.replace("{{text}}", text)
        .replace("{{transcript}}", transcript + "\n" + weighted_info)
        .replace("{{council_json}}", transcript)
        .replace("{{rag_context}}", rag_context)
        .replace("{{id}}", doc_id)
        .replace("{{weighted_score}}", str(council_result.weighted_score))
    )

    # <--- DEBUG LOG FOR JUDGE --->
    # log_agent_execution("JUDGE", full_sys, user_prompt)

    # 6. Run Agent
    judge_agent = Agent(LLM, output_type=S2Output, system_prompt=full_sys, retries=2)

    try:
        res = await safe_agent_run(judge_agent, user_prompt, deps=deps)
        return res.output
    except Exception as e:
        logger.error(f"Judge Failed: {e}")
        maj_key = (
            max(council_result.tally.keys(), key=lambda k: council_result.tally[k])
            if council_result.tally
            else "non"
        )
        # Ensure label is a valid literal type
        maj: Literal["conspiracy", "non"] = (
            "conspiracy" if maj_key == "conspiracy" else "non"
        )
        return S2Output(
            label=maj,
            rationale=f"Judge crashed ({str(e)})",
            confidence=0.0,
            key_evidence=[],
        )


# --- Dossier Synthesizer (CRITICAL FOR S2 INPUTS) ---
def synthesize_dossier(markers: List[Dict]) -> str:
    """Transforms S1 markers into a readable forensic summary for S2."""
    if not markers:
        return "No markers found."

    buckets = defaultdict(list)
    for m in markers:
        # Handle both dicts and S1Span objects
        txt = m.get("text") if isinstance(m, dict) else m.text
        lbl = m.get("type") if isinstance(m, dict) else m.label
        # Normalize label string (remove 'S1Label.')
        if lbl is None:
            lbl = "Unknown"
        elif hasattr(lbl, "value"):
            lbl = lbl.value
        buckets[str(lbl).capitalize()].append(f'"{txt}"')

    summary = []
    if buckets["Evidence"]:
        summary.append(f"EPISTEMICS: Relies on {', '.join(buckets['Evidence'])}")
    else:
        summary.append("EPISTEMICS: Assertion only (No evidence).")

    if buckets["Actor"]:
        summary.append(f"ACCUSED: {', '.join(buckets['Actor'])}")
    if buckets["Action"]:
        summary.append(f"METHODS: {', '.join(buckets['Action'])}")

    return "\n".join(summary)
