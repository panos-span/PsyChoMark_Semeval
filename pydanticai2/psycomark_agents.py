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

import logging
import os
import re
import json
import asyncio
import sys
import pathlib
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any, Literal, Union
from collections import Counter, defaultdict

# --- Make repo root importable FIRST ---
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pydanticai2.prompt_loader import S2_PROMPTS

# Pydantic & Pydantic-AI
from pydantic import BaseModel, Field, ConfigDict
import boto3
import threading
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
    before_sleep_log,  # <--- Import this
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

logger.remove()
logger.add(sys.stderr, level="DEBUG")

_GLOBAL_BEDROCK_SEMAPHORE = threading.Semaphore(1)

# Create a standard logger for tenacity to use
tenacity_logger = logging.getLogger("tenacity")
tenacity_logger.setLevel(logging.WARNING)

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
    stop=stop_after_attempt(20),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    reraise=True,
    before_sleep=before_sleep_log(tenacity_logger, logging.WARNING),
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
# 3. S1: The Discriminative Forensic Agent (DD-CoT Architecture)
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


# ===========================================================================
# 3.1 DD-CoT Schemas (Dynamic Discriminative Chain-of-Thought)
# ===========================================================================


class DDCoTSpan(BaseModel):
    """
    A single span with DISCRIMINATIVE reasoning.
    Key innovation: Explains WHY this label and NOT alternative labels.
    """

    text: str = Field(..., description="Verbatim span extracted from the document.")
    label: S1Label = Field(
        description="Assigned label (Actor/Action/Effect/Victim/Evidence)"
    )

    # Discriminative reasoning (the key DD-CoT innovation)
    why_this_label: str = Field(
        description="Why this span IS this label type (1-2 sentences)."
    )
    action_nucleus: Optional[str] = Field(
        None,
        description="For ACTION labels only: The specific main verb (e.g., 'rigged').",
    )
    why_not_other_labels: Dict[str, str] = Field(
        default_factory=dict,
        description="For each plausible alternative label, why it's NOT that. E.g., {'Victim': 'NOT Victim because it performs the action, not receives it'}",
    )

    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Extraction confidence (0-1).",
    )


class DDCoTExtraction(BaseModel):
    """
    Dynamic Discriminative CoT: Main generator output schema.
    Combines dynamic context assessment with discriminative extraction.
    """

    # Dynamic Assessment (adapts extraction strategy)
    text_complexity: Literal["simple", "moderate", "complex"] = Field(
        description="How ambiguous is this text? simple=clear markers, complex=many borderline cases"
    )
    dominant_narrative: Literal["conspiracy", "neutral", "debunking", "mixed"] = Field(
        description="What type of discourse? Helps calibrate extraction."
    )

    # Discriminative Extraction
    extractions: List[DDCoTSpan] = Field(
        description="List of spans with discriminative reasoning for each."
    )


class S1Reasoning(BaseModel):
    """
    LEGACY: Streamlined Chain-of-Thought Schema for span extraction.
    Kept for backward compatibility. Use DDCoTExtraction for new implementations.
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
    """LEGACY: Simple critique schema. Use EnhancedS1Critique for DD-CoT pipeline."""

    critiques: List[str] = Field(
        description="List of specific errors found (e.g. 'Missed Actor: The CIA'). Empty if perfect."
    )
    requires_refinement: bool = Field(description="True if changes are needed.")


class EnhancedS1Critique(BaseModel):
    """
    Enhanced Critic output for DD-CoT pipeline.
    Adds exhaustiveness and discrimination checks.
    """

    # Quality Errors (existing)
    verbatim_errors: List[str] = Field(
        default_factory=list,
        description="Spans that don't appear verbatim in the source text.",
    )
    granularity_errors: List[str] = Field(
        default_factory=list,
        description="Spans that are too short (e.g., single-word Actions).",
    )
    label_errors: List[str] = Field(
        default_factory=list,
        description="Wrong label assignments (e.g., 'The media' labeled as Evidence instead of Actor).",
    )

    # Exhaustiveness Check (NEW for DD-CoT)
    missed_spans: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Spans that SHOULD have been extracted but weren't. Format: [{'label': 'Actor', 'text': 'The government', 'reason': 'Clear agent of action'}]",
    )

    # Discrimination Check (NEW for DD-CoT)
    confusion_flags: List[str] = Field(
        default_factory=list,
        description="Label confusions detected (e.g., 'Actor↔Victim confusion on the people').",
    )

    requires_refinement: bool = Field(
        description="True if ANY errors were detected and refinement is needed."
    )


class S1Refinement(BaseModel):
    """LEGACY: Simple refinement output. Use DDCoTRefinement for DD-CoT pipeline."""

    final_spans: List[S1Span] = Field(description="The corrected list of spans.")


class DDCoTRefinement(BaseModel):
    """
    Enhanced Refinement output for DD-CoT pipeline.
    Maintains discriminative reasoning through refinement.
    """

    # Corrected extractions with DD-CoT reasoning
    refined_extractions: List[DDCoTSpan] = Field(
        description="The corrected list of spans with discriminative reasoning."
    )

    # Refinement summary for debugging
    fixes_applied: List[str] = Field(
        default_factory=list,
        description="List of fixes that were applied (for logging).",
    )


# ===========================================================================
# 3.2 S1 Agent Definitions (DD-CoT Architecture)
# ===========================================================================

# --- Legacy Agents (for backward compatibility) ---
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


# ===========================================================================
# 3.3 DD-CoT Agent Definitions (Optimal Architecture)
# ===========================================================================

# DD-CoT Generator: Uses discriminative reasoning
# Note: System prompt is loaded dynamically to support GEPA optimization
s1_ddcot_generator_agent: Optional[Agent[S1Deps, DDCoTExtraction]] = None


def get_s1_ddcot_generator(
    system_prompt_override: Optional[str] = None,
) -> Agent[S1Deps, DDCoTExtraction]:
    """
    Factory for DD-CoT Generator agent.
    Allows dynamic system prompt injection for GEPA optimization.
    """
    # Use provided override or load from optimized prompts
    if system_prompt_override:
        sys_prompt = system_prompt_override
    elif hasattr(S1_PROMPTS, "ddcot_gen_system"):
        sys_prompt = S1_PROMPTS.ddcot_gen_system
    else:
        # Fallback to builder function
        from pydanticai2.prompt_builder import build_s1_ddcot_system

        sys_prompt = build_s1_ddcot_system()

    return Agent(
        model=LLM,
        output_type=DDCoTExtraction,
        deps_type=S1Deps,
        retries=2,
        system_prompt=sys_prompt,
        model_settings=ModelSettings(temperature=0.7),  # Creative for extraction
    )


# DD-CoT Critic: Enhanced with exhaustiveness and discrimination checks
s1_ddcot_critic_agent: Optional[Agent[S1Deps, EnhancedS1Critique]] = None


def get_s1_ddcot_critic(
    system_prompt_override: Optional[str] = None,
) -> Agent[S1Deps, EnhancedS1Critique]:
    """
    Factory for DD-CoT Critic agent.
    Allows dynamic system prompt injection for GEPA optimization.
    """
    if system_prompt_override:
        sys_prompt = system_prompt_override
    elif hasattr(S1_PROMPTS, "ddcot_critic_system"):
        sys_prompt = S1_PROMPTS.ddcot_critic_system
    else:
        from pydanticai2.prompt_builder import build_s1_ddcot_critic_system

        sys_prompt = build_s1_ddcot_critic_system()

    return Agent(
        model=LLM,
        output_type=EnhancedS1Critique,
        deps_type=S1Deps,
        system_prompt=sys_prompt,
        model_settings=ModelSettings(temperature=0.0),  # Deterministic for critique
    )


# DD-CoT Refiner: Maintains discriminative reasoning through refinement
s1_ddcot_refiner_agent: Optional[Agent[S1Deps, DDCoTRefinement]] = None


def get_s1_ddcot_refiner(
    system_prompt_override: Optional[str] = None,
) -> Agent[S1Deps, DDCoTRefinement]:
    """
    Factory for DD-CoT Refiner agent.
    Allows dynamic system prompt injection for GEPA optimization.
    """
    if system_prompt_override:
        sys_prompt = system_prompt_override
    elif hasattr(S1_PROMPTS, "ddcot_refiner_system"):
        sys_prompt = S1_PROMPTS.ddcot_refiner_system
    else:
        from pydanticai2.prompt_builder import build_s1_ddcot_refiner_system

        sys_prompt = build_s1_ddcot_refiner_system()

    return Agent(
        model=LLM,
        output_type=DDCoTRefinement,
        deps_type=S1Deps,
        system_prompt=sys_prompt,
        model_settings=ModelSettings(temperature=0.0),  # Strict compliance
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
# 3.4 DD-CoT Pipeline Runner (Optimal Architecture)
# ===========================================================================


def ddcot_span_to_s1_span(ddcot_span: DDCoTSpan) -> S1Span:
    """Convert DDCoTSpan to legacy S1Span for backward compatibility."""
    return S1Span(
        label=ddcot_span.label,
        text=ddcot_span.text,
        why=ddcot_span.why_this_label,  # Preserve reasoning in 'why' field
    )


def validate_ddcot_extraction(
    extraction: DDCoTExtraction, raw_text: str
) -> Tuple[DDCoTExtraction, List[str]]:
    """
    Post-extraction validator for DD-CoT spans.

    Unlike the legacy @output_validator, this runs AFTER extraction and:
    1. Filters out hallucinated spans (not found in text)
    2. Flags short Actions (logged but not removed - let Critic handle)
    3. Returns cleaned extraction + list of issues found

    Returns:
        (cleaned_extraction, issues_list)
    """
    valid_spans = []
    issues = []

    for span in extraction.extractions:
        # Rule 1: Verbatim Check
        start, end = find_best_span(raw_text, span.text)

        if start == -1:
            issues.append(
                f"[HALLUCINATION] Span '{span.text[:50]}...' not found in text - REMOVED"
            )
            continue  # Skip hallucinated spans

        # Rule 2: Minimum Length for Actions (warn but don't remove)
        word_count = len(span.text.split())
        if span.label == S1Label.Action and word_count < 2:
            issues.append(
                f"[GRANULARITY] Action '{span.text}' is very short - flagged for Critic"
            )

        valid_spans.append(span)

    if issues:
        logger.warning(f"[DD-CoT Validator] Found {len(issues)} issues")
        for issue in issues:
            logger.debug(f"  - {issue}")

    cleaned = DDCoTExtraction(
        text_complexity=extraction.text_complexity,
        dominant_narrative=extraction.dominant_narrative,
        extractions=valid_spans,
    )
    return cleaned, issues


async def run_s1_ddcot(
    text: str,
    few_shots: Optional[List[Dict]] = None,
    # Overrides for GEPA Optimization
    gen_prompt_override: Optional[str] = None,
    gen_user_template_override: Optional[str] = None,
    critic_prompt_override: Optional[str] = None,
    critic_user_template_override: Optional[str] = None,
    refiner_prompt_override: Optional[str] = None,
    refiner_user_template_override: Optional[str] = None,
    # Control flags
    skip_critic: bool = False,  # For ablation testing
    return_full_extraction: bool = False,  # Return DDCoTExtraction instead
    return_usage: bool = False,  # <--- NEW ARGUMENT
) -> Union[
    Union[List[S1Span], "DDCoTExtraction"],
    Tuple[Union[List[S1Span], "DDCoTExtraction"], Dict[str, int]],
]:
    """
    DD-CoT Pipeline: Generator -> Critic -> Refiner.

    Optimal S1 architecture with:
    - Dynamic context assessment (text complexity, narrative type)
    - Discriminative reasoning (why IS this label, why NOT others)
    - Enhanced critic with exhaustiveness + discrimination checks
    - Maintains reasoning through refinement

    Returns:
        List[S1Span] by default (for backward compatibility)
        DDCoTExtraction if return_full_extraction=True
    """
    from pydanticai2.prompt_builder import (
        build_s1_ddcot_user_template,
        build_s1_ddcot_critic_user_template,
        build_s1_ddcot_refiner_user_template,
    )

    deps = S1Deps(raw_text=text, few_shots=few_shots or [])
    # sem = asyncio.Semaphore(1)

    # Initialize usage accumulator
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def _acc_usage(res):
        if hasattr(res, "usage"):
            u = res.usage()
            total_usage["input_tokens"] += u.request_tokens or 0
            total_usage["output_tokens"] += u.response_tokens or 0
            total_usage["total_tokens"] += u.total_tokens or 0

    with _GLOBAL_BEDROCK_SEMAPHORE:
        try:
            # STEP 1: DD-CoT GENERATOR
            gen_agent = get_s1_ddcot_generator(gen_prompt_override)

            if gen_prompt_override:
                effective_sys = assemble_s1_system_prompt(
                    gen_prompt_override, few_shots or []
                )
                gen_agent = Agent(
                    LLM,
                    output_type=DDCoTExtraction,
                    deps_type=S1Deps,
                    system_prompt=effective_sys,
                    model_settings=ModelSettings(temperature=0.7),
                )

            if gen_user_template_override:
                gen_user_msg = gen_user_template_override.replace("{{text}}", text)
            elif hasattr(S1_PROMPTS, "ddcot_gen_user_template"):
                gen_user_msg = S1_PROMPTS.ddcot_gen_user_template.replace(
                    "{{text}}", text
                )
            else:
                gen_user_msg = build_s1_ddcot_user_template().replace("{{text}}", text)

            if "{{few_shot_examples}}" in gen_user_msg:
                few_shot_xml = format_s1_fewshots_to_xml(few_shots or [])
                gen_user_msg = gen_user_msg.replace(
                    "{{few_shot_examples}}", few_shot_xml
                )

            logger.debug("[DD-CoT] Running Generator...")
            gen_result = await safe_agent_run(gen_agent, gen_user_msg, deps)
            _acc_usage(gen_result)  # <--- Track Usage
            draft_extraction: DDCoTExtraction = gen_result.output

            # === VALIDATION STEP: Clean hallucinations ===
            draft_extraction, validation_issues = validate_ddcot_extraction(
                draft_extraction, text
            )

            logger.info(
                f"[DD-CoT] Generator: {len(draft_extraction.extractions)} spans "
                f"({len(validation_issues)} validation issues)"
            )

            if skip_critic:
                if return_full_extraction:
                    val = draft_extraction
                else:
                    val = [
                        ddcot_span_to_s1_span(s) for s in draft_extraction.extractions
                    ]

                return (val, total_usage) if return_usage else val

            # STEP 2: ENHANCED CRITIC
            critic_agent = get_s1_ddcot_critic(critic_prompt_override)
            draft_json_str = json.dumps(
                [s.model_dump() for s in draft_extraction.extractions], indent=2
            )

            if critic_user_template_override:
                c_tmpl = critic_user_template_override
            elif hasattr(S1_PROMPTS, "ddcot_critic_user_template"):
                c_tmpl = S1_PROMPTS.ddcot_critic_user_template
            else:
                c_tmpl = build_s1_ddcot_critic_user_template()

            critique_user_msg = (
                c_tmpl.replace("{{text}}", text)
                .replace("{{draft_json}}", draft_json_str)
                .replace("{{complexity}}", draft_extraction.text_complexity)
                .replace("{{narrative}}", draft_extraction.dominant_narrative)
            )

            logger.debug("[DD-CoT] Running Critic...")
            critique_res = await safe_agent_run(critic_agent, critique_user_msg, deps)
            _acc_usage(critique_res)
            critique: EnhancedS1Critique = critique_res.output

            if not critique.requires_refinement:
                if return_full_extraction:
                    val = draft_extraction
                else:
                    val = [
                        ddcot_span_to_s1_span(s) for s in draft_extraction.extractions
                    ]

                return (val, total_usage) if return_usage else val

            # STEP 3: DD-CoT REFINER
            refiner_agent = get_s1_ddcot_refiner(refiner_prompt_override)
            critique_json_str = json.dumps(critique.model_dump(), indent=2)

            if refiner_user_template_override:
                r_tmpl = refiner_user_template_override
            elif hasattr(S1_PROMPTS, "ddcot_refiner_user_template"):
                r_tmpl = S1_PROMPTS.ddcot_refiner_user_template
            else:
                r_tmpl = build_s1_ddcot_refiner_user_template()

            refine_user_msg = (
                r_tmpl.replace("{{text}}", text)
                .replace("{{draft_json}}", draft_json_str)
                .replace("{{critique_json}}", critique_json_str)
            )

            logger.debug("[DD-CoT] Running Refiner...")
            refine_res = await safe_agent_run(refiner_agent, refine_user_msg, deps)
            _acc_usage(refine_res)
            refinement: DDCoTRefinement = refine_res.output

            # === VALIDATION STEP: Clean refined spans ===
            # Create temporary DDCoTExtraction for validation
            temp_extraction = DDCoTExtraction(
                text_complexity=draft_extraction.text_complexity,
                dominant_narrative=draft_extraction.dominant_narrative,
                extractions=refinement.refined_extractions,
            )
            validated_extraction, refine_issues = validate_ddcot_extraction(
                temp_extraction, text
            )

            logger.info(
                f"[DD-CoT] Refiner: {len(validated_extraction.extractions)} spans "
                f"({len(refine_issues)} validation issues)"
            )

            if return_full_extraction:
                val = validated_extraction
            else:
                val = [
                    ddcot_span_to_s1_span(s) for s in validated_extraction.extractions
                ]

            return (val, total_usage) if return_usage else val

        except Exception as e:
            logger.warning(f"[DD-CoT] Pipeline Failed: {e}")
            import traceback

            traceback.print_exc()
            # Return empty result with whatever usage we captured so far
            val = (
                DDCoTExtraction(
                    text_complexity="error", dominant_narrative="error", extractions=[]
                )
                if return_full_extraction
                else []
            )
            return (val, total_usage) if return_usage else val


# ===========================================================================
# 4. Utilities: Search & Verification (The "Map-Reduce" Engine)
# ===========================================================================

from difflib import SequenceMatcher

# Smart Quote Normalization Map
_SMART_TO_STRAIGHT = {
    ord("“"): ord('"'),
    ord("”"): ord('"'),
    ord("‘"): ord("'"),
    ord("’"): ord("'"),
    ord("–"): ord("-"),
    ord("—"): ord("-"),
    ord("…"): None,  # Will handle ellipsis separately
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
        # Unicode-safe end calculation: walk forward matching characters
        end = start
        snippet_idx = 0
        while snippet_idx < len(snippet) and end < len(raw_text):
            if raw_text[end].lower() == snippet[snippet_idx].lower():
                snippet_idx += 1
            end += 1
        # Fallback if walk failed
        if snippet_idx < len(snippet):
            end = start + len(snippet)
        return start, end

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

    # --- Strategy 5: SequenceMatcher Alignment (Last Resort) ---
    # Uses difflib to find best matching region via LCS alignment
    if len(snippet) >= 5:  # Only for reasonably sized spans
        matcher = SequenceMatcher(
            None, raw_text.lower(), snippet.lower(), autojunk=False
        )
        match = matcher.find_longest_match(0, len(raw_text), 0, len(snippet))

        # Accept if we matched at least 70% of the snippet
        if match.size >= len(snippet) * 0.7:
            # Extend boundaries to capture full phrase
            start = match.a
            end = match.a + match.size

            # Try to extend to word boundaries
            while start > 0 and not raw_text[start - 1].isspace():
                start -= 1
            while end < len(raw_text) and not raw_text[end].isspace():
                end += 1

            return start, end

    return -1, -1


# Wrapper for compatibility
def locate_span_in_text(full_text: str, substring: str) -> Tuple[int, int]:
    return find_best_span(full_text, substring, nth=0)


# ===========================================================================
# ENHANCED SPAN UTILITIES (Batch Processing & Deduplication)
# ===========================================================================


def precompute_span_positions(
    raw_text: str, candidates: List[str]
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Pre-compute all positions for all candidate spans at once.
    More efficient than calling find_best_span N times.

    Args:
        raw_text: The source document text
        candidates: List of span text strings to locate

    Returns:
        Dict mapping each snippet to list of (start, end) positions
    """
    positions = {}
    raw_lower = raw_text.lower()

    for snippet in set(candidates):  # Dedupe candidates
        if not snippet:
            continue
        positions[snippet] = []
        snip_lower = snippet.lower()
        start = 0

        while True:
            idx = raw_lower.find(snip_lower, start)
            if idx == -1:
                break
            positions[snippet].append((idx, idx + len(snippet)))
            start = idx + 1

    return positions


def find_span_with_context(
    raw_text: str, snippet: str, left_ctx: str = "", right_ctx: str = "", nth: int = 0
) -> Tuple[int, int]:
    """
    Use left/right context anchors to disambiguate location.
    Useful when snippet appears multiple times but context differs.

    Args:
        raw_text: Source document
        snippet: Text to locate
        left_ctx: Expected text immediately before snippet (5-10 chars)
        right_ctx: Expected text immediately after snippet (5-10 chars)
        nth: Which occurrence to find (0-indexed)

    Returns:
        (start, end) tuple, or (-1, -1) if not found
    """
    if not left_ctx and not right_ctx:
        # No context provided, fall back to standard search
        return find_best_span(raw_text, snippet, nth=nth)

    # Build a pattern that includes context
    pattern_parts = []
    if left_ctx:
        pattern_parts.append(re.escape(left_ctx[-10:]))  # Last 10 chars of left context
    pattern_parts.append(f"({re.escape(snippet)})")
    if right_ctx:
        pattern_parts.append(
            re.escape(right_ctx[:10])
        )  # First 10 chars of right context

    pattern = "".join(pattern_parts)

    try:
        matches = list(re.finditer(pattern, raw_text, re.IGNORECASE))
        if len(matches) > nth:
            match = matches[nth]
            # Return the capture group (the snippet itself)
            return match.start(1), match.end(1)
    except re.error:
        pass  # Pattern failed, fall back

    # Context search failed, use standard method
    return find_best_span(raw_text, snippet, nth=nth)


def deduplicate_overlapping_spans(
    spans: List[Dict], same_label_only: bool = True
) -> List[Dict]:
    """
    Remove spans that are subsets of other spans.

    When two spans overlap significantly:
    - If same_label_only=True: Only remove if labels match
    - Keeps the longer/more complete span

    Args:
        spans: List of span dicts with 'start', 'end', 'label', 'text' keys
        same_label_only: Only dedupe spans with identical labels

    Returns:
        Filtered list of spans with subsets removed
    """
    if not spans:
        return []

    # Sort by start position, then by length (longest first)
    sorted_spans = sorted(
        spans, key=lambda x: (x.get("start", 0), -(x.get("end", 0) - x.get("start", 0)))
    )

    kept = []
    for span in sorted_spans:
        span_start = span.get("start", -1)
        span_end = span.get("end", -1)
        span_label = span.get("label", "")

        if span_start < 0 or span_end < 0:
            continue  # Skip invalid spans

        # Check if this span is a subset of any kept span
        is_subset = False
        for k in kept:
            k_start = k.get("start", -1)
            k_end = k.get("end", -1)
            k_label = k.get("label", "")

            # Check label constraint
            if same_label_only and k_label != span_label:
                continue

            # Check if current span is contained within kept span
            if k_start <= span_start and k_end >= span_end:
                is_subset = True
                break

        if not is_subset:
            kept.append(span)

    return kept


def merge_adjacent_spans(spans: List[Dict], max_gap: int = 2) -> List[Dict]:
    """
    Merge spans of the same label that are nearly adjacent.
    Useful for fixing over-fragmented extractions.

    Args:
        spans: List of span dicts
        max_gap: Maximum characters between spans to consider merging

    Returns:
        List with adjacent same-label spans merged
    """
    if not spans:
        return []

    # Group by label
    by_label = {}
    for span in spans:
        label = span.get("label", "Unknown")
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(span)

    merged = []
    for label, label_spans in by_label.items():
        # Sort by start position
        label_spans.sort(key=lambda x: x.get("start", 0))

        current = None
        for span in label_spans:
            if current is None:
                current = span.copy()
            else:
                # Check if adjacent (with gap tolerance)
                gap = span.get("start", 0) - current.get("end", 0)
                if 0 <= gap <= max_gap:
                    # Merge: extend current span
                    current["end"] = span.get("end", current["end"])
                    current["text"] = (
                        current.get("text", "") + " " + span.get("text", "")
                    )
                else:
                    # Not adjacent, save current and start new
                    merged.append(current)
                    current = span.copy()

        if current:
            merged.append(current)

    # Sort final result by start position
    merged.sort(key=lambda x: x.get("start", 0))
    return merged


def verify_span_boundaries(spans: List[Dict], raw_text: str) -> List[Dict]:
    """
    Verify and fix span boundaries to align with word boundaries.
    Also re-slices text to ensure verbatim match.

    Args:
        spans: List of span dicts with start/end indices
        raw_text: Source document

    Returns:
        Spans with verified/corrected boundaries
    """
    verified = []
    for span in spans:
        start = span.get("start", -1)
        end = span.get("end", -1)

        if start < 0 or end < 0 or start >= len(raw_text) or end > len(raw_text):
            continue  # Invalid span

        # Adjust to word boundaries if needed
        # Trim leading/trailing whitespace
        while start < end and raw_text[start].isspace():
            start += 1
        while end > start and raw_text[end - 1].isspace():
            end -= 1

        if start >= end:
            continue  # Span collapsed to nothing

        # Re-slice text for verbatim match
        verified_span = span.copy()
        verified_span["start"] = start
        verified_span["end"] = end
        verified_span["text"] = raw_text[start:end]

        verified.append(verified_span)

    return verified


# ===========================================================================
# CROSS-ENCODER RERANKING (Runtime Enhancement - No RAG Rebuild Needed)
# ===========================================================================

# Lazy-loaded singleton for cross-encoder (avoids loading if not used)
_CROSS_ENCODER = None
_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def get_cross_encoder():
    """
    Lazy-load cross-encoder model for reranking.
    Uses a lightweight but effective MS MARCO model.
    """
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        try:
            from sentence_transformers import CrossEncoder

            logger.info(f"[Reranker] Loading cross-encoder: {_CROSS_ENCODER_MODEL}")
            _CROSS_ENCODER = CrossEncoder(_CROSS_ENCODER_MODEL)
            logger.success("[Reranker] Cross-encoder loaded successfully")
        except ImportError:
            logger.warning(
                "[Reranker] sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            _CROSS_ENCODER = None
        except Exception as e:
            logger.error(f"[Reranker] Failed to load cross-encoder: {e}")
            _CROSS_ENCODER = None
    return _CROSS_ENCODER


def rerank_documents(
    query: str,
    documents: List[Dict[str, Any]],
    top_k: int,
    text_field: str = "text",
) -> List[Dict[str, Any]]:
    """
    Rerank documents using cross-encoder for better semantic matching.

    Args:
        query: The query text
        documents: List of document dicts with text_field
        top_k: Number of top documents to return
        text_field: Key containing the document text

    Returns:
        Top-k documents sorted by cross-encoder relevance score
    """
    if not documents:
        return []

    if len(documents) <= top_k:
        return documents

    cross_encoder = get_cross_encoder()
    if cross_encoder is None:
        # Fallback: return first top_k without reranking
        logger.debug("[Reranker] Cross-encoder unavailable, returning original order")
        return documents[:top_k]

    try:
        # Create query-document pairs
        pairs = [(query, doc.get(text_field, "")) for doc in documents]

        # Score all pairs
        scores = cross_encoder.predict(pairs)

        # Sort by score (descending) and take top_k
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        reranked = [doc for doc, score in scored_docs[:top_k]]

        logger.debug(
            f"[Reranker] Reranked {len(documents)} → {top_k} docs. "
            f"Top score: {scored_docs[0][1]:.3f}, Bottom retained: {scored_docs[top_k-1][1]:.3f}"
        )

        return reranked

    except Exception as e:
        logger.warning(f"[Reranker] Reranking failed: {e}. Returning original order.")
        return documents[:top_k]


# ===========================================================================
# ENHANCED RETRIEVAL WITH RERANKING
# ===========================================================================


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


def retrieve_fewshots_reranked(
    collection: Collection,
    query_text: str,
    k: int = 4,
    overretrieve_factor: int = 3,
    filters: Optional[dict] = None,
) -> List[dict]:
    """
    Enhanced retrieval with cross-encoder reranking.

    1. Over-retrieves k * overretrieve_factor candidates using bi-encoder
    2. Reranks using cross-encoder for better semantic matching
    3. Returns top-k after reranking

    Args:
        collection: ChromaDB collection
        query_text: Query document text
        k: Final number of examples to return
        overretrieve_factor: How many extra candidates to retrieve for reranking
        filters: Optional metadata filters

    Returns:
        Top-k reranked examples
    """
    # Step 1: Over-retrieve candidates
    overretrieve_k = min(k * overretrieve_factor, 50)  # Cap at 50 to limit latency
    candidates = retrieve_fewshots(
        collection, query_text, k=overretrieve_k, filters=filters
    )

    if len(candidates) <= k:
        return candidates

    # Step 2: Rerank with cross-encoder
    reranked = rerank_documents(query_text, candidates, top_k=k, text_field="text")

    return reranked


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


def retrieve_stratified_s1_reranked(
    collection: Collection,
    query_text: str,
    k_total: int = 6,
    overretrieve_factor: int = 3,
) -> List[Dict]:
    """
    Retrieves balanced Conspiracy AND Non-Conspiracy examples with reranking.

    Each class is over-retrieved and reranked separately to maintain balance.
    """
    if not collection:
        return []

    half = k_total // 2

    # Over-retrieve and rerank each class separately
    pos = retrieve_fewshots_reranked(
        collection,
        query_text,
        k=half,
        overretrieve_factor=overretrieve_factor,
        filters={"label": "conspiracy"},
    )
    neg = retrieve_fewshots_reranked(
        collection,
        query_text,
        k=half,
        overretrieve_factor=overretrieve_factor,
        filters={"label": "non"},
    )

    # Interleave for balanced context
    stratified = []
    for p, n in zip(pos, neg):
        stratified.extend([p, n])
    if len(pos) > len(neg):
        stratified.extend(pos[len(neg) :])
    elif len(neg) > len(pos):
        stratified.extend(neg[len(pos) :])

    return stratified


def retrieve_hard_negatives_reranked(
    collection: Collection,
    query_text: str,
    k: int = 4,
    overretrieve_factor: int = 4,
) -> List[dict]:
    """
    Retrieves hard negative examples with cross-encoder reranking.

    Hard negatives are the most important for S2 Judge training,
    so we use a higher overretrieve factor for better selection.
    """
    return retrieve_fewshots_reranked(
        collection,
        query_text,
        k=k,
        overretrieve_factor=overretrieve_factor,
        filters={"is_hard_negative": True},
    )


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


# ===========================================================================
# 5.1 Enhanced S2 Schemas (Anti-Echo Chamber Architecture)
# ===========================================================================


class BlindVote(BaseModel):
    """
    Independent vote from a juror - NO access to other votes.
    Key anti-echo-chamber innovation: Each juror reasons independently.
    """

    verdict: Literal["conspiracy", "non"] = Field(
        description="Your independent verdict based ONLY on the evidence."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="How certain are you? 0.5 = coin flip, 1.0 = certain.",
    )
    key_signal: str = Field(
        description="The SINGLE most important signal that drove your decision (1 sentence)."
    )
    alternative_interpretation: str = Field(
        description="How could the OTHER side interpret this text? (Prevents confirmation bias)"
    )


class EnhancedS2Vote(BaseModel):
    """
    Enhanced vote with anti-echo-chamber safeguards.
    """

    juror: S2Juror
    verdict: Literal["conspiracy", "non"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(description="Main reasoning (2-3 sentences).")

    # Anti-Echo Chamber Fields
    key_signal: str = Field(
        default="", description="The single most decisive piece of evidence."
    )
    steelman_opposing: str = Field(
        default="",
        description="Best argument FOR the opposing verdict (prevents confirmation bias).",
    )
    uncertainty_flags: List[str] = Field(
        default_factory=list,
        description="What makes this case borderline? (e.g., 'sarcasm unclear', 'reporting vs endorsing')",
    )


class ParallelCouncilOutput(BaseModel):
    """
    Output from parallel (non-sequential) council voting.
    Prevents echo chamber by having all jurors vote independently.
    """

    votes: List[EnhancedS2Vote]
    tally: Dict[str, int]

    # Aggregated Metrics
    conspiracy_confidence_avg: float = Field(
        default=0.0, description="Average confidence of conspiracy votes."
    )
    non_confidence_avg: float = Field(
        default=0.0, description="Average confidence of non-conspiracy votes."
    )
    weighted_score: float = Field(
        default=0.0,
        description="Confidence-weighted score: positive = conspiracy, negative = non.",
    )

    # Dissent Analysis (for Judge calibration)
    dissent_strength: float = Field(
        default=0.0,
        description="How strong is the minority opinion? High = borderline case.",
    )
    consensus_level: Literal["unanimous", "strong", "split", "chaotic"] = Field(
        default="split",
        description="unanimous=4-0, strong=3-1, split=2-2, chaotic=many abstentions",
    )
    common_uncertainty_flags: List[str] = Field(
        default_factory=list,
        description="Uncertainty flags mentioned by multiple jurors.",
    )


class CalibratedJudgeOutput(BaseModel):
    """
    Enhanced Judge output with calibration signals.
    """

    label: Literal["conspiracy", "non"] = Field(
        description="Final verdict after weighing all evidence."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Calibrated confidence (lower if council was split).",
    )
    rationale: str = Field(
        description="Explanation referencing council votes AND any dissent considered."
    )

    # Calibration Fields
    dissent_considered: bool = Field(
        default=False,
        description="Did the Judge explicitly consider minority opinions?",
    )
    key_evidence: List[str] = Field(
        default_factory=list, description="1-3 verbatim quotes that sealed the verdict."
    )
    council_override: bool = Field(
        default=False,
        description="Did the Judge override the council majority? (Rare but allowed)",
    )
    borderline_flag: bool = Field(
        default=False,
        description="Is this a hard case that should be flagged for review?",
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


# ===========================================================================
# 5.2 PARALLEL COUNCIL RUNNER (Anti-Echo Chamber)
# ===========================================================================


def create_parallel_juror_agent(
    role: S2Juror,
    deps: S2Deps,
    override_sys: Optional[str] = None,
    temperature: float = 0.4,
) -> Agent[S2Deps, EnhancedS2Vote]:
    """
    Creates a juror agent that outputs EnhancedS2Vote.
    Key difference: Uses BlindVote-aware prompts that prevent echo chamber.
    """
    from pydanticai2.prompt_builder import (
        build_s2_parallel_prosecutor_system,
        build_s2_parallel_defense_system,
        build_s2_parallel_literalist_system,
        build_s2_parallel_profiler_system,
    )

    # Determine base system prompt
    base = ""
    if override_sys:
        base = override_sys
    elif S2_PROMPTS:
        # Check for parallel versions first, fallback to legacy
        if role == S2Juror.BELIEVER:
            base = getattr(S2_PROMPTS, "parallel_pros_sys", None) or S2_PROMPTS.pros_sys
        elif role == S2Juror.DEFENSE:
            base = getattr(S2_PROMPTS, "parallel_def_sys", None) or S2_PROMPTS.def_sys
        elif role == S2Juror.LITERALIST:
            base = getattr(S2_PROMPTS, "parallel_lit_sys", None) or S2_PROMPTS.lit_sys
        elif role == S2Juror.PROFILER:
            base = getattr(S2_PROMPTS, "parallel_prof_sys", None) or S2_PROMPTS.prof_sys
    else:
        # Fallback to builders
        if role == S2Juror.BELIEVER:
            base = build_s2_parallel_prosecutor_system()
        elif role == S2Juror.DEFENSE:
            base = build_s2_parallel_defense_system()
        elif role == S2Juror.LITERALIST:
            base = build_s2_parallel_literalist_system()
        elif role == S2Juror.PROFILER:
            base = build_s2_parallel_profiler_system()

    # Inject RAG context
    full_sys = assemble_s2_system_prompt(base, deps.rag_context)

    return Agent(
        LLM,
        output_type=EnhancedS2Vote,
        deps_type=S2Deps,
        system_prompt=full_sys,
        model_settings=ModelSettings(temperature=temperature),
        retries=2,
    )


async def run_s2_parallel_council(
    text: str,
    s1_spans: List[dict],
    marker_summary: str,
    rag_context: str = "",
    temperature: float = 0.4,
    # GEPA System Overrides (Parallel versions)
    prosecutor_sys_override: Optional[str] = None,
    defense_sys_override: Optional[str] = None,
    literalist_sys_override: Optional[str] = None,
    profiler_sys_override: Optional[str] = None,
    # GEPA User Template Override (shared for parallel voting)
    parallel_user_template_override: Optional[str] = None,
    return_usage: bool = False,  # <--- NEW
) -> Union[ParallelCouncilOutput, Tuple[ParallelCouncilOutput, Dict[str, int]]]:
    """
    PARALLEL Council: All jurors vote INDEPENDENTLY and SIMULTANEOUSLY.

    Anti-Echo Chamber guarantees:
    1. No juror sees another's vote
    2. Each juror must steelman the opposing view
    3. Uncertainty flags are collected for Judge calibration
    """
    from pydanticai2.prompt_builder import build_s2_parallel_user_template

    deps = S2Deps(
        raw_text=text,
        s1_markers=s1_spans,
        marker_summary=marker_summary,
        rag_context=rag_context,
    )

    # Shared user prompt for all jurors (parallel = same evidence, no prior votes)
    if parallel_user_template_override:
        user_template = parallel_user_template_override
    elif hasattr(S2_PROMPTS, "parallel_user"):
        user_template = S2_PROMPTS.parallel_user
    else:
        user_template = build_s2_parallel_user_template()

    user_msg = user_template.replace("{{text}}", text).replace(
        "{{marker_summary}}", marker_summary
    )

    # Initialize Usage
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    # async def _run_juror(
    #    role: S2Juror, sys_override: Optional[str]
    # ) -> Optional[EnhancedS2Vote]:
    #    try:
    #        agent = create_parallel_juror_agent(role, deps, sys_override, temperature)
    #        res = await safe_agent_run(agent, user_msg, deps)
    #        if res:
    #            vote = res.output
    #            vote.juror = role
    #            return vote
    #    except Exception as e:
    #        logger.warning(f"[Parallel Council] {role.value} failed: {e}")
    #    return None

    # Run ALL jurors in parallel (no sequential influence!)
    # tasks = [
    #    _run_juror(S2Juror.BELIEVER, prosecutor_sys_override),
    #    _run_juror(S2Juror.DEFENSE, defense_sys_override),
    #    _run_juror(S2Juror.LITERALIST, literalist_sys_override),
    #    _run_juror(S2Juror.PROFILER, profiler_sys_override),
    # ]
    # with _GLOBAL_BEDROCK_SEMAPHORE:
    #    results = await asyncio.gather(*tasks)
    # valid_votes = [v for v in results if v is not None]

    # [CHANGE 1] Define the list of juror configs to run
    juror_configs = [
        (S2Juror.BELIEVER, prosecutor_sys_override),
        (S2Juror.DEFENSE, defense_sys_override),
        (S2Juror.LITERALIST, literalist_sys_override),
        (S2Juror.PROFILER, profiler_sys_override),
    ]

    valid_votes = []

    # [CHANGE 2] Use the Global Semaphore around a SERIAL loop
    # This prevents the "Burst of 4" that kills your rate limit.
    with _GLOBAL_BEDROCK_SEMAPHORE:
        for role, sys_override in juror_configs:
            try:
                # Run one juror
                agent = create_parallel_juror_agent(
                    role, deps, sys_override, temperature
                )
                res = await safe_agent_run(agent, user_msg, deps)

                # Track Usage
                if hasattr(res, "usage"):
                    u = res.usage()
                    total_usage["input_tokens"] += u.request_tokens or 0
                    total_usage["output_tokens"] += u.response_tokens or 0
                    total_usage["total_tokens"] += u.total_tokens or 0

                if res:
                    vote = res.output
                    vote.juror = role
                    valid_votes.append(vote)

                # [OPTIONAL] Tiny cooldown between jurors to be nice to the API
                await asyncio.sleep(1)

            except Exception as e:
                logger.warning(f"[Parallel Council] {role.value} failed: {e}")

    # Compute aggregates
    tally = Counter(v.verdict for v in valid_votes)

    # Confidence averages by verdict
    consp_votes = [v for v in valid_votes if v.verdict == "conspiracy"]
    non_votes = [v for v in valid_votes if v.verdict == "non"]

    consp_conf_avg = (
        sum(v.confidence for v in consp_votes) / len(consp_votes)
        if consp_votes
        else 0.0
    )
    non_conf_avg = (
        sum(v.confidence for v in non_votes) / len(non_votes) if non_votes else 0.0
    )

    # Weighted score
    weighted_score = sum(
        v.confidence if v.verdict == "conspiracy" else -v.confidence
        for v in valid_votes
    )

    # Dissent analysis
    majority_count = max(tally.values()) if tally else 0
    minority_count = min(tally.values()) if len(tally) > 1 else 0
    total_votes = len(valid_votes)

    # Dissent strength: How strong is the minority?
    dissent_strength = minority_count / total_votes if total_votes > 0 else 0.0

    # Consensus level
    if total_votes == 0:
        consensus = "chaotic"
    elif majority_count == total_votes:
        consensus = "unanimous"
    elif majority_count >= 3:
        consensus = "strong"
    elif majority_count == 2 and minority_count == 2:
        consensus = "split"
    else:
        consensus = "split"

    # Collect common uncertainty flags
    all_flags = []
    for v in valid_votes:
        all_flags.extend(v.uncertainty_flags)
    flag_counts = Counter(all_flags)
    common_flags = [
        f for f, c in flag_counts.items() if c >= 2
    ]  # Mentioned by 2+ jurors

    logger.info(
        f"[Parallel Council] Votes: {dict(tally)}, "
        f"Consensus: {consensus}, Dissent: {dissent_strength:.2f}"
    )

    output = ParallelCouncilOutput(
        votes=valid_votes,
        tally=dict(tally),
        conspiracy_confidence_avg=consp_conf_avg,
        non_confidence_avg=non_conf_avg,
        weighted_score=weighted_score,
        dissent_strength=dissent_strength,
        consensus_level=consensus,
        common_uncertainty_flags=common_flags,
    )

    return (output, total_usage) if return_usage else output


# ===========================================================================
# 5.3 CALIBRATED JUDGE (Anti-Echo Chamber)
# ===========================================================================


async def run_s2_calibrated_judge(
    text: str,
    council_result: ParallelCouncilOutput,
    doc_id: str = "opt_sample",
    rag_context: str = "",
    judge_sys_override: Optional[str] = None,
    judge_user_template_override: Optional[str] = None,
    return_usage: bool = False,
) -> Union[CalibratedJudgeOutput, Tuple[CalibratedJudgeOutput, Dict[str, int]]]:
    """
    Calibrated Judge: Makes final decision with DISSENT AWARENESS.

    Key improvements over legacy judge:
    1. Explicitly considers minority opinions
    2. Lowers confidence when council is split
    3. Flags borderline cases for review
    4. Can override council if evidence warrants
    """
    from pydanticai2.prompt_builder import (
        build_s2_calibrated_judge_system,
        build_s2_calibrated_judge_user_template,
    )

    deps = S2Deps(raw_text=text, doc_id=doc_id, rag_context=rag_context)

    # Build vote transcript with steelman arguments visible
    vote_lines = []
    for v in council_result.votes:
        vote_lines.append(
            f"""
<vote juror="{v.juror.value}">
  <verdict>{v.verdict.upper()}</verdict>
  <confidence>{v.confidence:.2f}</confidence>
  <rationale>{v.rationale}</rationale>
  <key_signal>{v.key_signal}</key_signal>
  <steelman_opposing>{v.steelman_opposing}</steelman_opposing>
  <uncertainty_flags>{', '.join(v.uncertainty_flags) if v.uncertainty_flags else 'None'}</uncertainty_flags>
</vote>"""
        )

    transcript = "\n".join(vote_lines)

    # Council analysis for Judge context
    council_analysis = f"""
<council_analysis>
  <vote_tally>{council_result.tally}</vote_tally>
  <weighted_score>{council_result.weighted_score:.2f}</weighted_score>
  <consensus_level>{council_result.consensus_level}</consensus_level>
  <dissent_strength>{council_result.dissent_strength:.2f}</dissent_strength>
  <conspiracy_avg_confidence>{council_result.conspiracy_confidence_avg:.2f}</conspiracy_avg_confidence>
  <non_avg_confidence>{council_result.non_confidence_avg:.2f}</non_avg_confidence>
  <common_uncertainty_flags>{', '.join(council_result.common_uncertainty_flags) if council_result.common_uncertainty_flags else 'None'}</common_uncertainty_flags>
</council_analysis>
"""

    # Determine system prompt
    if judge_sys_override:
        base_sys = judge_sys_override
    elif hasattr(S2_PROMPTS, "calibrated_judge_sys"):
        base_sys = S2_PROMPTS.calibrated_judge_sys
    else:
        base_sys = build_s2_calibrated_judge_system()

    full_sys = assemble_s2_judge_system(base_sys, rag_context)

    # Determine user template
    if judge_user_template_override:
        usr_tmpl = judge_user_template_override
    elif hasattr(S2_PROMPTS, "calibrated_judge_user"):
        usr_tmpl = S2_PROMPTS.calibrated_judge_user
    else:
        usr_tmpl = build_s2_calibrated_judge_user_template()

    # Inject data
    user_prompt = (
        usr_tmpl.replace("{{text}}", text)
        .replace("{{transcript}}", transcript)
        .replace("{{council_analysis}}", council_analysis)
        .replace("{{rag_context}}", rag_context)
        .replace("{{id}}", doc_id)
    )

    # Run calibrated judge
    judge_agent = Agent(
        LLM,
        output_type=CalibratedJudgeOutput,
        system_prompt=full_sys,
        retries=2,
    )

    usage_dict = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    try:
        with _GLOBAL_BEDROCK_SEMAPHORE:
            res = await safe_agent_run(judge_agent, user_prompt, deps=deps)

            # Track Usage
            if hasattr(res, "usage"):
                u = res.usage()
                usage_dict["input_tokens"] = u.request_tokens or 0
                usage_dict["output_tokens"] = u.response_tokens or 0
                usage_dict["total_tokens"] = u.total_tokens or 0
            output = res.output

            # Post-process: Flag borderline if council was split
            if council_result.consensus_level in ["split", "chaotic"]:
                output.borderline_flag = True

            # Check if judge overrode council majority
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
                    f"[Calibrated Judge] OVERRIDE: Judge ruled {output.label} vs council {majority_verdict}"
                )

            return (output, usage_dict) if return_usage else output

    except Exception as e:
        logger.error(f"[Calibrated Judge] Failed: {e}")
        # Fallback to council majority
        maj_key = (
            max(council_result.tally.keys(), key=lambda k: council_result.tally[k])
            if council_result.tally
            else "non"
        )
        maj: Literal["conspiracy", "non"] = (
            "conspiracy" if maj_key == "conspiracy" else "non"
        )

        # Return fallback
        fallback = CalibratedJudgeOutput(
            label="non", confidence=0.0, rationale="Error", dissent_considered=False
        )
        return (fallback, usage_dict) if return_usage else fallback


# --- Dossier Synthesizer (CRITICAL FOR S2 INPUTS) ---
def synthesize_dossier(
    markers: List[Dict], complexity: str = "Unknown", narrative: str = "Unknown"
) -> str:
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
    # --- 1. Inject Dynamic Assessment (The New Capability) ---
    # This gives S2 jurors the "meta-context" they lacked before
    summary.append(
        f"DYNAMIC ASSESSMENT: Complexity={complexity.upper()} | Narrative={narrative.upper()}"
    )
    summary.append("-" * 40)
    if buckets["Evidence"]:
        summary.append(f"EPISTEMICS: Relies on {', '.join(buckets['Evidence'])}")
    else:
        summary.append("EPISTEMICS: Assertion only (No evidence).")

    if buckets["Actor"]:
        summary.append(f"ACCUSED: {', '.join(buckets['Actor'])}")
    if buckets["Action"]:
        summary.append(f"METHODS: {', '.join(buckets['Action'])}")

    return "\n".join(summary)
