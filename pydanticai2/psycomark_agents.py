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
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any, Literal
from collections import Counter, defaultdict
from botocore.exceptions import ClientError
import random

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
from prompt_loader import S1_PROMPTS
from prompt_builder import (
    build_s1_critic_user_template,
    build_s1_refiner_user_template,
)
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


async def _run_with_backoff(agent, prompt, deps, max_retries=8):
    """
    Wraps agent.run with exponential backoff + jitter for Bedrock Throttling.
    """
    delay = 2.0
    for attempt in range(max_retries + 1):
        try:
            return await agent.run(prompt, deps=deps)
        except Exception as e:
            is_throttle = False
            err_str = str(e)
            if isinstance(e, ClientError):
                code = e.response.get("Error", {}).get("Code", "")
                if "Throttling" in code or "TooManyRequests" in code:
                    is_throttle = True
            elif "ThrottlingException" in err_str or "429" in err_str:
                is_throttle = True

            if is_throttle and attempt < max_retries:
                sleep_time = delay * (2**attempt) + random.uniform(0.5, 1.5)
                logger.warning(
                    f"Bedrock Throttled. Sleeping {sleep_time:.2f}s (Attempt {attempt+1}/{max_retries})"
                )
                await asyncio.sleep(sleep_time)
                continue

            raise e


_provider = BedrockProvider(region_name=AWS_REGION, bedrock_client=_bedrock_client)
LLM = BedrockConverseModel(BEDROCK_MODEL_ID, provider=_provider)

# --- [NEW] Global Set to track what we've printed ---
_PROMPT_LOG_FLAGS = set()


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
    Discriminative Chain-of-Thought Schema (Forensic Audit Mode).
    Forces the model to 'Audit' candidates with explicit justification before filtering.
    """

    # Step 1: Broad Scan (High Recall)
    candidates: List[str] = Field(
        description="List of ALL potential entities, actions, or evidence phrases found in the text (brainstorming phase)."
    )

    # Step 2: The Audit (Negative Constraints)
    rejection_audit: List[S1Rejection] = Field(
        description="A detailed audit of candidates that failed the filter. You MUST explain WHY they were rejected."
    )

    # Step 3: Stance Check (Hard Negative Protection)
    stance_check: str = Field(
        description="Acknowledge if the text is debunking/reporting. Affirm that you will extract valid markers regardless of stance."
    )

    # Step 4: Final Survivors (High Precision)
    final_spans: List[S1Span] = Field(
        description="The final list of valid, verbatim markers that survived the audit."
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
    2. Logic Check: Enforces length constraints on Actions.

    If validation fails, raises ModelRetry to send the error back to the LLM.
    """
    raw_text = ctx.deps.text
    errors = []

    for span in result.final_spans:
        # --- Rule 1: Verbatim Constraint ---
        # We perform a robust check. If exact match fails, we try the robust finder
        # (defined later in this file, resolved at runtime).
        start, end = find_best_span(raw_text, span.text)

        if start == -1:
            # Hallucination detected
            errors.append(
                f"Span '{span.text}' NOT found in source text. You must extract verbatim text only."
            )
            continue

        # --- Rule 2: Logic/Granularity Constraints ---
        # "Actions" should be verb phrases, not full sentences.
        word_count = len(span.text.split())

        if span.label == S1Label.Action and word_count > 10:
            errors.append(
                f"Action '{span.text}' is too long ({word_count} words). "
                "Split this into specific Action (Verb) and Effect (Outcome), or shorten it."
            )

        if span.label == S1Label.Actor and word_count > 8:
            errors.append(
                f"Actor '{span.text}' is too long. Extract the precise entity name only."
            )

    if errors:
        # Combine errors into a single prompt for the retry
        error_msg = "\n- ".join(errors)
        raise ModelRetry(
            f"Validation Failed. Please fix the following errors and re-generate the JSON:\n- {error_msg}"
        )

    return result


# --- 1. Shared Prompt Assembler ---
def assemble_s1_system_prompt(base_instruction: str, few_shots: List[Dict]) -> str:
    """
    Combines the Base Instruction (Static/Optimized) with Dynamic Few-Shots.
    Used by BOTH Production (Decorator) and Optimization (Override).
    """
    # 1. Start with the Base (The part GEPA optimizes)
    full_prompt = base_instruction

    # 2. Append Stratified RAG Examples
    if few_shots:
        examples_xml = ["<reference_examples>"]
        for idx, ex in enumerate(few_shots):
            ex_type = (
                "CONSPIRACY" if ex.get("label") == "conspiracy" else "HARD_NEGATIVE"
            )
            examples_xml.append(
                f"""
  <example id="{idx+1}" type="{ex_type}">
    <input_text>{ex.get('text', '').strip()}</input_text>
    <extracted_spans>
      {json.dumps(ex.get('spans', []), indent=None)} 
    </extracted_spans>
  </example>"""
            )
        examples_xml.append("</reference_examples>")

        full_prompt += "\n\n" + "\n".join(examples_xml)

    return full_prompt


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
    few_shots: List[Dict] = None,
    # Overrides for GEPA Optimization
    gen_prompt_override: str = None,
    user_prompt_template_override: str = None,
    critic_prompt_override: str = None,
    critic_user_template_override: str = None,  # <--- NEW
    refiner_prompt_override: str = None,
    refiner_user_template_override: str = None,  # <--- NEW
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
            draft_result = await active_gen_agent.run(gen_user_msg, deps=deps)
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
            critique_res = await active_critic_agent.run(critique_user_msg, deps=deps)

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
            refine_res = await active_refiner_agent.run(refine_user_msg, deps=deps)
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
    collection: Collection, query_text: str, k: int = 8, filters: dict = None
) -> List[dict]:
    try:
        results = collection.query(query_texts=[query_text], n_results=k, where=filters)
        examples = []
        if results["documents"]:
            for i in range(len(results["documents"][0])):
                ex = {"text": results["documents"][0][i], **results["metadatas"][0][i]}
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


# --- Juror Agent Factory ---
# We use a factory because each Juror needs a different System Prompt
# --- Corrected Factory ---
def create_juror_agent(
    role: S2Juror, temperature: float = 0.4
) -> Agent[S2Deps, S2Vote]:  # <--- FIXED TYPE
    """
    Creates a specialized Juror Agent based on the 'role'.
    """

    # 1. Use 'role' to get the specific persona instructions
    system_prompt_str = get_juror_system_prompt(role)

    # [LOGGING] Print S2 System Prompt (Once per Role)
    log_key = f"s2_system_{role}"
    if log_key not in _PROMPT_LOG_FLAGS:
        logger.info(
            f"\n{'='*40}\n[DEBUG] S2 SYSTEM PROMPT ({role})\n{'='*40}\n{system_prompt_str}\n{'='*40}"
        )
        _PROMPT_LOG_FLAGS.add(log_key)

    # 2. Initialize Agent with that specific prompt
    return Agent(
        LLM,
        output_type=S2Vote,
        deps_type=S2Deps,
        system_prompt=system_prompt_str,  # <--- NOW USED
        model_settings=ModelSettings(temperature=temperature),
        retries=2,
    )


# --- System Prompt Selector ---
def get_juror_system_prompt(role: S2Juror) -> str:
    from prompt_builder import (
        build_s2_triage_system,  # Literalist
        build_s2_profiler_system,  # Profiler
        build_s2_defense_system,  # Defense
        build_s2_system,  # Believer (Standard)
    )

    if role == S2Juror.LITERALIST:
        return build_s2_triage_system()
    elif role == S2Juror.PROFILER:
        return build_s2_profiler_system()
    elif role == S2Juror.DEFENSE:
        return build_s2_defense_system()
    else:  # Believer/Standard
        return build_s2_system(include_cot=False)  # Faster, no CoT needed for voting


# --- The Council Runner (Parallel) ---
async def run_s2_council(
    text: str,
    s1_spans: List[dict],
    marker_summary: str,
    active_jurors: List[S2Juror] = [
        S2Juror.LITERALIST,
        S2Juror.BELIEVER,
        S2Juror.PROFILER,
    ],
    temperature: float = 0.4,  # <--- NEW PARAMETER
) -> S2CouncilOutput:

    deps = S2Deps(
        raw_text=text,
        s1_markers=s1_spans,
        marker_summary=marker_summary,  # Pass the string directly
    )

    # User Prompt is shared (The Evidence)
    # We use a simplified prompt for the jurors to keep it fast
    user_prompt = f"""
<case_file>
  <evidence_text>
{text}
  </evidence_text>

  <forensic_markers>
{marker_summary}
  </forensic_markers>

  <instruction>
    Review the evidence above according to your System Role.
    Render your Verdict.
  </instruction>
</case_file>
"""

    # [LOGGING] Print S2 User Prompt ONCE
    if "s2_user" not in _PROMPT_LOG_FLAGS:
        logger.info(
            f"\n{'='*40}\n[DEBUG] S2 USER PROMPT (First Run)\n{'='*40}\n{user_prompt}\n{'='*40}"
        )
        _PROMPT_LOG_FLAGS.add("s2_user")

    async def _run_juror(role: S2Juror):
        # Create the specialized agent
        agent = create_juror_agent(role, temperature=temperature)

        try:
            # [CRITICAL FIX] Wrap the execution in the semaphore
            # This forces the 4 jurors to queue up if the API is busy
            async with _SC_SEMAPHORE:
                res = await agent.run(user_prompt, deps=deps)

            # Stamp the vote with the juror's ID (Agent returns S2Vote, we ensure .juror is set)
            vote = res.output
            vote.juror = role
            return vote
        except Exception as e:
            logger.warning(f"Juror {role} failed: {e}")
            return None

    # Run in Parallel
    tasks = [_run_juror(role) for role in active_jurors]
    results = await asyncio.gather(*tasks)
    valid_votes = [r for r in results if r is not None]

    # Tally
    counts = Counter(v.verdict for v in valid_votes)

    return S2CouncilOutput(votes=valid_votes, tally=dict(counts))


async def run_s2_sequential_debate(
    text: str, s1_spans: List[dict], marker_summary: str, temperature: float = 0.4
) -> S2CouncilOutput:

    deps = S2Deps(raw_text=text, s1_markers=s1_spans, marker_summary=marker_summary)

    # [FIX] Local Semaphore
    sem = asyncio.Semaphore(1)
    votes = []

    base_prompt = f"<case_file>\n<evidence>{text}</evidence>\n<markers>{marker_summary}</markers>\n</case_file>"

    # Phase 1: Prosecutor
    prosecutor_vote = None
    try:
        async with sem:
            p_agent = create_juror_agent(S2Juror.BELIEVER, temperature)
            p_res = await p_agent.run(base_prompt, deps=deps)
            prosecutor_vote = p_res.output
            prosecutor_vote.juror = S2Juror.BELIEVER
            votes.append(prosecutor_vote)
    except Exception:
        pass

    # Phase 2: Defense
    try:
        def_prompt = base_prompt
        if prosecutor_vote and prosecutor_vote.verdict == "conspiracy":
            def_prompt += f"\n<prosecution_arg>{prosecutor_vote.rationale}</prosecution_arg>\n<instruction>Refute this.</instruction>"

        async with sem:
            d_agent = create_juror_agent(S2Juror.DEFENSE, temperature)
            d_res = await d_agent.run(def_prompt, deps=deps)
            d_vote = d_res.output
            d_vote.juror = S2Juror.DEFENSE
            votes.append(d_vote)
    except Exception:
        pass

    # Phase 3: Witnesses
    async def _run_witness(role):
        try:
            async with sem:
                w_agent = create_juror_agent(role, temperature)
                res = await w_agent.run(base_prompt, deps=deps)
                v = res.output
                v.juror = role
                return v
        except Exception:
            return None

    w_results = await asyncio.gather(
        *[_run_witness(r) for r in [S2Juror.LITERALIST, S2Juror.PROFILER]]
    )
    votes.extend([w for w in w_results if w])

    counts = Counter(v.verdict for v in votes)
    return S2CouncilOutput(votes=votes, tally=dict(counts))


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
        if hasattr(lbl, "value"):
            lbl = lbl.value
        buckets[lbl.capitalize()].append(f'"{txt}"')

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
