#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psycomark_agents.py — Pydantic-AI native S1/S2 agents that preserve your
existing (system,user) prompt shapes from prompt_builder.py.

- S1: span extraction with exact offsets (Actor, Action, Effect, Victim, Evidence)
- S1-Verifier: enforced natively via output_type validation + ModelRetry
- S2: conspiracy vs non (optionally 'cant_tell')

Compatibility:
- We *reuse* the system/user builders from prompt_builder.py so the LLM sees
  the same instructions as in your old runner. The only change is that responses
  are now *structured and validated* via pydantic-ai (no brittle string parsing).
- You can call the helpers here from a new runner, or migrate gradually.

Bedrock:
- Uses BedrockConverseModel via pydantic-ai (region/model from env).
- Falls back cleanly if you swap model IDs or regions.

Author: you
"""

from __future__ import annotations

import os
import re
import asyncio
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any, Literal
from prompt_builder import (
    to_s2_marker,
)  # utility for normalizing S1 spans into S2 markers
from prompt_builder import (
    build_s1_system,
    build_s1_system_aot,
    build_s1_user,
    build_s2_system,
    build_s2_user,
)
import chromadb
from chromadb import Collection
from chromadb.utils import embedding_functions
from pydantic import BaseModel, Field, field_validator, ConfigDict

# pydantic-ai core
from pydantic_ai import Agent, RunContext, ModelSettings
from pydantic_ai.models.bedrock import BedrockConverseModel
from loguru import logger
from collections import Counter
import json

# Bedrock provider/model (pydantic-ai)
from pydantic_ai.providers.bedrock import BedrockProvider

_SC_SEMAPHORE = asyncio.Semaphore(3)  # limit concurrent Bedrock calls

# ---------------------------------------------------------------------------
# Prompt caching wrapper for Bedrock Converse
# ---------------------------------------------------------------------------


# class _CachingBedrockClient:
#    """
#    Thin wrapper around the real boto3 Bedrock client.
#
#    It intercepts converse(**params) and, when enabled, injects a
#    cachePoint block into the 'system' content array so that the
#    static system prompt (instructions + few-shots) is cached.
#
#    This relies on Bedrock's prompt caching for Converse API:
#      - https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
#    """
#
#    def __init__(self, client, enable_cache: bool = True, cache_type: str = "default"):
#        self._client = client
#        self._enable_cache = enable_cache
#        self._cache_type = cache_type
#
#    def converse(self, **params):
#        # If caching is disabled, just delegate
#        if not self._enable_cache:
#            return self._client.converse(**params)
#
#        model_id = params.get("modelId", "") or ""
#
#        # Very light guard: only try to cache on Claude/Nova-like model IDs
#        # and when the account actually has prompt caching enabled.
#        # You can tighten/loosen this as needed.
#        if not any(k in model_id for k in ("anthropic", "claude", "nova")):
#            return self._client.converse(**params)
#
#        system = params.get("system")
#        if isinstance(system, list) and system:
#            # Check if we already added a cachePoint
#            has_cachepoint = any(
#                isinstance(block, dict) and "cachePoint" in block for block in system
#            )
#            if not has_cachepoint:
#                # Append a cachePoint block so that all preceding system content
#                # (our giant instructions + fewshots) becomes the cached prefix.
#                system = list(system) + [{"cachePoint": {"type": self._cache_type}}]
#                params["system"] = system
#
#        # You could also cache tools here, if you start using Bedrock tools:
#        # tool_config = params.get("toolConfig")
#        # ...
#
#        return self._client.converse(**params)
#
#    def __getattr__(self, name):
#        # Delegate all other attributes/methods to the real client
#        return getattr(self._client, name)
#
#
# class CachingBedrockConverseModel(BedrockConverseModel):
#    """
#    Drop-in replacement for BedrockConverseModel that wraps the underlying
#    boto3 client with _CachingBedrockClient.
#
#    This means pydantic-ai still builds the same Converse params; the only
#    difference is that just before hitting Bedrock we insert a cachePoint
#    in the system content (when enabled).
#    """
#
#    def __init__(
#        self,
#        model_id: str,
#        provider: BedrockProvider,
#        *,
#        enable_cache: bool | None = None,
#        cache_type: str = "default",
#    ):
#        # Normal BedrockConverseModel setup
#        super().__init__(model_id, provider=provider)
#
#        # Env toggle so you can easily disable if needed:
#        #   BEDROCK_PROMPT_CACHE=0 -> disabled
#        if enable_cache is None:
#            from os import getenv
#
#            enable_cache = getenv("BEDROCK_PROMPT_CACHE", "1") not in (
#                "0",
#                "false",
#                "False",
#            )
#
#        # Wrap the boto3 client
#        real_client = self.client
#        self.client = _CachingBedrockClient(
#            real_client,
#            enable_cache=enable_cache,
#            cache_type=cache_type,
#        )


# ===========================================================================
# Bedrock model wiring
# ===========================================================================
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-central-1")
BEDROCK_MODEL_ID = os.getenv(
    "MODEL_ID",
    # Make this match your active Bedrock Anthropic Sonnet ID for the region.
    # Keep it configurable; Bedrock model IDs vary per region/account.
    "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
)

_provider = BedrockProvider(region_name=AWS_REGION)
LLM = BedrockConverseModel(BEDROCK_MODEL_ID, provider=_provider)
# LLM = CachingBedrockConverseModel(
#    BEDROCK_MODEL_ID, provider=_provider, cache_type="default", enable_cache=True
# )


# ===========================================================================
# 0. RAG Utilities (Bedrock Titan Integration)
# ===========================================================================


class BedrockTitanEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """
    ChromaDB-compatible wrapper for Amazon Titan Text v2.
    """

    def __init__(self, region_name: str = AWS_REGION):
        import boto3

        self.bedrock = boto3.client(
            service_name="bedrock-runtime", region_name=region_name
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
    return client.get_collection(name=name, embedding_function=ef)


def retrieve_fewshots(
    collection: Collection, query_text: str, k: int = 8, filters: dict = None
) -> List[dict]:
    """
    Performs dynamic retrieval.
    Returns a list of dicts formatted for prompt_builder.
    """
    try:
        results = collection.query(query_texts=[query_text], n_results=k, where=filters)

        examples = []
        if results["documents"]:
            for i in range(len(results["documents"][0])):
                doc_text = results["documents"][0][i]
                meta = results["metadatas"][0][i]

                # Reconstruct the example object
                ex = {"text": doc_text, **meta}

                # Deserialize JSON fields stored as strings in Chroma
                if "spans_json" in ex:
                    ex["spans"] = json.loads(ex.pop("spans_json"))
                if "markers_json" in ex:
                    ex["markers"] = json.loads(ex.pop("markers_json"))

                examples.append(ex)
        return examples
    except Exception as e:
        logger.error(f"[RAG] Retrieval failed: {e}")
        return []


# ===========================================================================
# S1 — structured output types + agent
# ===========================================================================
# 1) define deps for S1
class S1Deps(BaseModel):
    model_config = ConfigDict(extra="ignore")  # ignore any unexpected kwargs
    raw_text: str
    doc_id: Optional[str] = None


class S1Label(str, Enum):
    Actor = "Actor"
    Action = "Action"
    Effect = "Effect"
    Victim = "Victim"
    Evidence = "Evidence"


class S1Span(BaseModel):
    label: S1Label
    text: str = Field(..., description="Verbatim snippet from RAW text")
    start: int | None = Field(None, description="0-indexed, inclusive")
    end: int | None = Field(None, description="0-indexed, end-exclusive")

    # [NEW] Add support for the "Why" rationale your new artifacts teach
    why: Optional[str] = Field(
        None, description="Brief forensic reason for this label (if generated)."
    )
    context: Optional[str] = Field(None, description="Local context snippet.")

    @field_validator("end")
    @classmethod
    def _end_after_start(cls, v, info):
        start = info.data.get("start", None)
        if start is not None and v <= start:
            raise ValueError("end must be greater than start")
        return v


class S1Output(BaseModel):
    spans: List[S1Span] = Field(default_factory=list)


class MarkerSummary(BaseModel):
    """
    Compact, S2-friendly summary for one marker label.

    We want:
      - a salience estimate (how central this marker is),
      - a conspiratoriality estimate (does it carry conspiratorial framing),
      - up to 3 short bullets with concrete cues.
    """

    label: S1Label
    salience: Literal["none", "low", "medium", "high"] = "low"
    conspiratoriality: Literal["none", "possible", "strong"] = "none"
    bullets: List[str] = Field(
        default_factory=list,
        description="Up to 3 short bullets (<15 words each) naming key cues.",
    )

    @field_validator("bullets")
    @classmethod
    def cap_bullets(cls, v: List[str]) -> List[str]:
        # Hard cap at 3 bullets; truncate silently if more are produced.
        return v[:3]


class S1ReviewOutput(BaseModel):
    """
    Output of the S1 self-consistency reviewer:
      - spans: final merged spans
      - summary: list[MarkerSummary], one per label that actually appears.
    """

    spans: List[S1Span] = Field(default_factory=list)
    summary: List[MarkerSummary] = Field(default_factory=list)


class S1ReviewDeps(BaseModel):
    """
    Deps provided to the reviewer agent:
      - raw_text: authoritative source for offset sanity
      - candidate_sets: k S1 outputs, as lists of dicts
    """

    model_config = ConfigDict(extra="ignore")
    raw_text: str
    doc_id: Optional[str] = None
    candidate_sets: List[List[Dict[str, Any]]] = Field(
        default_factory=list, description="K candidate span sets from S1 runs."
    )


# We create a single Agent instance and update its system prompt per run.
agent_s1 = Agent(
    LLM,
    output_type=S1Output,
    system_prompt="(placeholder)",
    deps_type=S1Deps,  # <-- IMPORTANT
    retries=4,
    output_retries=4,
    model_settings=ModelSettings(temperature=0.0),  # optional, reduces paraphrase
)


def _extract_raw_text_from_user_prompt(user_prompt: str) -> str:
    """We put the RAW text inside <text_to_analyze>...</text_to_analyze>."""
    m = re.search(r"<text_to_analyze>\s*(.*?)\s*</text_to_analyze>", user_prompt, re.S)
    return m.group(1) if m else user_prompt


# def _evidence_gate_ok(span_text: str) -> bool:
#    """
#    Evidence valid if ANY of:
#      (a) URL/domain present, OR
#      (b) Quoted material WITH attribution cue, OR
#      (c) Numeric facts WITH units (or %) AND attribution cue.
#    Mirrors the gate in your prompt text. :contentReference[oaicite:1]{index=1}
#    """
#    has_url = bool(_URL_RE.search(span_text))
#    has_quote = bool(_QUOTE_RE.search(span_text))
#    has_attrib = bool(_ATTRIB_RE.search(span_text))
#    has_numeric = bool(_NUMERIC_UNIT_RE.search(span_text))
#    if has_url:
#        return True
#    if has_quote and has_attrib:
#        return True
#    if has_numeric and has_attrib:
#        return True
#    # Also accept pure attribution with a named source (loose but useful).
#    if has_attrib:
#        return True
#    return False


SMART_TO_STRAIGHT = {
    "\u2018": "'",
    "\u2019": "'",
    "\u2032": "'",  # single quotes/prime
    "\u201c": '"',
    "\u201d": '"',
    "\u2033": '"',  # double quotes
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",  # dashes
    "\u00a0": " ",  # nbsp
}

# Expand the mapping of “smart” punctuation → straight ASCII
_SMART_TO_STRAIGHT = {
    ord("“"): ord('"'),
    ord("”"): ord('"'),
    ord("‘"): ord("'"),
    ord("’"): ord("'"),
    ord("‚"): ord("'"),
    ord("‛"): ord("'"),
    ord("«"): ord('"'),
    ord("»"): ord('"'),
    ord("–"): ord("-"),
    ord("—"): ord("-"),
    ord("−"): ord("-"),
    ord("…"): ord("."),  # we'll collapse runs anyway
}

# Collapse *all* whitespace kinds (spaces, NBSP, thin spaces, ZWSP sequences) to a single normal space
# Includes: \xA0 (NBSP), \u2000–\u200B (en/em/thin/ZWSP), \u202F (NNBSP)
_NORMALIZE_SPACE_RE = re.compile(r"[\s\u00A0\u2000-\u200B\u202F]+")


def _normalize_for_match(s: str) -> tuple[str, list[int]]:
    """
    Returns (norm, idx_map) where:
      - norm: s with smart quotes/dashes normalized and whitespace runs collapsed to ' '.
      - idx_map[i]: RAW index in original s corresponding to norm[i].
    """
    if not s:
        return "", []

    t = s.translate(_SMART_TO_STRAIGHT)

    norm_chars: list[str] = []
    idx_map: list[int] = []

    i, L = 0, len(t)
    while i < L:
        ch = t[i]
        # Collapse any whitespace-like cluster (incl. NBSP / thin / ZWSP) to a single ' '
        if _NORMALIZE_SPACE_RE.match(t, i):
            m = _NORMALIZE_SPACE_RE.match(t, i)
            norm_chars.append(" ")
            idx_map.append(i)  # map collapsed space to the first char of the run
            i = m.end()
        else:
            norm_chars.append(ch)
            idx_map.append(i)
            i += 1

    return "".join(norm_chars), idx_map


def _find_best_span(
    raw: str, snippet: str = None, *, nth: int = 0
) -> Optional[Tuple[int, int]]:
    """
    Locate the nth occurrence of `snippet` in RAW using:
      1) Exact match.
      2) Case-insensitive match (only if same-length exact RAW slice).
      3) Normalized (quotes+whitespace) match with reversible mapping to RAW.

    Returns (start, end) in RAW or None.
    """
    if not snippet:
        return None

    # 1) exact — iterate to nth
    start = 0
    for k in range(nth + 1):
        i = raw.find(snippet, start)
        if i < 0:
            break
        if k == nth:
            return i, i + len(snippet)
        start = i + 1  # advance at least 1 char

    # 2) case-insensitive (guarded) — iterate to nth
    lr, ls = raw.lower(), snippet.lower()
    start = 0
    for k in range(nth + 1):
        i = lr.find(ls, start)
        if i < 0:
            break
        j = i + len(snippet)
        if lr[i:j] == ls:
            if k == nth:
                return i, j
        start = i + 1

    # 3) normalized search with reverse map — iterate to nth
    norm_raw, raw_map = _normalize_for_match(raw)
    norm_snip, _ = _normalize_for_match(snippet)
    if not norm_snip:
        return None

    start = 0
    hit_idx = None
    for k in range(nth + 1):
        i = norm_raw.find(norm_snip, start)
        if i < 0:
            hit_idx = None
            break
        if k == nth:
            hit_idx = i
            break
        start = i + 1

    if hit_idx is None:
        return None

    # map back to RAW
    s = raw_map[hit_idx]
    end_norm_idx = hit_idx + len(norm_snip) - 1
    e = raw_map[end_norm_idx] + 1

    # safety: compare normalized slices
    raw_slice_norm, _ = _normalize_for_match(raw[s:e])
    snip_norm, _ = _normalize_for_match(snippet)
    if raw_slice_norm == snip_norm:
        return s, e
    return None


def create_s1_agent(
    system_prompt: str, temperature: float = 0.0
) -> Agent[S1Deps, S1Output]:
    """
    Fresh agent per document. Binds the same output validator to the new instance.
    """
    agent = Agent(
        LLM,
        output_type=S1Output,
        system_prompt=system_prompt,
        deps_type=S1Deps,  # <-- IMPORTANT
        retries=4,
        output_retries=4,
        model_settings=ModelSettings(
            temperature=temperature
        ),  # optional, reduces paraphrase
    )

    @agent.output_validator
    async def _bound_validator(ctx: RunContext[S1Deps], output: S1Output) -> S1Output:
        return await s1_verifier_impl(ctx, output)

    return agent


def create_s1_reviewer_agent(
    system_prompt: str, temperature: float = 0.0
) -> Agent[S1ReviewDeps, S1ReviewOutput]:
    """
    Fresh S1 reviewer agent; same pattern as create_s1_agent but different output_type.
    """
    agent = Agent(
        LLM,
        output_type=S1ReviewOutput,
        system_prompt=system_prompt,
        deps_type=S1ReviewDeps,
        retries=4,
        output_retries=4,
        model_settings=ModelSettings(temperature=temperature),
    )
    return agent


def make_s1_review_prompts(
    *,
    text: str,
    candidate_sets: List[List[Dict[str, Any]]],
) -> Tuple[str, str]:
    """
    Upgraded Reviewer: Acts as a Critic/Refiner over K AoT candidates.
    """
    # Reuse base system to keep definitions aligned
    base_system = build_s1_system(priors=None, conflicts=None, use_cot=True)

    # Inject the Critic-Refiner Rubric
    review_block = """
<critic_refiner_mandate>
  You are an expert Forensic Reviewer (Critic & Refiner).
  You have received K candidate extraction sets from independent analysis runs.
  
  Your Goal: Synthesize a single, high-fidelity JSON output.

  <critique_rubric>
    For every extracted span in the candidates, apply this filter:
    1. **Hallucination Check:** Is the text strictly present in the RAW document? (Reject if not).
    2. **Definition Check:** Does it actually fit the specific Role definition? (e.g., Reject neutral actors).
    3. **Boundary Accuracy:** Is the span precise? 
       - REJECT: "the shadowy figures" (includes unnecessary determiner)
       - ACCEPT: "shadowy figures"
    4. **Consensus Signal:** If multiple candidates found this span, it is likely valid. If only one found it, scrutinize its 'why' rationale deeply.
  </critique_rubric>

  <refinement_strategy>
    - Aggressively MERGE duplicates (prioritize the tightest valid boundary).
    - RESOLVE conflicts (e.g., if one run says 'Action' and another 'Effect', use the Context to decide).
    - PRESERVE the best 'why' rationale from the candidates or synthesize a better one.
  </refinement_strategy>
</critic_refiner_mandate>

<summary_style>
  (Keep the existing summary style instructions here...)
</summary_style>
""".strip()

    system_prompt = base_system + "\n\n" + review_block

    user_prompt = f"""
<text_to_analyze>
{text}
</text_to_analyze>

<candidate_marker_sets>
{json.dumps(candidate_sets, ensure_ascii=False, separators=(",", ":"))}
</candidate_marker_sets>
""".strip()

    return system_prompt, user_prompt


_WORD_RE = re.compile(r"\w")


def _tighten_gentle(
    raw: str, s: int, e: int, want_text: Optional[str] = None
) -> Tuple[int, int]:
    """
    Expand only to full word boundaries if equality (normalized) is preserved
    w.r.t. `want_text` (when provided) or the original RAW slice.
    """
    s0, e0 = s, e

    # expand left
    if s > 0 and _WORD_RE.match(raw[s]) and _WORD_RE.match(raw[s - 1]):
        while s > 0 and _WORD_RE.match(raw[s - 1]):
            s -= 1

    # expand right
    if e < len(raw) and _WORD_RE.match(raw[e - 1]) and _WORD_RE.match(raw[e]):
        while e < len(raw) and _WORD_RE.match(raw[e]):
            e += 1

    target = want_text if want_text is not None else raw[s0:e0]
    new_norm, _ = _normalize_for_match(raw[s:e])
    tgt_norm, _ = _normalize_for_match(target)
    return (s, e) if new_norm == tgt_norm else (s0, e0)


from collections import defaultdict


def _norm_key_text(s: str) -> str:
    # reuse your normalization so keys are stable across smart quotes / whitespace
    nx, _ = _normalize_for_match(s or "")
    return nx


_VICTIM_BARE_RE = re.compile(r"^(child|person|people|man|woman)$", re.IGNORECASE)


async def s1_verifier_impl(ctx: RunContext[S1Deps], output: S1Output) -> S1Output:
    dbg = True

    # authoritative RAW
    RAW = getattr(ctx.deps, "raw_text", "") or ""
    if not RAW:
        logger.warning(
            "[s1_verifier_impl] Warning: deps.raw_text missing; reconstructing from history"
        )
        try:
            msgs = getattr(ctx, "messages", []) or []
            user_chunks = []
            for m in msgs:
                if getattr(m, "role", "") == "user":
                    c = getattr(m, "content", "")
                    if isinstance(c, list):
                        texts = [getattr(p, "text", "") for p in c]
                        user_chunks.append("\n".join(texts))
                    else:
                        user_chunks.append(str(c))
            RAW = _extract_raw_text_from_user_prompt("\n".join(user_chunks))
        except Exception:
            RAW = ""

    spans_in = output.spans or []
    if dbg:
        logger.debug(f"[s1_verifier_impl] Received {len(spans_in)} candidate spans")
        pv = RAW
        logger.debug(f"[loc] len(raw)={len(RAW)} preview='{pv}…'")

    # NEW: track how many times we've already assigned this (label, text)
    assigned_count = defaultdict(
        int
    )  # key: (label, norm_text) -> nth index to use next

    seen, cleaned = set(), []
    action_spans: list[tuple[int, int]] = []  # NEW: track kept Action spans

    for i, m in enumerate(spans_in):
        label = m.label
        snippet = m.text or ""

        # choose nth occurrence deterministically
        key = (label, _norm_key_text(snippet))
        nth = assigned_count[key]

        # 1) locate-by-text (nth occurrence)
        hit = _find_best_span(RAW, snippet, nth=nth) if snippet else None
        if not hit:
            # optional: fallback to first occurrence if nth not found
            hit = _find_best_span(RAW, snippet, nth=0) if snippet else None
            if not hit:
                if dbg:
                    logger.debug(
                        f"[s1_verifier_impl] #{i} drop: locate-by-text failed (label={label}, text='{snippet}')"
                    )
                continue
            # if fallback used, don't bump nth; next one will try nth again

        else:
            # only bump when nth was honored
            assigned_count[key] += 1

        s, e = hit

        # 2) gentle boundary snap (preserve equality)
        ts, te = _tighten_gentle(RAW, s, e, want_text=snippet)
        if (ts, te) != (s, e) and dbg:
            logger.debug(f"[s1_verifier_impl] #{i} tightened -> ({ts},{te})")
        s, e = ts, te

        slice_txt = RAW[s:e]

        # 3) hard-equality safety net (normalized)
        raw_norm, _ = _normalize_for_match(slice_txt)
        snip_norm, _ = _normalize_for_match(snippet)
        if snippet and raw_norm != snip_norm:
            if dbg:
                logger.debug(f"[s1_verifier_impl] #{i} drop: hard equality failed")
            continue

        # (optional) Evidence gate could be applied here if you want;
        # left out because your current snippet doesn't include it.
        # --- NEW: drop embedded bare-noun Victims that sit fully inside an Action span ---
        if label == S1Label.Victim and _VICTIM_BARE_RE.fullmatch(slice_txt.strip()):
            if any(s >= a_s and e <= a_e for (a_s, a_e) in action_spans):
                if dbg:
                    logger.debug(
                        f"[s1_verifier_impl] #{i} drop: embedded bare-noun Victim inside Action"
                    )
                continue

        # de-dup identical offsets
        k2 = (label, s, e)
        if k2 in seen:
            if dbg:
                logger.debug(f"[s1_verifier_impl] #{i} drop: duplicate")
            continue
        seen.add(k2)

        cleaned.append(S1Span(label=label, text=slice_txt, start=s, end=e))
        if dbg:
            logger.debug(
                f"[s1_verifier_impl] #{i} kept [locate_by_text] {label} ({s},{e})='{slice_txt[:80]}'"
            )

        # NEW: remember kept Actions for subsequent Victim checks
        if label == S1Label.Action:
            action_spans.append((s, e))

    # --- NEW (optional, order-robust): prune embedded bare-noun Victims after we know all Actions ---
    if any(sp.label == S1Label.Action for sp in cleaned):
        actions = [(sp.start, sp.end) for sp in cleaned if sp.label == S1Label.Action]
        pruned = []
        for sp in cleaned:
            if sp.label == S1Label.Victim and _VICTIM_BARE_RE.fullmatch(
                sp.text.strip()
            ):
                if any(sp.start >= a_s and sp.end <= a_e for (a_s, a_e) in actions):
                    if dbg:
                        logger.debug(
                            f"[s1_verifier_impl] prune: embedded bare-noun Victim '{sp.text}'"
                        )
                    continue
            pruned.append(sp)
        cleaned = pruned
    # --- END NEW ---

    if dbg:
        logger.debug(f"[s1_verifier_impl] Final kept spans: {len(cleaned)}")
        for j, sp in enumerate(cleaned):
            logger.debug(
                f"[s1_verifier_impl]   #{j} {sp.label} ({sp.start},{sp.end})='{sp.text[:120]}'"
            )

    return S1Output(spans=cleaned)


# @agent_s1.output_validator
# async def s1_verifier(ctx: RunContext[S1Deps], output: S1Output) -> S1Output:
#    dbg = True
#    RAW = (getattr(ctx.deps, "raw_text", None) or "").strip()
#    if not RAW:
#        # fail fast instead of stitching from history (prevents cross-doc leakage)
#        raise ModelRetry(
#            "Internal: deps.raw_text missing; re-run with raw_text in S1Deps."
#        )
#
#    spans_in = output.spans or []
#    if dbg:
#        pv = RAW[:120].replace("\n", " ")
#        print(f"[s1_verifier] Received {len(spans_in)} candidate spans")
#        print(f"[loc] len(raw)={len(RAW)} preview='{pv}…'")
#
#    seen, cleaned = set(), []
#    had_non_evidence = any(m.label != S1Label.Evidence for m in spans_in)
#
#    for i, m in enumerate(spans_in):
#        snippet = (m.text or "").strip()
#        if not snippet:
#            if dbg:
#                print(f"[s1_verifier] #{i} drop: empty text")
#            continue
#
#        # Prefer provided offsets if valid, else locate-by-text (robust finder below)
#        if (
#            m.start is not None
#            and m.end is not None
#            and 0 <= m.start < m.end <= len(RAW)
#        ):
#            s, e = m.start, m.end
#            strategy = "provided_offsets"
#        else:
#            hit = _find_best_span(RAW, snippet)  # normalized locate; maps back to RAW
#            if not hit:
#                if dbg:
#                    print(f"[s1_verifier] #{i} drop: unable to locate in RAW")
#                continue
#            s, e = hit
#            strategy = "locate_by_text"
#        if dbg:
#            print(f"[s1_verifier] #{i} -> {strategy} ({s},{e})")
#
#        # gentle tighten: don’t cut mid-token; only snap if still exact-match compatible
#        s, e = _tighten_to_word(RAW, s, e, want=snippet)
#
#        slice_txt = RAW[s:e]
#
#        # HARD equality safety net (normalized)
#        if not _norm_equal(slice_txt, snippet):
#            if dbg:
#                print(f"[s1_verifier] #{i} drop: hard equality failed")
#            continue
#
#        if m.label == S1Label.Evidence and not _evidence_gate_ok(slice_txt):
#            if dbg:
#                print(f"[s1_verifier] #{i} drop: evidence gate")
#            continue
#
#        key = (m.label, s, e)
#        if key in seen:
#            if dbg:
#                print(f"[s1_verifier] #{i} drop: duplicate")
#            continue
#        seen.add(key)
#
#        cleaned.append(S1Span(label=m.label, start=s, end=e, text=slice_txt))
#        if dbg:
#            print(
#                f"[s1_verifier] #{i} kept [{strategy}] {m.label} ({s},{e})='{slice_txt[:80]}'"
#            )
#
#    # Retry only if the model attempted useful (non-Evidence) spans but none survived
#    if spans_in and not cleaned and had_non_evidence:
#        if dbg:
#            print("[s1_verifier] retry: proposed non-Evidence but none valid")
#        raise ModelRetry(
#            "Re-extract non-Evidence spans as verbatim substrings from RAW."
#        )
#
#    # Optional “strong cue” nudge: FAR less aggressive (off by default)
#    # if not cleaned and _likely_contains_markers(RAW):
#    #    raise ModelRetry("Markers likely present; extract verbatim spans; offsets optional.")
#
#    if dbg:
#        print(f"[s1_verifier] Final kept spans: {len(cleaned)}")
#        for j, sp in enumerate(cleaned):
#            print(
#                f"[s1_verifier]   #{j} {sp.label} ({sp.start},{sp.end})='{sp.text[:120]}'"
#            )
#
#    return S1Output(spans=cleaned)


# Convenience: build and run S1 using the same prompt text as your old runner.
def make_s1_prompts(
    *,
    text: str,
    priors: dict,
    conflicts: list,
    fewshots: list,
    include_cot: bool = True,
    want: int = 8,
    per_example_span_cap: int = 4,
) -> Tuple[str, str]:
    """(system,user) exactly as prompt_builder adapters do. :contentReference[oaicite:3]{index=3}"""
    sys_p = build_s1_system(priors=priors, conflicts=conflicts, use_cot=include_cot)
    usr_p = build_s1_user(
        text_input=text,
        s1_fewshots=fewshots,
        include_cot=include_cot,
        want=want,
        per_example_span_cap=per_example_span_cap,
    )
    return sys_p, usr_p


# --- New AoT Schema ---
class S1AoTOutput(BaseModel):
    """
    Algorithm of Thought output schema.
    Captures the reasoning steps separate from the final extraction.
    """

    strategy: List[str] = Field(
        ..., description="Step-by-step execution trace (Actor scan, Action scan, etc.)"
    )
    final_spans: List[S1Span] = Field(
        ..., description="The final validated list of markers."
    )


# --- AoT Agent Factory ---
def create_s1_aot_agent(
    system_prompt: str, temperature: float = 0.0
) -> Agent[S1Deps, S1AoTOutput]:
    """
    Creates an agent that thinks in steps (AoT) but validates offsets strictly.
    """
    agent = Agent(
        LLM,
        output_type=S1AoTOutput,
        system_prompt=system_prompt,
        deps_type=S1Deps,
        retries=4,
        output_retries=4,
        model_settings=ModelSettings(temperature=temperature),
    )

    # We bind the SAME heavy-duty validator, but adapt the input/output structure
    @agent.output_validator
    async def _bound_aot_validator(
        ctx: RunContext[S1Deps], output: S1AoTOutput
    ) -> S1AoTOutput:
        # 1. Unwrap: Pretend this is a standard S1Output for the validator
        temp_standard = S1Output(spans=output.final_spans)

        # 2. Validate: Run the offset-snapping and text-matching logic
        validated_standard = await s1_verifier_impl(ctx, temp_standard)

        # 3. Rewrap: Put the fixed spans back into the AoT object
        output.final_spans = validated_standard.spans
        return output

    return agent


async def run_s1(
    *,
    doc_id: str,
    text: str,
    priors: dict,
    conflicts: list,
    fewshots: list | None = None,
    rag_collection: Optional[Collection] = None,
    include_cot: bool = True,
    use_aot: bool = True,  # <--- NEW ARGUMENT
    temperature: float = 0.0,
) -> S1Output:

    # [Dynamic Retrieval Logic remains the same...]
    if rag_collection and not fewshots:
        fewshots = retrieve_fewshots(rag_collection, text, k=8)

    if use_aot:
        # --- AoT Branch ---
        # Note: AoT prompt does not use 'fewshots' in the system instructions
        # usually, but you can inject them if your build_s1_system_aot supports it.
        # Assuming build_s1_system_aot handles the core instructions:
        sys_p = build_s1_system_aot(priors=priors, conflicts=conflicts)

        # We reuse build_s1_user for the user message, as the <text_to_analyze> structure is identical
        # However, we disable CoT in the user prompt because AoT handles thinking in the JSON schema
        _, usr_p = make_s1_prompts(
            text=text,
            priors={},
            conflicts=[],
            fewshots=fewshots or [],
            include_cot=False,
        )

        agent = create_s1_aot_agent(sys_p, temperature=temperature)
        deps = S1Deps(raw_text=text, doc_id=doc_id)

        res = await agent.run(usr_p, deps=deps, message_history=[])

        # Flatten result: discard strategy, return standard S1Output for pipeline compatibility
        return S1Output(spans=res.output.final_spans)

    else:
        # --- Standard Branch (Existing Code) ---
        sys_p, usr_p = make_s1_prompts(
            text=text,
            priors=priors,
            conflicts=conflicts,
            fewshots=fewshots or [],
            include_cot=include_cot,
        )
        agent = create_s1_agent(sys_p, temperature=temperature)
        deps = S1Deps(raw_text=text, doc_id=doc_id)
        res = await agent.run(usr_p, deps=deps, message_history=[])
        return res.output


def _merge_s1_self_consistent(outputs: List[S1Output]) -> S1Output:
    """
    Simple self-consistency merge for S1:
      - key = (label, start, end)
      - keep spans that appear in >= majority of runs
      - text taken from the first occurrence (offsets already validated by verifier)
    """
    if not outputs:
        return S1Output(spans=[])

    k = len(outputs)
    # strict majority: for k=5 -> 3
    threshold = (k // 2) + 1

    counts: Counter[tuple] = Counter()
    exemplar: dict[tuple, S1Span] = {}

    for out in outputs:
        for sp in out.spans or []:
            key = (sp.label, sp.start, sp.end)
            counts[key] += 1
            # remember one exemplar
            if key not in exemplar:
                exemplar[key] = sp

    merged_spans: List[S1Span] = []
    for key, c in counts.items():
        if c >= threshold:
            merged_spans.append(exemplar[key])

    # Keep a deterministic ordering: by (label, start, end)
    merged_spans.sort(key=lambda sp: (str(sp.label), sp.start or 0, sp.end or 0))
    return S1Output(spans=merged_spans)


async def run_s1_self_consistent(
    *,
    doc_id: str,
    text: str,
    priors: dict,
    conflicts: list,
    fewshots: list,
    rag_collection: Optional[Collection] = None,  # [NEW]
    include_cot: bool = True,
    k: int = 5,
    temperature: float = 0.7,
    use_aot: bool = True,
) -> S1ReviewOutput:
    """
    Self-consistency for S1 with a reviewer agent:

      1. Run the base S1 extractor k times at non-zero temperature.
      2. Collect the k outputs as candidate_sets (lists of {label,text,start,end}).
      3. Call a reviewer agent that:
          - looks at RAW text + all candidates,
          - outputs final spans and a per-label marker summary.

    Returns:
      S1ReviewOutput(spans=[...], summary=[MarkerSummary(...), ...])
    """
    if k < 1:
        k = 1

    # --- Step 1: run base S1 k times (diversity from temperature) ---
    logger.info(
        f"[{doc_id}] S1-SC: Starting {k} Proposer runs (temp={temperature}, AoT={use_aot})..."
    )

    # --- Wrapper for a single concurrent run ---
    async def _single_s1_run(index: int) -> Optional[S1Output]:
        sub_id = f"{doc_id}::run{index+1}"
        async with _SC_SEMAPHORE:  # Rate limit protection
            try:
                logger.debug(f"[{doc_id}] Proposer {index+1}/{k} running...")
                return await run_s1(
                    doc_id=sub_id,
                    text=text,
                    priors=priors,
                    conflicts=conflicts,
                    fewshots=fewshots,
                    rag_collection=rag_collection,
                    include_cot=include_cot,
                    use_aot=use_aot,
                    temperature=temperature,
                )
            except Exception as e:
                logger.warning(f"[{doc_id}] ⚠️ Proposer {index+1} failed: {e}")
                return None

    # --- Step 1: Concurrent Execution ---
    # asyncio.gather runs all tasks in parallel
    results = await asyncio.gather(*[_single_s1_run(i) for i in range(k)])

    # Filter out failures
    runs = [r for r in results if r is not None]

    if not runs:
        logger.error(f"[{doc_id}] S1-SC: All Proposer runs failed.")
        return S1ReviewOutput(spans=[], summary=[])

    # --- Step 2: build candidate_sets for the reviewer ---
    candidate_sets: List[List[Dict[str, Any]]] = []
    total_spans = 0
    for out in runs:
        span_dicts: List[Dict[str, Any]] = []
        for sp in out.spans or []:
            span_dicts.append(
                {
                    "label": (
                        sp.label.value if hasattr(sp.label, "value") else str(sp.label)
                    ),
                    "text": sp.text,
                    "start": sp.start,
                    "end": sp.end,
                    "why": getattr(sp, "why", None),  # Include why for reviewer
                }
            )
        candidate_sets.append(span_dicts)
        total_spans += len(span_dicts)

    logger.info(
        f"[{doc_id}] Reviewer: Critiquing {len(runs)} sets containing {total_spans} raw spans..."
    )

    # --- Step 3: call reviewer agent to merge + summarize ---
    sys_p, usr_p = make_s1_review_prompts(text=text, candidate_sets=candidate_sets)
    reviewer = create_s1_reviewer_agent(system_prompt=sys_p, temperature=0.0)

    deps = S1ReviewDeps(
        raw_text=text,
        doc_id=doc_id,
        candidate_sets=candidate_sets,
    )

    res = await reviewer.run(usr_p, deps=deps, message_history=[])

    final_count = len(res.output.spans)
    logger.success(
        f"[{doc_id}] Reviewer: Consensus reached. Refined {total_spans} raw spans -> {final_count} final markers."
    )
    return res.output  # S1ReviewOutput


# ===========================================================================
# S2 — structured output types + agent
# ===========================================================================


class S2Deps(BaseModel):
    model_config = ConfigDict(extra="ignore")
    raw_text: str
    s1_markers: List[Dict[str, Any]] = []
    marker_summary: Optional[Dict[str, Any]] = None
    doc_id: Optional[str] = None


class S2Output(BaseModel):
    label: str = Field(..., description='One of: "conspiracy", "non""')
    rationale: str = Field(..., description="2 concise sentences naming decisive cues.")

    @field_validator("label")
    @classmethod
    def _label_ok(cls, v: str) -> str:
        v2 = (v or "").strip().lower()
        if v2 not in {"conspiracy", "non"}:
            raise ValueError("label must be conspiracy or non")
        return v2


agent_s2 = Agent(LLM, output_type=S2Output, system_prompt="(placeholder)", retries=4)


def make_s2_prompts(
    *,
    text: str,
    s1_spans: List[dict],
    fewshots: Optional[List[dict]],
    include_cot: bool = True,
    allow_cant_tell: bool = False,
    marker_summary: Optional[Dict[str, List[str]]] = None,
) -> Tuple[str, str]:
    """(system,user) identical to your adapter. :contentReference[oaicite:4]{index=4}"""
    sys_p = build_s2_system(include_cot=include_cot)
    usr_p = build_s2_user(
        text_input=text,
        s1_output=s1_spans,
        s2_fewshots=fewshots or [],
        include_cot=include_cot,
        marker_summary=marker_summary,
    )
    return sys_p, usr_p


def create_s2_agent(
    system_prompt: str, temperature: float = 0.0
) -> Agent[S2Deps, S2Output]:
    agent = Agent(
        LLM,
        output_type=S2Output,
        system_prompt=system_prompt,
        retries=3,
        output_retries=3,
        model_settings=ModelSettings(
            temperature=temperature
        ),  # optional, reduces paraphrase
    )

    return agent


async def run_s2(
    *,
    doc_id: str,
    text: str,
    s1_output_spans: List[dict],  # [{"type","startIndex","endIndex","text"}] preferred
    fewshots: Optional[List[dict]] = None,
    rag_collection: Optional[Collection] = None,  # [NEW]
    include_cot: bool = True,
    allow_cant_tell: bool = False,
    temperature: float = 0.0,
    marker_summary: Optional[Dict[str, Any]] = None,
) -> S2Output:
    # [NEW] Dynamic Retrieval Logic
    if rag_collection and not fewshots:
        # S2 benefits from fewer, highly specific examples (k=5 is often better than 8)
        fewshots = retrieve_fewshots(rag_collection, text, k=8)
    # Ensure S1 spans are in S2 schema (startIndex/endIndex/text)
    s1_norm = [
        to_s2_marker(m, text) for m in (s1_output_spans or [])
    ]  # :contentReference[oaicite:5]{index=5}

    # --- Logging Inputs for S2 ---
    # We log at DEBUG level for the full markers list to avoid clutter,
    # but INFO for the summary since it's the "Narrative".
    logger.debug(
        f"[{doc_id}] S2 Input Markers: {json.dumps(s1_norm, ensure_ascii=False)}"
    )
    if marker_summary:
        logger.info(
            f"[{doc_id}] S2 Input Summary: {json.dumps(marker_summary, ensure_ascii=False)}"
        )

    # ----------------------------------
    sys_p, usr_p = make_s2_prompts(
        text=text,
        s1_spans=s1_norm,
        fewshots=fewshots,
        include_cot=include_cot,
        allow_cant_tell=allow_cant_tell,
        marker_summary=marker_summary,
    )
    # Fresh agent per document (no .clone() needed)
    agent = create_s2_agent(sys_p, temperature=temperature)

    # Provide deps so the validator has authoritative inputs
    deps = S2Deps(
        raw_text=text, s1_markers=s1_norm, doc_id=doc_id, marker_summary=marker_summary
    )

    # Always run with a clean history for deterministic behavior
    res = await agent.run(usr_p, deps=deps, message_history=[])
    # If 'cant_tell' was disallowed in prompting, the model shouldn't return it;
    # the validator still accepts it, but your caller can downweight/repair if needed.
    return res.output


async def run_s2_self_consistent(
    *,
    doc_id: str,
    text: str,
    s1_output_spans: List[dict],
    fewshots: Optional[List[dict]] = None,
    rag_collection: Optional[Collection] = None,  # [NEW]
    include_cot: bool = True,
    allow_cant_tell: bool = False,
    k: int = 5,
    temperature: float = 0.7,
    marker_summary: Optional[Dict[str, Any]] = None,
) -> S2Output:
    """
    Self-consistency for S2:
      - Run S2 k times at non-zero temperature.
      - Majority vote over labels.
      - Rationale taken from the first run with the winning label.
    """
    if k <= 1:
        return await run_s2(
            doc_id=doc_id,
            text=text,
            s1_output_spans=s1_output_spans,
            fewshots=fewshots,
            rag_collection=rag_collection,  # Pass RAG collection
            include_cot=include_cot,
            allow_cant_tell=allow_cant_tell,
            temperature=temperature,
            marker_summary=marker_summary,
        )

    logger.info(f"[{doc_id}] S2-SC: Casting {k} votes (temp={temperature})...")

    # --- Wrapper for a single concurrent run ---
    async def _single_s2_run(index: int) -> Optional[S2Output]:
        async with _SC_SEMAPHORE:  # Rate limit protection
            try:
                return await run_s2(
                    doc_id=f"{doc_id}::sc{index+1}",
                    text=text,
                    s1_output_spans=s1_output_spans,
                    fewshots=fewshots,
                    rag_collection=rag_collection,
                    include_cot=include_cot,
                    allow_cant_tell=allow_cant_tell,
                    temperature=temperature,
                    marker_summary=marker_summary,
                )
            except Exception as e:
                logger.warning(f"[{doc_id}] S2 Run {index+1} failed: {e}")
                return None

    # --- Concurrent Execution ---
    results = await asyncio.gather(*[_single_s2_run(i) for i in range(k)])
    runs = [r for r in results if r is not None]

    if not runs:
        logger.error(f"[{doc_id}] S2-SC: All runs failed. Defaulting to 'non'.")
        return S2Output(label="non", rationale="All SC runs failed.")

    # TODO: Enhance voting logic if needed with multi Agent debate
    labels = [r.label for r in runs]
    label_counts = Counter(labels)
    logger.info(f"[{doc_id}] S2-SC Results: {dict(label_counts)}")
    top_label, top_count = label_counts.most_common(1)[0]

    # Tie-break: if counts tie (e.g., 2 vs 2), bias conservatively toward "non"
    # (or just keep first majority if you prefer).
    # Here we implement "non" bias for exact ties:
    max_count = max(label_counts.values())
    candidates = [lab for lab, c in label_counts.items() if c == max_count]
    if len(candidates) > 1:
        logger.info(f"[{doc_id}] Tie detected {candidates}. Bias -> non.")
        if "non" in candidates:
            top_label = "non"
        else:
            # deterministic: pick lexicographically
            top_label = sorted(candidates)[0]

    logger.success(f"[{doc_id}] S2 Final: {top_label}")

    # Rationale: take first run that produced the chosen label
    chosen_rationale = next(
        (r.rationale for r in runs if r.label == top_label),
        runs[0].rationale,
    )

    return S2Output(label=top_label, rationale=chosen_rationale)
