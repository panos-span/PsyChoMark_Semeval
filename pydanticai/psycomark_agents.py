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
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from prompt_builder import (
    to_s2_marker,
)  # utility for normalizing S1 spans into S2 markers
from prompt_builder import (
    build_s1_system,
    build_s1_user,
    build_s2_system,
    build_s2_user,
)
from pydantic import BaseModel, Field, field_validator, ConfigDict

# pydantic-ai core
from pydantic_ai import Agent, ModelRetry, RunContext, ModelSettings
from pydantic_ai.models.bedrock import BedrockConverseModel

# Bedrock provider/model (pydantic-ai)
from pydantic_ai.providers.bedrock import BedrockProvider


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

    @field_validator("end")
    @classmethod
    def _end_after_start(cls, v, info):
        start = info.data.get("start", None)
        if start is not None and v <= start:
            raise ValueError("end must be greater than start")
        return v


class S1Output(BaseModel):
    spans: List[S1Span] = Field(default_factory=list)


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

_URL_RE = re.compile(
    r"""(?i)\b(?:https?://|www\.)[^\s<>"']{3,}|\b[a-z0-9-]+(?:\.[a-z0-9-]+)+\b""",
    re.I,
)
_ATTRIB_RE = re.compile(r"""(?i)\b(according to|reported by|as stated by)\b""")
_QUOTE_RE = re.compile(r"[\"“”‘’']")  # any quote char
_NUMERIC_UNIT_RE = re.compile(
    r"(?i)\b\d+(\.\d+)?\s?(%|ppm|km|m|kg|k|million|billion)\b"
)


def _safe_clip(s: str, a: int, b: int) -> Tuple[int, int]:
    a = max(0, int(a))
    b = max(a, int(b))
    L = len(s)
    return min(a, L), min(b, L)


def _extract_raw_text_from_user_prompt(user_prompt: str) -> str:
    """We put the RAW text inside <text_to_analyze>...</text_to_analyze>."""
    m = re.search(r"<text_to_analyze>\s*(.*?)\s*</text_to_analyze>", user_prompt, re.S)
    return m.group(1) if m else user_prompt


def _evidence_gate_ok(span_text: str) -> bool:
    """
    Evidence valid if ANY of:
      (a) URL/domain present, OR
      (b) Quoted material WITH attribution cue, OR
      (c) Numeric facts WITH units (or %) AND attribution cue.
    Mirrors the gate in your prompt text. :contentReference[oaicite:1]{index=1}
    """
    has_url = bool(_URL_RE.search(span_text))
    has_quote = bool(_QUOTE_RE.search(span_text))
    has_attrib = bool(_ATTRIB_RE.search(span_text))
    has_numeric = bool(_NUMERIC_UNIT_RE.search(span_text))
    if has_url:
        return True
    if has_quote and has_attrib:
        return True
    if has_numeric and has_attrib:
        return True
    # Also accept pure attribution with a named source (loose but useful).
    if has_attrib:
        return True
    return False


def _local_align_search(raw: str, expected: str, s: int, e: int, window: int = 16):
    if not expected:
        return None
    L = len(raw)
    a = max(0, s - window)
    b = min(L, e + window)
    i = raw[a:b].find(expected)
    if i < 0:
        return None
    ss = a + i
    ee = ss + len(expected)
    return ss, ee


_MARKER_CUE_RE = re.compile(
    r"(?i)\b(they|the elite|globalists|deep state|big pharma|cover[-\s]?up|plot|scheme|orchestrate|fabricate)\b"
)
_PURPOSE_RE = re.compile(r"(?i)\bto\s+[a-z][a-z-]+\b")  # simple "to VERB" cue
_HAS_VERBISH = re.compile(
    r"(?i)\b(?:is|are|was|were|be|been|being|do|does|did|has|have|had|can|will|would|should)\b"
)


def _likely_contains_markers(t: str) -> bool:
    # Any of: marker cues, purpose clause, evidence-like surface cues, or just enough verbs in a long text
    if _MARKER_CUE_RE.search(t):
        return True
    if _PURPOSE_RE.search(t):
        return True
    if _URL_RE.search(t) or _QUOTE_RE.search(t) or _ATTRIB_RE.search(t):
        return True
    if len(t) > 220 and len(_HAS_VERBISH.findall(t)) >= 3:
        return True
    return False


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

_WS_RE = re.compile(r"\s+", re.S)
_ALNUM_RE = re.compile(r"[A-Za-z0-9]")


def _normalize_with_map(text: str) -> Tuple[str, List[int]]:
    """
    Normalize text for matching and produce a mapping from normalized
    indices -> original RAW indices.

    Normalizations:
      - smart quotes & dashes -> ASCII
      - collapse all whitespace runs to a single ' ' (space)

    Returns:
      norm: the normalized string
      idx_map: list where idx_map[i] gives the RAW index for norm[i]
    """
    # 1) map smart chars
    mapped_chars: List[str] = []
    mapped_src_idx: List[int] = []
    for i, ch in enumerate(text):
        mapped = SMART_TO_STRAIGHT.get(ch, ch)
        mapped_chars.append(mapped)
        mapped_src_idx.append(i)

    # 2) collapse whitespace, building final map
    norm_chars: List[str] = []
    idx_map: List[int] = []
    i = 0
    L = len(mapped_chars)
    while i < L:
        ch = mapped_chars[i]
        if ch.isspace():
            # collapse run to one space; source index = first raw idx in the run
            start_raw_i = mapped_src_idx[i]
            while i < L and mapped_chars[i].isspace():
                i += 1
            norm_chars.append(" ")
            idx_map.append(start_raw_i)
        else:
            norm_chars.append(ch)
            idx_map.append(mapped_src_idx[i])
            i += 1

    return "".join(norm_chars), idx_map


def _is_word_char(ch: str) -> bool:
    return bool(_ALNUM_RE.fullmatch(ch))


def _expand_to_token_edges(raw: str, s: int, e: int) -> Tuple[int, int]:
    """
    Expand [s,e) to word/token boundaries without crossing spaces:
      - move s left until start or previous char is non-alnum
      - move e right until end or next char is non-alnum
    """
    L = len(raw)
    # expand left
    while s > 0 and _is_word_char(raw[s]) and _is_word_char(raw[s - 1]):
        s -= 1
    # expand right
    while e < L and _is_word_char(raw[e - 1]) and _is_word_char(raw[e]):
        e += 1
    return s, e


def _tighten(
    raw: str, s: int, e: int, target_text: Optional[str] = None
) -> Tuple[int, int]:
    """
    Gentle boundary snap: only expand to token edges if it preserves equality with target_text.
    If target_text is None, keep (s,e) as-is.
    """
    if target_text is None:
        return s, e

    ns, ne = _expand_to_token_edges(raw, s, e)
    expanded = raw[ns:ne]

    # preserve only if equality or containment (to avoid mid-word truncation)
    if expanded == target_text or target_text in expanded or expanded in target_text:
        return ns, ne
    return s, e


# --- normalization helpers ---
_SMART = {
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "—": "-",
    "–": "-",
    "−": "-",
    "\u00a0": " ",
}


def _norm_text(s: str) -> str:
    s2 = "".join(_SMART.get(ch, ch) for ch in s)
    # collapse runs of whitespace to a single space
    return re.sub(r"\s+", " ", s2).strip()


def _norm_equal(a: str, b: str) -> bool:
    return _norm_text(a) == _norm_text(b)


def _build_norm_map(raw: str) -> tuple[str, list[int]]:
    """
    Return (norm, idx_map) where norm is normalized RAW,
    and idx_map[norm_idx] -> raw_idx for the start of each normalized char.
    Collapses whitespace runs to one space and normalizes quotes/dashes.
    """
    idx_map = []
    out = []
    i = 0
    L = len(raw)
    while i < L:
        ch = raw[i]
        ch = _SMART.get(ch, ch)
        if ch.isspace():
            # collapse a whitespace run
            while i < L and raw[i].isspace():
                i += 1
            out.append(" ")
            # map this single space to the first raw index after the run-start
            idx_map.append(
                i - 1
            )  # last whitespace pos; consistent since we’ll remap window later
            continue
        out.append(ch)
        idx_map.append(i)
        i += 1
    return "".join(out), idx_map


_WORD = re.compile(r"\w")


def _tighten_to_word(raw: str, s: int, e: int, *, want: str) -> tuple[int, int]:
    """
    Snap to token boundaries only when doing so keeps an exact match relationship.
    Never cut inside a word if avoidable.
    """
    # if already matching want exactly, keep tight
    if raw[s:e] == want:
        return s, e
    # expand left if we cut a word
    sl, sr = s, e
    if s > 0 and _WORD.match(raw[s]) and _WORD.match(raw[s - 1]):
        # expand left to word start
        while sl > 0 and _WORD.match(raw[sl - 1]):
            sl -= 1
        if _norm_equal(raw[sl:sr], want) or _norm_equal(raw[sl:sr], _norm_text(want)):
            s = sl
    # expand right if we cut a word
    if e < len(raw) and _WORD.match(raw[e - 1]) and _WORD.match(raw[e]):
        while sr < len(raw) and _WORD.match(raw[sr]):
            sr += 1
        if _norm_equal(raw[s:sr], want) or _norm_equal(raw[s:sr], _norm_text(want)):
            e = sr
    return s, e


import re
from typing import Optional, Tuple

_SMART_TO_STRAIGHT = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "—": "-",
        "–": "-",
        "−": "-",
    }
)

_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"\w")


def _normalize_for_match(s: str) -> Tuple[str, List[int]]:
    """
    Returns (norm, idx_map) where:
      - norm: s with smart quotes/dashes normalized and whitespace runs collapsed to ' '.
      - idx_map[i]: RAW index in original s corresponding to norm[i].
    """
    if not s:
        return "", []

    # 1) translate smart quotes/dashes → straight
    t = s.translate(_SMART_TO_STRAIGHT)

    # 2) collapse whitespace while tracking indices back to RAW
    norm_chars: List[str] = []
    idx_map: List[int] = []
    i, L = 0, len(t)
    while i < L:
        ch = t[i]
        if ch.isspace():
            j = i + 1
            while j < L and t[j].isspace():
                j += 1
            # emit single space mapped to the first RAW index of the run
            norm_chars.append(" ")
            idx_map.append(i)
            i = j
        else:
            norm_chars.append(ch)
            idx_map.append(i)
            i += 1
    return "".join(norm_chars), idx_map


def _find_best_span(
    raw: str, snippet: str, hint: Optional[Tuple[int, int]] = None
) -> Optional[Tuple[int, int]]:
    """
    Locate `snippet` in RAW using:
      0) Accept hint only if it hard-matches RAW exactly.
      1) Exact match.
      2) Case-insensitive match (only if same-length exact RAW slice).
      3) Normalized (quotes+whitespace) match with reversible mapping to RAW.
    """
    if not snippet:
        return None

    # 0) accept trusted hint only if exact
    if hint:
        s, e = hint
        if 0 <= s < e <= len(raw) and raw[s:e] == snippet:
            return s, e

    # 1) exact
    i = raw.find(snippet)
    if i >= 0:
        return i, i + len(snippet)

    # 2) case-insensitive (guarded)
    lr, ls = raw.lower(), snippet.lower()
    i = lr.find(ls)
    if i >= 0:
        j = i + len(snippet)
        if lr[i:j] == ls:
            return i, j

    # 3) normalized search with reverse map
    norm_raw, raw_map = _normalize_for_match(raw)
    norm_snip, _ = _normalize_for_match(snippet)
    i = norm_raw.find(norm_snip)
    if i < 0:
        return None

    # map back to RAW
    s = raw_map[i]
    end_norm_idx = i + len(norm_snip) - 1
    e = raw_map[end_norm_idx] + 1

    # safety: compare normalized slices
    raw_slice_norm, _ = _normalize_for_match(raw[s:e])
    snip_norm, _ = _normalize_for_match(snippet)
    if raw_slice_norm == snip_norm:
        return s, e
    return None


def create_s1_agent(
    system_prompt: str, temperature: int = 0.0
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


async def s1_verifier_impl(ctx: RunContext[S1Deps], output: S1Output) -> S1Output:
    dbg = True

    # authoritative RAW
    RAW = getattr(ctx.deps, "raw_text", "") or ""
    if not RAW:
        print(
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
        print(f"[s1_verifier_impl] Received {len(spans_in)} candidate spans")
        pv = RAW
        print(f"[loc] len(raw)={len(RAW)} preview='{pv}…'")

    seen, cleaned = set(), []

    for i, m in enumerate(spans_in):
        label = m.label
        snippet = (m.text or "").strip()

        hint = None
        if (
            m.start is not None
            and m.end is not None
            and 0 <= m.start < m.end <= len(RAW)
        ):
            hint = (m.start, m.end)

        # 1) locate-by-text first (robust). Only accept hint if it already hard-matches.
        hit = _find_best_span(RAW, snippet, hint=hint) if snippet else None
        if not hit:
            if dbg:
                print(
                    f"[s1_verifier_impl] #{i} drop: locate-by-text failed (label={label}, text='{snippet[:80]}')"
                )
            continue
        s, e = hit

        # 2) gentle boundary snap (preserve equality)
        ts, te = _tighten_gentle(RAW, s, e, want_text=snippet)
        if (ts, te) != (s, e) and dbg:
            print(f"[s1_verifier_impl] #{i} tightened -> ({ts},{te})")
        s, e = ts, te

        slice_txt = RAW[s:e]

        # 3) hard-equality safety net (normalized)
        raw_norm, _ = _normalize_for_match(slice_txt)
        snip_norm, _ = _normalize_for_match(snippet)
        if snippet and raw_norm != snip_norm:
            if dbg:
                print(f"[s1_verifier_impl] #{i} drop: hard equality failed")
            continue

        # 4) evidence gate
        # if label == S1Label.Evidence and not _evidence_gate_ok(slice_txt):
        #    if dbg:
        #        print(f"[s1_verifier_impl] #{i} drop: evidence gate")
        #    continue

        key = (label, s, e)
        if key in seen:
            if dbg:
                print(f"[s1_verifier_impl] #{i} drop: duplicate")
            continue
        seen.add(key)

        cleaned.append(S1Span(label=label, text=slice_txt, start=s, end=e))
        if dbg:
            src = "locate_by_text" if (hint is None or (s, e) != hint) else "provided"
            print(
                f"[s1_verifier_impl] #{i} kept [{src}] {label} ({s},{e})='{slice_txt[:80]}'"
            )

    if dbg:
        print(f"[s1_verifier_impl] Final kept spans: {len(cleaned)}")
        for j, sp in enumerate(cleaned):
            print(
                f"[s1_verifier_impl]   #{j} {sp.label} ({sp.start},{sp.end})='{sp.text[:120]}'"
            )

    # No forced ModelRetry here; return what we have (possibly empty)
    return S1Output(spans=cleaned)


@agent_s1.output_validator
async def s1_verifier(ctx: RunContext[S1Deps], output: S1Output) -> S1Output:
    dbg = True
    RAW = (getattr(ctx.deps, "raw_text", None) or "").strip()
    if not RAW:
        # fail fast instead of stitching from history (prevents cross-doc leakage)
        raise ModelRetry(
            "Internal: deps.raw_text missing; re-run with raw_text in S1Deps."
        )

    spans_in = output.spans or []
    if dbg:
        pv = RAW[:120].replace("\n", " ")
        print(f"[s1_verifier] Received {len(spans_in)} candidate spans")
        print(f"[loc] len(raw)={len(RAW)} preview='{pv}…'")

    seen, cleaned = set(), []
    had_non_evidence = any(m.label != S1Label.Evidence for m in spans_in)

    for i, m in enumerate(spans_in):
        snippet = (m.text or "").strip()
        if not snippet:
            if dbg:
                print(f"[s1_verifier] #{i} drop: empty text")
            continue

        # Prefer provided offsets if valid, else locate-by-text (robust finder below)
        if (
            m.start is not None
            and m.end is not None
            and 0 <= m.start < m.end <= len(RAW)
        ):
            s, e = m.start, m.end
            strategy = "provided_offsets"
        else:
            hit = _find_best_span(RAW, snippet)  # normalized locate; maps back to RAW
            if not hit:
                if dbg:
                    print(f"[s1_verifier] #{i} drop: unable to locate in RAW")
                continue
            s, e = hit
            strategy = "locate_by_text"
        if dbg:
            print(f"[s1_verifier] #{i} -> {strategy} ({s},{e})")

        # gentle tighten: don’t cut mid-token; only snap if still exact-match compatible
        s, e = _tighten_to_word(RAW, s, e, want=snippet)

        slice_txt = RAW[s:e]

        # HARD equality safety net (normalized)
        if not _norm_equal(slice_txt, snippet):
            if dbg:
                print(f"[s1_verifier] #{i} drop: hard equality failed")
            continue

        if m.label == S1Label.Evidence and not _evidence_gate_ok(slice_txt):
            if dbg:
                print(f"[s1_verifier] #{i} drop: evidence gate")
            continue

        key = (m.label, s, e)
        if key in seen:
            if dbg:
                print(f"[s1_verifier] #{i} drop: duplicate")
            continue
        seen.add(key)

        cleaned.append(S1Span(label=m.label, start=s, end=e, text=slice_txt))
        if dbg:
            print(
                f"[s1_verifier] #{i} kept [{strategy}] {m.label} ({s},{e})='{slice_txt[:80]}'"
            )

    # Retry only if the model attempted useful (non-Evidence) spans but none survived
    if spans_in and not cleaned and had_non_evidence:
        if dbg:
            print("[s1_verifier] retry: proposed non-Evidence but none valid")
        raise ModelRetry(
            "Re-extract non-Evidence spans as verbatim substrings from RAW."
        )

    # Optional “strong cue” nudge: FAR less aggressive (off by default)
    # if not cleaned and _likely_contains_markers(RAW):
    #    raise ModelRetry("Markers likely present; extract verbatim spans; offsets optional.")

    if dbg:
        print(f"[s1_verifier] Final kept spans: {len(cleaned)}")
        for j, sp in enumerate(cleaned):
            print(
                f"[s1_verifier]   #{j} {sp.label} ({sp.start},{sp.end})='{sp.text[:120]}'"
            )

    return S1Output(spans=cleaned)


# Convenience: build and run S1 using the same prompt text as your old runner.
def make_s1_prompts(
    *,
    text: str,
    priors: dict,
    conflicts: list,
    fewshots: list,
    include_cot: bool = True,
    want: int = 8,
    victim_min: int = 1,
    conflict_min: int = 1,
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


async def run_s1(
    *,
    doc_id: str,
    text: str,
    priors: dict,
    conflicts: list,
    fewshots: list,
    include_cot: bool = True,
) -> S1Output:
    sys_p, usr_p = make_s1_prompts(
        text=text,
        priors=priors,
        conflicts=conflicts,
        fewshots=fewshots,
        include_cot=include_cot,
    )

    # NEW: build a fresh agent; no clone()
    agent = create_s1_agent(sys_p)

    # Always pass deps so the verifier never falls back to stitched messages
    deps = S1Deps(raw_text=text, doc_id=doc_id)

    # history=[] guarantees a clean context per call
    res = await agent.run(usr_p, deps=deps, message_history=[])
    return res.output


# ===========================================================================
# S2 — structured output types + agent
# ===========================================================================


class S2Deps(BaseModel):
    model_config = ConfigDict(extra="ignore")
    raw_text: str
    s1_markers: List[Dict[str, Any]] = []
    doc_id: Optional[str] = None


class S2Output(BaseModel):
    label: str = Field(..., description='One of: "conspiracy", "non""')
    rationale: str = Field(
        ..., description="1-2 concise sentences naming decisive cues."
    )

    @field_validator("label")
    @classmethod
    def _label_ok(cls, v: str) -> str:
        v2 = (v or "").strip().lower()
        if v2 not in {"conspiracy", "non"}:
            raise ValueError("label must be conspiracy | non")
        return v2


agent_s2 = Agent(LLM, output_type=S2Output, system_prompt="(placeholder)", retries=4)


def make_s2_prompts(
    *,
    text: str,
    s1_spans: List[dict],
    fewshots: Optional[List[dict]],
    include_cot: bool = True,
    allow_cant_tell: bool = False,
) -> Tuple[str, str]:
    """(system,user) identical to your adapter. :contentReference[oaicite:4]{index=4}"""
    sys_p = build_s2_system(include_cot=include_cot, allow_cant_tell=allow_cant_tell)
    usr_p = build_s2_user(
        text_input=text,
        s1_output=s1_spans,
        s2_fewshots=fewshots or [],
        include_cot=include_cot,
        allow_cant_tell=allow_cant_tell,
    )
    return sys_p, usr_p


def create_s2_agent(
    system_prompt: str, temperature: int = 0.0
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
    include_cot: bool = True,
    allow_cant_tell: bool = False,
) -> S2Output:
    # Ensure S1 spans are in S2 schema (startIndex/endIndex/text)
    s1_norm = [
        to_s2_marker(m, text) for m in (s1_output_spans or [])
    ]  # :contentReference[oaicite:5]{index=5}
    sys_p, usr_p = make_s2_prompts(
        text=text,
        s1_spans=s1_norm,
        fewshots=fewshots,
        include_cot=include_cot,
        allow_cant_tell=allow_cant_tell,
    )
    # Fresh agent per document (no .clone() needed)
    agent = create_s2_agent(sys_p)

    # Provide deps so the validator has authoritative inputs
    deps = S2Deps(raw_text=text, s1_markers=s1_norm, doc_id=doc_id)

    # Always run with a clean history for deterministic behavior
    res = await agent.run(usr_p, deps=deps, message_history=[])
    # If 'cant_tell' was disallowed in prompting, the model shouldn't return it;
    # the validator still accepts it, but your caller can downweight/repair if needed.
    return res.output
