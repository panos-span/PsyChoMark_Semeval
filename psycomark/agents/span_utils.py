"""
psycomark.agents.span_utils — Deterministic Span Verification & Post-Processing.

Implements the non-LLM Verifier stage of the S1 pipeline:
    1. Exact substring match
    2. Case-insensitive match
    3. Normalized match (smart quotes, whitespace collapsing)
    4. Fuzzy match via Levenshtein distance
    5. SequenceMatcher alignment (last resort)

Also provides deduplication, merging, and boundary verification utilities.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from loguru import logger

try:
    from fuzzysearch import find_near_matches
except ImportError:
    find_near_matches = None


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

_SMART_QUOTES = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",  # single curly
        "\u201c": '"',
        "\u201d": '"',  # double curly
        "\u2013": "-",
        "\u2014": "-",  # en/em dash
    }
)


def _normalize_for_match(text: str) -> Tuple[str, List[int]]:
    """
    Normalise *text* for fuzzy matching.

    Returns ``(normalised_string, index_map)`` where ``index_map[i]``
    gives the original-text index for normalised position *i*.
    """
    norm_chars: list[str] = []
    idx_map: list[int] = []
    prev_space = False
    for i, ch in enumerate(text):
        ch = ch.translate(_SMART_QUOTES)
        if ch.isspace():
            if not prev_space:
                norm_chars.append(" ")
                idx_map.append(i)
            prev_space = True
        else:
            norm_chars.append(ch)
            idx_map.append(i)
            prev_space = False
    return "".join(norm_chars), idx_map


# ---------------------------------------------------------------------------
# Core Span Locator
# ---------------------------------------------------------------------------


def find_best_span(raw_text: str, snippet: str, nth: int = 0) -> Tuple[int, int]:
    """
    Robustly locate the *nth* occurrence of *snippet* in *raw_text*.

    Strategies (tried in order):
        1. Exact substring match
        2. Case-insensitive match
        3. Normalised match (quotes, whitespace)
        4. Levenshtein fuzzy match (~15 % edit distance)
        5. ``SequenceMatcher`` LCS alignment

    Returns ``(start, end)`` or ``(-1, -1)`` on failure.
    """
    if not snippet or not raw_text:
        return -1, -1

    # Strategy 1: Exact match
    start = -1
    for _ in range(nth + 1):
        start = raw_text.find(snippet, start + 1)
        if start == -1:
            break
    if start != -1:
        return start, start + len(snippet)

    # Strategy 2: Case-insensitive
    raw_lower = raw_text.lower()
    snip_lower = snippet.lower()
    start = -1
    for _ in range(nth + 1):
        start = raw_lower.find(snip_lower, start + 1)
        if start == -1:
            break
    if start != -1:
        end = start
        snippet_idx = 0
        while snippet_idx < len(snippet) and end < len(raw_text):
            if raw_text[end].lower() == snippet[snippet_idx].lower():
                snippet_idx += 1
            end += 1
        if snippet_idx < len(snippet):
            end = start + len(snippet)
        return start, end

    # Strategy 3: Normalised match
    raw_norm, raw_map = _normalize_for_match(raw_text)
    snip_norm, _ = _normalize_for_match(snippet)

    if snip_norm in raw_norm:
        start_norm = -1
        for _ in range(nth + 1):
            start_norm = raw_norm.find(snip_norm, start_norm + 1)
            if start_norm == -1:
                break
        if start_norm != -1:
            end_norm_idx = start_norm + len(snip_norm) - 1
            if end_norm_idx < len(raw_map):
                orig_start = raw_map[start_norm]
                if end_norm_idx + 1 < len(raw_map):
                    orig_end = raw_map[end_norm_idx + 1]
                else:
                    orig_end = raw_map[end_norm_idx] + 1
                return orig_start, orig_end

    # Strategy 4: Fuzzy match (Levenshtein)
    if find_near_matches and len(snippet) >= 3:
        max_dist = max(1, int(len(snippet) * 0.15)) if len(snippet) > 4 else 0
        if max_dist > 0:
            matches = find_near_matches(snippet, raw_text, max_l_dist=max_dist)
            if len(matches) > nth:
                m = matches[nth]
                return m.start, m.end

    # Strategy 5: SequenceMatcher alignment
    if len(snippet) >= 5:
        matcher = SequenceMatcher(
            None, raw_text.lower(), snippet.lower(), autojunk=False
        )
        blocks = [b for b in matcher.get_matching_blocks() if b.size > 2]
        if blocks:
            start_cand = blocks[0].a
            end_cand = blocks[-1].a + blocks[-1].size
            matched_len = sum(b.size for b in blocks)
            coverage = matched_len / len(snippet)
            is_compact = (end_cand - start_cand) <= len(snippet) * 1.5
            if coverage > 0.6 and is_compact:
                start, end = start_cand, end_cand
                while start > 0 and not raw_text[start - 1].isspace():
                    start -= 1
                while end < len(raw_text) and not raw_text[end].isspace():
                    end += 1
                return start, end

    return -1, -1


def find_span_with_context(
    raw_text: str,
    snippet: str,
    left_ctx: str = "",
    right_ctx: str = "",
    nth: int = 0,
) -> Tuple[int, int]:
    """
    Context-anchored span finder.

    Uses preceding / following context to disambiguate duplicate spans.
    Falls back to :func:`find_best_span` if context matching fails.
    """
    if not left_ctx and not right_ctx:
        return find_best_span(raw_text, snippet, nth=nth)

    def _to_flex_pattern(s: str) -> str:
        parts = [re.escape(p) for p in s.split()]
        return r"\s+".join(parts)

    pattern_parts: list[str] = []
    if left_ctx:
        pattern_parts.append(f"(?:{_to_flex_pattern(left_ctx)})\\s*")
    pattern_parts.append(f"({_to_flex_pattern(snippet)})")
    if right_ctx:
        pattern_parts.append(f"\\s*(?:{_to_flex_pattern(right_ctx)})")

    try:
        matches = list(re.finditer("".join(pattern_parts), raw_text, re.IGNORECASE))
        if len(matches) > nth:
            return matches[nth].start(1), matches[nth].end(1)
    except re.error:
        pass

    return find_best_span(raw_text, snippet, nth=nth)


# ---------------------------------------------------------------------------
# Batch Utilities
# ---------------------------------------------------------------------------


def precompute_span_positions(
    raw_text: str, candidates: List[str]
) -> Dict[str, List[Tuple[int, int]]]:
    """Pre-compute all positions for a batch of candidate span strings."""
    positions: Dict[str, List[Tuple[int, int]]] = {}
    raw_lower = raw_text.lower()
    for snippet in set(candidates):
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


# ---------------------------------------------------------------------------
# Deduplication & Merging
# ---------------------------------------------------------------------------


def deduplicate_overlapping_spans(
    spans: List[Dict], same_label_only: bool = True
) -> List[Dict]:
    """Remove spans that are subsets of other (longer) spans."""
    if not spans:
        return []
    sorted_spans = sorted(
        spans,
        key=lambda x: (x.get("start", 0), -(x.get("end", 0) - x.get("start", 0))),
    )
    kept: list[Dict] = []
    for span in sorted_spans:
        s, e, lbl = span.get("start", -1), span.get("end", -1), span.get("label", "")
        if s < 0 or e < 0:
            continue
        is_subset = any(
            k.get("start", -1) <= s
            and k.get("end", -1) >= e
            and (not same_label_only or k.get("label", "") == lbl)
            for k in kept
        )
        if not is_subset:
            kept.append(span)
    return kept


def merge_adjacent_spans(spans: List[Dict], max_gap: int = 2) -> List[Dict]:
    """Merge same-label spans separated by at most *max_gap* characters."""
    if not spans:
        return []
    by_label: Dict[str, list] = {}
    for span in spans:
        by_label.setdefault(span.get("label", "Unknown"), []).append(span)

    merged: list[Dict] = []
    for label, group in by_label.items():
        group.sort(key=lambda x: x.get("start", 0))
        current: Optional[Dict] = None
        for span in group:
            if current is None:
                current = span.copy()
            else:
                gap = span.get("start", 0) - current.get("end", 0)
                if 0 <= gap <= max_gap:
                    current["end"] = span.get("end", current["end"])
                    current["text"] = (
                        current.get("text", "") + " " + span.get("text", "")
                    )
                else:
                    merged.append(current)
                    current = span.copy()
        if current:
            merged.append(current)

    merged.sort(key=lambda x: x.get("start", 0))
    return merged


def verify_span_boundaries(spans: List[Dict], raw_text: str) -> List[Dict]:
    """Align span boundaries to word boundaries and re-slice text."""
    verified: list[Dict] = []
    for span in spans:
        start, end = span.get("start", -1), span.get("end", -1)
        if start < 0 or end <= start or start >= len(raw_text) or end > len(raw_text):
            continue
        while start < end and raw_text[start].isspace():
            start += 1
        while end > start and raw_text[end - 1].isspace():
            end -= 1
        if start >= end:
            continue
        v = span.copy()
        v["start"], v["end"], v["text"] = start, end, raw_text[start:end]
        verified.append(v)
    return verified
