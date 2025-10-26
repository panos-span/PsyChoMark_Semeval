#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_fewshot_examples.py

Produces best_fewshot_examples.json and fewshot_policy.json in the latest derived folder:
- S2: balanced conspiracy/non, optional cant_tell-as-negative with rationale
- S1: prior-aware (length/start), targets ambiguous pairs via overlap stats,
      adds small quota of outliers, enforces subreddit diversity, short snippets.

Inputs (from latest derived run):
  train.jsonl, dev.jsonl
  overlap_pair_stats_ci.json or overlap_pair_stats.json (optional, used if present)
  length_position_priors.json (optional; robust fallbacks)
  boundary_context.json (optional; not required)
  hard_examples.json (optional; helps S1/S2 selection, but not required)

Outputs:
  best_fewshot_examples.json
  fewshot_policy.json
"""

from __future__ import annotations

import argparse
import json
import re
from string import punctuation
import math
import random
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# -----------------------------
# Defaults / Policy
# -----------------------------
ALLOWED_S1 = {"Actor", "Action", "Effect", "Victim", "Evidence"}
_STOP = {
    "the",
    "a",
    "an",
    "to",
    "of",
    "and",
    "in",
    "on",
    "for",
    "with",
    "at",
    "by",
    "from",
    "that",
    "this",
    "it",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "as",
    "or",
    "if",
    "but",
    "so",
    "do",
    "did",
    "does",
}
_PUNCT_RE = re.compile(rf"^[{re.escape(punctuation)}\s]+$")
LABELS = ["Actor", "Action", "Effect", "Victim", "Evidence"]

# --- D) Weak-Evidence pruning params ---
EVIDENCE_MIN_LEN = 5
EVIDENCE_MAX_LEN = 90
EVIDENCE_MAX_STOPWORD_RATIO = 0.65
EVIDENCE_IDF_MIN_PCTL = 40  # require avg-IDF >= this percentile of corpus IDFs
EVIDENCE_MIN_ALPHA_RATIO = 0.6  # proportion of [a-z] characters in span
HEDGES = {
    "apparently",
    "reportedly",
    "allegedly",
    "maybe",
    "perhaps",
    "seems",
    "seem",
    "seemed",
    "supposedly",
    "rumor",
    "rumour",
    "claims",
    "claim",
    "claimed",
    "likely",
    "unlikely",
    "could",
    "might",
    "can",
    "could be",
    "might be",
    "i think",
    "i believe",
    "i guess",
    "according to someone",
    "i heard",
    "some say",
    "people say",
    "i read somewhere",
}
_URL_RE = re.compile(r"https?://|www\.")
_QUOTE_RE = re.compile(r"^[\"'“”‘’].+[\"'“”‘’]$")

# Policy (FROZEN unless overridden with flags)
CANT_TELL_IN_S2 = False
CANT_TELL_RATIONALE_DEFAULT = "Insufficient evidence for a concrete conspiracy claim; statements are ambiguous or hedged."

# tighter span constraints for few-shots
MIN_SPAN = 3
MAX_SPAN = 90
MIN_LABELS_PER_EXAMPLE = 2  # teach multi-label windows (esp. conflict pairs)

# --- E) Negative few-shots (no markers) ---
S1_NEG_MIN = 1
S1_NEG_MAX = 2
NEG_PAD = 90  # context around a gap center
NEG_MAX_LEN = 220  # hard length cap
NEG_MIN_LEN = 120  # aim for enough context
NEG_IDF_MIN_PCTL = 35  # slightly laxer than Evidence (D)


def _snippet_quality_negative(snippet: str, idf: dict, idf_cutoff: float) -> float:
    """Higher is better; penalize fluff/links/quotes; reward salience."""
    txt = snippet.strip()
    if not txt:
        return -1e6
    L = len(txt)
    if L < 40:
        return -1e6
    if _URL_RE.search(txt) or _QUOTE_RE.match(txt):
        return -1e3
    # salience via avg-idf (stopwords removed)
    sal = _avg_idf(txt, idf)
    if sal < idf_cutoff:
        return -5.0
    # lighter stopword penalty than Evidence
    toks = re.findall(r"\w+", txt)
    sw = sum(1 for t in toks if t.lower() in _STOP)
    sw_pen = (sw / max(1, len(toks))) * 0.8
    # reward medium length
    len_bonus = 1.0 if NEG_MIN_LEN <= L <= NEG_MAX_LEN else 0.0
    return sal + len_bonus - sw_pen


def _window_from_center(
    text: str, c: int, pad: int = NEG_PAD, hard_cap: int = NEG_MAX_LEN
) -> tuple[int, int]:
    left = max(0, c - pad)
    right = min(len(text), c + pad)
    if (right - left) > hard_cap:
        overflow = (right - left) - hard_cap
        trim_l = overflow // 2
        trim_r = overflow - trim_l
        left = min(max(0, left + trim_l), len(text))
        right = max(min(len(text), right - trim_r), 0)
    return left, right


def _gen_negative_snippets_from_gaps(
    text: str, forbid: list[tuple[int, int]], want: int = 4
) -> list[tuple[int, int]]:
    """
    Create negative windows from the largest unlabeled gaps in `text`.
    `forbid` contains (start, end) spans to avoid.
    """
    if not text:
        return []
    # merge/normalize forbid spans
    forbid = sorted(
        [(max(0, s), max(s, e)) for s, e in forbid if e > s], key=lambda x: x[0]
    )
    merged = []
    for s, e in forbid:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    forbid = [(s, e) for s, e in merged]
    # compute gaps
    gaps = []
    last = 0
    for s, e in forbid:
        if s > last:
            gaps.append((last, s))
        last = max(last, e)
    if last < len(text):
        gaps.append((last, len(text)))
    # choose centers from largest gaps
    gaps.sort(key=lambda x: (x[1] - x[0]), reverse=True)
    picks = []
    for gL, gR in gaps[: max(6, want * 2)]:
        if (gR - gL) < 40:  # too tiny
            continue
        c = (gL + gR) // 2
        L, R = _window_from_center(text, c)
        picks.append((L, R))
        if len(picks) >= want:
            break
    return picks


def _window_overlaps_any(L: int, R: int, spans: list[tuple[int, int]]) -> bool:
    for s, e in spans:
        if max(0, min(R, e) - max(L, s)) > 0:
            return True
    return False


def _tokenize_simple(s: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+", s.lower())


# --- F) Word-boundary tightening & G) crisp windows ---
CLEAN_PAD = 90  # default pad for clean/prototype/per-label windows
CLEAN_CAP = 240  # hard cap on clean windows
CONFLICT_CAP = 220  # hard cap on ambiguous windows (already used in B)

_BOUNDARY_PUNCT = set(punctuation) | {"“", "”", "‘", "’", "—", "–"}
_WHITES = {" ", "\t", "\n", "\r"}


def _is_boundary_char(ch: str) -> bool:
    return (ch in _WHITES) or (not ch.isalnum())


def _trim_span_punct(text: str, s: int, e: int) -> tuple[int, int]:
    """Trim leading/trailing punctuation/quotes; keep within [0,len]."""
    s0, e0 = s, e
    while s < e and text[s] in _BOUNDARY_PUNCT:
        s += 1
    while e > s and text[e - 1] in _BOUNDARY_PUNCT:
        e -= 1
    return (s, e) if (e > s) else (s0, e0)


def _snap_to_word_boundaries(text: str, s: int, e: int) -> tuple[int, int] | None:
    """
    Snap [s,e) to loose word boundaries to avoid mid-token edges.
    Returns None if snapping collapses the span.
    """
    if s < 0 or e <= s or e > len(text):
        return None
    # move left until boundary (or start)
    while s > 0 and not _is_boundary_char(text[s - 1]) and text[s - 1].isalnum():
        s -= 1
    # move right until boundary (or end)
    while e < len(text) and not _is_boundary_char(text[e]) and text[e].isalnum():
        e += 1
    if e <= s:
        return None
    return (s, e)


def _span_boundary_ok(text: str, s: int, e: int) -> bool:
    """
    Stronger boundary check: discourage mid-token edges and trailing punct.
    """
    if s < 0 or e <= s or e > len(text):
        return False
    if text[s:e].strip() == "":
        return False
    # no trailing punctuation
    if text[e - 1] in _BOUNDARY_PUNCT:
        return False
    # left/right "looks like" boundaries
    left_ok = (s == 0) or _is_boundary_char(text[s - 1])
    right_ok = (e == len(text)) or _is_boundary_char(text[e])
    return left_ok and right_ok


def _clamp_window(left: int, right: int, cap: int, n: int) -> tuple[int, int]:
    """Clamp [left,right) to <= cap length by trimming both sides."""
    left = max(0, left)
    right = min(n, right)
    if right - left <= cap:
        return left, right
    overflow = (right - left) - cap
    trim_l = overflow // 2
    trim_r = overflow - trim_l
    left = min(max(0, left + trim_l), n)
    right = max(min(n, right - trim_r), 0)
    if right <= left:  # degenerate, fallback to left..left+cap
        right = min(n, left + cap)
    return left, right


def _make_snippet(
    text: str, s: int, e: int, pad: int = CLEAN_PAD, cap: int = CLEAN_CAP
) -> tuple[str, int, int, int, int]:
    """
    Build a crisp snippet around [s,e):
      1) trim boundary punctuation
      2) snap to word boundaries (loose)
      3) pad symmetrically by `pad`, then clamp to `cap`
    Returns: (snippet, left, right, new_s, new_e) where new_s/new_e are offsets within snippet.
    """
    n = len(text)
    s, e = _trim_span_punct(text, s, e)
    snapped = _snap_to_word_boundaries(text, s, e)
    if snapped is not None:
        s, e = snapped
    left = max(0, s - pad)
    right = min(n, e + pad)
    left, right = _clamp_window(left, right, cap, n)
    snippet = text[left:right]
    return snippet, left, right, s - left, e - left


def _build_idf_stats(texts: list[str]) -> dict:
    """Compute doc-frequency and IDF for a simple corpus."""
    N = max(1, len(texts))
    df = Counter()
    for t in texts:
        df.update(set(_tokenize_simple(t)))
    idf = {}
    for w, d in df.items():
        idf[w] = math.log((N + 1) / (d + 0.5)) + 1.0
    return idf


def _avg_idf(span_text: str, idf: dict) -> float:
    toks = _tokenize_simple(span_text)
    toks = [t for t in toks if t not in _STOP]
    if not toks:
        return 0.0
    return sum(idf.get(t, 0.0) for t in toks) / len(toks)


# -----------------------------
# Helpers
# -----------------------------

# --- Per-label balance & diversity (C) ---
S1_PER_LABEL_DEFAULT = 4
MIN_PROTOTYPES_PER_LABEL = 2
MAX_AMBIG_PER_LABEL = 1
MAX_SUBREDDIT_REPEAT = 2  # per (subreddit, label)


def _labels_in_example(ex: dict) -> set[str]:
    return {m["label"] for m in ex.get("spans", []) if "label" in m}


def _snippet_hash(text: str) -> str:
    # stable dedup key by text content
    import hashlib

    return hashlib.md5((text or "").encode("utf-8")).hexdigest()


def _example_subreddit(ex: dict, fallback: str = "unknown") -> str:
    # try to carry subreddit via meta or ex; else fallback
    return ex.get("meta", {}).get("subreddit") or ex.get("subreddit") or fallback


def _is_ambiguous_reason(ex: dict) -> bool:
    r = (ex.get("meta") or {}).get("reason", "")
    return r.startswith("ambiguous_pair_")


# --- Clean prototype selection params ---
PROTOTYPE_PER_LABEL = 2
_STOP = {
    "the",
    "a",
    "an",
    "to",
    "of",
    "and",
    "in",
    "on",
    "for",
    "with",
    "as",
    "at",
    "by",
    "from",
    "that",
    "this",
    "it",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "into",
    "about",
}


def _span_boundary_ok(text: str, s: int, e: int) -> bool:
    if s < 0 or e <= s or e > len(text):
        return False
    # no trailing punctuation and reasonable characters inside
    snip = text[s:e]
    if not snip.strip():
        return False
    if snip.strip().lower() in {"the", "a", "an"}:
        return False
    if snip[-1] in punctuation:
        return False
    # coarse word-boundary checks
    left_ok = (s == 0) or text[s - 1].isspace()
    right_ok = (e == len(text)) or text[e].isspace()
    return left_ok and right_ok


def _is_weak_evidence_span(
    text: str, s: int, e: int, idf: dict, idf_cutoff: float
) -> bool:
    """Heuristic: reject hedgy/boilerplate, low-salience Evidence spans."""
    if s < 0 or e <= s or e > len(text):
        return True
    snip = text[s:e]
    L = e - s
    # length
    if L < EVIDENCE_MIN_LEN or L > EVIDENCE_MAX_LEN:
        return True
    # obvious boilerplate
    low = snip.strip().lower()
    if _URL_RE.search(low) or _QUOTE_RE.match(snip.strip()):
        return True
    # hedges
    for h in HEDGES:
        if h in low:
            return True
    # stopword & alpha ratio
    toks = re.findall(r"\w+", snip)
    if toks:
        sw = sum(1 for t in toks if t.lower() in _STOP)
        if (sw / len(toks)) > EVIDENCE_MAX_STOPWORD_RATIO:
            return True
    alpha = sum(1 for ch in snip if ch.isalpha())
    if (alpha / max(1, len(snip))) < EVIDENCE_MIN_ALPHA_RATIO:
        return True
    # salience by IDF
    if _avg_idf(snip, idf) < idf_cutoff:
        return True
    return False


def _span_quality(label: str, s: int, e: int, text: str, priors: dict) -> float:
    """Higher is better. Prior-aware + cleanliness penalties."""
    L = e - s
    if L < 3 or L > 90:
        return -1e6
    if not _span_boundary_ok(text, s, e):
        return -1e4
    # stopword penalty
    toks = re.findall(r"\w+", text[s:e])
    sw = sum(1 for t in toks if t.lower() in _STOP)
    sw_pen = (sw / max(1, len(toks))) * 1.5
    # prior length sweet-spot (q50~q90)
    p = priors.get(label, {})
    q50 = p.get("q50_len", 25)
    q90 = p.get("q90_len", 60)
    sweet = 0.0
    if 0.6 * q50 <= L <= q90:
        sweet += 2.5
    # start-position closeness to beta-mode if available
    pos_bonus = 0.0
    if "start_beta" in p:
        a = p["start_beta"].get("alpha", 1.0)
        b = p["start_beta"].get("beta", 1.0)
        mode = (
            ((a - 1) / (a + b - 2))
            if (a > 1 and b > 1)
            else (a / (a + b) if a > 0 and b > 0 else 0.5)
        )
        rel = s / max(1, len(text))
        pos_bonus = 1.2 * (1.0 - min(1.0, abs(rel - mode) / 0.25))
    return 5.0 + sweet + pos_bonus - sw_pen


def _example_is_valid_single(ex: dict) -> bool:
    """Validation that allows a single span/label (for prototypes)."""
    txt = ex.get("text", "") or ""
    spans = ex.get("spans", []) or []
    if not txt or not spans:
        return False
    for sp in spans:
        if not _valid_span(sp, txt):
            return False
    # forbid identical offsets across (possible) multiple labels
    offs = {(int(sp["start"]), int(sp["end"])) for sp in spans}
    if len(offs) < len(spans):
        return False
    return True


def _cross_label_iou_exceeds(
    cands_df: pd.DataFrame, doc_id: str, s: int, e: int, lab: str, thr: float = 0.05
) -> bool:
    """Check if (s,e) has IoU>thr with any OTHER label span from same doc."""
    sub = cands_df[cands_df["doc_id"] == doc_id]
    for _, r in sub.iterrows():
        if r["label"] == lab:
            continue
        s2, e2 = int(r["start"]), int(r["end"])
        inter = max(0, min(e, e2) - max(s, s2))
        union = (e - s) + (e2 - s2) - inter
        iou = inter / union if union > 0 else 0.0
        if iou > thr:
            return True
    return False


def _iou(a_s: int, a_e: int, b_s: int, b_e: int) -> float:
    inter = max(0, min(a_e, b_e) - max(a_s, b_s))
    union = (a_e - a_s) + (b_e - b_s) - inter
    return inter / union if union > 0 else 0.0


def _identical_substring(text: str, s1: int, e1: int, s2: int, e2: int) -> bool:
    """True if spans have the same surface string after stripping whitespace/punct."""
    t1 = text[s1:e1].strip().strip(punctuation).lower()
    t2 = text[s2:e2].strip().strip(punctuation).lower()
    return bool(t1) and (t1 == t2)


def _window_quality_for_conflict(text: str, spans: list[dict]) -> float:
    """Reward short windows and low cross-label crowding; higher is better."""
    coords = [(m["start"], m["end"]) for m in spans]
    left = min(s for s, _ in coords)
    right = max(e for _, e in coords)
    length = right - left
    # short windows preferred; 220 is target cap
    len_pen = max(0, (length - 220)) / 50.0
    # mild penalty if there are more than 2 spans squeezed in
    crowd_pen = max(0, len(spans) - 2) * 0.6
    return 3.0 - len_pen - crowd_pen


def _is_all_stopwords(txt: str) -> bool:
    toks = [t for t in re.split(r"\s+", txt.strip()) if t]
    if not toks:
        return True
    return all(t.lower() in _STOP for t in toks)


def _valid_span(
    sp, text: str, min_len: int = MIN_SPAN, max_len: int = MAX_SPAN
) -> bool:
    try:
        s = int(sp.get("start", -1))
        e = int(sp.get("end", -1))
        lbl = sp.get("label")
    except Exception:
        return False
    if lbl not in ALLOWED_S1 or e <= s:
        return False
    if (e - s) < min_len or (e - s) > max_len:
        return False
    # extract raw span safely
    if not (0 <= s < len(text)) or not (0 < e <= len(text)):
        return False
    snip = text[s:e]
    if not snip or _PUNCT_RE.match(snip):
        return False
    if _is_all_stopwords(snip):
        return False
    return True


# ---- Evidence quality + cross-label duplicate filters ----
EVIDENCE_URL_RE = re.compile(r"(https?://|www\.)", re.I)
EVIDENCE_TLD_RE = re.compile(r"\.(?:com|org|net|io|gov|edu)\b", re.I)
EVIDENCE_QUOTE_RE = re.compile(r"[\"“”‘’].+?[\"“”‘’]")
EVIDENCE_ATTRIB_RE = re.compile(
    r"\b(according to|reported by|the report|the study|data (?:show|shows)|"
    r"leaked (?:emails|documents|docs)|foia|as cited by)\b",
    re.I,
)
EVIDENCE_NUMERIC_RE = re.compile(
    r"\b\d{2,}(?:\.\d+)?%?\b|\b\d+\s+(?:emails|pages|studies)\b", re.I
)

_WEAK_EVIDENCE_STRS = {
    "this article",
    "this source",
    "sources",
    "source",
    "see above",
    "that",
    "this",
    "it",
}


def _evidence_is_strong(snip: str) -> bool:
    s = (snip or "").strip()
    if not s:
        return False
    return any(
        pat.search(s)
        for pat in (
            EVIDENCE_URL_RE,
            EVIDENCE_TLD_RE,
            EVIDENCE_QUOTE_RE,
            EVIDENCE_ATTRIB_RE,
            EVIDENCE_NUMERIC_RE,
        )
    )


def _evidence_is_weak(snip: str) -> bool:
    return (snip or "").strip().lower() in _WEAK_EVIDENCE_STRS


def _has_bad_cross_label_dup(example: dict) -> bool:
    """
    Return True if the example contains two *different* labels sharing the exact [start,end),
    EXCEPT when one of them is Evidence (Evidence is allowed to overlap anything).
    Also explicitly catch Action==Effect identical offsets.
    """
    spans = example.get("spans", []) or []
    by_coord = defaultdict(set)  # (s,e) -> {labels}
    for m in spans:
        try:
            lab = (m.get("label") or "").strip()
            s = int(m.get("start"))
            e = int(m.get("end"))
        except Exception:
            continue
        if e <= s:
            continue
        by_coord[(s, e)].add(lab)

    for (s, e), labs in by_coord.items():
        if len(labs) >= 2:
            # if overlap involves Evidence + something else → allowed
            if labs - {"Evidence"} and "Evidence" in labs and len(labs) == 2:
                continue
            # explicitly catch Action==Effect duplicates
            if {"Action", "Effect"}.issubset(labs):
                return True
            # any other cross-label identical span (no Evidence exemption)
            return True
    return False


def _example_has_only_weak_evidence(example: dict, full_text: str) -> bool:
    """True iff there is at least one Evidence span, and *all* Evidence spans are weak."""
    spans = example.get("spans", []) or []
    evidences = []
    for m in spans:
        if (m.get("label") or "").strip() != "Evidence":
            continue
        try:
            s = int(m.get("start"))
            e = int(m.get("end"))
        except Exception:
            continue
        if e <= s or not (0 <= s < len(full_text)) or not (0 < e <= len(full_text)):
            continue
        evidences.append(full_text[s:e])
    if not evidences:
        return False
    # at least one evidence exists; check if *all* are weak
    strong = any(_evidence_is_strong(snip) for snip in evidences)
    weak_only = all(_evidence_is_weak(snip) for snip in evidences)
    return (not strong) and weak_only


def _example_is_valid(ex: dict) -> bool:
    """All spans must fit the text and be non-trivial; prefer >=2 labels."""
    txt = ex.get("text", "") or ""
    spans = ex.get("spans", []) or []
    if not txt or not spans:
        return False
    labs = set()
    for sp in spans:
        if not _valid_span(sp, txt):
            return False
        labs.add(sp.get("label"))
    # forbid identical spans carrying different labels (e.g., Action==Effect on same offsets)
    offsets = {(int(sp["start"]), int(sp["end"])) for sp in spans}
    if len(offsets) < len(spans):
        return False
    # encourage multi-label windows (but allow single when Evidence-only is unavoidable)
    return (len(labs) >= MIN_LABELS_PER_EXAMPLE) or (labs == {"Evidence"})


def _example_has_valid_span(ex) -> bool:
    text = ex.get("text", "") or ""
    spans = ex.get("spans", []) or []
    return any(_valid_span(sp, text) for sp in spans)


def load_lexicons(out_dir: Path):
    # defaults if file missing
    abs_default = [
        "always",
        "never",
        "everyone",
        "no one",
        "impossible",
        "undeniable",
        "without a doubt",
        "completely",
        "totally",
        "entirely",
        "absolutely",
        "certainly",
        "no doubt",
        "no doubts",
    ]
    hed_default = [
        "maybe",
        "perhaps",
        "possibly",
        "likely",
        "unlikely",
        "appears",
        "seems",
        "suggests",
        "might",
        "could",
        "may",
        "arguably",
    ]
    path = out_dir / "lexicons.json"
    if path.exists():
        try:
            js = json.loads(path.read_text(encoding="utf-8"))
            A = js.get("ABSOLUTIST") or abs_default
            H = js.get("HEDGES") or hed_default
            return A, H
        except Exception:
            pass
    return abs_default, hed_default


# --- NEW: Prior- & boundary-aware pickers for S1 ---
def _dist_to_priors(priors: dict, label: str, s: int, e: int, tlen: int) -> float:
    span_len = max(1, e - s)
    start_pos = s / max(1, tlen)
    # lower is better
    dz = prior_len_z(priors, label, span_len)  # ~|z|
    dp = prior_pos_dist(priors, label, start_pos)  # ~distance in [0,1]
    return 1.5 * dz + 1.0 * dp


def pick_s1_prior_examples(df_lab: pd.DataFrame, priors: dict, k_near=1, k_outlier=1):
    if df_lab.empty:
        return []
    df = df_lab.copy()
    df["tlen"] = df["text"].str.len().fillna(1).astype(int)
    df["prior_dist"] = df.apply(
        lambda r: _dist_to_priors(priors, r["label"], r["start"], r["end"], r["tlen"]),
        axis=1,
    )
    # near-prior: lowest distance
    near = (
        df.sort_values("prior_dist", ascending=True)
        .head(k_near)
        .to_dict(orient="records")
    )
    # outlier: highest distance
    out = (
        df.sort_values("prior_dist", ascending=False)
        .head(k_outlier)
        .to_dict(orient="records")
    )
    return near + out


def _has_boundary_cue(ctx: dict, label: str, text: str, s: int, e: int) -> bool:
    cues = []
    for key in ("before_1w", "after_1w", "before_2w", "after_2w"):
        cues.extend(ctx.get(label, {}).get(key, [])[:5])  # top-5 per side
    window = text[max(0, s - 40) : min(len(text), e + 40)].lower()
    return any(c and c.lower() in window for c in cues if isinstance(c, str))


def pick_s1_boundary_examples(df_lab: pd.DataFrame, boundary_ctx: dict, k=1):
    if df_lab.empty:
        return []
    rows = []
    for _, r in df_lab.iterrows():
        if _has_boundary_cue(
            boundary_ctx, r["label"], r["text"] or "", int(r["start"]), int(r["end"])
        ):
            rows.append(r.to_dict())
    rng = np.random.default_rng(42)
    rng.shuffle(rows)
    return rows[:k]


def find_latest_dir(pointer: Path) -> Path:
    if not pointer.exists():
        raise FileNotFoundError(
            f"Pointer file not found: {pointer}. Run data_pipeline.py first."
        )
    d = Path(pointer.read_text().strip())
    if not d.exists():
        raise FileNotFoundError(f"Derived folder listed in pointer not found: {d}")
    return d


def alias_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "_id" not in df.columns and "doc_id" in df.columns:
        df["_id"] = df["doc_id"]
    return df


def safe_markers(row) -> List[dict]:
    m = row.get("markers", [])
    return m if isinstance(m, list) else []


def iou_char(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    s1, e1 = a
    s2, e2 = b
    inter = max(0, min(e1, e2) - max(s1, s2))
    if inter <= 0:
        return 0.0
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / union if union > 0 else 0.0


def top_pairs(
    pair_stats: Dict[str, dict], key="iou@0.5", topk=2
) -> List[Tuple[str, str]]:
    if not pair_stats:
        return []
    items = [(p, d.get(key, 0.0)) for p, d in pair_stats.items()]
    items = sorted(items, key=lambda x: x[1], reverse=True)
    return [tuple(sorted(p.split("/"))) for p, _ in items[:topk]]


def load_json(path: Path, default=None):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {} if default is None else default


def prior_len_z(priors, label, span_len) -> float:
    """Return |z| distance under lognormal prior; robust to missing fits."""
    p = priors.get(label, {})
    if "length_lognorm" in p:
        mu = p["length_lognorm"].get("mu", 0.0)
        sig = max(1e-6, p["length_lognorm"].get("sigma", 1.0))
        return abs((math.log(max(1, span_len)) - mu) / sig)
    # fallback: compare to q90 if present -> pseudo z in [0, ~1.5+]
    q90 = p.get("q90_len") or p.get("q90", None)
    if q90:
        return max(0.0, (span_len - float(q90)) / (float(q90) + 1e-6))
    return 1.0  # neutral


def prior_pos_dist(priors, label, start_pos) -> float:
    """Distance to Beta mode for start position; robust to missing fits."""
    p = priors.get(label, {})
    if "start_beta" in p:
        a = p["start_beta"].get("alpha", 1.0)
        b = p["start_beta"].get("beta", 1.0)
        mode = (
            (a - 1) / (a + b - 2)
            if (a > 1 and b > 1)
            else (a / (a + b) if (a > 0 and b > 0) else 0.5)
        )
        return abs(start_pos - mode)
    return 0.5  # neutral


def doc_has_target_overlap(
    markers: List[dict], target_pairs: List[Tuple[str, str]]
) -> int:
    spans = [
        (m["label"], m["start"], m["end"])
        for m in (markers or [])
        if m.get("label") in ALLOWED_S1
    ]
    for i in range(len(spans)):
        l1, s1, e1 = spans[i]
        for j in range(i + 1, len(spans)):
            l2, s2, e2 = spans[j]
            if (
                tuple(sorted([l1, l2])) in target_pairs
                and iou_char((s1, e1), (s2, e2)) >= 0.5
            ):
                return 1
    return 0


def markers_compact(markers: List[dict], max_per_label=2) -> List[dict]:
    out = []
    per_lab = defaultdict(int)
    for m in markers or []:
        lbl = m.get("label")
        if lbl not in ALLOWED_S1:
            continue
        if per_lab[lbl] >= max_per_label:
            continue
        if {"label", "start", "end"} <= m.keys():
            out.append({"label": lbl, "start": int(m["start"]), "end": int(m["end"])})
            per_lab[lbl] += 1
    return out


# --- utilities near the top of the script ---
CUE = {
    "Action": [
        "ban",
        "cover",
        "expose",
        "direct",
        "rig",
        "steal",
        "weaponize",
        "air dropping",
        "orchestrate",
    ],
    "Effect": [
        "agenda",
        "control",
        "population",
        "plan",
        "goal",
        "narrative",
        "cover-up",
        "depopulation",
        "scheme",
    ],
    "Evidence": [
        "according to",
        "http",
        "www",
        ".com/",
        ".org/",
        "report",
        "leaked",
        "email",
        "study",
        "data",
        "quote",
        '"',
    ],
    "Actor": [
        "they",
        "cabal",
        "deep state",
        "elites",
        "agency",
        "organization",
        "party",
        "admin",
        "MSM",
    ],
    "Victim": [
        "public",
        "people",
        "citizen",
        "taxpayer",
        "voters",
        "children",
        "us",
        "society",
    ],
}
STOP = set("the a an of to in on for with by and or".split())


def _trim_snippet(text, s, e, radius=130):
    left = max(0, s - radius)
    right = min(len(text), e + radius)
    return text[left:right], left, right


def _distinct_by_doc(seen, doc_id):
    if doc_id in seen:
        return False
    seen.add(doc_id)
    return True


# --------- NEW: quality heuristics & guards ---------
def _is_url_like(s: str) -> bool:
    s = (s or "").lower()
    return (
        ("http://" in s)
        or ("https://" in s)
        or (".com/" in s)
        or (".org/" in s)
        or (".net/" in s)
    )


def _looks_evidence_fragment(txt: str) -> bool:
    s = (txt or "").lower()
    cues = [
        "according to",
        "report",
        "study",
        "leaked",
        "data",
        "evidence",
        "the report shows",
        "the data show",
        "source:",
    ]
    if _is_url_like(s):
        return True
    if any(c in s for c in cues):
        return True
    # numbers-heavy is often evidence-like
    digits = sum(ch.isdigit() for ch in s)
    return digits >= max(3, len(s) // 6)


def _pick_evidence_from_train(train_df, want=3, seed=42, min_len=10, max_len=120):
    rng = np.random.default_rng(seed)
    out = []
    for _, r in train_df.iterrows():
        t = r.get("text") or ""
        for m in r.get("markers") or []:
            if m.get("label") == "Evidence":
                s, e = int(m.get("start", -1)), int(m.get("end", -1))
                if e <= s:
                    continue
                span = t[s:e].strip()
                if not (min_len <= len(span) <= max_len):
                    continue
                if not _looks_evidence_fragment(span):
                    continue
                left, right = max(0, s - 120), min(len(t), e + 120)
                snippet = t[left:right].strip()
                off_s, off_e = s - left, e - left
                out.append(
                    {
                        "doc_id": r.get("doc_id"),
                        "text": snippet,
                        "spans": [{"label": "Evidence", "start": off_s, "end": off_e}],
                        "meta": {"reason": "evidence_boost"},
                    }
                )
    rng.shuffle(out)
    return out[:want]


def _pick_victim_from_train(train_df, want=2, seed=42, min_len=3, max_len=40):
    rng = np.random.default_rng(seed)
    out = []
    for _, r in train_df.iterrows():
        t = r.get("text") or ""
        for m in r.get("markers") or []:
            if m.get("label") == "Victim":
                s, e = int(m.get("start", -1)), int(m.get("end", -1))
                if e <= s:
                    continue
                span = t[s:e].strip()
                if not (min_len <= len(span) <= max_len):
                    continue
                # short, clean NP
                if span.lower().startswith(("the ", "a ", "an ")):
                    # keep determiners only if group name
                    if len(span.split()) <= 2:
                        continue
                left, right = max(0, s - 120), min(len(t), e + 120)
                snippet = t[left:right].strip()
                off_s, off_e = s - left, e - left
                out.append(
                    {
                        "doc_id": r.get("doc_id"),
                        "text": snippet,
                        "spans": [{"label": "Victim", "start": off_s, "end": off_e}],
                        "meta": {"reason": "victim_boost"},
                    }
                )
    rng.shuffle(out)
    return out[:want]


def _has_identical_action_effect(spans_in_snippet: list, snippet_text: str) -> bool:
    """
    Return True if Action and Effect substrings are exactly identical (case/space-insensitive),
    which is a harmful training signal for the model (teaches the same span for both).
    """
    a_txt, e_txt = None, None
    for m in spans_in_snippet:
        if m["label"] == "Action":
            a_txt = snippet_text[m["start"] : m["end"]].strip().lower()
        elif m["label"] == "Effect":
            e_txt = snippet_text[m["start"] : m["end"]].strip().lower()
    return bool(a_txt and e_txt and a_txt == e_txt)


# -----------------------------
# S2 selection
# -----------------------------
def pick_balanced_by_label(
    df,
    k_per_label=2,
    label_col="doc_label",
    text_col="text",
    diversity_col="subreddit",
    seed=42,
    len_min=160,
    len_max=1000,
):
    out = []
    rng = np.random.default_rng(seed)
    for lab in ["conspiracy", "non"]:
        sub = df[df[label_col] == lab].copy()
        if sub.empty:
            continue
        sub["_rand"] = rng.random(size=len(sub))
        if diversity_col in sub.columns:
            sub = sub.sort_values([diversity_col, "_rand"])
        else:
            sub = sub.sort_values(["_rand"])
        picked = []
        seen = defaultdict(int)
        for _, r in sub.iterrows():
            txt = r.get(text_col, "") or ""
            if not (len_min <= len(txt) <= len_max):
                continue
            sr = r.get(diversity_col, "")
            if seen[sr] >= 2:  # light diversity cap
                continue
            picked.append(r.to_dict())
            seen[sr] += 1
            if len(picked) >= k_per_label:
                break
        out.extend(picked)
    return out


def coerce_s2_item(r, label_override=None, rationale=None, tag=None):
    item = {
        "doc_id": r.get("doc_id"),
        "text": r.get("text", ""),
        "label": label_override if label_override else r.get("doc_label"),
        "rationale": rationale if rationale else "",
    }
    if tag:
        item["source_label"] = tag
    return item


# --- add near other helpers ---
def pick_negative_s1_snippet(dev_df, min_len=120, max_len=320, seed=42):
    """Return a dict with text and empty spans to teach 'no markers'."""
    pool = dev_df[
        dev_df["markers"].apply(lambda m: not isinstance(m, list) or len(m) == 0)
    ]
    if pool.empty:
        return None
    rows = pool.sample(n=min(50, len(pool)), random_state=seed, replace=False)
    for _, r in rows.iterrows():
        t = (r.get("text") or "").strip()
        if min_len <= len(t) <= max_len:
            return {
                "doc_id": r.get("doc_id"),
                "text": t,
                "spans": [],  # empty JSON to demonstrate valid 'no markers'
                "meta": {"reason": "negative_no_markers"},
            }
    return None


def _shorten_rationale(r: str, max_chars=160) -> str:
    r = (r or "").strip().split("\n")[0]
    return (r[:max_chars] + "…") if len(r) > max_chars else r


def enforce_min_yes_fewshots(s2_list, min_yes=4, df_all=None, seed=42):
    yes = [x for x in s2_list if (x.get("label") or "").lower() == "conspiracy"]
    if len(yes) >= min_yes:
        # clean rationales
        for x in s2_list:
            if "rationale" in x:
                x["rationale"] = _shorten_rationale(x["rationale"])
        return s2_list

    # sample additional 'conspiracy' rows from data
    needed = min_yes - len(yes)
    if df_all is not None:
        pool = df_all[df_all["doc_label"] == "conspiracy"].copy()
        if not pool.empty:
            pool = pool.sample(n=min(needed * 3, len(pool)), random_state=seed)
            added = 0
            for _, r in pool.iterrows():
                if any(x["doc_id"] == r["doc_id"] for x in s2_list):
                    continue
                s2_list.append(
                    {
                        "doc_id": r["doc_id"],
                        "text": r.get("text", ""),
                        "label": "conspiracy",
                        "rationale": "Text clearly alleges a covert plan with actors and evidence.",
                    }
                )
                added += 1
                if added >= needed:
                    break
    # clean rationales for all
    for x in s2_list:
        if "rationale" in x:
            x["rationale"] = _shorten_rationale(x["rationale"])
    return s2_list


# -----------------------------
# S1 scoring / selection
# -----------------------------
MIN_PER_LABEL = 2
ALL_LABS = {"Actor", "Action", "Effect", "Victim", "Evidence"}


def _label_counts(exs):
    c = Counter()
    for ex in exs:
        for m in ex.get("spans", []):
            lab = m.get("label")
            if lab in ALL_LABS:
                c[lab] += 1
    return c


def backfill_missing_labels(train_df, s1_examples, max_per_label=MIN_PER_LABEL):
    """
    Relaxed backfill: if any label has < max_per_label examples, sample additional
    (sanity-filtered) spans from train_df to guarantee coverage.
    """
    have = _label_counts(s1_examples)
    need = [lab for lab in ALL_LABS if have[lab] < max_per_label]
    if not need:
        return []

    added = []
    seen_texts = {ex.get("text", "") for ex in s1_examples}

    for _, r in train_df.iterrows():
        if not need:
            break
        text = r.get("text") or ""
        tlen = len(text)
        for m in r.get("markers") or []:
            L = m.get("label")
            if L not in need:
                continue
            s, e = int(m.get("start", 0)), int(m.get("end", 0))
            span_len = max(1, e - s)

            # relaxed but sane filters
            if span_len < 3:
                continue
            if s == 0 and e >= (tlen - 3):
                continue
            if text in seen_texts:
                continue

            left = max(0, s - 120)
            right = min(tlen, e + 120)
            snippet = text[left:right].strip()
            off_s, off_e = s - left, e - left

            ex = {
                "doc_id": r.get("doc_id"),
                "text": snippet,
                "spans": [{"label": L, "start": off_s, "end": off_e}],
                "meta": {"reason": "backfill_relaxed", "subreddit": r.get("subreddit")},
            }
            added.append(ex)
            seen_texts.add(text)
            have[L] += 1
            if have[L] >= max_per_label:
                need.remove(L)
                if not need:
                    break
    return added


def build_s1_candidates(
    train_df: pd.DataFrame, priors: dict, target_pairs: List[Tuple[str, str]]
) -> pd.DataFrame:
    cand = []
    for _, r in train_df.iterrows():
        t = r.get("text") or ""
        tlen = max(1, len(t))
        has_tpair = doc_has_target_overlap(r.get("markers"), target_pairs)
        for m in r.get("markers") or []:
            L = m.get("label")
            if L not in ALLOWED_S1:
                continue
            s, e = int(m["start"]), int(m["end"])
            span_len = max(1, e - s)

            # --- QUALITY FILTERS ---
            if span_len < 4:
                continue
            if s == 0 and e >= (tlen - 5):
                continue
            span_txt = t[s:e].strip().lower()
            if span_txt.endswith((".", ",", ";", "’", "”")) or span_txt.startswith(
                ("the ", "a ", "an ")
            ):
                continue
            # -----------------------

            start_pos = s / tlen
            score = 0.0
            # priors (closer is better)
            score += 1.5 * (1.0 - min(3.0, prior_len_z(priors, L, span_len)) / 3.0)
            score += 1.5 * (1.0 - min(1.0, prior_pos_dist(priors, L, start_pos)))
            # overlap bonus for docs with target pair conflict
            score += 1.0 * has_tpair
            if L == "Evidence":
                score -= 0.15  # reduce baseline bias
                # prefer Evidence that co-occurs with other labels in the same doc
                if has_tpair:
                    score += 0.30
            # compact-length bonus vs q90
            q90 = priors.get(L, {}).get("q90_len") or priors.get(L, {}).get(
                "q90_per_label", {}
            ).get(L)
            if q90 is not None:
                score += 0.25 * (span_len <= float(q90))

            # >>> NEW: Evidence quality shaping
            if L == "Evidence":
                snip = t[s:e]
                if _evidence_is_strong(snip):
                    score += 0.7  # prefer concrete citations/quotes/URLs/numerics
                elif _evidence_is_weak(snip):
                    score -= 0.6  # downweight vague placeholders

            cand.append(
                {
                    "doc_id": r["doc_id"],
                    "subreddit": r.get("subreddit", "NA"),
                    "label": L,
                    "start": s,
                    "end": e,
                    "score": float(score),
                    "text": t,
                }
            )
    return pd.DataFrame(cand)


def build_s1_fewshots(
    cands_df: pd.DataFrame,
    policy: dict | None = None,
    max_per_label: int = 4,
    max_per_pair: int = 1,
    pad: int = CLEAN_PAD,
) -> List[dict]:
    """
    Selector over pre-scored candidates (from build_s1_candidates):
      1) Per-label top-k with subreddit diversity, then 1 outlier per label.
      2) Conflict exemplars for target ambiguous pairs (Action–Effect, Actor–Victim):
         - spans from the same doc
         - IoU in [0.15, 0.60]
         - combined window ≤ ~320 chars (pad around union)
         - skip identical substrings for Action/Effect
      3) Return raw few-shot examples; later pruning/backfill in main() will refine further.
    """
    # --- Build corpus IDF once (for Evidence pruning)
    try:
        _all_texts = cands_df["text"].dropna().astype(str).tolist()
    except Exception:
        _all_texts = []
    _idf = _build_idf_stats(_all_texts) if _all_texts else {}
    # compute IDF cutoff (percentile) over vocabulary
    _idf_vals = sorted(_idf.values()) if _idf else [0.0]

    def _pctl(vals, p):
        if not vals:
            return 0.0
        k = max(0, min(len(vals) - 1, int(round((p / 100.0) * (len(vals) - 1)))))
        return vals[k]

    _idf_cutoff = _pctl(_idf_vals, EVIDENCE_IDF_MIN_PCTL)
    _neg_idf_cutoff = _pctl(_idf_vals, NEG_IDF_MIN_PCTL)

    if cands_df is None or cands_df.empty:
        return []

    picked: List[dict] = []

    # 1) Clean prototypes (single-label, low cross-label IoU) — pick first
    for lab in LABELS:
        df_lab = cands_df[cands_df["label"] == lab].copy()
        if df_lab.empty:
            continue
        # re-score by cleanliness + priors
        df_lab["_proto_score"] = df_lab.apply(
            lambda r: _span_quality(
                lab,
                int(r["start"]),
                int(r["end"]),
                r["text"],
                policy.get("priors", {}) if policy else {},
            ),
            axis=1,
        )
        # subreddit-aware ordering, then proto score, then original score
        df_lab["_rank_sub"] = df_lab.groupby("subreddit")["_proto_score"].rank(
            ascending=False, method="first"
        )
        df_lab = df_lab.sort_values(
            ["_rank_sub", "_proto_score", "score"], ascending=[True, False, False]
        )

        per_lab = []
        seen_texts = set()
        for _, r in df_lab.iterrows():
            doc_id = r["doc_id"]
            text = r["text"]
            s = int(r["start"])
            e = int(r["end"])
            L = e - s
            if L < MIN_SPAN or L > MAX_SPAN:
                continue
            # avoid cross-label overlap in same doc
            if _cross_label_iou_exceeds(cands_df, doc_id, s, e, lab, thr=0.05):
                continue
            # Evidence: strong-only
            if lab == "Evidence" and _is_weak_evidence_span(
                text, s, e, _idf, _idf_cutoff
            ):
                continue
            # short, crisp snippet (F+G)
            snippet, ls, rs, ns, ne = _make_snippet(
                text, s, e, pad=min(pad, CLEAN_PAD), cap=CLEAN_CAP
            )
            # boundary sanity
            if not _span_boundary_ok(snippet, ns, ne):
                continue
            span = {"label": lab, "start": ns, "end": ne}
            ex = {
                "doc_id": doc_id,
                "text": snippet,
                "spans": [span],
                "meta": {"reason": "prototype_clean", "subreddit": r.get("subreddit")},
            }
            if not _example_is_valid_single(ex):
                continue
            if snippet in seen_texts:
                continue
            per_lab.append(ex)
            seen_texts.add(snippet)
            if len(per_lab) >= PROTOTYPE_PER_LABEL:
                break
        picked.extend(per_lab)

    # 2) Per-label top-k (multi-label windows) to reach coverage after prototypes
    for lab in LABELS:
        have_lab = sum(
            1 for ex in picked if any(m["label"] == lab for m in ex.get("spans", []))
        )
        need = max(0, max_per_label - have_lab)
        if need == 0:
            continue
        df_lab = cands_df[cands_df["label"] == lab].copy()
        if df_lab.empty:
            continue
        df_lab["_rank"] = df_lab.groupby("subreddit")["score"].rank(
            ascending=False, method="first"
        )
        df_lab = df_lab.sort_values(["_rank", "score"], ascending=[True, False])
        per_lab = []
        seen_texts = {ex.get("text", "") for ex in picked}
        for _, r in df_lab.iterrows():
            doc_id = r["doc_id"]
            text = r["text"]
            s = int(r["start"])
            e = int(r["end"])
            L = e - s
            if L < MIN_SPAN or L > MAX_SPAN:
                continue
            ls = max(0, s - pad)
            rs = min(len(text), e + pad)
            snippet, ls, rs, ns, ne = _make_snippet(
                text, s, e, pad=min(pad, CLEAN_PAD), cap=CLEAN_CAP
            )
            if not _span_boundary_ok(snippet, ns, ne):
                continue
            spans = [{"label": lab, "start": ns, "end": ne}]
            ex = {
                "doc_id": doc_id,
                "text": snippet,
                "spans": spans,
                "meta": {"reason": "per_label_top", "subreddit": r.get("subreddit")},
            }
            # here we keep the original validator (prefers >=2 labels), but will be complemented by overlaps later
            if _example_is_valid(ex) and snippet not in seen_texts:
                per_lab.append(ex)
                seen_texts.add(snippet)
            if len(per_lab) >= need:
                break
        picked.extend(per_lab)

    # -- E) Build negatives (no markers)
    neg_candidates: list[dict] = []
    # 1) If a negatives source (full docs with markers) is provided in policy/meta, use it
    # You can thread this via `policy.get("neg_source_docs")` if you parsed --neg-source earlier.
    neg_docs = None
    if isinstance(policy, dict):
        neg_docs = policy.get(
            "neg_source_docs"
        )  # list of dicts: {"doc_id","text","markers":[],"subreddit":...}
    if neg_docs:
        for d in neg_docs:
            if d is None:
                continue
            if d.get("markers"):  # must be empty or falsy
                continue
            text = (d.get("text") or "").strip()
            if not text:
                continue
            # take a centered window over the whole doc if it's long
            c = len(text) // 2
            L, R = _window_from_center(text, c)
            snippet = text[L:R].strip()
            if not (NEG_MIN_LEN <= len(snippet) <= NEG_MAX_LEN):
                continue
            q = _snippet_quality_negative(snippet, _idf, _neg_idf_cutoff)
            if q < 0.0:
                continue
            neg_candidates.append(
                {
                    "doc_id": d.get("doc_id"),
                    "text": snippet,
                    "spans": [],
                    "meta": {
                        "reason": "negative_no_markers",
                        "subreddit": d.get("subreddit"),
                    },
                }
            )
    else:
        # 2) Fallback: synthesize from gaps within docs that do have labeled spans
        by_doc_spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
        doc_text: dict[str, str] = {}
        doc_sub: dict[str, str] = {}
        for _, r in cands_df.iterrows():
            did = r["doc_id"]
            s, e = int(r["start"]), int(r["end"])
            by_doc_spans[did].append((s, e))
            doc_text[did] = r["text"]
            if "subreddit" in r:
                doc_sub[did] = r["subreddit"]
        for did, spans in by_doc_spans.items():
            text = (doc_text.get(did) or "").strip()
            if not text:
                continue
            # derive a few windows from the largest gaps
            windows = _gen_negative_snippets_from_gaps(text, spans, want=4)
            for L, R in windows:
                if (R - L) < NEG_MIN_LEN:
                    continue
                if _window_overlaps_any(L, R, spans):
                    continue
                snippet = text[L:R].strip()
                if len(snippet) > NEG_MAX_LEN:
                    snippet = snippet[:NEG_MAX_LEN].rstrip()
                q = _snippet_quality_negative(snippet, _idf, _neg_idf_cutoff)
                if q < 0.0:
                    continue
                neg_candidates.append(
                    {
                        "doc_id": did,
                        "text": snippet,
                        "spans": [],
                        "meta": {
                            "reason": "negative_no_markers",
                            "subreddit": doc_sub.get(did),
                        },
                    }
                )

    # keep top few negatives by quality & uniqueness
    # (quality is embedded via _snippet_quality_negative; re-score here quickly)
    scored_neg = []
    seen_neg = set()
    for n in neg_candidates:
        t = n.get("text", "")
        h = _snippet_hash(t)
        if h in seen_neg:
            continue
        seen_neg.add(h)
        q = _snippet_quality_negative(t, _idf, _neg_idf_cutoff)
        scored_neg.append((q, n))
    scored_neg.sort(key=lambda x: x[0], reverse=True)
    neg_top = [n for _, n in scored_neg[: max(4, S1_NEG_MAX * 3)]]

    # 3) Conflict exemplars (ambiguous pairs, sanitized)
    target_pairs = []
    if policy and isinstance(policy, dict):
        target_pairs = policy.get("targets", {}).get("ambiguous_pairs_top2", []) or []
    target_pairs = [
        tuple(sorted(p))
        for p in target_pairs
        if isinstance(p, (list, tuple)) and len(p) == 2
    ]

    if target_pairs:
        # group candidate rows by doc, so we can pair spans inside the same doc
        by_doc = {}
        for _, r in cands_df.iterrows():
            did = r.get("doc_id")
            if did not in by_doc:
                by_doc[did] = []
            by_doc[did].append(
                {
                    "label": r["label"],
                    "start": int(r["start"]),
                    "end": int(r["end"]),
                    "text": r["text"],  # full doc text
                    "score": float(r.get("score", 0.0)),
                }
            )

        IOU_MIN, IOU_MAX = 0.10, 0.35  # tighter range
        PAD_CONFLICT = min(pad, 90)  # shorter windows
        EFF_MAX_PER_PAIR = int(max(0, max_per_pair))

        per_pair_added = Counter()
        seen_docs_for_pair = set()  # avoid many from the same doc/pair

        # deterministic but varied over docs
        for doc_id, spans in by_doc.items():
            # quick index by label
            by_lab = defaultdict(list)
            for m in spans:
                by_lab[m["label"]].append(m)

            for a, b in target_pairs:
                # normalize tuple order to match per_pair_added keys
                key_pair = tuple(sorted((a, b)))
                if per_pair_added[key_pair] >= EFF_MAX_PER_PAIR:
                    continue
                if a not in by_lab or b not in by_lab:
                    continue

                # consider top-k candidates per label for speed/diversity
                cand_a = sorted(by_lab[a], key=lambda x: x["score"], reverse=True)[:6]
                cand_b = sorted(by_lab[b], key=lambda x: x["score"], reverse=True)[:6]

                best_ex = None
                best_score = -1e9

                for ma in cand_a:
                    s1, e1 = ma["start"], ma["end"]
                    for mb in cand_b:
                        s2, e2 = mb["start"], mb["end"]

                        # IoU gate
                        iou = _iou(s1, e1, s2, e2)
                        if iou < IOU_MIN or iou > IOU_MAX:
                            continue

                        # reject identical substrings for ANY pair (not just Action/Effect)
                        if _identical_substring(ma["text"], s1, e1, s2, e2):
                            continue

                        # tight union window + short pad, then clamp to CONFLICT_CAP
                        base_left = max(0, min(s1, s2) - PAD_CONFLICT)
                        base_right = min(len(ma["text"]), max(e1, e2) + PAD_CONFLICT)
                        left, right = _clamp_window(
                            base_left, base_right, CONFLICT_CAP, len(ma["text"])
                        )
                        snippet = (ma["text"][left:right]).strip()
                        off_a_s, off_a_e = s1 - left, e1 - left
                        off_b_s, off_b_e = s2 - left, e2 - left

                        spans_pair = [
                            {"label": a, "start": off_a_s, "end": off_a_e},
                            {"label": b, "start": off_b_s, "end": off_b_e},
                        ]

                        ex = {
                            "doc_id": doc_id,
                            "text": snippet,
                            "spans": spans_pair,
                            "meta": {
                                "reason": f"ambiguous_pair_{a}_{b}",
                                "subreddit": r.get("subreddit"),
                                "iou": round(iou, 3),
                            },
                        }

                        # window quality gate (short & focused)
                        if _window_quality_for_conflict(snippet, spans_pair) < 1.0:
                            continue

                        if not _example_is_valid(ex):
                            continue

                        # score: prefer higher base scores + tighter union
                        union_len = max(e1, e2) - min(s1, s2)
                        score = ma["score"] + mb["score"] - 0.01 * union_len

                        if score > best_score:
                            best_score = score
                            best_ex = ex

                if best_ex is not None:
                    # avoid many from same doc+pair
                    seen_key = (doc_id, key_pair[0], key_pair[1])
                    if seen_key not in seen_docs_for_pair:
                        picked.append(best_ex)
                        per_pair_added[key_pair] += 1
                        seen_docs_for_pair.add(seen_key)

    # 4) Per-label balance & diversity guard (C)
    def _reason(ex):
        return (ex.get("meta") or {}).get("reason", "")

    # Build fast lookup pools for CLEAN backfills (not ambiguous, low overlap)
    priors = (policy or {}).get("priors", {}) if isinstance(policy, dict) else {}
    pools_by_label: dict[str, list[tuple[float, dict]]] = {lab: [] for lab in LABELS}
    used_hashes = {_snippet_hash(ex.get("text", "")) for ex in picked if ex.get("text")}

    for _, r in cands_df.iterrows():
        lab = r["label"]
        if lab not in pools_by_label:
            continue
        text = r["text"]
        s = int(r["start"])
        e = int(r["end"])
        # avoid building exact duplicates
        pad_local = 90
        ls = max(0, s - pad_local)
        rs = min(len(text), e + pad_local)
        snippet, ls, rs, ns, ne = _make_snippet(
            text, s, e, pad=CLEAN_PAD, cap=CLEAN_CAP
        )
        if _snippet_hash(snippet) in used_hashes:
            continue
        # prefer clean single-label span windows
        if not _span_boundary_ok(snippet, ns, ne):
            continue
        span = {"label": lab, "start": ns, "end": ne}
        ex = {
            "doc_id": r["doc_id"],
            "text": snippet,
            "spans": [span],
            "meta": {
                "reason": "prototype_clean_backfill",
                "subreddit": r.get("subreddit"),
            },
        }
        # must satisfy single-span validator
        if not _example_is_valid_single(ex):
            continue
        # quality score (same as prototypes)
        q = _span_quality(lab, s, e, text, priors)
        if q < 0:
            continue
        pools_by_label[lab].append((q, ex))

    # sort pools by descending quality
    for lab in LABELS:
        pools_by_label[lab].sort(key=lambda x: x[0], reverse=True)

    # Helpers for current counts
    def _count_for_label(label: str, items: list[dict]) -> int:
        return sum(1 for ex in items if label in _labels_in_example(ex))

    def _count_ambig_for_label(label: str, items: list[dict]) -> int:
        return sum(
            1
            for ex in items
            if _is_ambiguous_reason(ex) and label in _labels_in_example(ex)
        )

    # Subreddit diversity counter
    sublab_counts = defaultdict(int)
    for ex in picked:
        subs = _example_subreddit(ex)
        for lab in _labels_in_example(ex):
            sublab_counts[(subs, lab)] += 1

    # Ensure per-label structure: >=2 prototypes, <=1 ambiguous, total == max_per_label
    max_per_label = (
        S1_PER_LABEL_DEFAULT if not isinstance(max_per_label, int) else max_per_label
    )

    # 4a) If ambiguous > MAX_AMBIG_PER_LABEL for any label, prune excess (keep higher-scored if available)
    #     Priority keep-order: prototype_clean > per_label_top > ambiguous_pair
    def _priority_rank(ex):
        r = _reason(ex)
        if r.startswith("prototype_clean"):
            return 3
        if r == "per_label_top":
            return 2
        if r.startswith("ambiguous_pair_"):
            return 1
        return 0

    # Prune over-ambiguous per label
    for lab in LABELS:
        amb = [
            (i, ex)
            for i, ex in enumerate(picked)
            if _is_ambiguous_reason(ex) and lab in _labels_in_example(ex)
        ]
        if len(amb) > MAX_AMBIG_PER_LABEL:
            # remove lowest priority among ambiguous (any order; they’re equal)
            to_remove = len(amb) - MAX_AMBIG_PER_LABEL
            for idx, ex in amb[::-1][:to_remove]:
                picked.pop(idx)

    # 4b) Backfill to ensure >= MIN_PROTOTYPES_PER_LABEL and total == max_per_label
    def _try_add_clean(label: str):
        # pull from pool respecting subreddit cap & dedup
        pool = pools_by_label.get(label, [])
        for qi, (q, ex) in enumerate(pool):
            h = _snippet_hash(ex.get("text", ""))
            if h in used_hashes:
                continue
            subs = _example_subreddit(ex)
            if sublab_counts[(subs, label)] >= MAX_SUBREDDIT_REPEAT:
                continue
            picked.append(ex)
            used_hashes.add(h)
            sublab_counts[(subs, label)] += 1
            pool.pop(qi)
            return True
        return False

    # Enforce prototypes ≥ MIN and totals == max_per_label
    for lab in LABELS:
        # count prototypes already present
        def _is_proto(ex):
            return _reason(ex).startswith("prototype_clean")

        have_proto = sum(
            1 for ex in picked if _is_proto(ex) and lab in _labels_in_example(ex)
        )
        # backfill prototypes first
        while have_proto < MIN_PROTOTYPES_PER_LABEL:
            if not _try_add_clean(lab):
                break
            have_proto += 1

        # now fill up to max_per_label
        while _count_for_label(lab, picked) < max_per_label:
            if not _try_add_clean(lab):
                break

        # if we still exceed ambiguous cap (because conflicts added both labels),
        # we already pruned in 4a, so nothing else to do here.

    # 4c) If any label exceeds max_per_label, trim lowest-priority for that label
    for lab in LABELS:
        while _count_for_label(lab, picked) > max_per_label:
            # remove lowest-priority item contributing this label
            idx_to_remove = None
            worst_rank = 1e9
            for i, ex in enumerate(picked):
                if lab not in _labels_in_example(ex):
                    continue
                rank = _priority_rank(ex)
                # prefer removing ambiguous first, then per_label_top; never remove prototypes if avoidable
                if rank < worst_rank:
                    worst_rank = rank
                    idx_to_remove = i
            if idx_to_remove is None:
                break
            picked.pop(idx_to_remove)

    # 4c.1) Ensure 1–2 negatives at the FRONT (don’t count toward label quotas)
    neg_needed = S1_NEG_MIN
    neg_limit = S1_NEG_MAX

    front_neg = []
    neg_added = 0

    # reuse hashes from already-picked examples
    used_hashes = {_snippet_hash(ex.get("text", "")) for ex in picked if ex.get("text")}

    # Pass 1: add up to neg_limit unique, high-quality negatives
    for q, cand in scored_neg:  # scored_neg is a list of (score, ex) sorted desc
        if neg_added >= neg_limit:
            break
        t = cand.get("text", "")
        if not t:
            continue
        h = _snippet_hash(t)
        if h in used_hashes:
            continue
        front_neg.append(cand)
        used_hashes.add(h)
        neg_added += 1

    # Pass 2: if we didn’t meet the minimum, keep scanning the pool to hit neg_needed
    if neg_added < neg_needed:
        for q, cand in scored_neg:
            if neg_added >= neg_needed:
                break
            t = cand.get("text", "")
            if not t:
                continue
            h = _snippet_hash(t)
            if h in used_hashes:
                continue
            front_neg.append(cand)
            used_hashes.add(h)
            neg_added += 1

    # Prepend if we found any; otherwise leave as-is (no safe unique negatives available)
    if front_neg:
        picked = front_neg + picked

    # Safety net: drop any weak Evidence that slipped through
    _picked_clean = []
    for ex in picked:
        ok = True
        for m in ex.get("spans", []):
            if m.get("label") == "Evidence":
                s, e = int(m["start"]), int(m["end"])
                # Convert local offsets to original snippet text coordinates (already local here)
                if _is_weak_evidence_span(ex["text"], s, e, _idf, _idf_cutoff):
                    ok = False
                    break
        if ok:
            _picked_clean.append(ex)
    picked = _picked_clean

    # 4d) Final small de-duplication by text (defensive)
    seen_texts = set()
    out = []
    for ex in picked:
        t = ex.get("text", "")
        if not t or t in seen_texts:
            continue
        seen_texts.add(t)
        out.append(ex)

    return out


def pick_s1_for_label(
    df_lab: pd.DataFrame,
    k=2,
    outlier_k=1,
    priors: dict | None = None,
    subreddit_diversity=True,
):
    if df_lab.empty:
        return []
    topk = df_lab.sort_values("score", ascending=False)
    chosen = []
    if subreddit_diversity and "subreddit" in df_lab.columns:
        seen = set()
        for _, row in topk.iterrows():
            if row["subreddit"] in seen:
                continue
            chosen.append(row.to_dict())
            seen.add(row["subreddit"])
            if len(chosen) >= k:
                break
    if len(chosen) < k:
        chosen = topk.head(k).to_dict(orient="records")

    # add outliers (furthest from priors)
    def prior_z_len(label, span_len):
        return prior_len_z(priors or {}, label, span_len)

    def prior_dist_pos(label, start_pos):
        return prior_pos_dist(priors or {}, label, start_pos)

    df_lab = df_lab.copy()
    df_lab["z_len"] = df_lab.apply(
        lambda r: prior_z_len(r["label"], r["end"] - r["start"]), axis=1
    )
    df_lab["pos_dist"] = df_lab.apply(
        lambda r: prior_dist_pos(r["label"], r["start"] / max(1, len(r["text"]))),
        axis=1,
    )
    outliers = df_lab.sort_values(["z_len", "pos_dist"], ascending=False).head(
        outlier_k
    )
    chosen += outliers.to_dict(orient="records")
    return chosen[: k + outlier_k]


def make_snippet(item: dict, pad=120) -> dict:
    """Crop around span, normalize start/end to snippet space."""
    t = item["text"]
    s, e = int(item["start"]), int(item["end"])
    left = max(0, s - pad)
    right = min(len(t), e + pad)
    snippet = (t[left:s] + t[s:e] + t[e:right]).strip()
    new_start = len(t[left:s])
    new_end = new_start + (e - s)
    ex = {
        "doc_id": item["doc_id"],
        "text": snippet,
        "spans": [{"label": item["label"], "start": new_start, "end": new_end}],
        "meta": {
            "source_window": [left, right],
            "subreddit": r.get("subreddit"),
            "reason": "prior_closeness+ambiguous_pair_bonus",
        },
    }
    return ex if _example_is_valid(ex) else {}


def compute_prior_features(priors, label, start, end, text_len):
    length = max(1, int(end) - int(start))
    start_pos = int(start) / max(1, int(text_len))
    return {
        "len": int(length),
        "start_pos": float(start_pos),
        "z_len": float(prior_len_z(priors, label, length)),
        "pos_dist": float(prior_pos_dist(priors, label, start_pos)),
    }


def detect_boundary_hit(boundary_ctx, label, text, start, end):
    if not boundary_ctx:
        return {"hit": False, "cues": []}
    window = (text or "").lower()[max(0, start - 50) : min(len(text), end + 50)]
    cues = []
    for key in ("before_1w", "after_1w", "before_2w", "after_2w"):
        for c in (boundary_ctx.get(label, {}).get(key, []) or [])[:5]:
            if isinstance(c, str) and c.lower() in window:
                cues.append(c)
    return {"hit": len(cues) > 0, "cues": list(dict.fromkeys(cues))}


# --- S2 heuristic pickers ---
def _marker_density(mks):  # compact proxy: more markers -> more conspiratorial framing
    return 0 if not isinstance(mks, list) else min(10, len(mks))


def is_hedged_no(row_text: str, hedges) -> bool:
    s = (row_text or "").lower()
    return any(h in s for h in hedges)


def is_speculative_yes(row_text: str, absolutist) -> bool:
    s = (row_text or "").lower()
    return any(a in s for a in absolutist) and (
        "they" in s or "agenda" in s or "cover up" in s
    )


def is_debunk(text: str) -> bool:
    s = (text or "").lower()
    cues = [
        "debunk",
        "fact-check",
        "myth",
        "false claim",
        "no evidence",
        "according to",
        "the report shows",
        "the data show",
    ]
    conspiracese = [
        "agenda",
        "cover up",
        "cabal",
        "they are trying",
        "secret plan",
        "deep state",
    ]
    return any(c in s for c in cues) and not any(w in s for w in conspiracese)


def is_borderline_positive(text: str) -> bool:
    s = (text or "").lower()
    soft_cues = [
        "they",
        "agenda",
        "cover up",
        "hidden",
        "behind the scenes",
        "orchestrated",
        "weaponize",
        "deep state",
    ]
    hedges = ["maybe", "perhaps", "could be", "might be", "likely"]
    return any(c in s for c in soft_cues) and any(h in s for h in hedges)


def pick_s2_buckets(
    df_all,
    ABSOLUTIST,
    HEDGES,
    k_yes=3,
    k_no=3,
    k_hedged_no=2,
    k_spec_yes=2,
    k_debunk=2,  # NEW
    k_borderline_yes=2,  # NEW
    seed=42,
):
    # clear YES/NO by label
    yy = df_all[df_all["doc_label"] == "conspiracy"].copy()
    nn = df_all[df_all["doc_label"] == "non"].copy()

    # enrich with marker density if available
    if "markers" in df_all.columns:
        yy["_md"] = yy["markers"].apply(
            lambda m: 0 if not isinstance(m, list) else min(10, len(m))
        )
        nn["_md"] = nn["markers"].apply(
            lambda m: 0 if not isinstance(m, list) else min(10, len(m))
        )
        yy = yy.sort_values("_md", ascending=False)
        nn = nn.sort_values("_md", ascending=True)

    clear_yes = yy.head(k_yes).to_dict(orient="records")
    clear_no = nn.head(k_no).to_dict(orient="records")

    # hedged NO (non + hedges)
    hed_pool = nn[
        nn["text"].apply(lambda t: any(h in (t or "").lower() for h in HEDGES))
    ]
    hedged_no = (
        hed_pool.sample(n=min(k_hedged_no, len(hed_pool)), random_state=seed).to_dict(
            orient="records"
        )
        if not hed_pool.empty
        else []
    )

    # speculative YES (conspiracy + absolutist language)
    spec_pool = yy[
        yy["text"].apply(lambda t: any(a in (t or "").lower() for a in ABSOLUTIST))
    ]
    spec_yes = (
        spec_pool.sample(n=min(k_spec_yes, len(spec_pool)), random_state=seed).to_dict(
            orient="records"
        )
        if not spec_pool.empty
        else []
    )

    # NEW: debunk pool (non + debunk cues)
    deb_pool = nn[nn["text"].apply(is_debunk)]
    debunk_list = (
        deb_pool.sample(n=min(k_debunk, len(deb_pool)), random_state=seed).to_dict(
            orient="records"
        )
        if not deb_pool.empty
        else []
    )

    # NEW: borderline positive (soft conspiracy with hedges)
    bl_pool = yy[yy["text"].apply(is_borderline_positive)]
    borderline_yes_list = (
        bl_pool.sample(
            n=min(k_borderline_yes, len(bl_pool)), random_state=seed
        ).to_dict(orient="records")
        if not bl_pool.empty
        else []
    )

    return clear_yes, clear_no, hedged_no, spec_yes, debunk_list, borderline_yes_list


def pick_hard_S2(hard_df, df_all, max_borderline=3, max_misleading=3):
    if hard_df is None or hard_df.empty:
        return []
    J = hard_df.copy()
    J["doc_id"] = J["doc_id"].astype(str)
    right = df_all.set_index("doc_id")[["text", "doc_label", "subreddit"]]
    common = (set(J.columns) & set(right.columns)) - {"doc_id"}
    if common:
        right = right.drop(columns=list(common))
    J = J.merge(right, on="doc_id", how="left")
    J = J[J["doc_label"].isin(["conspiracy", "non"])]

    borderline = J[
        J["reasons"].apply(lambda rs: any("High" in r or "Entropy" in r for r in rs))
    ].head(max_borderline)
    misleading = J[
        J["reasons"].apply(lambda rs: any("Baseline Confident Error" in r for r in rs))
    ].head(max_misleading)

    out = []
    for _, r in borderline.iterrows():
        out.append(
            {
                "doc_id": r["doc_id"],
                "text": r["text"],
                "label": r["doc_label"],
                "rationale": "Borderline framing; avoid over-reading ambiguity.",
            }
        )
    for _, r in misleading.iterrows():
        # keep the gold label but make rationale explicit
        lab = r["doc_label"]
        rat = (
            "Speculative framing asserted as fact."
            if lab == "conspiracy"
            else "Non-conspiratorial despite suggestive phrasing."
        )
        out.append(
            {"doc_id": r["doc_id"], "text": r["text"], "label": lab, "rationale": rat}
        )
    return out


# ===================== HIGH-VALUE S1 FEWSHOT BUILDER =====================

GOOD_EVIDENCE_CUES = (
    "according to",
    "as reported by",
    "said",
    "says",
    "stated",
    "report",
    "study",
    "paper",
    "data",
    "evidence",
    "leaked",
    "http://",
    "https://",
    "www.",
    "doi.org",
    "arxiv.org",
)


def _labels_of_ex(ex):
    return [
        m["label"] for m in ex.get("spans", []) if _valid_span(m, ex.get("text", ""))
    ]


def rebalance_s1_examples(
    examples: List[dict],
    cap_per_label: Dict[str, int] | None = None,
    min_per_label: int = 2,
    prefer_multilabel: bool = True,
) -> List[dict]:
    """
    Enforce per-label caps (e.g., cap Evidence) after backfill/boosters/balancing.
    Prefer to drop low-value examples first:
      1) single-span Evidence-only
      2) examples without Action/Effect
      3) otherwise any example that reduces the over-cap label without breaking min coverage
    """
    if not examples:
        return examples

    if cap_per_label is None:
        cap_per_label = {"Evidence": 8}  # sensible default for ~25 total few-shots

    # current counts
    def counts(exs):
        c = Counter()
        for ex in exs:
            for lab in _labels_of_ex(ex):
                c[lab] += 1
        return c

    exs = [ex for ex in examples if _example_is_valid(ex)]
    c = counts(exs)

    # If nothing is over cap, return early
    over = {lab: c[lab] - cap for lab, cap in cap_per_label.items() if c[lab] > cap}
    if not over:
        return exs

    # Rank examples by "drop-ability" score
    def drop_score(ex):
        labs = _labels_of_ex(ex)
        L = set(labs)
        n_spans = len(labs)

        # high score = easier to drop
        s = 0.0
        # Evidence-only single span: drop first
        if L == {"Evidence"} and n_spans == 1:
            s += 10.0
        # evidence-heavy (>= 2 evidence and no Action/Effect)
        if labs.count("Evidence") >= 2 and not (("Action" in L) or ("Effect" in L)):
            s += 6.0
        # lacks Action/Effect entirely
        if ("Action" not in L) and ("Effect" not in L):
            s += 3.0
        # prefer to keep multi-label examples if requested
        if prefer_multilabel and n_spans <= 1:
            s += 1.0
        # tiny tie-breaker: shorter texts easier to drop
        s += 0.001 * (len(ex.get("text", "")) <= 200)
        return s

    ranked = sorted(exs, key=drop_score, reverse=True)

    # Greedily drop while any label is above its cap, but never violate min_per_label
    kept = exs[:]
    for ex in ranked:
        c = counts(kept)
        # still over any caps?
        over_labels = [lab for lab, cap in cap_per_label.items() if c[lab] > cap]
        if not over_labels:
            break
        ex_labs = _labels_of_ex(ex)
        # if removing this example reduces some over-cap label without breaking mins for others, drop it
        would_break_min = any(
            (lab not in over_labels) and (c[lab] - ex_labs.count(lab) < min_per_label)
            for lab in set(ex_labs)
        )
        reduces_over = any(
            (lab in over_labels) and (ex_labs.count(lab) > 0) for lab in set(ex_labs)
        )
        if reduces_over and not would_break_min:
            kept.remove(ex)

    return kept


def _is_good_evidence_span(text: str, s: int, e: int) -> bool:
    """Keep only Evidence that looks like citation/quote/attribution/URL/numbered fact."""
    snip = (text or "")[max(0, s) : max(0, e)].strip().lower()
    if len(snip) < 8:
        return False
    if any(k in snip for k in GOOD_EVIDENCE_CUES):
        return True
    # quote or number-heavy signals
    if any(q in snip for q in ['"', "“", "”", "'", "’"]) and len(snip.split()) >= 2:
        return True
    # light numeric density (e.g., percentages, years) can be evidential if >= 2 tokens
    nums = sum(t.isdigit() for t in snip.replace("%", " ").split())
    return nums >= 2 and len(snip.split()) >= 4


def _no_identical_crosslabel(spans: List[dict]) -> bool:
    """Reject examples where different labels share exact same [start,end) window."""
    seen = {}
    for m in spans:
        key = (int(m["start"]), int(m["end"]))
        lab = m.get("label")
        if key in seen and seen[key] != lab:
            return False
        seen[key] = lab
    return True


def _tight_minimal_span_ok(text: str, s: int, e: int) -> bool:
    """Tightness/quality gate for all labels."""
    if e - s < 4:
        return False
    snip = (text or "")[s:e]
    if not snip or not snip.strip():
        return False
    # ban pure punctuation / leading articles unless needed
    if snip.strip().lower() in {"the", "a", "an"}:
        return False
    if all(ch in " \t\n\r.,;:!?“”’'\"-()[]{}" for ch in snip):
        return False
    return True


def _normalize_and_filter_doc_markers(row: dict) -> List[dict]:
    """Coerce markers from a train/dev row into clean S1 spans with stronger gates."""
    text = row.get("text") or ""
    out = []
    for m in row.get("markers") or []:
        lab = m.get("label")
        if lab not in ALLOWED_S1:
            continue
        try:
            s = int(m.get("start"))
            e = int(m.get("end"))
        except Exception:
            continue
        # core gates
        if not _tight_minimal_span_ok(text, s, e):
            continue
        if lab == "Evidence" and not _is_good_evidence_span(text, s, e):
            continue
        out.append({"label": lab, "start": s, "end": e})
    return out


def _score_doc_for_fewshot(markers: List[dict], text_len: int) -> Tuple[float, dict]:
    """
    Score doc desirability:
      + Action&Effect present and distinct
      + Actor&Victim co-mention
      + more labels up to 4 (diminishing returns)
    Penalize identical cross-label spans.
    """
    labs = [m["label"] for m in markers]
    L = set(labs)
    score, detail = 0.0, {}

    # base coverage
    score += min(4, len(L)) * 1.0
    detail["label_types"] = len(L)

    # Action & Effect split and distinct windows
    has_a = any(m["label"] == "Action" for m in markers)
    has_e = any(m["label"] == "Effect" for m in markers)
    if has_a and has_e:
        # distinctness bonus if IoU < 0.6 on at least one A/E pair
        ae_distinct = False
        A = [m for m in markers if m["label"] == "Action"]
        E = [m for m in markers if m["label"] == "Effect"]
        for a in A:
            for b in E:
                inter = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
                union = (a["end"] - a["start"]) + (b["end"] - b["start"]) - inter
                iou = (inter / union) if union > 0 else 0.0
                if iou < 0.60:  # clearly split
                    ae_distinct = True
                    break
            if ae_distinct:
                break
        score += 2.0 if ae_distinct else 1.0
        detail["action_effect_split"] = ae_distinct

    # Actor & Victim co-mention bonus
    if ("Actor" in L) and ("Victim" in L):
        score += 1.0
        detail["actor_victim_comention"] = True

    # evidence bonus (only if already multi-label)
    if "Evidence" in L and len(L) >= 3:
        score += 0.5

    # penalize identical cross-label spans
    if not _no_identical_crosslabel(markers):
        score -= 2.0
        detail["identical_crosslabel_penalty"] = True

    # mild length prior: prefer snippets not covering whole doc
    score += 0.2 * min(1.0, text_len / 800.0)
    return float(score), detail


def _window_around_labels(
    text: str, spans: List[dict], pad: int = 120
) -> Tuple[str, List[dict]]:
    """Create a snippet covering the union of chosen spans; renormalize start/end to snippet space."""
    if not spans:
        return "", []
    L = len(text)
    left = min(m["start"] for m in spans)
    right = max(m["end"] for m in spans)
    left = max(0, left - pad)
    right = min(L, right + pad)
    snippet = (text[left:right]).strip()
    out = []
    for m in spans:
        out.append(
            {
                "label": m["label"],
                "start": int(m["start"] - left),
                "end": int(m["end"] - left),
            }
        )
    return snippet, out


def build_high_value_s1_examples(
    train_df: pd.DataFrame,
    max_total: int = 18,
    min_per_label: int = 2,
    seed: int = 42,
) -> List[dict]:
    """
    Build a compact bank of few-shots where each example:
      - has ≥ 2 labels (Action required), and ideally includes Action+Effect split
      - Evidence spans are gated to citations/quotes/attributions
      - renormalized offsets within a short snippet window
      - no identical spans across labels
    Also ensures per-label coverage (≥ min_per_label) and caps total to max_total.
    """
    rng = np.random.default_rng(seed)
    scored = []
    for _, row in train_df.iterrows():
        text = row.get("text") or ""
        marks = _normalize_and_filter_doc_markers(row)
        if not marks:
            continue
        # need at least 2 labels and must include Action
        labs = {m["label"] for m in marks}
        if len(labs) < 2 or ("Action" not in labs):
            continue
        # local distinctness filter: keep at most 3 per label, prefer longer spans
        by_lab = defaultdict(list)
        for m in marks:
            by_lab[m["label"]].append(m)
        for lab in by_lab:
            by_lab[lab] = sorted(
                by_lab[lab], key=lambda x: (x["end"] - x["start"]), reverse=True
            )[:3]
        kept = [m for arr in by_lab.values() for m in arr]
        # score doc desirability
        sc, detail = _score_doc_for_fewshot(kept, len(text))
        if sc <= 0:
            continue
        snippet, spans_norm = _window_around_labels(text, kept, pad=120)
        # final gates on the snippet example
        if not snippet or not spans_norm:
            continue
        if not _no_identical_crosslabel(spans_norm):
            continue
        scored.append(
            {
                "doc_id": row.get("doc_id"),
                "text": snippet,
                "spans": spans_norm,
                "_score": sc,
                "_labs": {m["label"] for m in spans_norm},
                "_detail": detail,
            }
        )

    if not scored:
        return []

    # Diversity: prefer high score, then promote examples containing conflict pairs
    def _key(ex):
        bonus = 0.0
        labs = ex["_labs"]
        if {"Action", "Effect"} <= labs:
            bonus += 0.5
        if {"Actor", "Victim"} <= labs:
            bonus += 0.3
        return -(ex["_score"] + bonus)

    scored.sort(key=_key)

    # Greedy pick with per-label coverage
    picked, label_counts = [], Counter()
    for ex in scored:
        # accept if it helps fill coverage or we still have room
        helps = any(label_counts[l] < min_per_label for l in ex["_labs"])
        if helps or len(picked) < max_total:
            picked.append(ex)
            for l in ex["_labs"]:
                label_counts[l] += 1
        if len(picked) >= max_total and all(
            label_counts[l] >= min_per_label for l in ALLOWED_S1
        ):
            break

    # trim to max_total
    picked = picked[:max_total]
    # strip helper keys
    for ex in picked:
        ex.pop("_score", None)
        ex.pop("_labs", None)
        ex.pop("_detail", None)
    return picked


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--derived-root", default="data/derived")
    ap.add_argument("--latest-pointer", default="data/derived/psycomark_latest.txt")
    ap.add_argument("--shots-s2-per-class", type=int, default=16)
    ap.add_argument("--shots-s1-per-label", type=int, default=16)
    ap.add_argument("--shots-s1-outliers", type=int, default=6)
    ap.add_argument(
        "--s1-min-evidence",
        type=int,
        default=3,
        help="Minimum number of Evidence few-shots to include.",
    )
    ap.add_argument(
        "--s1-min-victim",
        type=int,
        default=2,
        help="Minimum number of Victim few-shots to include.",
    )
    ap.add_argument(
        "--s1-include-empty-negative",
        action="store_true",
        help="If set, include a negative S1 exemplar with empty spans.",
    )
    ap.add_argument(
        "--max-n-fewshot",
        type=int,
        default=30,
        help="Max number of S1 fewshot examples to include (after balancing)",
    )
    ap.add_argument(
        "--cant-tell-negs",
        type=int,
        default=2,
        help="How many cant_tell to inject as negative S2 (label=non).",
    )
    ap.add_argument("--cant-tell-rationale", default=CANT_TELL_RATIONALE_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--s2-thresh",
        default="auto",
        help="Probability threshold for 'Yes'. Float in [0,1] or 'auto' to tune on dev.",
    )

    ap.add_argument(
        "--preserve-existing-s1",
        action="store_true",
        default=False,
        help="If set, keep existing S1 exemplars and only update S2.",
    )
    args = ap.parse_args()
    min_needed = args.shots_s1_per_label * len(LABELS)
    if args.max_n_fewshot < min_needed:
        print(
            f"[S1] max_n_fewshot too small ({args.max_n_fewshot}) for per-label coverage; "
            f"bumping to {min_needed}."
        )
        args.max_n_fewshot = min_needed
    max_n = args.max_n_fewshot  # keep using this local
    max_n = args.max_n_fewshot

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = find_latest_dir(Path(args.latest_pointer))
    ABSOLUTIST, HEDGES = load_lexicons(out_dir)
    print(f"--- Using latest derived run: {out_dir.name} ---")

    # Load data
    train_df = pd.read_json(out_dir / "train.jsonl", lines=True)
    dev_df = pd.read_json(out_dir / "dev.jsonl", lines=True)
    train_df = alias_ids(train_df)
    dev_df = alias_ids(dev_df)

    # Pools / convenience views
    df_all = pd.concat([train_df, dev_df], ignore_index=True).copy()
    # print df_all columns
    print(f"Data loaded: train={len(train_df)}, dev={len(dev_df)}, total={len(df_all)}")
    print(f"Data columns: {df_all.columns.tolist()}")

    # Load optional stats
    priors = load_json(out_dir / "length_position_priors.json", default={})
    pair_stats = load_json(out_dir / "overlap_pair_stats_ci.json", default={})
    if not pair_stats:
        pair_stats = load_json(out_dir / "overlap_pair_stats.json", default={})
    boundary = load_json(out_dir / "boundary_context.json", default={})

    # Pick top ambiguous pairs (fallback to Action/Effect & Actor/Victim)
    pairs = top_pairs(pair_stats, key="iou@0.5", topk=2)
    if not pairs:
        pairs = [("Action", "Effect"), ("Actor", "Victim")]

    # Build span candidates with prior/overlap shaping
    cands = build_s1_candidates(train_df=train_df, priors=priors, target_pairs=pairs)

    # Hard examples (optional)
    hard_path = out_dir / "hard_examples.json"
    hard_df = pd.read_json(hard_path) if hard_path.exists() else pd.DataFrame()

    # Existing fewshots (optional)
    fs_path = out_dir / "best_fewshot_examples.json"
    existing = load_json(fs_path, default={"s1": [], "s2": []})

    # ----------------- S2 selection (binary only) -----------------
    # Build S2 pool (exclude cant_tell)
    pool_s2 = df_all[df_all["doc_label"].isin(["conspiracy", "non"])].copy()

    # Prefer hard examples if available; else use whole pool
    if not hard_df.empty:
        # Ensure doc_id is string on both sides
        hard_df["doc_id"] = hard_df["doc_id"].astype(str)
        df_all["doc_id"] = df_all["doc_id"].astype(str)

        # Keep the 'text' already in hard_df; only join NEW cols
        right = df_all.set_index("doc_id")[["doc_label", "subreddit"]]
        base = hard_df.set_index("doc_id").join(right, how="left").reset_index()

        # If any hard rows lack text, fill from df_all
        if "text" not in base.columns:
            base["text"] = ""
        text_map = df_all.set_index("doc_id")["text"].astype(str).to_dict()
        need_text = base["text"].isna() | (base["text"].astype(str).str.len() == 0)
        base.loc[need_text, "text"] = (
            base.loc[need_text, "doc_id"].map(text_map).fillna("")
        )

        # Keep only binary labels
        base = base[base["doc_label"].isin(["conspiracy", "non"])].copy()
        pool_for_pick = base
    else:
        pool_for_pick = pool_s2

    print(f"[S2] candidate pool (binary): {len(pool_for_pick)}")

    # Balanced buckets: clear Yes/No + hedged No + speculative Yes
    cy, cn, hed_no, spec_yes, debunk_list, borderline_yes_list = pick_s2_buckets(
        df_all,
        k_yes=3,
        k_no=3,
        k_hedged_no=2,
        k_spec_yes=2,
        k_debunk=2,
        k_borderline_yes=2,
        seed=args.seed,
        ABSOLUTIST=ABSOLUTIST,
        HEDGES=HEDGES,
    )

    def _mk(item, label_override=None, rationale=""):
        return coerce_s2_item(item, label_override=label_override, rationale=rationale)

    s2_main = []
    s2_main += [
        _mk(
            r, rationale="Explicit claim of covert actors/actions with supporting cues."
        )
        for r in cy
    ]
    s2_main += [
        _mk(r, rationale="Statement is non-conspiratorial and descriptive.") for r in cn
    ]
    s2_main += [
        _mk(
            r,
            label_override="non",
            rationale="Hedged/uncertain language without endorsement.",
        )
        for r in hed_no
    ]
    s2_main += [
        _mk(
            r,
            label_override="conspiracy",
            rationale="Speculative assertion framed as fact with absolutist cues.",
        )
        for r in spec_yes
    ]

    # NEW: debunk & borderline
    s2_main += [
        _mk(r, label_override="non", rationale="Debunking/neutral fact-based critique.")
        for r in debunk_list
    ]
    s2_main += [
        _mk(
            r,
            label_override="conspiracy",
            rationale="Borderline positive: hidden-agent cues with hedges.",
        )
        for r in borderline_yes_list
    ]

    s2_hard = pick_hard_S2(hard_df, df_all, max_borderline=2, max_misleading=2)
    s2_main += s2_hard

    # Optional: inject cant_tell as negative with rationale
    s2_ct = []
    if args.cant_tell_negs > 0:
        all_raw = df_all[["doc_id", "text", "doc_label", "subreddit"]].copy()
        ct_pool = all_raw[all_raw["doc_label"] == "cant_tell"].copy()
        if not ct_pool.empty:
            rng = np.random.default_rng(args.seed)
            ct_pool["_rand"] = rng.random(size=len(ct_pool))
            order = (
                ["subreddit", "_rand"] if "subreddit" in ct_pool.columns else ["_rand"]
            )
            for _, r in ct_pool.sort_values(order).head(args.cant_tell_negs).iterrows():
                s2_ct.append(
                    coerce_s2_item(
                        r,
                        label_override="non",
                        rationale=args.cant_tell_rationale,
                        tag="cant_tell",
                    )
                )
    # Freeze policy
    assert all(
        x["label"] in {"conspiracy", "non"} for x in (s2_main + s2_ct)
    ), "Policy breach: S2 must be binary."
    assert (
        CANT_TELL_IN_S2 is False
    ), "Policy breach: cant_tell cannot appear as S2 label."

    s2_fewshots = s2_main + s2_ct

    # attach compact markers to S2 fewshots so the S2 prompt examples mirror runtime conditioning
    def attach_markers(ex):
        did = ex["doc_id"]
        row = df_all[df_all["doc_id"] == did].head(1)
        mks = (
            markers_compact(row.iloc[0].get("markers", []), max_per_label=2)
            if not row.empty
            else []
        )
        ex["markers"] = mks
        return ex

    s2_fewshots = [attach_markers(x) for x in (s2_main + s2_ct)]
    s2_fewshots = enforce_min_yes_fewshots(
        s2_fewshots, min_yes=6, df_all=df_all, seed=args.seed
    )

    # ----------------- S1 selection (spans) -----------------
    # Build conflict targets and candidates first
    pairs = top_pairs(pair_stats, key="iou@0.5", topk=2)
    cands = build_s1_candidates(
        train_df=train_df,
        priors=priors,
        target_pairs=pairs,
    )

    if args.preserve_existing_s1 and existing.get("s1"):
        s1_examples = existing["s1"]
        print(f"[S1] Preserving existing S1 few-shots: {len(s1_examples)}")
    else:
        # NEW: selector over candidates (per-label + ambiguous-pair exemplars)
        s1_examples = build_s1_fewshots(
            cands_df=cands,
            policy={"targets": {"ambiguous_pairs_top2": pairs}},
            max_per_label=args.shots_s1_per_label,  # e.g., 4
            max_per_pair=3,  # ensure conflict coverage
        )

    # --- NEW: prune low-signal S1 fewshots ---
    before = len(s1_examples)
    pruned = []
    dropped_dup = dropped_weak_ev = 0
    for ex in s1_examples:
        if _has_bad_cross_label_dup(ex):
            dropped_dup += 1
            continue
        full_text = ex.get("text") or ""
        if _example_has_only_weak_evidence(ex, full_text):
            dropped_weak_ev += 1
            continue
        pruned.append(ex)
    s1_examples = pruned
    print(
        f"[S1] pruned few-shots: dropped {dropped_dup} cross-label dup(s), "
        f"{dropped_weak_ev} weak-evidence example(s); kept={len(s1_examples)} / {before}"
    )

    # Limit how many ambiguous-pair exemplars we add per pair
    MAX_PER_AMBIG_PAIR = 4
    ambig_added = defaultdict(int)  # (label_a,label_b) sorted tuple -> count

    # Audit log (unchanged shape)
    print(
        "✅ Final S1 fewshot label counts:",
        dict(
            Counter(
                l
                for ex in s1_examples
                for l in [m["label"] for m in ex.get("spans", [])]
            )
        ),
    )

    # Add overlap exemplars for top ambiguous pairs (Action/Effect, Actor/Victim)
    if pairs and not cands.empty:
        for a, b in pairs:
            df_ab = cands[cands["label"].isin([a, b])].copy()
            if df_ab.empty:
                continue
            seen_docs = set()
            for doc_id, g in df_ab.groupby("doc_id"):
                if doc_id in seen_docs:
                    continue
                g = g.sort_values(["start", "end"])
                spans = g.to_dict(orient="records")
                ok = False
                for i in range(len(spans)):
                    for j in range(i + 1, len(spans)):
                        if spans[i]["label"] == spans[j]["label"]:
                            continue
                        s1_, e1_ = int(spans[i]["start"]), int(spans[i]["end"])
                        s2_, e2_ = int(spans[j]["start"]), int(spans[j]["end"])
                        inter = max(0, min(e1_, e2_) - max(s1_, s2_))
                        union = (e1_ - s1_) + (e2_ - s2_) - inter
                        iou = (inter / union) if union > 0 else 0.0
                        if iou >= 0.15 and abs(max(e1_, e2_) - min(s1_, s2_)) <= 320:
                            t = spans[i]["text"]
                            left = max(0, min(s1_, s2_) - 120)
                            right = min(len(t), max(e1_, e2_) + 120)
                            snippet = (t[left:right]).strip()
                            off1s, off1e = s1_ - left, e1_ - left
                            off2s, off2e = s2_ - left, e2_ - left
                            spans_pair = [
                                {
                                    "label": spans[i]["label"],
                                    "start": off1s,
                                    "end": off1e,
                                },
                                {
                                    "label": spans[j]["label"],
                                    "start": off2s,
                                    "end": off2e,
                                },
                            ]
                            # Skip if Action & Effect are identical substrings
                            if {a, b} == {
                                "Action",
                                "Effect",
                            } and _has_identical_action_effect(spans_pair, snippet):
                                continue
                            key = tuple(sorted([a, b]))
                            if ambig_added[key] >= MAX_PER_AMBIG_PAIR:
                                break
                            ex = {
                                "doc_id": doc_id,
                                "text": snippet,
                                "spans": spans_pair,
                                "meta": {
                                    "reason": f"ambiguous_pair_{a}_{b}",
                                    "subreddit": r.get("subreddit"),
                                },
                            }
                            if _example_is_valid(ex):
                                s1_examples.append(ex)
                                ambig_added[key] += 1
                                seen_docs.add(doc_id)
                                ok = True
                                break
                    if ok:
                        break

    # --- NEW: drop S1 few-shots that have no valid spans (or junk spans) ---
    _before = len(s1_examples)
    s1_examples = [ex for ex in s1_examples if _example_has_valid_span(ex)]
    print(
        f"[S1] filtered few-shots: kept={len(s1_examples)} dropped={_before - len(s1_examples)} (invalid/empty)"
    )

    # --- NEW: prune low-signal S1 fewshots ---
    before = len(s1_examples)
    pruned = []
    dropped_dup = dropped_weak_ev = 0
    for ex in s1_examples:
        # 3a) drop cross-label identical spans (except Evidence overlaps)
        if _has_bad_cross_label_dup(ex):
            dropped_dup += 1
            continue
        # 3b) drop examples whose Evidence is *only* weak placeholders
        #     (if the example has Evidence spans and none are strong)
        full_text = ex.get("text") or ""
        if _example_has_only_weak_evidence(ex, full_text):
            dropped_weak_ev += 1
            continue
        pruned.append(ex)
    s1_examples = pruned
    print(
        f"[S1] pruned few-shots: dropped {dropped_dup} cross-label dup(s), "
        f"{dropped_weak_ev} weak-evidence example(s); kept={len(s1_examples)} / {before}"
    )

    # ---- Build audit log for reproducibility ----
    audit = {
        "seed": args.seed,
        "shots": {
            "s1_per_label": args.shots_s1_per_label,
            "s1_outliers_per_label": args.shots_s1_outliers,
            "s2_per_class": args.shots_s2_per_class,
        },
        "s1_examples": [],
        "s2_examples": [],
    }

    # S1 audit
    for ex in s1_examples:
        rec = {
            "doc_id": ex.get("doc_id"),
            "reason": ex.get("meta", {}).get("reason", ""),
            "text_len": len(ex.get("text", "")),
        }
        rec["spans"] = []
        for sp in ex.get("spans", []):
            if not _valid_span(sp, ex.get("text", "")):
                continue
            L = sp.get("label")
            s = int(sp.get("start", 0))
            e = int(sp.get("end", 0))
            pf = compute_prior_features(priors, L, s, e, rec["text_len"])
            b = detect_boundary_hit(boundary, L, ex.get("text", ""), s, e)
            rec["spans"].append(
                {
                    "label": L,
                    "start": s,
                    "end": e,
                    **pf,
                    "boundary_hit": b["hit"],
                    "boundary_cues": b["cues"],
                }
            )
        audit["s1_examples"].append(rec)

    # S2 audit
    for ex in s2_fewshots:
        audit["s2_examples"].append(
            {
                "doc_id": ex.get("doc_id"),
                "label": ex.get("label"),
                "markers_count": len(ex.get("markers", [])),
                "source_label": ex.get("source_label", ""),
            }
        )

    # Negative S1 exemplar (teaches: empty JSON is valid)
    if args.s1_include_empty_negative:
        neg_s1 = pick_negative_s1_snippet(dev_df, seed=args.seed)
        if neg_s1:
            s1_examples.append(neg_s1)

    # --- NEW: guarantee per-label coverage via relaxed backfill (Actor/Action gaps etc.) ---
    extras = backfill_missing_labels(
        train_df=train_df,
        s1_examples=s1_examples,
        max_per_label=args.shots_s1_per_label,  # usually 2
    )
    if extras:
        s1_examples.extend(extras)
        print(f"[S1] Backfill added {len(extras)} examples to meet per-label coverage.")

    # --- NEW: Ensure minimum Evidence & Victim coverage with quality filters ---
    def _count_label(exlist, lab):
        return sum(
            1 for ex in exlist for m in ex.get("spans", []) if m.get("label") == lab
        )

    need_evi = max(0, args.s1_min_evidence - _count_label(s1_examples, "Evidence"))
    need_vic = max(0, args.s1_min_victim - _count_label(s1_examples, "Victim"))

    if need_evi > 0:
        boosters = _pick_evidence_from_train(train_df, want=need_evi, seed=args.seed)
        # avoid pathological Action==Effect duplicates inside boosters (paranoia)
        boosters = [
            b
            for b in boosters
            if not _has_identical_action_effect(b.get("spans", []), b.get("text", ""))
        ]
        s1_examples.extend(boosters)

    if need_vic > 0:
        s1_examples.extend(
            _pick_victim_from_train(train_df, want=need_vic, seed=args.seed)
        )

    # --- Ensure all 5 S1 markers are covered at least 2x ---
    min_per_label = 2
    label_counter = Counter()
    for ex in s1_examples:
        for m in ex.get("spans", []):
            label = m.get("label")
            if label:
                label_counter[label] += 1

    # Log current status
    print("🔍 Fewshot label counts before trimming:", dict(label_counter))

    # Filter to balance if needed
    final_s1_examples = []
    used = set()
    per_label_buffer = {lab: [] for lab in LABELS}

    for ex in s1_examples:
        if not _example_is_valid(ex):
            continue
        added = False
        text_ex = ex.get("text", "")
        for m in ex.get("spans", []):
            if not _valid_span(m, text_ex):
                continue
            lab = m.get("label")
            if lab in ALLOWED_S1 and text_ex not in used:
                per_label_buffer[lab].append(ex)
                used.add(ex["text"])
                added = True
        if not added:
            final_s1_examples.append(ex)

    # Now collect balanced
    balanced = []
    # Always take up to shots_s1_per_label per label first
    for lab in LABELS:
        take = per_label_buffer[lab][: args.shots_s1_per_label]
        balanced.extend(take)

    # Then fill with remainder until max_n (if any room left)
    seen_texts = set(e["text"] for e in balanced)
    for ex in s1_examples:
        if len(balanced) >= max_n:
            break
        if ex["text"] not in seen_texts:
            balanced.append(ex)
            seen_texts.add(ex["text"])

    # Add remainder until max_n using round-robin with per-label cap
    OVERALL_CAP_PER_LABEL = math.ceil(
        args.max_n_fewshot / len(LABELS)
    )  # e.g., 4 if 20/5
    seen_texts = set(e["text"] for e in balanced)

    # Bucket remaining examples by their *first* valid span label
    buckets = {lab: [] for lab in LABELS}
    for ex in s1_examples:
        if ex.get("text", "") in seen_texts:
            continue
        text_ex = ex.get("text", "")
        # pick the first valid span's label to bucket
        lab_for_ex = None
        for m in ex.get("spans", []):
            if _valid_span(m, text_ex):
                lab_for_ex = m["label"]
                break
        if lab_for_ex in buckets:
            buckets[lab_for_ex].append(ex)

    def _labels_of_ex(ex):
        return [
            m["label"]
            for m in ex.get("spans", [])
            if _valid_span(m, ex.get("text", ""))
        ]

    counts = Counter(l for ex in balanced for l in _labels_of_ex(ex))

    # Round-robin add until we reach max_n, respecting per-label caps
    while len(balanced) < max_n:
        progressed = False
        for lab in LABELS:
            if counts[lab] >= OVERALL_CAP_PER_LABEL:
                continue
            while buckets[lab]:
                ex = buckets[lab].pop(0)
                if ex.get("text", "") in seen_texts:
                    continue
                balanced.append(ex)
                seen_texts.add(ex["text"])
                for l in _labels_of_ex(ex):
                    counts[l] += 1
                progressed = True
                break
            if len(balanced) >= max_n:
                break
        if not progressed:
            break

    # Overwrite
    s1_examples = [ex for ex in balanced[:max_n] if _example_is_valid(ex)]

    # --- NEW: cap Evidence (and any other label you want) post-hoc ---
    s1_examples = rebalance_s1_examples(
        s1_examples,
        cap_per_label={"Evidence": max(6, args.s1_min_evidence)},  # e.g., cap at 6–8
        min_per_label=args.shots_s1_per_label,  # keep at least the target minimum per label
        prefer_multilabel=True,
    )

    print(
        "✅ Final S1 fewshot label counts:",
        dict(
            Counter(
                l
                for ex in s1_examples
                for l in [m["label"] for m in ex.get("spans", [])]
            )
        ),
    )

    # ----------------- Write outputs -----------------
    out_fs = {"s1": s1_examples, "s2": s2_fewshots}
    fs_path.write_text(
        json.dumps(out_fs, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # --- NEW: also emit a fewshot_bank.json and compact artifacts for prompt_sweep ---
    fewshot_bank_path = out_dir / "fewshot_bank.json"
    fewshot_bank_path.write_text(
        json.dumps(
            {"s1": s1_examples, "s2": s2_fewshots}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )

    conflicts = {"pairs": pairs if "pairs" in locals() else []}
    (out_dir / "conflicts.json").write_text(
        json.dumps(conflicts, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # minimally derived priors for display (use q90 as you already compute it)
    pri_disp = {}
    for lab, d in (priors or {}).items():
        if lab in ALLOWED_S1:
            pri_disp[lab] = {
                "q50_len": d.get("q50_len"),
                "q90_len": d.get("q90_len"),
                "start_mode": (d.get("start_beta") or {}).get("mode") or None,
            }
    (out_dir / "priors_prompt.json").write_text(
        json.dumps(pri_disp, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # boundary prompts: compress any available cues
    bp = {}
    for lab in ALLOWED_S1:
        b = (boundary or {}).get(lab, {})
        befo = (b.get("before_1w") or [])[:3]
        aftr = (b.get("after_1w") or [])[:3]
        if befo or aftr:
            bp[lab] = {"before": befo, "after": aftr}
    (out_dir / "boundary_prompts.json").write_text(
        json.dumps(bp, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    policy_meta = {
        "cant_tell": {
            "s2_training_excluded": True,
            "fewshot_negatives_added": int(
                sum(1 for ex in s2_fewshots if ex.get("source_label") == "cant_tell")
            ),
            "rationale": args.cant_tell_rationale,
        },
        "seed": args.seed,
        "shots": {
            "s2_per_class": args.shots_s2_per_class,
            "s1_per_label": args.shots_s1_per_label,
            "s1_outliers_per_label": args.shots_s1_outliers,
        },
        "targets": {
            "ambiguous_pairs_top2": pairs if "pairs" in locals() else [],
        },
        "artifacts": {
            "fewshot_bank": str(fewshot_bank_path.name),
            "conflicts": "conflicts.json",
            "priors_prompt": "priors_prompt.json",
            "boundary_prompts": "boundary_prompts.json",
        },
    }
    (out_dir / "fewshot_policy.json").write_text(
        json.dumps(policy_meta, indent=2), encoding="utf-8"
    )

    print("\n[select_fewshot_examples] Wrote:")
    print(f"  - {fs_path}")
    print(f"  - {out_dir / 'fewshot_policy.json'}")
    print(f"  S1 count: {len(s1_examples)}  | S2 count: {len(s2_fewshots)}")


if __name__ == "__main__":
    main()
