#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_prompt_artifacts.py

Generates prompt artifacts for SemEval-2026 PsyCoMark:
- S1 priors (length percentiles, start-mode)
- S1 conflict pairs (most overlapping label pairs)
- Few-shot banks for S1 (span extraction) and S2 (doc classification)

Outputs:
- JSON artifact with priors + conflicts (path via --output-file)
- fewshot_bank.json with {"s1":[...], "s2":[...]} (path via --fewshot-out)

Notes (S1 changes in this rewrite):
- Per-role quality gates are stricter but slightly relaxed for scarce roles
  (Evidence/Effect) to improve availability.
- Diversity caps are applied *per-role* to avoid starving scarce labels.
- Dynamic, availability-aware span targets; hard guard prevents over-target
  ride-alongs (a doc can't inflate already-satiated labels unless it also
  helps under-target labels, and never pushes an over-target label higher).
- Gentle repair swaps to keep max-min label span skew <= 4 by default.
- One span per role per example (configurable) for clarity of demonstrations.
"""
from __future__ import annotations
import sys, os, json, math, random, pathlib, re
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import Counter, defaultdict

import numpy as np

# ---------------- Repo paths & .env ----------------
_THIS = pathlib.Path(__file__).resolve()
ROOT = _THIS.parents[3]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _load_dotenv_into_environ():
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    # map aliases to AWS_*
    alias = {
        "ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
        "SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
        "SESSION_TOKEN": "AWS_SESSION_TOKEN",
        "REGION": "AWS_DEFAULT_REGION",
    }
    for src, dst in alias.items():
        if src in os.environ and dst not in os.environ:
            os.environ[dst] = os.environ[src]
    print(f"[env] loaded from {env_path if env_path.exists() else '(none)'}")


_load_dotenv_into_environ()

# Late import to allow path/env setup
try:
    from src.psycomark.llm.bedrock_chat import BedrockChat
except Exception:
    BedrockChat = None  # optional

# ---------------- Constants & utils ----------------
ALLOWED_S1 = {"Actor", "Action", "Effect", "Evidence", "Victim"}
# Slightly upweight scarce roles by default
ROLE_PRIORITY = {"Evidence": 4, "Effect": 3, "Action": 3, "Actor": 2, "Victim": 2}

_WS = re.compile(r"\s+")
STOP = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "as",
    "of",
    "to",
    "for",
    "in",
    "on",
    "at",
    "by",
    "with",
    "than",
    "that",
    "this",
    "these",
    "those",
    "be",
    "is",
    "are",
    "was",
    "were",
    "it",
    "its",
    "their",
    "your",
}
PRONOUNS = {
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "her",
    "our",
    "their",
    "yours",
    "ours",
    "theirs",
    "i'd",
    "i'm",
    "i'll",
    "i've" "i’ve",
    "thou",  # <-- ADD THIS
}
BAD_SOLO = {"wednesday", "amount", "video", "names", "names.", "and", "the"}

EVID_PAT = re.compile(
    r"(https?://|\bwww\.|according to\b|report\b|reports\b|reported\b|"
    r"\b(said|says)\b|Reuters|AP|OIG|study\b|paper\b|dataset\b|"
    r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s?(%|\$|cases|people|bn|million|billion)|"
    r"[“\"“].+?[”\"”])",
    re.I,
)
VERB_HEAD_RE = re.compile(r"^(?:to\s+)?[A-Za-z]+(?:ed|ing|es|s)?\b")
DATE_ONLY_RE = re.compile(
    r"^\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|"
    r"January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},?\s+\d{2,4}",
    re.I,
)


def _tok(s: str) -> list[str]:
    return [t for t in re.split(_WS, s.strip()) if t]


def _dedup_texts(examples: list[dict], key="text", min_dist=0.20) -> list[dict]:
    seen, out = [], []
    for ex in examples:
        toks = set(_tok(ex.get(key, "")))
        ok = True
        for t2 in seen:
            inter = len(toks & t2)
            union = len(toks | t2) or 1
            if inter / union >= (1.0 - min_dist):
                ok = False
                break
        if ok:
            seen.append(toks)
            out.append(ex)
    return out


def _cap_per_key(items: list[dict], key: str, k: int) -> list[dict]:
    if not key or k <= 0:
        return items
    bucket, out = defaultdict(int), []
    for ex in items:
        v = ex.get(key) or "UNK"
        if bucket[v] < k:
            out.append(ex)
            bucket[v] += 1
    return out


def _normalize_row(r: dict) -> dict:
    spans = r.get("spans") or r.get("markers") or []
    label = r.get("label") or r.get("doc_label") or (r.get("gold") or {}).get("label")
    return {
        "text": r.get("text") or "",
        "spans": spans,
        "label": label,
        "subreddit": r.get("subreddit") or r.get("source") or "unknown",
        "_id": r.get("_id") or r.get("doc_id"),
    }


def load_jsonl(path: Path) -> list[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return []


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved JSON → {path}")


# ---------------- Priors & conflicts ----------------


def _normalize_markers(row_or_spans) -> list[dict]:
    spans = row_or_spans
    if isinstance(row_or_spans, dict):
        spans = row_or_spans.get("spans") or row_or_spans.get("markers") or []
    if not isinstance(spans, list):
        return []
    out = []
    for m in spans or []:
        if not isinstance(m, dict):
            continue
        lab = (m.get("label") or m.get("type") or "").strip()
        if lab not in ALLOWED_S1:
            continue
        try:
            s = int(m.get("start", m.get("startIndex")))
            e = int(m.get("end", m.get("endIndex")))
        except Exception:
            continue
        if e is None or s is None or e <= s:
            continue
        out.append({"label": lab, "start": s, "end": e})
    return out


def calculate_statistical_priors(data: list[dict]) -> dict:
    buckets = defaultdict(lambda: {"lengths": [], "positions": []})
    for item in data:
        text = item.get("text") or ""
        n = len(text)
        if n <= 0:
            continue
        for m in _normalize_markers(item):
            ln = m["end"] - m["start"]
            if ln <= 0:
                continue
            rel = m["start"] / max(1, n)
            b = round(math.floor(rel * 10) / 10.0, 1)
            buckets[m["label"]]["lengths"].append(ln)
            buckets[m["label"]]["positions"].append(b)
    priors = {}
    for lab, vals in buckets.items():
        L, P = vals["lengths"], vals["positions"]
        if not L:
            continue
        q50, q90 = int(np.percentile(L, 50)), int(np.percentile(L, 90))
        mean_len = float(np.mean(L))
        most = 0.5
        if P:
            cnt = Counter(P)
            most = max(sorted(cnt.items()), key=lambda kv: (kv[1], kv[0]))[0]
        priors[lab] = {
            "q50_len": q50,
            "q90_len": q90,
            "mean_len": round(mean_len, 1),
            "mode_pos": most,
        }
    print("Calculated S1 priors for labels:", sorted(priors.keys()))
    return priors


def _overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return max(a[0], b[0]) < min(a[1], b[1])


def analyze_span_conflicts(data: list[dict], top_n: int = 2) -> list[list[str]]:
    counts = Counter()
    for item in data:
        spans = sorted(_normalize_markers(item), key=lambda m: m["start"])
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                m1, m2 = spans[i], spans[j]
                if _overlap((m1["start"], m1["end"]), (m2["start"], m2["end"])):
                    pair = tuple(sorted([m1["label"], m2["label"]]))
                    counts[pair] += 1
    pairs = [list(p) for p, _ in counts.most_common(top_n)]
    print(f"Top {len(pairs)} conflict pairs:", pairs)
    return pairs


# ---------------- Span hygiene ----------------
_PUNCT_EDGE = re.compile(
    r"^[\s\.,:;!?\"'“”‘’\-\(\)\[\]]+|[\s\.,:;!?\"'“”‘’\-\(\)\[\]]+$"
)


def _trim_edges(text: str, s: int, e: int) -> tuple[int, int]:
    orig_s, orig_e = s, e
    # Move 's' forward past whitespace/punctuation
    while s < e and (text[s].isspace() or text[s] in '.,:;!?"“”‘’-()[]'):
        s += 1
    # Move 'e' backward past whitespace/punctuation
    while e > s and (text[e - 1].isspace() or text[e - 1] in '.,:;!?"“”‘’-()[]'):
        e -= 1

    # After trimming, check if the span is just a stopword or pronoun
    if s < e:
        seg_low = text[s:e].lower()
        toks = seg_low.split()
        if not toks:  # Was only whitespace/punct
            return s, s  # Return empty span
        # If ALL tokens are stopwords/pronouns, reject the span
        if all(t in STOP or t in PRONOUNS for t in toks):
            return s, s  # Return empty span

    # If the span is valid (or became empty), return the new boundaries
    if s >= e:  # If span became empty
        return orig_s, orig_s

    return s, e


# ---- Stronger role validators ----

ACTION_BAD_LONE = {"all", "when", "agenda", "caves"}
EFFECT_BAD_NOUNS = {"basic premise", "better place", "goal", "purpose"}
ACTOR_REJECT = {"asia", "europe", "africa", "america", "americans", "public"}

VERB_PHRASE_HEAD = re.compile(r"^(?:to\s+)?[A-Za-z]+(?:ed|ing|es|s)\b")


def _is_good_actor(s: str) -> bool:
    t = (s or "").strip().strip(' .,:;!?"“”’')
    if len(t) < 3:
        return False
    low = t.lower()
    if low in PRONOUNS or low in ACTOR_REJECT:
        return False
    # reject possessives like "JBP's"
    if re.match(r"^\w+'s$", t):
        return False
    # allow explicit conspiratorial collectives
    if re.search(r"\b(elites?|globalists?|deep state|cabal|world government)\b", low):
        return True
    # prefer named entities / multi-word capitalized
    return bool(re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}$", t))


def _is_good_action(seg: str) -> bool:
    t = seg.strip().strip('".,;:!?')
    low = t.lower()
    toks = low.split()
    if not VERB_PHRASE_HEAD.match(low):
        return False
    # must be ≥2 tokens unless imperative/proper form like "Ban TikTok"
    if len(toks) == 1 and not re.match(
        r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$", seg.strip()
    ):
        return False
    if low in ACTION_BAD_LONE:
        return False
    # filter generic [gerund + noun] that aren’t action-y
    if re.match(r"^\w+ing\s+\w+$", low) and not re.match(
        r"^(planning|trying|seeking|ordering|directing|leaking|recruiting|suppressing|censoring|covering|manipulating|spying|shooting|banning)\b",
        low,
    ):
        return False
    return True


def _is_good_effect(s: str) -> bool:
    t = (s or "").strip().strip(' .,:;!?"“”’')
    low = t.lower()
    if len(t) < 3:
        return False
    if low in EFFECT_BAD_NOUNS:
        return False
    # direct purpose/goal patterns
    if re.match(r"^to\s+\w{3,}", low):
        return True
    if re.search(r"\b(so that|in order to)\b", low):
        return True
    # causal/resultive cues
    return bool(
        re.search(
            r"\b(result|purpose|goal|aim|intent|outcome|thereby|thus|hence|"
            r"lead(s|ing)?\s+to|cause(s|d|ing)?|enable(s|d|ing)?|result(s|ed)?\s+in|"
            r"control|enslavement|depopulation|silence|censor(ship)?)\b",
            low,
        )
    )


VICTIM_BAD = {"public", "everyone", "people", "the public"}


def _is_good_victim(seg: str) -> bool:
    t = seg.strip().strip('".,;:!?')
    low = t.lower()
    if not t or low in PRONOUNS or low in VICTIM_BAD:
        return False
    if DATE_ONLY_RE.match(t):
        return False
    # allow specific groups or proper noun targets; reject bare "people" / extremely generic
    if low in {"people", "public", "everyone"}:
        return False
    return True


def _is_good_evidence(seg: str) -> bool:
    t = seg.strip()
    if DATE_ONLY_RE.match(t):
        return False
    if t.lower() in BAD_SOLO:
        return False
    if EVID_PAT.search(t):
        return True
    # NEW: bare attribution like "he said", "officials said", "court records show"
    if re.search(r"\b(said|says|stated|records show|documents show)\b", t, re.I):
        return True
    return False


def _span_ok_by_role(label: str, seg: str) -> bool:
    return {
        "Actor": _is_good_actor,
        "Action": _is_good_action,
        "Effect": _is_good_effect,
        "Victim": _is_good_victim,
        "Evidence": _is_good_evidence,
    }.get(label, lambda _: False)(seg)


def _has_AE_overlap(spans: list[dict]) -> bool:
    def L(m):
        return (m.get("label") or m.get("type") or "").strip()

    def S(m):
        return int(m.get("start", m.get("startIndex", -1)))

    def E(m):
        return int(m.get("end", m.get("endIndex", -1)))

    pairs = [
        (L(m), S(m), E(m))
        for m in spans
        if isinstance(m, dict) and L(m) and E(m) > S(m)
    ]
    for i in range(len(pairs)):
        li, si, ei = pairs[i]
        for j in range(i + 1, len(pairs)):
            lj, sj, ej = pairs[j]
            if {li, lj} == {"Action", "Effect"} and max(si, sj) < min(ei, ej):
                return True
    return False


def _clip_to_sentences(t: str, max_chars: int = 1200) -> str:
    if len(t) <= max_chars:
        return t
    cut = t[:max_chars]
    dot = cut.rfind(". ")
    return cut[: dot + 1] if dot >= max_chars * 0.6 else cut


def _clip_and_reindex_spans(
    text: str, spans: list[dict], max_chars: int
) -> tuple[str, list[dict]]:
    if not isinstance(text, str) or not text:
        return "", []
    if not isinstance(spans, list):
        spans = []
    clipped = _clip_to_sentences(text, max_chars=max_chars)
    if clipped is text:
        out = []
        seen = set()
        for m in spans or []:
            try:
                s = int(m["start"])
                e = int(m["end"])
                lab = m.get("label") or m.get("type")
            except Exception:
                continue
            if e <= s:
                continue
            ss, ee = _trim_edges(text, s, e)
            if ee <= ss:
                continue
            seg = text[ss:ee]
            key = (lab, ss, ee, seg)
            if key in seen:
                continue
            seen.add(key)
            out.append({"label": lab, "start": ss, "end": ee, "text": seg})
        return text, out
    start_idx = text.find(clipped)
    if start_idx < 0:
        start_idx = 0
        clipped = text[:max_chars]
    end_idx = start_idx + len(clipped)
    reindexed, seen = [], set()
    for m in spans or []:
        try:
            s = int(m["start"])
            e = int(m["end"])
            lab = m.get("label") or m.get("type")
        except Exception:
            continue
        if e <= start_idx or s >= end_idx:
            continue
        ns = max(s, start_idx) - start_idx
        ne = min(e, end_idx) - start_idx
        ss, ee = _trim_edges(clipped, ns, ne)
        if ee <= ss:
            continue
        seg = clipped[ss:ee]
        key = (lab, ss, ee, seg)
        if key in seen:
            continue
        seen.add(key)
        reindexed.append({"label": lab, "start": int(ss), "end": int(ee), "text": seg})
    return clipped, reindexed


# ---------------- S1 few-shot builder ----------------


def _s1_snippet_score(item: dict) -> float:
    text = (item.get("text") or "").strip()
    spans = item.get("spans") or item.get("markers") or []
    labs = [s.get("label") for s in spans if isinstance(s, dict) and s.get("label")]
    L = len(labs)
    uniq = len(set(labs))

    # bonuses
    has_victim = 1.0 if "Victim" in labs else 0.0
    has_evidence = 1.0 if "Evidence" in labs else 0.0

    # A/E overlap
    has_ae = 0.0
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            li, lj = spans[i].get("label"), spans[j].get("label")
            if {li, lj} == {"Action", "Effect"} and max(
                int(spans[i]["start"]), int(spans[j]["start"])
            ) < min(int(spans[i]["end"]), int(spans[j]["end"])):
                has_ae = 1.0
                break
        if has_ae:
            break

    penalty = 0.0
    if len(text) > 1400:
        penalty += 0.2
    if re.search(r"https?://|www\.", text, re.I):
        penalty += 0.1

    # slightly upweight Evidence (to lift its count without overwhelming)
    return (
        0.55 * L
        + 0.8 * uniq
        + 0.5 * has_victim
        + 0.6 * has_ae
        + 0.6 * has_evidence
        - penalty
    )


def build_s1_fewshot_snippets(
    train_rows: list[dict],
    *,
    want: int = 20,
    victim_min: int = 2,
    conflict_min: int = 1,
    diversity_key: str | None = "subreddit",
    max_per_diverse: int = 2,
    max_chars: int = 1200,
    per_label_min: dict[str, int] | None = None,
    prefer_underrepresented: bool = True,  # kept for API
    max_per_role: int = 1,  # one span per role per example for clarity
    seed: int = 42,
) -> list[dict]:
    import statistics as _stats

    rng = random.Random(seed)

    # -------- helpers --------
    span_ok_by_role = globals().get("_span_ok_by_role", lambda lab, txt: True)
    role_priority = globals().get("ROLE_PRIORITY", ROLE_PRIORITY)

    def _rank_within_role(m):
        L = m["type"]
        ln = int(m["endIndex"]) - int(m["startIndex"])
        score = (role_priority.get(L, 0) * 2) - 0.01 * ln
        if L == "Evidence":
            score += 1.2 if _is_good_evidence(m["text"]) else 0.0
        return score

    def _cap_per_role(spans, k=1):
        bucket = defaultdict(int)
        out = []
        for m in spans:
            L = m["type"]
            if bucket[L] < k:
                out.append(m)
                bucket[L] += 1
        return out

    # -------- 1) candidate pools --------
    cands_pos, cands_neg = [], []
    for r in train_rows or []:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        spans_norm = _normalize_markers(r)
        if spans_norm:
            score = float(_s1_snippet_score({"text": text, "markers": spans_norm}))
            cands_pos.append({**r, "score": score, "_norm_spans": spans_norm})
        else:
            if len(text) < 600 and not re.search(r"https?://|www\.", text, re.I):
                cands_neg.append({**r, "score": 0.0})

    # -------- 2) clip & reindex, build canonical spans --------
    cands_pos.sort(key=lambda x: x["score"], reverse=True)
    cleaned_pos: list[dict] = []
    seen_ids = set()

    for ex in cands_pos:
        ex_id = ex.get("_id") or ex.get("doc_id")
        if ex_id in seen_ids:
            continue
        seen_ids.add(ex_id)

        clipped_text, re_spans = _clip_and_reindex_spans(
            (ex.get("text") or ""), ex["_norm_spans"], max_chars=max_chars
        )
        if not clipped_text or not re_spans:
            continue

        role_groups = defaultdict(list)
        for m in re_spans:
            lab = (m.get("label") or m.get("type") or "").strip()
            if lab not in ALLOWED_S1:
                continue
            s = int(m["start"])
            e = int(m["end"])
            if e <= s or s < 0 or e > len(clipped_text):
                continue
            seg = clipped_text[s:e]
            if _span_ok_by_role(lab, seg):
                role_groups[lab].append(
                    {"type": lab, "startIndex": s, "endIndex": e, "text": seg}
                )

        if not role_groups:
            continue

        spans_canon: list[dict] = []
        for L, items in role_groups.items():
            items = sorted(items, key=_rank_within_role, reverse=True)
            spans_canon.append(items[0])  # keep best per role
        extras = [m for L, items in role_groups.items() for m in items[1:]]
        spans_canon += sorted(extras, key=_rank_within_role, reverse=True)

        spans_canon = _cap_per_role(spans_canon, k=max_per_role)
        spans_canon = sorted(spans_canon, key=_rank_within_role, reverse=True)[:4]
        if not spans_canon:
            continue

        cleaned_pos.append(
            {
                "_id": ex_id,
                "subreddit": ex.get("subreddit"),
                "text": clipped_text,
                "spans": spans_canon,
                "score": ex.get("score", 0.0),
            }
        )

    # -------- 3) dedup + per-role diversity --------
    cleaned_pos = _dedup_texts(cleaned_pos, key="text", min_dist=0.30)

    if diversity_key:
        per_label_cleaned = []
        for L in ["Actor", "Action", "Effect", "Evidence", "Victim"]:
            bucket = [e for e in cleaned_pos if any(m["type"] == L for m in e["spans"])]
            bucket = _cap_per_key(bucket, diversity_key, max_per_diverse)
            per_label_cleaned.extend(bucket)
        # dedup by _id after merging per-label buckets
        seen = set()
        cleaned = []
        for e in per_label_cleaned:
            if e["_id"] in seen:
                continue
            seen.add(e["_id"])
            cleaned.append(e)
        cleaned_pos = cleaned

    if not cleaned_pos:
        # fallback: negatives only
        fallback = []
        for ex in cands_neg[:want]:
            txt = (ex.get("text") or "").strip()
            if not txt:
                continue
            clipped_txt, _ = _clip_and_reindex_spans(txt, [], max_chars=max_chars)
            if not clipped_txt:
                continue
            fallback.append(
                {
                    "text": clipped_txt,
                    "answer": [],
                    "subreddit": ex.get("subreddit"),
                    "_id": ex.get("_id") or ex.get("doc_id"),
                }
            )
        return fallback

    # -------- availability (cap 1 per role per doc) --------
    labels = ["Actor", "Action", "Effect", "Evidence", "Victim"]

    def _avail_capped(pool):
        av = {L: 0 for L in labels}
        for e in pool:
            seen = set()
            for m in e["spans"]:
                L = m["type"]
                if L in labels and L not in seen:
                    av[L] += 1
                    seen.add(L)
        return av

    availability = _avail_capped(cleaned_pos)
    print(f"[S1] availability per label: {availability} (after hygiene + caps)")

    # -------- per-label minima (doc-level) --------
    if per_label_min is None:
        per_label_min = {
            "Actor": 2,
            "Action": 2,
            "Effect": 3,
            "Evidence": 3,
            "Victim": 2,
        }

    picked: list[dict] = []
    picked_ids = set()

    # seed by minima
    for L, q in per_label_min.items():
        bucket = sorted(
            [e for e in cleaned_pos if any(m["type"] == L for m in e["spans"])],
            key=lambda x: x.get("score", 0.0),
            reverse=True,
        )
        for ex in bucket:
            if len(picked) >= want:
                break
            if ex["_id"] in picked_ids:
                continue
            picked.append(ex)
            picked_ids.add(ex["_id"])
            if sum(1 for e in picked if any(m["type"] == L for m in e["spans"])) >= q:
                break

    # Victim presence
    if victim_min > 0 and not any(
        "Victim" in {m["type"] for m in e["spans"]} for e in picked
    ):
        for ex in sorted(
            [e for e in cleaned_pos if any(m["type"] == "Victim" for m in e["spans"])],
            key=lambda x: x.get("score", 0.0),
            reverse=True,
        ):
            if ex["_id"] in picked_ids:
                continue
            picked.append(ex)
            picked_ids.add(ex["_id"])
            break

    # AE overlap guarantee
    if conflict_min > 0 and not any(_has_AE_overlap(e["spans"]) for e in picked):
        ae_pool = [
            e
            for e in cleaned_pos
            if _has_AE_overlap(e["spans"]) and e["_id"] not in picked_ids
        ]
        if ae_pool:
            ae_pool.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            picked.append(ae_pool[0])
            picked_ids.add(ae_pool[0]["_id"])

    # -------- 5) dynamic balancing --------
    _span_lens = [min(len(e["spans"]), 4) for e in cleaned_pos[: max(want, 50)]]
    avg_spans = int(round(_stats.mean(_span_lens))) if _span_lens else 3
    avg_spans = min(max(avg_spans, 2), 4)
    total_target_spans = want * avg_spans

    # initial equal-share targets, adjusted by availability
    base = total_target_spans // len(labels)
    rem = total_target_spans % len(labels)
    target_per_label = {L: min(base, availability[L]) for L in labels}
    for L, _ in sorted(availability.items(), key=lambda kv: (kv[1], kv[0]))[:rem]:
        target_per_label[L] = min(target_per_label[L] + 1, availability[L])

    MIN_FLOOR = 5 if want >= 20 else 3
    for L in labels:
        target_per_label[L] = max(
            min(target_per_label[L], availability[L]), min(MIN_FLOOR, availability[L])
        )

    # if total below target due to availability, try to top up smallest targets that still have spare availability
    current_sum = sum(target_per_label.values())
    spare = {L: max(0, availability[L] - target_per_label[L]) for L in labels}
    while current_sum < total_target_spans:
        choices = [L for L in labels if spare[L] > 0]
        if not choices:
            break
        L_pick = min(choices, key=lambda L: (target_per_label[L], L))
        target_per_label[L_pick] += 1
        spare[L_pick] -= 1
        current_sum += 1

    TOL = 5
    hard_cap = {L: target_per_label[L] + (TOL // 2) for L in labels}

    def _span_hist(docs):
        h = defaultdict(int)
        for e in docs:
            for m in e["spans"]:
                h[m["type"]] += 1
        return h

    def _doc_span_contrib_capped(ex, cap_per_role=1):
        c = defaultdict(int)
        for m in ex["spans"]:
            L = m["type"]
            if c[L] < cap_per_role:
                c[L] += 1
        return c

    def _has_victim_doc(docs):
        return any("Victim" in {m["type"] for m in e["spans"]} for e in docs)

    def _has_ae_doc(docs):
        return any(_has_AE_overlap(e["spans"]) for e in docs)

    def _constraints_ok(docs):
        return _has_victim_doc(docs) and (not conflict_min or _has_ae_doc(docs))

    def _deficits(hist):
        return {L: max(0, target_per_label[L] - hist.get(L, 0)) for L in labels}

    def _doc_is_helpful(contrib, hist, target):
        adds_under = any(
            contrib.get(L, 0) > 0 and hist.get(L, 0) < target[L] for L in labels
        )
        adds_over = any(
            contrib.get(L, 0) > 0 and hist.get(L, 0) > target[L] for L in labels
        )
        adds_at = any(
            contrib.get(L, 0) > 0 and hist.get(L, 0) == target[L] for L in labels
        )
        if adds_over:
            return False
        if adds_at and not adds_under:
            return False
        return True

    remaining = [e for e in cleaned_pos if e["_id"] not in picked_ids]

    # 5a: round-robin fill on most under-target labels
    hist = _span_hist(picked)
    deficits = _deficits(hist)

    def _best_for_label(L_focus, hist, pool):
        best, best_score, best_tb = None, -1e18, -1e18
        for ex in pool:
            contrib = _doc_span_contrib_capped(ex, cap_per_role=1)
            # hard caps
            if any(hist.get(L, 0) + contrib.get(L, 0) > hard_cap[L] for L in labels):
                continue
            if not _doc_is_helpful(contrib, hist, target_per_label):
                continue
            # helpfulness: upweight L_focus
            help_score = 0.0
            for L in labels:
                deficit = max(0, target_per_label[L] - hist.get(L, 0))
                if deficit <= 0:
                    continue
                inc = min(contrib.get(L, 0), deficit)
                if inc <= 0:
                    continue
                w = 2.5 if L == L_focus else 1.0
                help_score += w * inc
            tb = 0.001 * ex.get("score", 0.0)
            if not _has_ae_doc(picked) and _has_AE_overlap(ex["spans"]):
                tb += 0.5
            if help_score > best_score or (help_score == best_score and tb > best_tb):
                best, best_score, best_tb = ex, help_score, tb
        return best

    rr_guard = 0
    while (
        len(picked) < min(want, len(cleaned_pos))
        and sum(deficits.values()) > 0
        and remaining
    ):
        max_def = max(deficits.values())
        need = [L for L, d in deficits.items() if d == max_def and d > 0]
        picked_any = False
        for L_focus in sorted(need):
            ex = _best_for_label(L_focus, hist, remaining)
            if not ex:
                continue
            picked.append(ex)
            picked_ids.add(ex["_id"])
            remaining.remove(ex)
            hist = _span_hist(picked)
            deficits = _deficits(hist)
            picked_any = True
            if len(picked) >= want:
                break
        rr_guard += 1
        if not picked_any or rr_guard > 5 * want:
            break

    # 5b: gentle fill while still under targets
    while len(picked) < min(want, len(cleaned_pos)) and remaining:
        hist = _span_hist(picked)
        deficits = _deficits(hist)
        if sum(deficits.values()) == 0:
            break
        best, best_gain, best_tb = None, -1e18, -1e18
        for ex in list(remaining):
            contrib = _doc_span_contrib_capped(ex, cap_per_role=1)
            if any(hist.get(L, 0) + contrib.get(L, 0) > hard_cap[L] for L in labels):
                continue
            if not _doc_is_helpful(contrib, hist, target_per_label):
                continue
            gain = sum(
                min(contrib.get(L, 0), deficits[L]) for L in labels if deficits[L] > 0
            )
            if gain <= 0:
                continue
            tb = 0.001 * ex.get("score", 0.0)
            if not _has_ae_doc(picked) and _has_AE_overlap(ex["spans"]):
                tb += 0.5
            if gain > best_gain or (gain == best_gain and tb > best_tb):
                best, best_gain, best_tb = ex, gain, tb
        if not best:
            break
        picked.append(best)
        picked_ids.add(best["_id"])
        remaining.remove(best)

    # 5c: repair swaps to reduce skew within tolerance
    def _skew(h):
        return (max(h.values()) - min(h.values())) if h else 0

    hist = _span_hist(picked)
    if _skew(hist) > TOL:
        over = max(hist, key=hist.get)
        under = min(hist, key=hist.get)
        offenders = sorted(
            [
                (i, sum(1 for m in picked[i]["spans"] if m["type"] == over))
                for i in range(len(picked))
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        helpers = [
            e
            for e in cleaned_pos
            if e["_id"] not in picked_ids
            and any(m["type"] == under for m in e["spans"])
            and all(m["type"] != over for m in e["spans"])
        ]
        if not helpers:
            helpers = [
                e
                for e in cleaned_pos
                if e["_id"] not in picked_ids
                and sum(1 for m in e["spans"] if m["type"] == under)
                > sum(1 for m in e["spans"] if m["type"] == over)
            ]
        for idx, _cnt in offenders[:6]:
            if not helpers:
                break
            cand = helpers.pop(0)
            trial = picked[:idx] + [cand] + picked[idx + 1 :]
            new_h = _span_hist(trial)
            if (
                _skew(new_h) <= _skew(hist)
                and all(new_h[L] <= hard_cap[L] for L in labels)
                and _constraints_ok(trial)
            ):
                picked_ids.remove(picked[idx]["_id"])
                picked[idx] = cand
                picked_ids.add(cand["_id"])
                hist = new_h
                if _skew(hist) <= TOL:
                    break

    # -------- 6) optional negatives tail (≤2) --------
    rng.shuffle(cands_neg)
    negs = []
    for ex in cands_neg:
        if len(picked) + len(negs) >= want:
            break
        txt = (ex.get("text") or "").strip()
        if not txt:
            continue
        clipped_txt, _ = _clip_and_reindex_spans(txt, [], max_chars=max_chars)
        if not clipped_txt:
            continue
        negs.append(
            {
                "_id": ex.get("_id") or ex.get("doc_id"),
                "subreddit": ex.get("subreddit"),
                "text": clipped_txt,
                "spans": [],
            }
        )
        if len(negs) >= 2:
            break

    final_docs = (picked + negs)[:want]

    # ---- Post-select scrub + refill (keeps balance & quality) ----
    def _is_weak_span(m: dict) -> bool:
        txt = (m.get("text") or "").strip()
        low = txt.lower()
        if len(txt) < 4 and not re.match(r"^[A-Z][a-z]+$", txt):
            return True
        if low in {"and", "or", "but", "when", "all", "the", "a"}:
            return True
        # role-specific checks piggyback on validators
        lab = m.get("type")
        if lab == "Action" and not _is_good_action(txt):
            return True
        if lab == "Effect" and not _is_good_effect(txt):
            return True
        if lab == "Actor" and not _is_good_actor(txt):
            return True
        if lab == "Victim" and not _is_good_victim(txt):
            return True
        if lab == "Evidence" and not _is_good_evidence(txt):
            return True
        return False

    def _hist(docs):
        from collections import defaultdict as _dd

        h = _dd(int)
        for e in docs:
            for m in e.get("spans", []):
                h[m["type"]] += 1
        return h

    # scrub spans inside each selected doc
    for ex in final_docs:
        if "spans" in ex:
            ex["spans"] = [m for m in ex["spans"] if not _is_weak_span(m)]
            # ensure per-role cap still applies
            bucket = defaultdict(int)
            kept = []
            for m in sorted(ex["spans"], key=_rank_within_role, reverse=True):
                L = m["type"]
                if bucket[L] < max_per_role:
                    kept.append(m)
                    bucket[L] += 1
            ex["spans"] = kept

    # drop any doc that lost all spans (we'll refill)
    final_docs = [e for e in final_docs if e.get("spans")]

    # refill to maintain 'want' using remaining pool, prioritizing under-target labels
    labels = ["Actor", "Action", "Effect", "Evidence", "Victim"]
    current = _hist(final_docs)
    remaining_pool = [
        e for e in cleaned_pos if e["_id"] not in {d["_id"] for d in final_docs}
    ]

    def _contrib_capped(ex):
        from collections import defaultdict as _dd

        c = _dd(int)
        for m in ex["spans"]:
            L = m["type"]
            if L in labels and c[L] < 1:  # cap 1 per role per doc for balancing
                c[L] += 1
        return c

    while len(final_docs) < want and remaining_pool:
        deficits = {L: max(0, target_per_label[L] - current.get(L, 0)) for L in labels}
        max_def = max(deficits.values()) if deficits else 0
        focus = {L for L, d in deficits.items() if d == max_def and d > 0} or set(
            labels
        )

        best, best_gain, best_pen, best_tb = None, -1e18, 1e18, -1e18
        for ex in remaining_pool:
            contrib = _contrib_capped(ex)
            if not any(contrib.get(L, 0) > 0 for L in focus):
                continue
            # avoid exceeding hard caps
            overs = 0
            for L in labels:
                fut = current.get(L, 0) + contrib.get(L, 0)
                if fut > hard_cap[L]:
                    overs += fut - hard_cap[L]
            if overs > 0:
                continue
            gain = sum(min(contrib.get(L, 0), deficits.get(L, 0)) for L in labels)
            tb = 0.001 * ex.get("score", 0.0)
            if not any(
                _has_AE_overlap(d.get("spans", [])) for d in final_docs
            ) and _has_AE_overlap(ex["spans"]):
                tb += 0.5
            if (
                (gain > best_gain)
                or (gain == best_gain and overs < best_pen)
                or (gain == best_gain and overs == best_pen and tb > best_tb)
            ):
                best, best_gain, best_pen, best_tb = ex, gain, overs, tb

        if not best:
            break
        final_docs.append(best)
        remaining_pool.remove(best)
        current = _hist(final_docs)

    # -------- output schema + validation --------
    out: list[dict] = []
    for ex in final_docs:
        spans = ex.get("spans", [])
        spans = _cap_per_role(
            sorted(spans, key=_rank_within_role, reverse=True), k=max_per_role
        )
        spans = spans[:4]
        out.append(
            {
                "text": ex["text"],
                "answer": spans,
                "subreddit": ex.get("subreddit"),
                "_id": ex.get("_id"),
            }
        )

    for ex in out:
        for m in ex["answer"]:
            assert all(
                k in m for k in ("type", "startIndex", "endIndex", "text")
            ), "malformed span"

    return out


# ---------------- S2 few-shot builder (unchanged logic) ----------------
_CONSP_CUES = [
    r"\bdeep state\b",
    r"\belite(s)?\b",
    r"\bglobalist(s)?\b",
    r"\bcover[- ]?up\b",
    r"\bfalse flag\b",
    r"\bnew world order\b",
    r"\bthey\b.*\bcontrol\b",
    r"\bpuppet(s)? master\b",
    r"\bagenda\b",
    r"\bdo your own research\b",
    r"\bfollow the money\b",
]
_REPORTING_CUES = [r"\b(according to|Reuters|AP|BBC|NYT|study|report)\b"]


def _s2_endorsement_score(text: str) -> float:
    t = (text or "").strip().lower()
    if not t:
        return 0.0
    cons = sum(bool(re.search(p, t)) for p in _CONSP_CUES)
    reps = sum(bool(re.search(p, t)) for p in _REPORTING_CUES)
    affect = (
        1
        if re.search(r"\b(tyranny|enslavement|genocide|destroy|depopulation)\b", t)
        else 0
    )
    us_vs_them = (
        1 if re.search(r"\b(we|us)\b.*\b(they|elite|globalist|deep state)\b", t) else 0
    )
    penalty = 0.1 if re.search(r"https?://|www\.", t) else 0.0
    return 0.9 * cons + 0.6 * affect + 0.5 * us_vs_them - 0.3 * reps - penalty


def build_s2_fewshot_examples(
    train_rows: list[dict],
    *,
    want: int = 10,
    diversity_key: str | None = "subreddit",
    max_per_diverse: int = 2,
    max_chars: int = 1200,
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)
    pos, neg = [], []
    for r in train_rows:
        text = (r.get("text") or "").strip()
        lab = (r.get("label") or (r.get("gold") or {}).get("label") or "").lower()
        if not text or lab not in ("conspiracy", "non"):
            continue
        score = (
            _s2_endorsement_score(text)
            if lab == "conspiracy"
            else -_s2_endorsement_score(text)
        )
        ex = {
            **r,
            "score": score,
            "text": _clip_to_sentences(text, max_chars=max_chars),
        }
        (pos if lab == "conspiracy" else neg).append(ex)
    pos = sorted(
        _dedup_texts(pos, key="text", min_dist=0.2),
        key=lambda x: x["score"],
        reverse=True,
    )
    neg = sorted(
        _dedup_texts(neg, key="text", min_dist=0.2),
        key=lambda x: x["score"],
        reverse=True,
    )
    if diversity_key:
        pos = _cap_per_key(pos, diversity_key, max_per_diverse)
        neg = _cap_per_key(neg, diversity_key, max_per_diverse)

    n_pos = want // 2
    n_neg = want - n_pos
    sel = pos[:n_pos] + neg[:n_neg]
    if len(pos) < n_pos:
        sel = pos + neg[: (n_neg + (n_pos - len(pos)))]
    if len(neg) < n_neg:
        sel = pos[: (n_pos + (n_neg - len(neg)))] + neg

    out = []
    for e in sel[:want]:
        lab = (e.get("label") or (e.get("gold") or {}).get("label")).lower()
        out.append(
            {
                "text": e["text"],
                "answer": {"label": lab, "rationale": "TBD"},
                "subreddit": e.get("subreddit"),
                "_id": e.get("_id") or e.get("doc_id"),
            }
        )
    return out


# ------------- S2 rationale rewriter (optional) -------------
PLACEHOLDER_RATS = {
    "",
    "tbd",
    "n/a",
    "na",
    "none",
    "—",
    "-",
    "concise example rationale.",
    "concise rationale.",
    "example rationale.",
}


def _extract_label_rationale(ex) -> tuple[str, str]:
    a = ex.get("answer")
    label, rat = None, None
    if isinstance(a, dict):
        label, rat = a.get("label"), a.get("rationale")
    elif isinstance(a, str):
        rat = a
    label = (label or ex.get("label") or "non").strip().lower()
    if label not in {"conspiracy", "non"}:
        label = "non"
    rat = (rat or "").strip()
    return label, rat


def _needs_rewrite(rationale: str) -> bool:
    if not rationale:
        return True
    if rationale.strip().lower() in PLACEHOLDER_RATS:
        return True
    return len(rationale) < 12


def _set_answer(ex: dict, label: str, rationale: str):
    ex["answer"] = {"label": label, "rationale": rationale.strip()}


def _rewrite_rationales_if_needed(bank_s2: list[dict], bc: "BedrockChat" | None) -> int:
    if not bc:
        return 0
    rewritten = 0
    SYS = (
        "You are an expert annotator for SemEval PsyCoMark. "
        "Write a crisp 1–2 sentence rationale naming decisive cues "
        "(us–them roles, intentional secret causality, self-sealing logic, affect, endorsement vs reporting)."
    )
    for ex in bank_s2:
        label, rat = _extract_label_rationale(ex)
        if not _needs_rewrite(rat):
            _set_answer(ex, label, rat)
            continue
        text = (ex.get("text") or "").strip()
        if not text:
            _set_answer(ex, label, rat)
            continue
        user = (
            "<framework>1) roles; 2) secret causality; 3) self-sealing; 4) affect; 5) endorsement vs reporting.</framework>\n"
            f"<label>{label}</label>\n<text>{text[:1200]}</text>\nReturn only the rationale (1–2 sentences)."
        )
        try:
            out = bc.chat(
                system_prompt=SYS, user_prompt=user, max_tokens=192, temperature=0.2
            )
            cand = out.get("answer") if isinstance(out, dict) else out
            cand = (cand or "").strip()
            m = re.search(r"<answer>\s*(.*?)\s*</answer>", cand, re.S | re.I)
            if m:
                cand = m.group(1).strip()
            try:
                maybe = json.loads(cand)
                if isinstance(maybe, dict) and "rationale" in maybe:
                    cand = str(maybe.get("rationale") or "").strip()
            except Exception:
                pass
            if not cand:
                cand = rat or "Names decisive cues briefly."
            _set_answer(ex, label, cand)
            rewritten += 1
        except Exception as e:
            print(f"[rationales] skip (error: {e})")
            _set_answer(ex, label, rat)
    return rewritten


def _balance_s2_examples(pool: list[dict], want: int = 10) -> list[dict]:
    cons = [ex for ex in pool if _extract_label_rationale(ex)[0] == "conspiracy"]
    nons = [ex for ex in pool if _extract_label_rationale(ex)[0] == "non"]
    half = max(1, want // 2)
    out = []
    rng = random.Random(0)
    rng.shuffle(cons)
    rng.shuffle(nons)
    out.extend(cons[: min(half, len(cons))])
    out.extend(nons[: min(want - len(out), len(nons))])
    remain = [ex for ex in pool if ex not in out]
    rng.shuffle(remain)
    out.extend(remain[: max(0, want - len(out))])
    return out[:want]


# ---------------- CLI ----------------


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Generate PsyCoMark prompt artifacts + few-shot banks."
    )
    ap.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Annotated train .jsonl (text + markers/doc_label).",
    )
    ap.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Where to save priors/conflicts JSON.",
    )
    ap.add_argument(
        "--fewshot-out",
        type=Path,
        required=True,
        help="Where to save fewshot_bank.json",
    )
    ap.add_argument(
        "--s1-shots", type=int, default=20, help="Number of S1 few-shot examples."
    )
    ap.add_argument(
        "--s2-shots", type=int, default=10, help="Number of S2 few-shot examples."
    )
    ap.add_argument(
        "--s1-victim-min", type=int, default=2, help="Min Victim examples in S1 bank."
    )
    ap.add_argument(
        "--s1-conflict-min",
        type=int,
        default=1,
        help="Min Action–Effect overlap examples in bank.",
    )
    ap.add_argument(
        "--diversity-key",
        type=str,
        default="subreddit",
        help="Field for diversity (e.g., subreddit).",
    )
    ap.add_argument(
        "--max-per-diverse",
        type=int,
        default=2,
        help="Max examples per diversity bucket.",
    )
    ap.add_argument(
        "--s1-max-chars", type=int, default=1200, help="Cap S1 snippet length."
    )
    ap.add_argument(
        "--s2-max-chars", type=int, default=1200, help="Cap S2 text length."
    )
    ap.add_argument(
        "--top-n-conflicts",
        type=int,
        default=2,
        help="Top-N overlapping label pairs to log.",
    )
    ap.add_argument(
        "--rewrite-rationales",
        action="store_true",
        default=True,
        help="Use Claude via Bedrock to rewrite S2 rationales.",
    )
    ap.add_argument(
        "--model-id", type=str, default="anthropic.claude-sonnet-4-5-20250929-v1:0"
    )
    ap.add_argument("--region", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    print(f"Loading training data from: {args.input_file}")
    training = load_jsonl(args.input_file)
    if not training:
        print("Input file is empty or could not be loaded. Exiting.")
        return
    training = [_normalize_row(r) for r in training if r.get("text")]
    print(f"[debug] normalized: {len(training)} valid rows after filtering")

    print("\n--- Generating S1 priors & conflicts ---")
    priors = calculate_statistical_priors(training)
    conflicts = analyze_span_conflicts(training, top_n=args.top_n_conflicts)
    artifacts = {
        "s1_priors": priors,
        "s1_conflicts": conflicts,
        "metadata": {
            "source_file": str(args.input_file),
            "num_docs_analyzed": len(training),
            "seed": args.seed,
        },
    }
    save_json(artifacts, args.output_file)

    print("\n--- Building few-shot banks ---")
    s1_bank = build_s1_fewshot_snippets(
        training,
        want=args.s1_shots,
        seed=args.seed,
        victim_min=args.s1_victim_min,
        conflict_min=args.s1_conflict_min,
        max_per_diverse=args.max_per_diverse,
        diversity_key=args.diversity_key,
        max_chars=args.s1_max_chars,
        max_per_role=1,  # keep 1 span per role per example
        prefer_underrepresented=True,
    )

    def _assert_s1_bank(s1_bank):
        h = Counter()
        for ex in s1_bank:
            for m in ex.get("answer", []):
                h[m["type"]] += 1
        if h:
            assert (
                max(h.values()) - min(h.values()) <= 4
            ), f"Unbalanced S1 fewshots: {h}"
        assert any(
            _has_AE_overlap(ex.get("answer", [])) for ex in s1_bank
        ), "No Action–Effect overlap example found"

    _assert_s1_bank(s1_bank)

    s2_bank = build_s2_fewshot_examples(
        training,
        want=args.s2_shots,
        seed=args.seed,
        max_chars=args.s2_max_chars,
    )

    fewshot_bank = {"s1": s1_bank, "s2": s2_bank}

    bc = None
    if args.rewrite_rationales and BedrockChat is not None:
        model_id = os.getenv("MODEL_ID") or args.model_id
        region = os.getenv("AWS_DEFAULT_REGION") or args.region
        bc = BedrockChat(model_id=model_id, region_name=region)
        print(f"[debug] BedrockChat initialized with model={model_id} region={region}")

    fewshot_bank["s2"] = _balance_s2_examples(fewshot_bank["s2"], want=args.s2_shots)
    rewritten = _rewrite_rationales_if_needed(fewshot_bank["s2"], bc)
    print(f"[rationales] rewrote {rewritten} example(s)")

    save_json(fewshot_bank, args.fewshot_out)

    # Coverage/logging
    def _lab_counts_s1(items):
        c = Counter()
        for e in items:
            for a in e.get("answer") or []:
                lab = a.get("label") or a.get("type")
                if lab:
                    c[lab] += 1
        return dict(c)

    print("[fewshot] S1 label counts:", _lab_counts_s1(s1_bank))
    has_victim = any(
        any(
            (a.get("label") or a.get("type")) == "Victim"
            for a in (e.get("answer") or [])
        )
        for e in s1_bank
    )
    has_conflict = any(_has_AE_overlap(e.get("answer") or []) for e in s1_bank)
    print(f"[fewshot] S1 Victim present: {has_victim}")
    print(f"[fewshot] S1 has Action-Effect conflict: {has_conflict}")
    cons = sum(1 for e in fewshot_bank["s2"] if e["answer"]["label"] == "conspiracy")
    print(f"[fewshot] S2 balance: {cons} / {len(fewshot_bank['s2'])}")
    print("\n✅ Artifact generation complete.")


if __name__ == "__main__":
    main()
