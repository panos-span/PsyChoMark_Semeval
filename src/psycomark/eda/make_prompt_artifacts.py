import os
from enum import Enum
from typing import Optional, Iterable, Set, List, Literal, Dict
from collections import Counter
from dataclasses import dataclass
import argparse, json, random, re, time
from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict, field_validator
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.providers.bedrock import BedrockProvider
from pydantic_ai.models.bedrock import BedrockConverseModel
import pathlib
import sys
import copy
import asyncio, inspect

_JSON_RE = re.compile(r"\{[\s\S]*\}$")  # last JSON object in text


def agent_run_sync(agent, *args, **kwargs):
    # prefer run_sync when available
    if hasattr(agent, "run_sync") and callable(agent.run_sync):
        return agent.run_sync(*args, **kwargs)
    coro = agent.run(*args, **kwargs)
    return asyncio.run(coro) if inspect.iscoroutine(coro) else coro


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

# Try to import the blocks from prompt_builder; fall back to local copies
try:
    from pydanticai.prompt_builder import (
        playbook_block,
        psycho_theory_preamble,
        data_profile_block,
    )
except Exception:

    def playbook_block() -> str:
        return """
<psycomark_playbook version="1.0">
  <cues_actor>vague/collective agents alleging secret coordination: "they", "the elite", "globalists", "deep state", "big pharma".</cues_actor>
  <cues_action>intentional control/hostility/cover-up verbs: plot, scheme, infiltrate, engineer, manipulate, cover up, weaponize.</cues_action>
  <cues_effect>extreme stakes or grand outcomes: total control, enslavement, depopulation, tyranny.</cues_effect>
  <cues_epistemics>self-sealing logic: counter-evidence framed as disinformation; "do your own research"; "connect the dots".</cues_epistemics>
  <pitfalls>Do not rely on keywords alone; distinguish reporting/debunking from endorsement.</pitfalls>
</psycomark_playbook>
""".strip()

    def psycho_theory_preamble() -> str:
        return """
<psycholinguistic_preamble version="1.0">
  <role>You are an expert computational psycholinguist. Align your reasoning with psycholinguistic and evolutionary accounts of conspiratorial rhetoric for SemEval-2026 PsyCoMark.</role>
  <marker_definitions>
    <Actor>Agents alleged to secretly orchestrate events; the conspirators.</Actor>
    <Action>Deliberate acts attributed to the Actor (what they do). Verb phrase; exclude outcomes/goals.</Action>
    <Effect>Consequence/goal/purpose of the Action (why/result). Often purpose/result clause.</Effect>
    <Victim>Entity harmed/targeted by the Action.</Victim>
    <Evidence>Support claims: links; quoted+attributed material; numeric facts+units+named source.</Evidence>
  </marker_definitions>
</psycholinguistic_preamble>
""".strip()


# Bedrock model wiring (same pattern you use in psycomark_agents.py)
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "eu-central-1")
BEDROCK_MODEL_ID = os.getenv("MODEL_ID", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")
_provider = BedrockProvider(region_name=AWS_REGION)
LLM = BedrockConverseModel(BEDROCK_MODEL_ID, provider=_provider)

LABELS = {"Actor", "Action", "Effect", "Victim", "Evidence"}

# --- span quality scoring & subtype detectors ---

URL_RE = re.compile(r"https?://|www\.", re.I)
QUOTE_ATTR_RE = re.compile(
    r"['\"“”‘’].+['\"“”‘’]\s*(?:—|-)?\s*(?:said|told|according to|reported|writes|stated)\b",
    re.I,
)
NAMED_SOURCE_RE = re.compile(
    r"\b(Reuters|AP|BBC|NYT|WHO|CDC|Kremlin|Ministry|Harvard|Stanford|Boudry|Meigs|Dunbar)\b",
    re.I,
)
NUMERIC_SOURCE_RE = re.compile(
    r"\b\d+(\.\d+)?\s*(%|percent|pp|people|cases|pages?)\b", re.I
)
VERBISH_RE = re.compile(
    r"\b(are|is|was|were|be|being|been|do|does|did|have|has|had|plot|scheme|infiltrate|engineer|manipulate|cover(?:\s*up)?|weaponize|poison|hide|fabricate|rig|lie|deceive|control|suppress|ban|force|impose)\b",
    re.I,
)
PURPOSE_RE = re.compile(
    r"\b(to|so that|in order to|for the purpose of|result|effect|cause)\b", re.I
)
VAGUE_ACTOR_RE = re.compile(
    r"\b(they|the elite|globalists|deep state|big pharma)\b", re.I
)
NAMED_PERSON_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")

S2_CUE_RE = re.compile(
    r"(deep state|globalist|elite|agenda|cover[- ]?up|false flag|"
    r"hoax|they\s+want|they're trying|new world order|"
    r"pedo|chemtrail|MK[-\s]?Ultra|shadow government)",
    re.I,
)
S2_DEBUNK_RE = re.compile(
    r"\b(debunk|myth|not true|no evidence|conclusion is wrong|"
    r"conspiracy theory(?:ies)? as such)\b",
    re.I,
)

DATE_ONLY_RE = re.compile(
    r"^\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|"
    r"January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},?\s+\d{2,4}",
    re.I,
)

TIME_ANCHOR_RE = re.compile(
    r"\b(yesterday|today|tonight|tomorrow|last|next|prior|before|after|"
    r"hour|hours|day|days|week|weeks|month|months|year|years)\b",
    re.I,
)


def _looks_like_time_anchor(text: str) -> bool:
    """
    Heuristic: short phrase that is mostly about time, not an entity.
    Used to drop Victim/Effect spans that are just time anchors.
    """
    if not isinstance(text, str):
        return False
    t = text.strip()
    if not t:
        return False

    # Pure date like "April 15, 2019"
    if DATE_ONLY_RE.match(t):
        return True

    toks = t.split()
    if len(toks) <= 6:
        if TIME_ANCHOR_RE.search(t):
            return True
        # bare years / numeric time expressions
        if re.search(r"\b(19|20)\d{2}\b", t):
            return True
        if re.search(r"\b\d{1,2}\s*(hours?|days?|weeks?|months?|years?)\b", t, re.I):
            return True
    return False


def s2_signals(text: str) -> dict:
    cues = len(S2_CUE_RE.findall(text))
    debunk = bool(S2_DEBUNK_RE.search(text))
    qmarks = text.count("?")
    length = len(text)
    return {
        "cues": cues,
        "debunk": debunk,
        "qmarks": qmarks,
        "length": length,
    }


# --- NEW: complexity helpers (use raw start/end from training file) ---
def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return (a_start < b_end) and (b_start < a_end)


def _count_overlaps(raw_markers: list[dict]) -> int:
    pairs = 0
    ms = [
        (m.get("label") or m.get("type"), m.get("start"), m.get("end"))
        for m in (raw_markers or [])
    ]
    ms = [
        (lab, s, e)
        for lab, s, e in ms
        if isinstance(s, int) and isinstance(e, int) and e > s
    ]
    n = len(ms)
    for i in range(n):
        _, s1, e1 = ms[i]
        for j in range(i + 1, n):
            _, s2, e2 = ms[j]
            if _overlap(s1, e1, s2, e2):
                pairs += 1
    return pairs


def _count_nested(raw_markers: list[dict]) -> int:
    nest = 0
    ms = [(m.get("start"), m.get("end")) for m in (raw_markers or [])]
    ms = [(s, e) for s, e in ms if isinstance(s, int) and isinstance(e, int) and e > s]
    for i, (s1, e1) in enumerate(ms):
        for j, (s2, e2) in enumerate(ms):
            if i == j:
                continue
            if s1 <= s2 and e2 <= e1:  # j is nested in i
                nest += 1
    return nest


def _complexity_score(raw_markers: list[dict]) -> tuple:
    """
    Higher is better. Returns a tuple for stable sorting:
      (span_count, overlap_pairs, nested_pairs, unique_labels, action_effect_cooccur)
    """
    span_count = len(raw_markers or [])
    overlaps = _count_overlaps(raw_markers)
    nested = _count_nested(raw_markers)
    labs = {m.get("label") or m.get("type") for m in (raw_markers or [])}
    uniq = len(labs)
    ae = int(("Action" in labs) and ("Effect" in labs))
    return (span_count, overlaps, nested, uniq, ae)


def evidence_subtypes(text: str) -> set[str]:
    t = text.strip()
    subs = set()
    if URL_RE.search(t):
        subs.add("url")
    if NAMED_SOURCE_RE.search(t):
        subs.add("named")
    if QUOTE_ATTR_RE.search(t):
        subs.add("quote_attr")
    if NUMERIC_SOURCE_RE.search(t):
        subs.add("numeric_source")
    return subs


def actor_subtypes(text: str) -> set[str]:
    t = text.strip()
    subs = set()
    if VAGUE_ACTOR_RE.search(t):
        subs.add("vague_collective")
    if NAMED_SOURCE_RE.search(t):
        subs.add("named_org")
    if NAMED_PERSON_RE.search(t):
        subs.add("named_person")
    return subs


def score_span(label: str, text: str) -> int:
    t = text.strip()
    s = 0
    if label == "Evidence":
        s += len(evidence_subtypes(t)) * 2
    elif label == "Action":
        if VERBISH_RE.search(t):
            s += 2
        if len(t.split()) >= 2:
            s += 1
    elif label == "Effect":
        if PURPOSE_RE.search(t):
            s += 2
        if len(t.split()) >= 3:
            s += 1
    elif label == "Actor":
        s += len(actor_subtypes(t))
    elif label == "Victim":
        if re.search(r"\b(victim|people|citizens|children|us|we|them)\b", t, re.I):
            s += 1
    return s


def doc_signals(doc) -> dict:
    labs = {s.label for s in doc.spans}
    return {
        "has_AE": ("Action" in labs and "Effect" in labs),
        "has_Victim": ("Victim" in labs),
        "ev_subtypes": set().union(
            *(evidence_subtypes(s.text) for s in doc.spans if s.label == "Evidence")
        )
        or set(),
        "actor_subtypes": set().union(
            *(actor_subtypes(s.text) for s in doc.spans if s.label == "Actor")
        )
        or set(),
        "mean_span_score": (
            (
                sum(score_span(s.label, s.text) for s in doc.spans)
                / max(1, len(doc.spans))
            )
            if doc.spans
            else 0
        ),
    }


@dataclass
class Span:
    label: str
    text: str


@dataclass
class Doc:
    text: str
    spans: list[Span]
    doc_id: str | None = None


def _norm_marker(m: dict) -> Optional[Span]:
    lab = (m.get("type") or m.get("label") or "").strip()
    txt = (m.get("text") or "").strip()
    if lab in LABELS and txt:
        return Span(lab, txt)
    return None


def _load_s1_docs(path: str) -> list[Doc]:
    docs: list[Doc] = []

    # DEBUG counters for ID keys
    id_key_counts = {"doc_id": 0, "_id": 0, "id": 0}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            # debug: count which id-like keys exist in raw JSON
            for k in id_key_counts:
                if k in obj:
                    id_key_counts[k] += 1

            text = obj.get("text") or ""
            raw_markers = obj.get("markers") or obj.get("spans") or []
            spans: list[Span] = []
            for m in raw_markers:
                nm = _norm_marker(m)
                if nm:
                    spans.append(nm)

            # NEW: pick doc_id from either doc_id or _id
            did = obj.get("doc_id") or obj.get("_id")

            d = Doc(text=text, spans=spans, doc_id=did)
            # keep raw markers (with start/end) for complexity computation
            setattr(d, "_raw_markers", raw_markers)
            docs.append(d)

    # DEBUG summary
    with_id = sum(1 for d in docs if d.doc_id is not None)
    print(
        f"[debug S1-load] loaded {len(docs)} docs from {path} | "
        f"with_doc_id={with_id} | id_key_counts={id_key_counts}"
    )
    if with_id:
        sample_ids = sorted({str(d.doc_id) for d in docs if d.doc_id is not None})[:5]
        print(f"[debug S1-load] sample S1 doc_ids: {sample_ids}")

    return docs


def _has_ae(spans: Iterable[Span]) -> bool:
    labs = {s.label for s in spans}
    return "Action" in labs and "Effect" in labs


def _pick_clean_negatives(docs: list[Doc], k: int = 2) -> list[Doc]:
    cues = re.compile(r"(http|www\.|@\w|%|\d{4})|deep state|globalist|they|elite", re.I)
    cands = [
        d
        for d in docs
        if not d.spans and 60 <= len(d.text) <= 220 and not cues.search(d.text)
    ]
    random.shuffle(cands)
    return cands[:k]


def _label_counts(doc: Doc) -> Counter:
    return Counter(s.label for s in doc.spans)


def _cap_one_per_label(spans: list[Span]) -> list[Span]:
    out, seen = [], set()
    for s in spans:
        if s.label in seen:
            continue
        seen.add(s.label)
        out.append(s)
    return out


def _pick_hard_negatives(docs: list[Doc], k: int = 1) -> list[Doc]:
    cues = re.compile(
        r"(they|elite|deep state|globalist|agenda|cover[- ]?up|manipulat|engineer|weaponiz)",
        re.I,
    )
    cands = [
        d
        for d in docs
        if not d.spans and 80 <= len(d.text) <= 240 and cues.search(d.text)
    ]
    random.shuffle(cands)
    return cands[:k]


def top_up_s1(chosen_pos: list, positives: list, pos_budget: int) -> list:
    if len(chosen_pos) >= pos_budget:
        return chosen_pos
    seen = {id(x) for x in chosen_pos}
    for d in positives:
        if id(d) in seen:
            continue
        if not d.get("spans"):
            continue
        chosen_pos.append(d)
        seen.add(id(d))
        if len(chosen_pos) >= pos_budget:
            break
    return chosen_pos


# --- complexity helpers ---
def _overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
    return (s1 < e2) and (s2 < e1)


def _count_overlaps(raw_markers: list[dict]) -> int:
    ms = [(m.get("start"), m.get("end")) for m in (raw_markers or [])]
    ms = [(s, e) for s, e in ms if isinstance(s, int) and isinstance(e, int) and e > s]
    c = 0
    for i in range(len(ms)):
        s1, e1 = ms[i]
        for j in range(i + 1, len(ms)):
            s2, e2 = ms[j]
            if _overlap(s1, e1, s2, e2):
                c += 1
    return c


def _count_nested(raw_markers: list[dict]) -> int:
    ms = [(m.get("start"), m.get("end")) for m in (raw_markers or [])]
    ms = [(s, e) for s, e in ms if isinstance(s, int) and isinstance(e, int) and e > s]
    c = 0
    for i, (s1, e1) in enumerate(ms):
        for j, (s2, e2) in enumerate(ms):
            if i == j:
                continue
            if s1 <= s2 and e2 <= e1:
                c += 1
    return c


def _topup_s1_bank(
    s1_bank: list[dict],
    extra_positive_candidates: list[dict],
    target_k: int,
    rng_seed: int = 13,
) -> list[dict]:
    """
    Ensure we end up with exactly target_k S1 fewshots by topping up
    with additional positive candidates (no-marker negatives are already capped).

    - s1_bank: current fewshot list after complex selection + negative trimming.
    - extra_positive_candidates: pool of additional S1-positive docs (with spans)
      that we haven't used yet.
    """
    if len(s1_bank) >= target_k or not extra_positive_candidates:
        return s1_bank

    import random

    random.seed(rng_seed)
    # avoid duplicates
    used_ids = {ex.get("doc_id") or ex.get("_id") for ex in s1_bank}
    remaining = [
        ex
        for ex in extra_positive_candidates
        if (ex.get("doc_id") or ex.get("_id")) not in used_ids
    ]
    random.shuffle(remaining)

    needed = target_k - len(s1_bank)
    s1_bank.extend(remaining[:needed])
    return s1_bank


def _complexity_tuple(raw_markers: list[dict]) -> tuple[int, int, int, int, int]:
    # span_count, overlaps, nested, unique_labels, AE_cooccur
    labs = {m.get("label") or m.get("type") for m in (raw_markers or [])}
    return (
        len(raw_markers or []),
        _count_overlaps(raw_markers),
        _count_nested(raw_markers),
        len(labs),
        int(("Action" in labs) and ("Effect" in labs)),
    )


def build_s1_diversified_fewshots(
    train_jsonl: str,
    *,
    total_examples: int = 10,
    negatives: int = 2,
    include_why: bool = False,
    max_label_skew: int = 4,
    rng_seed: int = 7,
    # --- complex/normal mix controls ---
    complex_k: int = 3,  # set 3 or 4
    min_spans: int = 6,
    min_overlap: int = 1,
    cap_one_per_label_normal: bool = True,
    max_spans_per_ex_normal: int = 10,
) -> list[dict]:
    """
    Target composition (when total_examples=10, negatives=2):
      - complex_k in {3,4} complex examples (many spans, overlap)
      - normal_k = (total_examples - negatives - complex_k) normal examples (diverse but compact)
      - negatives = 2 (1 hard + 1 clean)
    Falls back to keep the same totals if pools are insufficient.
    """
    rng = random.Random(rng_seed)

    # ---------- load & prep ----------
    docs: list[Doc] = _load_s1_docs(
        train_jsonl
    )  # preserves _raw_markers for complexity and doc_id if present
    print(f"[mix] loaded {len(docs)} S1 docs from {train_jsonl}")
    positives = [d for d in docs if d.spans]
    if not positives:
        # all negatives if nothing annotated
        need = total_examples
        negs = _pick_hard_negatives(docs, k=min(1, need)) + _pick_clean_negatives(
            docs, k=max(0, need - 1)
        )
        out = [
            {
                "text": d.text,
                "spans": [],
                "doc_id": getattr(d, "doc_id", None),
            }
            for d in negs[:need]
        ]
        rng.shuffle(out)
        return out

    # quality signals
    sig = {id(d): doc_signals(d) for d in positives}

    # complexity signals
    comp = {id(d): _complexity_tuple(getattr(d, "_raw_markers", [])) for d in positives}
    complex_pool = [
        d
        for d in positives
        if comp[id(d)][0] >= min_spans and comp[id(d)][1] >= min_overlap
    ]
    complex_pool.sort(key=lambda d: comp[id(d)], reverse=True)

    # normal pool (exclude complex to avoid duplicates)
    normal_pool = [d for d in positives if d not in complex_pool]
    normal_pool.sort(
        key=lambda d: (
            sig[id(d)]["mean_span_score"],
            sig[id(d)]["has_AE"],
            sig[id(d)]["has_Victim"],
        ),
        reverse=True,
    )

    pos_budget = max(0, total_examples - negatives)
    print(
        f"[mix] targets -> complex_k={complex_k} normal_k={max(0, pos_budget - complex_k)} neg={negatives}"
    )
    print(
        f"[mix] pools   -> complex_pool={len(complex_pool)} normal_pool={len(normal_pool)} positives={len(positives)}"
    )

    complex_target = min(complex_k, pos_budget)
    normal_target = pos_budget - complex_target

    chosen: list[Doc] = []
    label_bank_counts: Dict[str, int] = {lab: 0 for lab in LABELS}

    def _labels_after_cap(d: Doc) -> Set[str]:
        # count labels assuming 1-per-label cap (for skew bookkeeping only)
        seen, out = set(), set()
        for s in d.spans:
            if s.label in seen:
                continue
            seen.add(s.label)
            out.add(s.label)
        return out

    def _skew_ok(labels_to_add: Set[str]) -> bool:
        for lab in labels_to_add:
            if label_bank_counts.get(lab, 0) + 1 > max_label_skew:
                return False
        return True

    # ---------- pick complex first ----------
    for d in complex_pool:
        if len(chosen) >= complex_target:
            break
        labs_cap = _labels_after_cap(d)
        if not _skew_ok(labs_cap):
            continue
        chosen.append(d)
        for lab in labs_cap:
            label_bank_counts[lab] += 1

    # if complex pool too small, backfill its deficit into normal_target
    if len(chosen) < complex_target:
        normal_target += complex_target - len(chosen)

    # ---------- pick normal with diversity (evidence/actor subtypes, Victim, AE) ----------
    need_ev = {"url", "named", "quote_attr", "numeric_source"}
    need_actor = {"vague_collective", "named_org", "named_person"}
    covered_labels: Set[str] = set()
    normals: list[Doc] = []

    # pass 1: evidence subtypes
    for d in normal_pool:
        if len(normals) >= normal_target:
            break
        if "Evidence" not in {s.label for s in d.spans}:
            continue
        subs = sig[id(d)]["ev_subtypes"]
        if not subs or subs.isdisjoint(need_ev):
            continue
        labs_cap = _labels_after_cap(d)
        if not _skew_ok(labs_cap):
            continue
        normals.append(d)
        covered_labels |= {s.label for s in d.spans}
        need_ev -= subs
        for lab in labs_cap:
            label_bank_counts[lab] += 1

    # pass 2: actor subtypes
    for d in normal_pool:
        if len(normals) >= normal_target:
            break
        if d in normals:
            continue
        if "Actor" not in {s.label for s in d.spans}:
            continue
        subs = sig[id(d)]["actor_subtypes"]
        if not subs or subs.isdisjoint(need_actor):
            continue
        labs_cap = _labels_after_cap(d)
        if not _skew_ok(labs_cap):
            continue
        normals.append(d)
        covered_labels |= {s.label for s in d.spans}
        need_actor -= subs
        for lab in labs_cap:
            label_bank_counts[lab] += 1

    # pass 3: ensure Victim
    if not any("Victim" in {s.label for s in x.spans} for x in normals):
        for d in normal_pool:
            if len(normals) >= normal_target:
                break
            if d in normals:
                continue
            if not sig[id(d)]["has_Victim"]:
                continue
            labs_cap = _labels_after_cap(d)
            if not _skew_ok(labs_cap):
                continue
            normals.append(d)
            covered_labels |= {s.label for s in d.spans}
            for lab in labs_cap:
                label_bank_counts[lab] += 1
            break

    # pass 4: ensure AE co-occur
    if not any({"Action", "Effect"} <= {s.label for s in x.spans} for x in normals):
        for d in normal_pool:
            if len(normals) >= normal_target:
                break
            if d in normals:
                continue
            if not sig[id(d)]["has_AE"]:
                continue
            labs_cap = _labels_after_cap(d)
            if not _skew_ok(labs_cap):
                continue
            normals.append(d)
            covered_labels |= {s.label for s in d.spans}
            for lab in labs_cap:
                label_bank_counts[lab] += 1
            break

    # pass 5: top-up normals by quality
    if len(normals) < normal_target:
        seen = {id(x) for x in normals}
        for d in normal_pool:
            if len(normals) >= normal_target:
                break
            if id(d) in seen:
                continue
            labs_cap = _labels_after_cap(d)
            if not _skew_ok(labs_cap):
                continue
            normals.append(d)
            seen.add(id(d))
            for lab in labs_cap:
                label_bank_counts[lab] += 1

    # final positive set in target mix
    def _ensure_role_docs(
        chosen_docs: list[Doc], role: str, target_docs: int
    ) -> list[Doc]:
        """
        Ensure that at least `target_docs` of the chosen positives contain `role`,
        by swapping in good candidates from the remaining positives when possible.
        Does not change the total positive count.
        """

        def _has_role(d: Doc) -> bool:
            return any(s.label == role for s in d.spans)

        docs_with_role = [d for d in chosen_docs if _has_role(d)]
        if len(docs_with_role) >= target_docs:
            return chosen_docs

        chosen_set = set(chosen_docs)
        # candidates: positives that have the role but aren't currently chosen
        candidates = [d for d in positives if d not in chosen_set and _has_role(d)]
        if not candidates:
            return chosen_docs

        # prefer high-quality candidates (by mean_span_score) first
        candidates.sort(
            key=lambda d: sig.get(id(d), {}).get("mean_span_score", 0.0),
            reverse=True,
        )

        # docs we are allowed to drop: those without the role
        cand_drops = [d for d in chosen_docs if not _has_role(d)]
        if not cand_drops:
            return chosen_docs

        # Prefer dropping non-complex docs with lower mean span score
        def _is_complex(d: Doc) -> bool:
            return d in complex_pool

        cand_drops.sort(
            key=lambda d: (
                _is_complex(d),  # False (normal) first
                sig.get(id(d), {}).get("mean_span_score", 0.0),
            )
        )

        for cand in candidates:
            if len(docs_with_role) >= target_docs or not cand_drops:
                break
            drop = cand_drops.pop(0)
            if drop in chosen_docs:
                chosen_docs.remove(drop)
            chosen_docs.append(cand)
            docs_with_role.append(cand)

        return chosen_docs

    chosen = chosen + normals
    # if still short, attempt strict top-up (with skew) then relaxed top-up (ignore skew)
    if len(chosen) < pos_budget:
        remaining = [d for d in positives if d not in chosen]
        remaining.sort(
            key=lambda d: (comp[id(d)], sig[id(d)]["mean_span_score"]), reverse=True
        )
        # pass A: respect skew
        for d in list(remaining):
            if len(chosen) >= pos_budget:
                break
            labs_cap = _labels_after_cap(d)
            if not _skew_ok(labs_cap):
                continue
            chosen.append(d)
            for lab in labs_cap:
                label_bank_counts[lab] += 1
            remaining.remove(d)
    if len(chosen) < pos_budget:
        # pass B: RELAX skew to ensure we meet the budget
        print(
            f"[mix] relaxing skew for final top-up: need {pos_budget - len(chosen)} more"
        )
        for d in remaining:
            if len(chosen) >= pos_budget:
                break
            chosen.append(d)
        # no label_bank_counts update needed here for diagnostics

    # --- NEW: gently enforce doc-level Victim/Evidence coverage if possible ---
    # chosen = _ensure_role_docs(chosen, "Victim", target_docs=2)
    # chosen = _ensure_role_docs(chosen, "Evidence", target_docs=2)

    # ---------- negatives ----------
    hard = _pick_hard_negatives(docs, k=min(1, negatives))
    clean = _pick_clean_negatives(docs, k=max(0, negatives - len(hard)))
    picked = chosen + hard + clean
    print(
        f"[mix] built -> positives={len(chosen)} (target {pos_budget})  "
        f"negatives={len(hard)+len(clean)} (target {negatives})  "
        f"total={len(picked)} (target {total_examples})"
    )

    # ---------- emit JSON ----------
    def _emit_doc(d: Doc, complex_doc: bool) -> dict:
        spans = list(d.spans)
        if not complex_doc:
            # normal: compact prompt styling
            if cap_one_per_label_normal:
                seen: Set[str] = set()
                capped: list[Span] = []
                for s in spans:
                    if s.label in seen:
                        continue
                    seen.add(s.label)
                    capped.append(s)
                spans = capped
            if max_spans_per_ex_normal and len(spans) > max_spans_per_ex_normal:
                spans = spans[:max_spans_per_ex_normal]
        # else: complex keeps ALL normalized spans (show density/overlap)

        spans_json: list[dict] = []
        for s in spans:
            item = {"label": s.label, "text": s.text}
            if include_why:
                item["why"] = ""  # filled later
            spans_json.append(item)

        return {
            "text": d.text,
            "spans": spans_json,
            "doc_id": getattr(d, "doc_id", None),
        }

    # mark which picked are complex (from complex_pool) using object ids
    complex_ids = {id(d) for d in complex_pool}
    out = [_emit_doc(d, complex_doc=(id(d) in complex_ids)) for d in picked]

    rng.shuffle(out)
    return out


class S1Label(str, Enum):
    Actor = "Actor"
    Action = "Action"
    Effect = "Effect"
    Victim = "Victim"
    Evidence = "Evidence"


class S1WhyDeps(BaseModel):
    model_config = ConfigDict(extra="ignore")
    raw_text: str
    span_text: str
    label: S1Label
    doc_id: Optional[str] = None


class S1WhyOut(BaseModel):
    why: str = Field(..., description="Concise rationale for why span_text fits label.")
    context: str = Field("", description="verbatim snippet from RAW near the span")


S1_WHY_SYSTEM = f"""
{psycho_theory_preamble()}

{playbook_block()}

{data_profile_block()}

<task name="rationale_with_context">
  You are given:
    - RAW: the full preprocessed submission statement (Reddit text after preprocessing).
    - TEXT: one extracted span from RAW.
    - LABEL: one of [Actor, Action, Effect, Victim, Evidence].

  Your job is to SUPPORT why TEXT fits LABEL, and to show a short local context from RAW.

  Marker guidance:
    - Actor: agents alleged to secretly orchestrate events (people, groups, institutions, vague collectives like "they", "the elite").
    - Action: deliberate, controllable acts attributed to an Actor (verb phrase; what they DO, not the outcome).
    - Effect: consequence, goal, or purpose of an Action (what happens or is intended as a result).
    - Victim: entity harmed or targeted by the Action.
    - Evidence: explicit support for claims (URLs, named sources, reports, quotes with attribution, or numeric facts tied to a source).

  Rules:
    - SUPPORTIVE ONLY — never write "Not LABEL" or otherwise contradict LABEL. If unsure, give the best plausible justification for LABEL.
    - The WHY must reference at least one concrete lexical cue or phrase from TEXT or its immediate context (not just "fits the definition").
    - Do NOT invent new spans or indices; TEXT and context must be copied from RAW.
    - The context must be from the same sentence as TEXT when possible; otherwise use the nearest neighboring sentence.
    - No fabricated URLs, dates, or named sources; only what appears in RAW.
    - Length limits:
        * why: at most 25 words.
        * context: at most 25 words, copied verbatim from RAW (you may truncate a sentence as long as it remains grammatical).

  Output: ONLY strict JSON (no extra text, no comments):
  {{
    "why": "<concise explanation, <=25 words, supporting why TEXT fits LABEL and mentioning a concrete cue>",
    "context": "<verbatim local snippet from RAW (<=25 words) or empty string>"
  }}
</task>
""".strip()


def build_s1_why_user(raw_text: str, span_text: str, label: str) -> str:
    return (
        "<inputs>"
        f"<label>{label}</label>"
        "<span_text>" + span_text + "</span_text>"
        "<raw_text>" + raw_text + "</raw_text>"
        "</inputs>\n"
        '<format>Return ONLY JSON: {"why": "...", "context": "..."} </format>'
    )


agent_s1_why = Agent(
    LLM,
    output_type=S1WhyOut,
    system_prompt=S1_WHY_SYSTEM,
    deps_type=S1WhyDeps,
    retries=4,
    output_retries=4,
    model_settings=ModelSettings(temperature=0.0, max_output_tokens=1024),
)


def _fill_one_why(
    raw_text: str, span_text: str, label: str, doc_id: str | None
) -> tuple[str, str]:
    """
    Returns (why, context) per Option B schema.
    """
    user = build_s1_why_user(raw_text, span_text, label)
    deps = S1WhyDeps(raw_text=raw_text, span_text=span_text, label=label, doc_id=doc_id)
    res = agent_s1_why.run_sync(user, deps=deps, message_history=[])
    why = (res.output.why or "").strip()
    ctx = (getattr(res.output, "context", "") or "").strip()
    print(f"[s1 why] label={label} span='{span_text}' why='{why}' ctx='{ctx}'")
    return why, ctx


def fill_s1_whys_with_bedrock_pydantic(
    examples: list[dict],
    *,
    sleep_between: float = 0.0,
) -> list[dict]:
    """
    Fills each span with {why, context}. If the model omits context, we still write an empty string.
    Caches (why, context) per (label, span_text, text_hash) to avoid duplicate LLM calls.
    """
    import hashlib, time

    def _thash(t: str) -> str:
        return hashlib.sha1(t.encode("utf-8")).hexdigest()[:10]

    cache: dict[tuple[str, str, str], tuple[str, str]] = {}
    out = []

    for i, ex in enumerate(examples):
        text = ex["text"]
        spans = ex.get("spans", [])
        text_h = _thash(text)

        new_spans = []
        for s in spans:
            lab = s["label"]
            span_txt = s["text"]

            # Skip if both already present (idempotent)
            if s.get("why") and ("context" in s):
                new_spans.append(s)
                continue

            key = (lab, span_txt, text_h)
            pair = cache.get(key)
            if not pair:
                try:
                    pair = _fill_one_why(text, span_txt, lab, doc_id=f"s1_ex_{i}")
                except Exception as e:
                    print(f"[warn] S1 why gen failed: {e}")
                    pair = ("Fits the label per the playbook and definitions.", "")
                cache[key] = pair
                if sleep_between > 0:
                    time.sleep(sleep_between)

            why, ctx = pair
            ns = dict(s)
            ns["why"] = (ns.get("why") or why or "").strip()
            # if caller previously set context, keep it; else write model's context (may be empty)
            ns["context"] = (ns.get("context") or ctx or "").strip()
            new_spans.append(ns)

        # Preserve original metadata (doc_id, etc.)
        new_ex = dict(ex)
        new_ex["spans"] = new_spans
        out.append(new_ex)

    return out


# Uses the generated "why" plus lightweight per-label rules.


def _why_says_not(s: dict) -> bool:
    w = (s.get("why") or "").lower().strip()
    return (
        w.startswith("not ")
        or "not an action" in w
        or "not evidence" in w
        or "not an actor" in w
        or "not an effect" in w
        or "not a victim" in w
    )


def _is_good_action(txt: str) -> bool:
    t = (txt or "").strip()
    # prefer deliberate, controllable verb phrases (use existing VERBISH_RE)
    return bool(VERBISH_RE.search(t))


def _is_good_effect(txt: str) -> bool:
    t = (txt or "").strip()
    # purpose/result cues or reasonably long NP/VP
    return bool(PURPOSE_RE.search(t)) or len(t.split()) >= 4


def _is_good_evidence(txt: str) -> bool:
    t = (txt or "").strip()
    # URL, named outlet/person, quote+attribution, or numeric+unit
    return bool(
        URL_RE.search(t)
        or NAMED_SOURCE_RE.search(t)
        or QUOTE_ATTR_RE.search(t)
        or NUMERIC_SOURCE_RE.search(t)
    )


def _is_good_actor(txt: str) -> bool:
    t = (txt or "").strip()
    # discourage lone "I" and very short fragments as conspirators
    if t.lower() == "i":
        return False
    if len(t.split()) < 2:
        # very short actor mentions are almost always noisy in the training data
        return False
    return True


def _filter_spans(spans: list[dict]) -> list[dict]:
    """
    Light hygiene for S1 fewshots:
      - drop spans with < 2 tokens
      - drop Victim/Effect spans that look like pure time anchors
      - keep Evidence stricter (no bare dates/URLs/etc.)
      - deduplicate noisy spans (same text, same or different labels)
    """
    cleaned: list[dict] = []

    # --- 1) basic hygiene ---
    for s in spans or []:
        lab = (s.get("label") or "").strip()
        txt = (s.get("text") or "").strip()
        if not lab or not txt:
            continue

        toks = txt.split()
        # very short spans are usually junk for teaching the model
        if len(toks) < 2:
            continue

        # Victim/Effect spans that are actually time anchors
        if lab in {"Victim", "Effect"} and _looks_like_time_anchor(txt):
            continue

        # Evidence: reject tiny things that are clearly date/place/URL noise
        if lab == "Evidence":
            if len(toks) <= 2 and (
                re.search(r"\bhttps?://|\bwww\.", txt)
                or DATE_ONLY_RE.match(txt)
                or TIME_ANCHOR_RE.search(txt)
            ):
                continue

        cleaned.append({**s, "label": lab, "text": txt})

    if not cleaned:
        return []

    # --- 2) group spans by text to handle duplicates/multi-label conflicts ---
    by_text: dict[str, list[dict]] = {}
    for s in cleaned:
        t = s["text"]
        by_text.setdefault(t, []).append(s)

    resolved: list[dict] = []
    for txt, group in by_text.items():
        # de-duplicate same-label duplicates first
        dedup_group: list[dict] = []
        seen_labels: set[str] = set()
        for s in group:
            lab = s["label"]
            if lab in seen_labels:
                # drop exact same (label, text) duplicates such as repeated "Bill Clinton's"
                continue
            seen_labels.add(lab)
            dedup_group.append(s)

        if len(dedup_group) == 1:
            resolved.extend(dedup_group)
            continue

        # If the same text appears with multiple labels, prefer Evidence when present
        labels = {s["label"] for s in dedup_group}
        if "Evidence" in labels:
            ev_span = next(s for s in dedup_group if s["label"] == "Evidence")
            resolved.append(ev_span)
        else:
            # otherwise just keep the first variant; this keeps the example simple
            resolved.append(dedup_group[0])

    return resolved


def _apply_s1_post_filter(
    bank: list[dict],
    *,
    drop_empty_docs: bool = True,
) -> list[dict]:
    cleaned: list[dict] = []
    for ex in bank:
        spans_in = ex.get("spans", [])
        spans_out = _filter_spans(spans_in)

        # if we expected markers but filtered them all, optionally drop the doc
        if spans_in and not spans_out and drop_empty_docs:
            continue

        cleaned.append(
            {
                "text": ex.get("text", ""),
                "spans": spans_out,
                # NEW: preserve doc_id (and any other metadata you care about)
                "doc_id": ex.get("doc_id"),
            }
        )
    return cleaned


class S2Deps(BaseModel):
    model_config = ConfigDict(extra="ignore")
    raw_text: str
    gold_label: Literal["conspiracy", "non"]
    doc_id: str | None = None


class S2RationaleOut(BaseModel):
    rationale: str = Field(..., description="2 concise sentences naming decisive cues.")


S2_RATIONALE_SYSTEM = f"""
{psycho_theory_preamble()}

{playbook_block()}

<task name="rationale_only">
  You are given:
    - RAW document text, and
    - its GOLD label in {{conspiracy, non}} (do NOT predict or change it).
    Optionally, you may also receive extracted S1 markers:
      Actor, Action, Effect, Victim, Evidence.

  Your job:
    - Write ONE concise rationale (< 40 words) that justifies WHY the gold label is appropriate.
    - The rationale must be consistent with the gold label, even if the text is noisy or ambiguous.

  Using markers (if provided):
    - Treat S1 markers as noisy hints about roles in the narrative, not as ground truth.
    - When useful, explicitly refer to them in abstract terms:
        e.g., "vague Actor", "hostile Action", "grand Effect", "cited Evidence", "Victim framing".
    - Always base the rationale on the full document and the authorial stance,
      even when markers highlight conspiratorial-looking spans.
    - Markers alone never force "conspiracy" or "non"; they are evidence to interpret.

  Handling disagreement between markers and gold label:
    - If markers suggest a conspiratorial pattern but the gold label is "non",
      explain that the text reports, questions, or critiques such claims
      instead of endorsing a hidden mechanism.
    - If markers are sparse but the gold label is "conspiracy",
      focus on the strongest cues for Actor+Action+Effect or self-sealing epistemics.

  Label-specific guidance:
    - If label == "conspiracy":
        - Point to how the text communicates a conspiratorial mechanism:
          at least one cue about a coordinated Actor, intentional Action,
          grand/collective Effect, or self-sealing epistemics.
        - You MAY mention that multiple markers (Actor/Action/Effect/Evidence)
          align to a hidden plot.
    - If label == "non":
        - Explain which conspiratorial cues are missing, weak, or framed as critique/debunking.
        - If conspiracy language or markers appear, note that it is quoted, reported,
          fictional, or explicitly questioned rather than endorsed.

  Style and format:
    - Write a single sentence, less than 40 words.
    - Do NOT quote long fragments of the document.
    - Do NOT mention "gold label", "markers", or "S1" explicitly; talk in semantic terms.
    - Return ONLY JSON:
      {{"rationale": "<your 1-sentence explanation>"}}
</task>
""".strip()


def build_s2_rat_user(raw_text: str, gold_label: str) -> str:
    return (
        "<inputs>"
        f"<gold_label>{gold_label}</gold_label>"
        "<raw_text>" + raw_text + "</raw_text>"
        "</inputs>\n"
        '<format>Return ONLY JSON: {"rationale": "..."}</format>'
    )


agent_s2_rationale = Agent(
    LLM,
    output_type=S2RationaleOut,
    system_prompt=S2_RATIONALE_SYSTEM,
    deps_type=S2Deps,
    retries=4,
    output_retries=4,
    model_settings=ModelSettings(temperature=0.0, max_output_tokens=512),
)


def make_s2_item_with_rationale(text: str, gold_label: str, doc_id: str):
    user = build_s2_rat_user(text, gold_label)
    deps = S2Deps(raw_text=text, gold_label=gold_label, doc_id=doc_id)
    res = agent_s2_rationale.run_sync(user, deps=deps, message_history=[])
    rationale = (res.output.rationale or "").strip()
    print(f"[debug] S2 rationale for doc_id={doc_id} label={gold_label}: {rationale}")
    return {"text": text, "label": gold_label.lower(), "rationale": rationale}


def _pick_with_subreddit_diversity(cands: list[dict], k: int, score_fn) -> list[dict]:
    # sort by score
    sorted_cands = sorted(cands, key=score_fn, reverse=True)
    picked: list[dict] = []
    seen_subs: set[str] = set()

    for r in sorted_cands:
        if len(picked) >= k:
            break
        sub = (r.get("subreddit") or "").lower()
        # prefer new subreddits, but allow repeats if we run out
        if sub not in seen_subs or len(seen_subs) >= k:
            picked.append(r)
            if sub:
                seen_subs.add(sub)
    return picked


def align_s1_s2_fewshots_by_doc_id(
    s1_bank: list[dict],
    s2_bank: list[dict],
    *,
    max_aligned: int = 2,
) -> tuple[list[dict], int]:
    """
    Attach S1 markers as `markers` on up to `max_aligned` S2 fewshots
    that share the same doc_id. We do NOT touch the rationale text here.
    """
    # build doc_id -> S1 example map
    s1_by_id: dict[str, dict] = {}
    for ex in s1_bank:
        did = ex.get("doc_id")
        if did is None:
            continue
        did = str(did)
        if did and did not in s1_by_id:
            s1_by_id[did] = ex

    print(
        f"[debug align-docid] enter: S1={len(s1_bank)} | S2={len(s2_bank)} | "
        f"S1_with_doc_id={len(s1_by_id)} | max_aligned={max_aligned}"
    )

    aligned = 0
    new_s2: list[dict] = []

    for ex in s2_bank:
        did = ex.get("doc_id")
        if aligned < max_aligned and did is not None:
            did_str = str(did)
            if did_str in s1_by_id:
                s1_ex = s1_by_id[did_str]
                markers = s1_ex.get("spans") or []
                if markers:
                    ex = dict(ex)  # shallow copy
                    ex["markers"] = markers
                    aligned += 1
                    print(
                        f"[debug align-docid] attached markers for doc_id={did_str} | "
                        f"spans={len(markers)} | aligned={aligned}"
                    )
        new_s2.append(ex)

    print(f"[debug align-docid] exit: total_aligned={aligned}")
    return new_s2, aligned


def _score_conspiracy_row(row: dict) -> float:
    text = (row.get("text") or row.get("doc_text") or "").strip()
    sig = s2_signals(text)
    score = 0.0
    score += 3.0 * sig["cues"]
    score += min(sig["length"] / 400.0, 1.0)
    if sig["debunk"]:
        score -= 4.0
    score -= 0.5 * sig["qmarks"]
    return score


def _score_non_row(row: dict) -> float:
    text = (row.get("text") or row.get("doc_text") or "").strip()
    sig = s2_signals(text)
    base = min(sig["length"] / 400.0, 1.0)
    bonus = 0.0
    if sig["debunk"]:
        bonus += 2.0
    if sig["cues"] > 0 and not sig["debunk"]:
        bonus += 1.0
    penalty_q = 0.2 * sig["qmarks"]
    return base + bonus - penalty_q


def _build_s2_fewshots_for_rows(
    rows: list[dict],
    k: int,
    *,
    attach_markers: bool,
    s1_by_id: dict[str, dict] | None = None,
    rng_seed: int = 7,
) -> list[dict]:
    import random
    from collections import Counter

    if k <= 0 or not rows:
        return []

    random.seed(rng_seed)

    cons = [r for r in rows if r.get("label") == "conspiracy"]
    nonc = [r for r in rows if r.get("label") == "non"]

    cons_sorted = sorted(cons, key=_score_conspiracy_row, reverse=True)
    nonc_sorted = sorted(nonc, key=_score_non_row, reverse=True)

    target_cons = max(1, k // 2)
    target_non = k - target_cons

    picked: list[dict] = []
    picked.extend(cons_sorted[:target_cons])
    picked.extend(nonc_sorted[:target_non])

    if len(picked) < k:
        rest = [r for r in rows if r not in picked]
        random.shuffle(rest)
        picked.extend(rest[: max(0, k - len(picked))])

    picked = picked[:k]
    random.shuffle(picked)

    fewshots: list[dict] = []
    for i, r in enumerate(picked):
        text = (r.get("text") or r.get("doc_text") or "").strip()
        gold_label = (r.get("label") or "").strip().lower() or "non"
        did_str = str(r.get("doc_id") or f"s2_{i}")

        try:
            item = make_s2_item_with_rationale(
                text=text,
                gold_label=gold_label,
                doc_id=did_str,
            )
        except Exception as e:
            print(f"[fewshot S2] ERROR creating rationale for doc_id={did_str}: {e!r}")
            fallback_rat = (
                "The author explicitly endorses a hidden, coordinated conspiracy with vague hostile actors and extreme stakes."
                if gold_label == "conspiracy"
                else "The text reports or analyses without endorsing a hidden, coordinated conspiracy mechanism."
            )
            item = {
                "text": text,
                "label": gold_label,
                "rationale": fallback_rat,
            }

        if attach_markers and s1_by_id:
            s1_ex = s1_by_id.get(did_str)
            if s1_ex:
                markers = s1_ex.get("spans") or s1_ex.get("markers") or []
                if markers:
                    item["markers"] = markers

        item["task"] = "s2"
        item["doc_id"] = did_str
        fewshots.append(item)

    print(
        "[report] _build_s2_fewshots_for_rows: "
        f"label_counts={Counter(fs['label'] for fs in fewshots)} | "
        f"n={len(fewshots)} | attach_markers={attach_markers}"
    )

    return fewshots


def build_s2_fewshots_with_llm_pydantic(
    train_docclf_jsonl: str,
    *,
    k: int = 8,
    rng_seed: int = 7,
    s1_bank: list[dict] | None = None,
) -> list[dict]:
    """
    Hybrid S2 fewshots:
      - some from docs that appear in S1 fewshot bank (with markers attached),
      - some from the rest of S2 train (no markers).

    This teaches the model how to use S1 markers when present,
    without restricting all S2 fewshots to the tiny S1-overlap subset.
    """
    import json
    import random
    from collections import Counter

    try:
        k_int = int(k)
    except Exception:
        print(
            f"[debug S2-k-type] k={k!r} (type={type(k)}) was not int-castable; falling back to 8"
        )
        k_int = 8

    random.seed(rng_seed)

    # --- map S1 fewshot doc_ids -> examples (for markers) ---
    s1_by_id: dict[str, dict] = {}
    if s1_bank:
        for ex in s1_bank:
            did = ex.get("doc_id") or ex.get("_id")
            if did is None:
                continue
            did_str = str(did)
            spans = ex.get("spans") or ex.get("markers") or []
            if not spans:
                continue
            if did_str not in s1_by_id:
                s1_by_id[did_str] = ex

    print(f"[debug S2-hybrid] S1 fewshot docs with spans={len(s1_by_id)}")

    # --- load all S2 rows ---
    all_rows: list[dict] = []
    with open(train_docclf_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = (obj.get("text") or obj.get("doc_text") or "").strip()
            if not text:
                continue

            did = obj.get("doc_id") or obj.get("_id")
            if did is None:
                continue
            did_str = str(did)

            label = (obj.get("label") or obj.get("doc_label") or "").strip().lower()
            if label not in {"conspiracy", "non"}:
                continue

            obj["doc_id"] = did_str
            obj["label"] = label
            all_rows.append(obj)

    print(f"[debug S2-hybrid] total S2 rows loaded={len(all_rows)}")
    if not all_rows:
        print("[warn S2-hybrid] No S2 rows; S2 fewshot bank will be empty.")
        return []

    # --- split into overlap vs non-overlap with S1 fewshots ---
    overlap_rows = [r for r in all_rows if r["doc_id"] in s1_by_id]
    non_overlap_rows = [r for r in all_rows if r["doc_id"] not in s1_by_id]

    print(
        f"[debug S2-hybrid] overlap_rows={len(overlap_rows)} "
        f"| non_overlap_rows={len(non_overlap_rows)}"
    )

    # how many marker-rich fewshots?
    marker_k = min(len(overlap_rows), max(3, k_int // 2))
    plain_k = max(0, k_int - marker_k)

    print(f"[debug S2-hybrid] marker_k={marker_k} | plain_k={plain_k} | k={k_int}")

    few_marker = _build_s2_fewshots_for_rows(
        overlap_rows,
        marker_k,
        attach_markers=True,
        s1_by_id=s1_by_id,
        rng_seed=rng_seed,
    )

    few_plain = _build_s2_fewshots_for_rows(
        non_overlap_rows,
        plain_k,
        attach_markers=False,
        s1_by_id=None,
        rng_seed=rng_seed + 1,
    )

    fewshots = few_marker + few_plain
    random.shuffle(fewshots)

    label_counts = Counter(fs["label"] for fs in fewshots)
    with_markers = sum(1 for fs in fewshots if fs.get("markers"))
    print(
        "[report] S2 fewshots (hybrid): "
        f"label_counts={dict(label_counts)} | n={len(fewshots)} | with_markers={with_markers}"
    )

    return fewshots[:k_int]


def _dedupe_s2_fewshots(examples: list[dict]) -> list[dict]:
    """
    Deduplicate S2 fewshots by (text, label, rationale).
    This keeps your bank compact and avoids overweighting any single doc.
    """
    seen: set[tuple[str, str, str]] = set()
    out: list[dict] = []
    for ex in examples:
        text = (ex.get("text") or "").strip()
        label = str(ex.get("label", "")).strip().lower()
        rat = (ex.get("rationale") or "").strip()
        key = (text, label, rat)
        if key in seen:
            continue
        seen.add(key)
        out.append(ex)
    return out


def align_s1_s2_fewshots_on_text(
    s1_bank: list[dict],
    s2_bank: list[dict],
    *,
    max_aligned: int = 2,
) -> tuple[list[dict], int]:
    """
    For up to `max_aligned` docs where S1 and S2 fewshots share the same text,
    attach S1 spans as a `markers` block on the S2 fewshot.

    We deliberately do NOT modify the rationale text here; that remains
    entirely generated by the Bedrock model so the style/distribution is
    consistent.
    """
    by_text: dict[str, dict] = {
        ex.get("text", ""): ex
        for ex in s1_bank
        if isinstance(ex, dict) and ex.get("text") and isinstance(ex.get("spans"), list)
    }

    print(
        f"[debug align-text] enter: S1={len(s1_bank)} | S2={len(s2_bank)} | "
        f"S1_unique_texts={len(by_text)} | max_aligned={max_aligned}"
    )

    aligned = 0
    new_s2: list[dict] = []

    for ex in s2_bank:
        t = ex.get("text", "")
        if aligned < max_aligned and t in by_text:
            s1_ex = by_text[t]
            markers = s1_ex.get("spans") or []
            if markers:
                ex = dict(ex)  # shallow copy
                ex["markers"] = markers
                aligned += 1
                print(
                    f"[debug align-text] attached markers on shared text | "
                    f"spans={len(markers)} | aligned={aligned}"
                )

        new_s2.append(ex)

    print(f"[debug align-text] exit: total_aligned={aligned}")
    return new_s2, aligned


def main():
    p = argparse.ArgumentParser()

    # S1
    p.add_argument("--s1-train-jsonl", required=True)
    p.add_argument("--s1-k", type=int, default=10)
    p.add_argument("--s1-neg-k", type=int, default=2)
    p.add_argument(
        "--s1-why-mode", choices=["none", "placeholder", "bedrock"], default="bedrock"
    )
    p.add_argument("--bedrock-sleep", type=float, default=0.0)

    # S2 (gold label given)
    p.add_argument("--build-s2-fewshots", action="store_true", default=True)
    p.add_argument("--s2-train-docclf", type=str)
    p.add_argument("--s2-k", type=int, default=8)
    # --- NEW: complexity controls for S1 ---
    p.add_argument(
        "--s1-complex",
        action="store_true",
        default=True,
        help="Prefer complex S1 examples (many spans, overlapping/nested spans).",
    )
    p.add_argument(
        "--s1-complex-k",
        type=int,
        default=4,
        help="Number of complex examples to include when in complex mode.",
    )
    p.add_argument(
        "--s1-min-spans",
        type=int,
        default=5,
        help="Minimum number of spans in a doc to be considered complex.",
    )
    p.add_argument(
        "--s1-min-overlap",
        type=int,
        default=1,
        help="Minimum number of overlapping span pairs to be considered complex.",
    )
    p.add_argument(
        "--s1-max-spans-per-ex",
        type=int,
        default=10,
        help="Soft cap of spans per example when NOT in complex mode.",
    )
    p.add_argument(
        "--s1-cap-one-per-label",
        action="store_true",
        default=False,
        help="If set, keep at most one span per label per example (NOT recommended for complex mode).",
    )

    # Out
    p.add_argument(
        "--fewshot-out", required=False, default="data/fewshots/psycomark_fewshots.json"
    )

    args = p.parse_args()

    include_why = args.s1_why_mode in {"placeholder", "bedrock"}
    s1_bank = build_s1_diversified_fewshots(
        args.s1_train_jsonl,
        total_examples=args.s1_k,
        negatives=args.s1_neg_k,
        include_why=include_why,
        rng_seed=7,
        complex_k=args.s1_complex_k,
        min_spans=args.s1_min_spans,
        min_overlap=args.s1_min_overlap,
        cap_one_per_label_normal=(not args.s1_complex and args.s1_cap_one_per_label),
        max_spans_per_ex_normal=args.s1_max_spans_per_ex,
    )
    print(f"[fewshot] built S1 fewshot bank: {len(s1_bank)} examples")

    if args.s1_why_mode == "bedrock":
        s1_bank = fill_s1_whys_with_bedrock_pydantic(
            s1_bank,
            sleep_between=args.bedrock_sleep,
        )
        print("[fewshot] filled S1 whys with Bedrock LLM")

    # --- Post-filter only in NON-complex mode; complex mode keeps dense spans ---
    if not args.s1_complex:
        before_ex = len(s1_bank)
        before_sp = sum(len(ex.get("spans", [])) for ex in s1_bank)
        s1_bank = _apply_s1_post_filter(s1_bank, drop_empty_docs=True)
        # optional soft cap when not complex
        if args.s1_max_spans_per_ex > 0:
            for ex in s1_bank:
                if len(ex["spans"]) > args.s1_max_spans_per_ex:
                    ex["spans"] = ex["spans"][: args.s1_max_spans_per_ex]
        after_ex = len(s1_bank)
        after_sp = sum(len(ex.get("spans", [])) for ex in s1_bank)
        ae_docs = sum(
            1
            for ex in s1_bank
            if {"Action", "Effect"} <= {s["label"] for s in ex.get("spans", [])}
        )
        victim_docs = sum(
            1
            for ex in s1_bank
            if any(s["label"] == "Victim" for s in ex.get("spans", []))
        )
        print(
            f"[report] S1 post-filter: ex {before_ex}->{after_ex} | spans {before_sp}->{after_sp} | AE in {ae_docs} | Victim in {victim_docs}"
        )
    else:
        # complexity visibility report
        # light filter: keep docs but drop obviously bad spans
        s1_bank = _apply_s1_post_filter(s1_bank, drop_empty_docs=False)
        # complexity visibility report
        max_sp = max((len(ex.get("spans", [])) for ex in s1_bank), default=0)
        avg_sp = sum(len(ex.get("spans", [])) for ex in s1_bank) / max(1, len(s1_bank))
        print(
            f"[report] S1 complex mode: examples={len(s1_bank)} | max_spans/ex={max_sp} | avg_spans/ex={avg_sp:.1f}"
        )

    # --- Post-filter only in NON-complex mode; complex mode keeps dense spans ---
    if not args.s1_complex:
        before_ex = len(s1_bank)
        before_sp = sum(len(ex.get("spans", [])) for ex in s1_bank)
        s1_bank = _apply_s1_post_filter(s1_bank, drop_empty_docs=True)
        # optional soft cap when not complex
        if args.s1_max_spans_per_ex > 0:
            for ex in s1_bank:
                if len(ex["spans"]) > args.s1_max_spans_per_ex:
                    ex["spans"] = ex["spans"][: args.s1_max_spans_per_ex]
        after_ex = len(s1_bank)
        after_sp = sum(len(ex.get("spans", [])) for ex in s1_bank)
        ae_docs = sum(
            1
            for ex in s1_bank
            if {"Action", "Effect"} <= {s["label"] for s in ex.get("spans", [])}
        )
        victim_docs = sum(
            1
            for ex in s1_bank
            if any(s["label"] == "Victim" for s in ex.get("spans", []))
        )
        print(
            f"[report] S1 post-filter: ex {before_ex}->{after_ex} | spans {before_sp}->{after_sp} | AE in {ae_docs} | Victim in {victim_docs}"
        )
    else:
        # light filter: keep docs but drop obviously bad spans
        s1_bank = _apply_s1_post_filter(s1_bank, drop_empty_docs=False)
        max_sp = max((len(ex.get("spans", [])) for ex in s1_bank), default=0)
        avg_sp = sum(len(ex.get("spans", [])) for ex in s1_bank) / max(1, len(s1_bank))
        print(
            f"[report] S1 complex mode: examples={len(s1_bank)} | max_spans/ex={max_sp} | avg_spans/ex={avg_sp:.1f}"
        )

    # --- LIMIT explicit "no markers" examples and then TOP UP back to s1_k ---
    max_neg_for_prompts = 2
    negatives = [ex for ex in s1_bank if not ex.get("spans")]
    if len(negatives) > max_neg_for_prompts:
        keep_ids = {id(ex) for ex in negatives[:max_neg_for_prompts]}
        trimmed_bank = []
        for ex in s1_bank:
            if ex.get("spans") or id(ex) in keep_ids:
                trimmed_bank.append(ex)
        print(
            f"[report] S1 negatives trimmed: kept {max_neg_for_prompts} explicit no-marker examples "
            f"from {len(negatives)}"
        )
        s1_bank = trimmed_bank

    # Tag remaining no-marker examples explicitly
    for ex in s1_bank:
        if not ex.get("spans"):
            ex["no_markers"] = True

    # --- TOP UP S1 back to args.s1_k with extra positive docs (with spans) ---
    if len(s1_bank) < args.s1_k:
        extra_pool: list[dict] = []
        for d in _load_s1_docs(args.s1_train_jsonl):
            if not d.spans:
                continue
            extra_pool.append(
                {
                    "text": d.text,
                    "spans": [{"label": s.label, "text": s.text} for s in d.spans],
                    "doc_id": getattr(d, "doc_id", None),
                }
            )

        used_ids = {str(ex.get("doc_id")) for ex in s1_bank if ex.get("doc_id")}
        extra_pool = [
            ex
            for ex in extra_pool
            if not ex.get("doc_id") or str(ex["doc_id"]) not in used_ids
        ]
        random.shuffle(extra_pool)

        needed = max(0, args.s1_k - len(s1_bank))
        topup = extra_pool[:needed]
        s1_bank.extend(topup)
        print(
            f"[report] S1 top-up: added {len(topup)} extra positives | "
            f"final_n={len(s1_bank)} (target={args.s1_k})"
        )

    print(f"[report] S1 fewshots final: n={len(s1_bank)} (target={args.s1_k})")

    s2_bank: list[dict] = []
    if args.build_s2_fewshots and args.s2_train_docclf:
        print(
            f"[fewshot] building S2 fewshot bank with LLM from {args.s2_train_docclf}"
        )
        s2_bank = build_s2_fewshots_with_llm_pydantic(
            args.s2_train_docclf,
            k=8,
            rng_seed=7,
            s1_bank=s1_bank,
        )

        # 1) dedupe
        orig_n = len(s2_bank)
        s2_bank = _dedupe_s2_fewshots(s2_bank)

        # 2) align 1–2 examples with S1 markers on shared texts
        aligned = 0
        if s1_bank:
            # First try doc_id-based alignment (most robust)
            s2_bank, aligned = align_s1_s2_fewshots_by_doc_id(
                s1_bank, s2_bank, max_aligned=2
            )
            # If we still have quota, fall back to text-based matching
            if aligned < 2:
                s2_bank, extra = align_s1_s2_fewshots_on_text(
                    s1_bank, s2_bank, max_aligned=2 - aligned
                )
                aligned += extra

        # 3) quick report
        from collections import Counter as _Counter

        label_counts = _Counter(str(ex.get("label", "")).lower() for ex in s2_bank)
        avg_len = sum(len(ex.get("text", "") or "") for ex in s2_bank) / max(
            1, len(s2_bank)
        )
        print(
            f"[report] S2 fewshots: label_counts={dict(label_counts)} | "
            f"avg_len={avg_len:.1f} | deduped_from={orig_n} | aligned_with_markers={aligned}"
        )

    out = {"s1": s1_bank, "s2": s2_bank}
    Path(args.fewshot_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.fewshot_out).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[fewshot] wrote → {args.fewshot_out} | S1={len(s1_bank)} | S2={len(s2_bank)}"
    )


if __name__ == "__main__":
    main()
