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
    - its GOLD label belongs in {{conspiracy, non}} (do NOT predict or change it).
    Optionally, you may also receive extracted markers:
    (Actor, Action, Effect, Victim, Evidence) from a separate S1 system.

  Your job:
    - Write ONE concise rationale (less than 40 words) that justifies WHY the gold label is appropriate.

  Using markers (if provided):
    - Treat S1 markers as noisy hints about roles in the narrative, not as ground truth.
    - You MAY refer to them in the rationale (e.g., vague Actor, hostile Action, grand Effect, explicit Evidence).
    - Always base the rationale on the full document and authorial stance, even when markers are present.
    - Markers alone never force a "conspiracy" or "non" decision.

  Label-specific guidance:
    - If label is "conspiracy":
        Explain how the text communicates a conspiratorial mechanism:
        at least one lexical cue or phrase that signals Actor, intentional Action,
        grand/collective Effect, or self-sealing epistemics.
    - If label is "non":
        Explain which conspiratorial cues are missing, weak, or framed as critique/debunking.
        If conspiracy language appears, note that it is reported, questioned, or rejected.

  Return ONLY JSON:
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


def build_s2_fewshots_with_llm_pydantic(
    train_docclf_jsonl: str,
    *,
    k: int = 6,
    rng_seed: int = 7,
    s1_bank: list[dict] | None = None,
    aligned_k: int = 2,
) -> list[dict]:
    # --- Normalize numeric args defensively ---
    try:
        k_int = int(k)
    except Exception:
        print(f"[debug S2-k] could not cast k={k!r} (type={type(k)}), defaulting to 8")
        k_int = 8

    try:
        aligned_k = int(aligned_k)
    except Exception:
        print(
            f"[debug S2-aligned_k] could not cast aligned_k={aligned_k!r} "
            f"(type={type(aligned_k)}), defaulting to 2"
        )
        aligned_k = 2

    random.seed(rng_seed)
    rows: list[dict] = []

    # DEBUG counters for ID keys in S2 file
    id_key_counts = {"doc_id": 0, "_id": 0, "id": 0}

    with open(train_docclf_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue

            # debug: raw id-like keys
            for key in id_key_counts:
                if key in obj:
                    id_key_counts[key] += 1

            text = (obj.get("text") or "").strip()
            if not text:
                continue

            # NEW: normalize doc_id here
            did = obj.get("doc_id") or obj.get("_id")
            if did is not None:
                obj["doc_id"] = did

            # Normalize label: prefer `label`, fallback to `doc_label`
            lab = obj.get("label") or obj.get("doc_label")
            if not lab:
                continue

            lab = str(lab).strip().lower()

            # --- Hard override: Pizzagate / Comet Pizza should be "conspiracy" ---
            low = text.lower()
            if any(
                kw in low
                for kw in (
                    "pizzagate",
                    "comet pizza",
                    "comet ping pong",
                )
            ):
                lab = "conspiracy"

            if lab not in {"conspiracy", "non"}:
                continue

            obj["label"] = lab
            rows.append(obj)

    # DEBUG: S2 load summary
    from collections import Counter  # already imported at top, but safe

    rows_with_doc_id = sum(1 for r in rows if r.get("doc_id") is not None)
    label_dist = Counter(r.get("label") for r in rows)
    print(
        f"[debug S2-load] loaded {len(rows)} rows from {train_docclf_jsonl} | "
        f"with_doc_id={rows_with_doc_id} | id_key_counts={id_key_counts}"
    )
    print(f"[debug S2-load] label distribution before scoring: {dict(label_dist)}")

    # --- 1) Try to pick up to `aligned_k` docs that overlap with S1 fewshots by doc_id ---
    aligned_rows: list[dict] = []
    aligned_ids: set[str] = set()

    if s1_bank and aligned_k > 0:
        # doc_ids that appear in the S1 fewshots
        s1_ids = [
            str(ex.get("doc_id")) for ex in s1_bank if ex.get("doc_id") is not None
        ]
        s1_ids = [i for i in s1_ids if i]  # drop empties
        s1_unique = set(s1_ids)

        print(
            f"[debug S2-align-pre] s1_bank={len(s1_bank)} | "
            f"S1 with doc_id={len(s1_ids)} | unique_s1_ids={len(s1_unique)}"
        )

        if s1_ids:
            # map doc_id -> first matching row in rows
            rows_by_id: dict[str, dict] = {}
            for r in rows:
                did = str(r.get("doc_id") or "")
                if did and did not in rows_by_id:
                    rows_by_id[did] = r

            overlap_ids = s1_unique & set(rows_by_id.keys())
            print(
                f"[debug S2-align-pre] S2 rows_with_doc_id={len(rows_by_id)} | "
                f"overlap_ids={len(overlap_ids)}"
            )
            if overlap_ids:
                print(
                    f"[debug S2-align-pre] sample overlap_ids="
                    f"{sorted(list(overlap_ids))[:5]}"
                )

            for did in s1_ids:
                if len(aligned_rows) >= aligned_k:
                    break
                if did in rows_by_id:
                    r = rows_by_id[did]
                    lab = r.get("label")
                    if lab in {"conspiracy", "non"}:
                        aligned_rows.append(r)
                        aligned_ids.add(did)

        print(
            f"[debug S2-align-pre] aligned_rows={len(aligned_rows)} | "
            f"aligned_ids_sample={sorted(list(aligned_ids))[:5]}"
        )

    # --- 2) Remaining pool for ordinary scoring (exclude already aligned) ---
    remaining_rows = [r for r in rows if str(r.get("doc_id") or "") not in aligned_ids]

    # Split by gold label
    cons = [r for r in remaining_rows if r.get("label") == "conspiracy"]
    nonc = [r for r in remaining_rows if r.get("label") == "non"]

    def _score_conspiracy(r: dict) -> float:
        sig = s2_signals(r.get("text", ""))
        # strong cues + decent length, penalize a lot of "??"
        return sig["cues"] * 3 + min(sig["length"] / 400.0, 1.0) - 0.3 * sig["qmarks"]

    def _score_non(r: dict) -> float:
        sig = s2_signals(r.get("text", ""))
        hard = sig["cues"] > 0  # non-conspiracy but with conspiratorial language
        base = min(sig["length"] / 400.0, 1.0)
        return base + (1.5 if hard else 0.0)

    # how many more we need beyond the aligned ones
    print(
        f"[debug S2-k-type] k={k!r} (type={type(k)}) | "
        f"aligned_rows={len(aligned_rows)}"
    )
    # how many more we need beyond the aligned ones
    remaining_k = max(0, k_int - len(aligned_rows))

    if cons and nonc and remaining_k > 0:
        half = remaining_k // 2
        cons_sorted = sorted(cons, key=_score_conspiracy, reverse=True)
        nonc_sorted = sorted(nonc, key=_score_non, reverse=True)
        sample_rest = (
            cons_sorted[: min(half, len(cons_sorted))]
            + nonc_sorted[: min(remaining_k - half, len(nonc_sorted))]
        )
        random.shuffle(sample_rest)
    else:
        sample_rest = remaining_rows[:remaining_k]

    sample = aligned_rows + sample_rest
    random.shuffle(sample)

    out: list[dict] = []
    for i, r in enumerate(sample):
        t = r.get("text", "")
        gold = r.get("label", "non")
        try:
            item = make_s2_item_with_rationale(t, gold, doc_id=str(r.get("doc_id", i)))
            # keep doc_id around; useful later for debugging
            if "doc_id" in r:
                item["doc_id"] = r["doc_id"]
        except Exception as e:
            print(f"[fewshot] ERROR building S2 item: {e!r}")
            continue
        out.append(item)

    print(
        f"[debug S2-final] fewshot_k={k} | aligned_rows_used={len(aligned_rows)} | "
        f"total_sampled={len(out)}"
    )

    return out


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
            aligned_k=2,  # force up to 2 aligned docs
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
