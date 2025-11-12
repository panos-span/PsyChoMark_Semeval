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
    from pydanticai.prompt_builder import playbook_block, psycho_theory_preamble
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


def _norm_marker(m: dict) -> Optional[Span]:
    lab = (m.get("type") or m.get("label") or "").strip()
    txt = (m.get("text") or "").strip()
    if lab in LABELS and txt:
        return Span(lab, txt)
    return None


def _load_s1_docs(path: str) -> list[Doc]:
    docs: list[Doc] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("text") or ""
            raw_markers = obj.get("markers") or obj.get("spans") or []
            spans: list[Span] = []
            for m in raw_markers:
                nm = _norm_marker(m)
                if nm:
                    spans.append(nm)
            d = Doc(text=text, spans=spans)
            # keep raw markers (with start/end) for complexity computation
            setattr(d, "_raw_markers", raw_markers)
            docs.append(d)
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
    )  # preserves _raw_markers for complexity
    positives = [d for d in docs if d.spans]
    if not positives:
        # all negatives if nothing annotated
        need = total_examples
        negs = _pick_hard_negatives(docs, k=min(1, need)) + _pick_clean_negatives(
            docs, k=max(0, need - 1)
        )
        out = [{"text": d.text, "spans": []} for d in negs[:need]]
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
    # if still short (very sparse data), fill from remaining positives by complexity/quality
    if len(chosen) < pos_budget:
        remaining = [d for d in positives if d not in chosen]
        # prefer complex first then quality
        remaining.sort(
            key=lambda d: (comp[id(d)], sig[id(d)]["mean_span_score"]), reverse=True
        )
        for d in remaining:
            if len(chosen) >= pos_budget:
                break
            labs_cap = _labels_after_cap(d)
            if not _skew_ok(labs_cap):
                continue
            chosen.append(d)
            for lab in labs_cap:
                label_bank_counts[lab] += 1

    # ---------- negatives ----------
    hard = _pick_hard_negatives(docs, k=min(1, negatives))
    clean = _pick_clean_negatives(docs, k=max(0, negatives - len(hard)))
    picked = chosen + hard + clean

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
        return {"text": d.text, "spans": spans_json}

    # mark which picked are complex (from complex_pool)
    complex_set = set(complex_pool)
    out = [_emit_doc(d, complex_doc=(d in complex_set)) for d in picked]

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


S1_WHY_SYSTEM = f"""
{psycho_theory_preamble()}

{playbook_block()}

<task name="rationale">
  Given the RAW document, a span TEXT, and its LABEL (Actor|Action|Effect|Victim|Evidence),
  write ONE short rationale (<=25 words) explaining why TEXT fits LABEL.
  - Do NOT invent new spans or indices.
  - Action = controllable verb phrase; Effect = purpose/result; Evidence = attribution/URL/named source/quote/numeric+source.
  Output: ONLY JSON -> {{"why":"..."}}
</task>
""".strip()


def build_s1_why_user(raw_text: str, span_text: str, label: str) -> str:
    return (
        "<inputs>"
        f"<label>{label}</label>"
        "<span_text>" + span_text + "</span_text>"
        "<raw_text>" + raw_text + "</raw_text>"
        "</inputs>\n"
        '<format>Return ONLY JSON: {"why": "..."}</format>'
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


def _fill_one_why(raw_text: str, span_text: str, label: str, doc_id: str | None) -> str:
    user = build_s1_why_user(raw_text, span_text, label)
    deps = S1WhyDeps(raw_text=raw_text, span_text=span_text, label=label, doc_id=doc_id)
    res = agent_s1_why.run_sync(user, deps=deps, message_history=[])
    out = res.output  # <-- pydantic model S1WhyOut
    print(f"[debug] filled why for label={label} span='{span_text}': {out.why}")
    return (out.why or "").strip() or "Fits the label per playbook and definitions."


def fill_s1_whys_with_bedrock_pydantic(
    examples: list[dict],
    *,
    sleep_between: float = 0.0,
) -> list[dict]:
    cache: dict[tuple[str, str], str] = {}
    out = []
    for i, ex in enumerate(examples):
        text = ex["text"]
        spans = ex.get("spans", [])
        new_spans = []
        for s in spans:
            lab = s["label"]
            span_txt = s["text"]
            if s.get("why"):
                new_spans.append(s)
                continue
            key = (lab, span_txt)
            why = cache.get(key)
            if not why:
                try:
                    why = _fill_one_why(text, span_txt, lab, doc_id=f"s1_ex_{i}")
                except Exception as e:
                    print(f"[warn] S1 why gen failed: {e}")
                    why = "Fits the label per the playbook and definitions."
                cache[key] = why
                if sleep_between > 0:
                    time.sleep(sleep_between)
            ns = dict(s)
            ns["why"] = why
            new_spans.append(ns)
        out.append({"text": text, "spans": new_spans})
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
    # discourage lone "I" as conspirator; allow named org/persons or vague collectives (handled in training upstream)
    return t.lower() != "i"


def _filter_spans(spans: list[dict]) -> list[dict]:
    out = []
    for s in spans or []:
        lab = s.get("label")
        txt = s.get("text") or ""
        if _why_says_not(s):
            continue
        if lab == "Action" and not _is_good_action(txt):
            continue
        if lab == "Effect" and not _is_good_effect(txt):
            continue
        if lab == "Evidence" and not _is_good_evidence(txt):
            continue
        if lab == "Actor" and not _is_good_actor(txt):
            continue
        out.append(s)
    return out


def _apply_s1_post_filter(bank: list[dict]) -> list[dict]:
    cleaned = []
    for ex in bank:
        spans_in = ex.get("spans", [])
        spans_out = _filter_spans(spans_in)
        # keep negatives (empty spans) as-is; drop positives that became empty
        if spans_in and not spans_out:
            continue
        cleaned.append({"text": ex.get("text", ""), "spans": spans_out})
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
  The GOLD label for this document is provided. Do NOT predict a label.
  Write ONE concise rationale (≤40 words) that justifies WHY the gold label is appropriate,
  referencing mechanism cues (Actor+Action+Effect) if present, or their *absence* if label is 'non'.
  Return ONLY JSON: {{"rationale":"..."}}
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


def build_s2_fewshots_with_llm_pydantic(
    train_docclf_jsonl: str,
    *,
    k: int = 6,
    rng_seed: int = 7,
) -> list[dict]:
    random.seed(rng_seed)
    rows = []
    with open(train_docclf_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except:
                pass

    # Expect rows like {"text": "...", "label": "conspiracy"|"non"}
    cons = [r for r in rows if r.get("label") == "conspiracy"]
    nonc = [r for r in rows if r.get("label") == "non"]
    if cons and nonc:
        half = k // 2
        sample = random.sample(cons, min(half, len(cons))) + random.sample(
            nonc, min(k - half, len(nonc))
        )
        random.shuffle(sample)
    else:
        sample = random.sample(rows, min(k, len(rows)))

    out = []
    for i, r in enumerate(sample):
        t = r.get("text", "")
        gold = r.get("label", "non")
        try:
            item = make_s2_item_with_rationale(t, gold, doc_id=f"{i}")
            out.append(item)
        except Exception as e:
            print(f"[warn] S2 rationale gen failed: {e}")
            # Fallback: minimal deterministic rationale
            if gold == "conspiracy":
                out.append(
                    {
                        "label": gold,
                        "rationale": "Text alleges coordinated actors with intentional actions toward a grand effect.",
                        "text": t,
                    }
                )
            else:
                out.append(
                    {
                        "label": gold,
                        "rationale": "No coherent conspiratorial mechanism (actor+intentional action+aimed effect) is asserted.",
                        "text": t,
                    }
                )
    return out


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
        complex_mode=args.s1_complex,
        min_spans=args.s1_min_spans,
        min_overlap=args.s1_min_overlap,
        cap_one_per_label=(not args.s1_complex and args.s1_cap_one_per_label),
        max_spans_per_ex=args.s1_max_spans_per_ex,
    )
    print(f"[fewshot] built S1 fewshot bank: {len(s1_bank)} examples")

    if args.s1_why_mode == "bedrock":
        s1_bank = fill_s1_whys_with_bedrock_pydantic(
            s1_bank,
            sleep_between=args.bedrock_sleep,
        )
        print(f"[fewshot] filled S1 whys with Bedrock LLM")

    # --- Post-filter only in NON-complex mode; complex mode keeps dense spans ---
    if not args.s1_complex:
        before_ex = len(s1_bank)
        before_sp = sum(len(ex.get("spans", [])) for ex in s1_bank)
        s1_bank = _apply_s1_post_filter(s1_bank)
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
        max_sp = max((len(ex.get("spans", [])) for ex in s1_bank), default=0)
        avg_sp = sum(len(ex.get("spans", [])) for ex in s1_bank) / max(1, len(s1_bank))
        print(
            f"[report] S1 complex mode: examples={len(s1_bank)} | max_spans/ex={max_sp} | avg_spans/ex={avg_sp:.1f}"
        )

    s2_bank = []
    if args.build_s2_fewshots:
        print(
            f"[fewshot] building S2 fewshot bank with LLM from {args.s2_train_docclf}"
        )
        if not args.s2_train_docclf:
            raise SystemExit("--build-s2-fewshots requires --s2-train-docclf")
        s2_bank = build_s2_fewshots_with_llm_pydantic(
            args.s2_train_docclf,
            k=args.s2_k,
            rng_seed=7,
        )
        print(f"[fewshot] built S2 fewshot bank: {len(s2_bank)} examples")

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
