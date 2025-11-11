#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import re
import html
from pathlib import Path
from typing import Optional, List, Dict
from pathlib import Path as _Path


# --------- Artifact IO ----------
def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


# ---------- Few-shot & artifact loaders ----------
def load_artifacts(path: str | _Path) -> dict:
    """Loads priors & conflicts produced by make_prompt_artifacts.py"""
    p = _Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"s1_priors": {}, "s1_conflicts": []}


def load_fewshot_bank(path: str | _Path) -> dict:
    """Loads {"s1":[...], "s2":[...]}"""
    p = _Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"s1": [], "s2": []}


def playbook_block() -> str:
    return """
<psycomark_playbook version="1.0">
  <narrative_roles>
    <malevolent_actor features="vague ‘they’; abstract collectives (‘elite’, ‘globalists’, ‘deep state’); hyper-competence/secret coordination"
                      function="Constructs omnipresent enemy; vilifies out-group"
                      cues="they; the elite; globalists; deep state; big pharma"/>
    <victim_us       features="inclusive ‘we/us’; competitive victimhood; moral-emotional framing"
                      function="Fosters in-group solidarity; moral high ground"
                      cues="we the people; our way of life; we suffer"/>
    <savior_campaigner features="authoritative ‘I’; privileged knowledge claims"
                      function="Legitimizes authority; promises rescue"
                      cues="I alone can fix; I know the truth"/>
  </narrative_roles>

  <causality_intent>
    <action_language features="verbs of secrecy/control/hostility; linear intentional causation"
                     function="Eliminates randomness; centers intentional harm"
                     cues="plot; scheme; infiltrate; engineer; manipulate; cover up; weaponize"/>
    <effect_language features="extreme stakes; high negative affect"
                     function="Maximizes threat/urgency; fuels engagement"
                     cues="total control; tyranny; enslavement; destruction; depopulation"/>
  </causality_intent>

  <epistemic_stance>
    <rhetoric_of_evidence features="reframe counter-evidence as cover-up/disinformation; cui bono; loaded language; thought-terminating clichés"
                          function="Self-sealing logic; hermeneutics of suspicion"
                          cues="who benefits; follow the money; do your own research; connect the dots"/>
    <certainty_doubt_paradox features="absolute certainty for claim + radical skepticism of institutions; ‘faith in intuition’"
                             function="Epistemic closure and in-group privilege"
                             cues="truth is obvious; facts are clear; media is lying; don’t trust experts"/>
  </epistemic_stance>

  <shortcuts_and_pitfalls>
    <ts_001>Avoid keyword-only flags; context decides endorsing vs reporting/debunking.</ts_001>
    <ts_002>Self-sealing must include the reframing move (counter-evidence leads to ‘cover-up’).</ts_002>
  </shortcuts_and_pitfalls>
</psycomark_playbook>
""".strip()


# --- add near the top of prompt_builder.py (next to playbook_block) ---
# 1) preamble keeps the role + theory
def psycho_theory_preamble() -> str:
    return """
<psycholinguistic_preamble version="1.0">
  <role>You are an expert computational psycholinguist. Align your reasoning with psycholinguistic and evolutionary accounts of conspiratorial rhetoric for SemEval-2026 PsyCoMark Subtask 1 (marker extraction).</role>
  <marker_definitions>
    <Actor>Agents alleged to secretly orchestrate events; the conspirators.</Actor>
    <Action>Deliberate acts attributed to the Actor (what they do). Verb phrase; exclude outcomes/goals.</Action>
    <Effect>Consequence/goal/purpose of the Action (why/result). Often purpose/result clause.</Effect>
    <Victim>Entity harmed/targeted by the Action.</Victim>
    <Evidence>Support claims: links; quoted+attributed material; numeric facts+units+named source.</Evidence>
  </marker_definitions>
</psycholinguistic_preamble>
""".strip()


# TODO: Check the span extraction method if it is valid or if we should ask the LLM to handle it
# --------- S1 builders ----------
def build_s1_system(
    priors: dict | None = None,
    conflicts: list[tuple[str, str]] | None = None,
    use_cot: bool = True,
) -> str:
    """
    Pydantic-AI mode:
      - No JSON schema / <answer> formatting rules (handled by output_type).
      - Single inclusion of theory + playbook.
      - Domain guidance only (Evidence gate, Action↔Effect split, overlap policy, priors).
    """
    priors_str = json.dumps(priors or {}, ensure_ascii=False, separators=(",", ":"))
    conflict_pairs_str = json.dumps(
        conflicts or [], ensure_ascii=False, separators=(",", ":")
    )

    header = psycho_theory_preamble() + "\n" + playbook_block()

    rules = f"""
<rules>
  <evidence_gate>
    Evidence ONLY if at least ONE holds:
      (a) Contains a URL/domain,
      (b) Quoted material WITH attribution verb AND named source,
      (c) Numeric facts WITH units AND named source.
  </evidence_gate>

  <span_rules>
    - Keep spans token-tight; include particles only if integral (e.g., "set up", "cover up").
    - Prefer minimal spans that still fully express the role.
  </span_rules>

  <action_effect_split>
    - Action = what is done (verb phrase).
    - Effect = consequence/purpose/result (often NP or "to …"/"so that …").
    - Do not merge Action and Effect.
  </action_effect_split>

  <overlap_policy>
    - Forbid Actor ↔ Victim overlaps; if uncertain, prefer Actor unless Victim clearly superior.
    - Allow short Victim NP inside Action; keep both if well-formed.
    - Evidence may overlap others only if part of a quote/citation per <evidence_gate>.
    - Ambiguous pairs hint: {conflict_pairs_str}
  </overlap_policy>

  <statistical_priors>
    {priors_str}
  </statistical_priors>

  <notes>
    - Choose exact substrings first; offsets will be auto-filled from your text.
    - Don't over-mark generic function words.
    - If a label is absent, output none for that label.
  </notes>
</rules>
""".strip()

    workflow = ""
    if use_cot:
        workflow = """
<workflow>
  1) Scan roles: Actor, Action, Effect, Victim; then explicitly scan for Evidence.
  2) Apply <evidence_gate>.
  3) Enforce Action<->Effect split.
  4) Tighten boundaries; keep particles only if integral.
  5) Apply <overlap_policy>.
</workflow>""".strip()

    output_contract = """
    <verbatim_rule>
        Every span's "text" MUST be a verbatim substring of <text_to_analyze>.
        DO NOT paraphrase, summarize, or invent. Copy exact characters from RAW.
    </verbatim_rule>
    <output_contract>Provide verbatim text</output_contract>
    """.strip()

    return (
        header
        + "\n"
        + rules
        + ("\n" + workflow if workflow else "")
        + "\n"
        + output_contract
    ).strip()


import html, json, re
from typing import List, Dict, Any

_LABELS = {"Actor", "Action", "Effect", "Victim", "Evidence"}


def _is_example_xml(s: str) -> bool:
    return isinstance(s, str) and "<example>" in s and "</example>" in s


def _clip_text(s: str, max_chars: int) -> str:
    return s[:max_chars] if s and len(s) > max_chars else (s or "")


def _dedup_by_text(items: List[dict]) -> List[dict]:
    seen, out = set(), []
    for it in items:
        t = it.get("text", "")
        if t not in seen and t:
            seen.add(t)
            out.append(it)
    return out


def _cap_spans_per_example(spans: List[dict], k: int) -> List[dict]:
    if not spans:
        return []
    kept, seen_txt = [], set()
    for m in spans:
        txt = (m.get("text") or "").strip()
        if not txt or txt in seen_txt:
            continue
        kept.append(m)
        seen_txt.add(txt)
        if len(kept) >= k:
            break
    return kept


def _has_ae_conflict(spans: List[dict]) -> bool:
    # simple heuristic: same sentence-ish or overlapping Action & Effect present
    has_a = any(m.get("label") == "Action" for m in spans)
    has_e = any(m.get("label") == "Effect" for m in spans)
    return has_a and has_e


def _norm_span(m: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Normalize incoming span to: {"label","text","start?","end?"}
      accepts evaluator-style: {"type","startIndex","endIndex","text"?}
      accepts old: {"label","start","end","text"?}
    """
    if not isinstance(m, dict):
        return None
    label = m.get("label") or m.get("type")
    if not label or str(label) not in _LABELS:
        return None
    text = m.get("text")
    start = m.get("start", m.get("startIndex"))
    end = m.get("end", m.get("endIndex"))
    out = {"label": str(label)}
    if text is not None:
        out["text"] = str(text)
    if start is not None and end is not None:
        try:
            out["start"] = int(start)
            out["end"] = int(end)
        except Exception:
            # ignore bad indices; validator will locate by text
            out.pop("start", None)
            out.pop("end", None)
    return out


def build_s1_user(
    *,
    text_input: str,
    s1_fewshots: list | None,
    include_cot: bool = True,
    want: int = 8,  # total few-shots to keep
    victim_min: int = 1,  # ensure at least one Victim example
    conflict_min: int = 1,  # ensure at least one Action–Effect example
    neg_cap: int = 2,  # cap negatives to avoid “all-[]” priming
    per_example_span_cap: int = 4,  # reduce noisy gold to concise spans
    max_text_chars: int = 1200,  # clip long few-shot texts
) -> str:
    """
    Robust few-shot packer (pydantic-AI ready):
    - Accepts dict few-shots or pre-rendered <example>...</example> strings.
    - Normalizes spans to {'label','text','start?','end?'}.
    - Caps spans/example; dedups by text; limits negatives.
    - Guarantees at least one Victim example and one Action–Effect example when available.
    - Emits <few_shots>…</few_shots> with JSON spans in current schema.
    """
    rendered_blocks: List[str] = []
    raw_structured: List[dict] = []

    # --- Normalize incoming few-shots ---
    for ex in s1_fewshots or []:
        if isinstance(ex, str) and _is_example_xml(ex):
            rendered_blocks.append(ex.strip())
            continue
        if not isinstance(ex, dict):
            continue

        text = _clip_text((ex.get("text") or "").strip(), max_chars=max_text_chars)
        spans_raw = ex.get("answer") or ex.get("spans") or ex.get("markers") or []
        spans = [m for m in (_norm_span(m) for m in spans_raw) if m]

        raw_structured.append(
            {
                "text": text,
                "spans": spans,
                "subreddit": ex.get("subreddit"),
                "_id": ex.get("_id") or ex.get("doc_id"),
            }
        )

    # Deduplicate by text
    raw_structured = _dedup_by_text(raw_structured)

    # Split positives / negatives and cap spans per example
    pos_prepped = []
    for e in raw_structured:
        ss = (
            _cap_spans_per_example(e["spans"], k=per_example_span_cap)
            if e["spans"]
            else []
        )
        pos_prepped.append({**e, "spans": ss})
    positives = [e for e in pos_prepped if e["spans"]]
    negatives = [e for e in pos_prepped if not e["spans"]]

    # Greedy label coverage first
    kept: List[dict] = []
    have_labels = set()
    for e in positives:
        labs = {m["label"] for m in e["spans"]}
        if not labs.issubset(have_labels):
            kept.append(e)
            have_labels |= labs
        if len(kept) >= want:
            break

    # Top-up with remaining positives
    if len(kept) < want:
        for e in positives:
            if e in kept:
                continue
            kept.append(e)
            if len(kept) >= want:
                break

    # Add up to neg_cap negatives if still under want
    if len(kept) < want and negatives:
        room = min(neg_cap, want - len(kept))
        kept.extend(negatives[:room])

    # Guarantees: at least one Victim and one Action–Effect example (if available)
    def ensure_victim(items: List[dict]) -> List[dict]:
        if any(
            any(m["label"] == "Victim" for m in it.get("spans", [])) for it in items
        ):
            return items
        for e in positives:
            if any(m["label"] == "Victim" for m in e["spans"]) and e not in items:
                return ([e] + items)[:want]
        return items

    def ensure_ae(items: List[dict]) -> List[dict]:
        if any(_has_ae_conflict(it.get("spans", [])) for it in items):
            return items
        for e in positives:
            if _has_ae_conflict(e["spans"]) and e not in items:
                return ([e] + items)[:want]
        return items

    kept = ensure_victim(kept) if victim_min > 0 else kept
    kept = ensure_ae(kept) if conflict_min > 0 else kept

    kept = kept[:want]

    # --- Render <few_shots> blocks in the CURRENT schema ---
    for ex in kept:
        spans = ex.get("spans", [])
        txt = ex.get("text", "")
        block = (
            "<example>\n"
            "<text>\n" + html.escape(txt) + "\n</text>\n"
            "<spans>\n"
            + json.dumps(spans, ensure_ascii=False, separators=(",", ":"))
            + "\n</spans>\n"
            "</example>"
        )
        rendered_blocks.append(block)

    fewshots_xml = (
        "<few_shots>\n" + "\n".join(rendered_blocks) + "\n</few_shots>\n\n"
        if rendered_blocks
        else ""
    )

    cot_hint = (
        "<thinking>Please follow <workflow> precisely before finalizing.</thinking>"
        if include_cot
        else ""
    )
    target_hint = f"<target>Please extract up to {want} concise, token-tight spans if present.</target>"

    raw = text_input or ""
    return f"""{fewshots_xml}{cot_hint}
{target_hint}
<constraint>Return only substrings that already exist in RAW. No paraphrases.</constraint>
<verbatim_rule>Every span's text MUST be a verbatim substring of <text_to_analyze>.</verbatim_rule>
<text_to_analyze>
{raw}
</text_to_analyze>""".strip()


# --- S2 prompt adapter: builds prompts from tech flags and passes cant_tell policy ---
def build_s2_prompts_adapter(
    *,
    text: str,
    markers: list,
    fewshots: list | None,
    tech: str,
    allow_cant_tell: bool = False,
) -> tuple[str, str]:
    use_cot = "cot" in tech
    sys_prompt = build_s2_system(
        include_cot=use_cot,
        allow_cant_tell=allow_cant_tell,
        # the following are accepted but ignored (kept for backward-compat)
        policy_text=None,
        boundary_note=None,
        prompt_arts=None,
    )
    user_prompt = build_s2_user(
        text_input=text,
        s1_output=markers,
        s2_fewshots=fewshots or [],
        include_cot=use_cot,
        allow_cant_tell=allow_cant_tell,
    )
    return sys_prompt, user_prompt


def build_s2_system(
    *,
    include_cot: bool = True,
    allow_cant_tell: bool = False,
) -> str:
    """
    Pydantic-AI mode:
      - No JSON/format schema; just label policy and rationale guidance.
    """
    labels = ["conspiracy", "non"]
    if allow_cant_tell:
        labels.append("cant_tell")

    policy = f"""
<classification_policy>
  - Choose one label from: {", ".join(labels)}.
  - Base the decision on conspiracist cues vs. ordinary discourse.
  - Rationale: 1-2 concise sentences naming decisive cues (no summaries).
</classification_policy>""".strip()

    cot = (
        """
<workflow>
  1) Identify conspiracist narrative cues (coordination/omnipotent actors, secret plots, us-vs-them).
  2) Contrast with ordinary skepticism or factual critique.
  3) Decide label; compose a brief rationale naming decisive cues.
</workflow>
""".strip()
        if include_cot
        else ""
    )

    header = psycho_theory_preamble() + "\n" + playbook_block()  # included once here

    return (header + "\n" + policy + ("\n" + cot if cot else "")).strip()


def build_s2_user(
    *,
    text_input: str,
    s1_output: List[dict] | None,
    s2_fewshots: List[dict] | None = None,
    include_cot: bool = False,
    allow_cant_tell: bool = False,
) -> str:
    """
    Pydantic-AI mode:
      - Embed RAW text and normalized S1 markers as evidence.
      - Few-shots contain {label, rationale} only (compact).
    """
    raw = text_input or ""

    examples_xml = ""
    if s2_fewshots:
        ex_parts = []
        valid = {"conspiracy", "non"} if allow_cant_tell else {"conspiracy", "non"}
        for ex in s2_fewshots:
            lab = str(ex.get("label", "")).lower()
            if lab not in valid:
                continue
            rationale = ex.get("rationale", "")
            etext = ex.get("text", "")
            ex_parts.append(
                "<example>"
                f"<label>{lab}</label>"
                f"<rationale>{rationale}</rationale>"
                f"<text>{etext}</text>"
                "</example>"
            )
        if ex_parts:
            examples_xml = "<few_shots>\n" + "\n".join(ex_parts) + "\n</few_shots>"

    cot_hint = (
        "<thinking>Follow <workflow> first; then decide one label and a brief rationale.</thinking>"
        if include_cot
        else ""
    )

    markers_xml = "<extracted_markers>[]</extracted_markers>"
    if s1_output:
        markers_xml = (
            "<extracted_markers>\n"
            + json.dumps(s1_output, ensure_ascii=False, separators=(",", ":"))
            + "\n</extracted_markers>"
        )

    return f"""
{examples_xml}
{cot_hint}
<text_to_analyze>
{raw}
</text_to_analyze>

{markers_xml}
""".strip()


def extract_answer_json(x):
    """
    Returns either:
      - list[dict] (for S1 spans), or
      - dict (for S2 {"label":..., "rationale":..., "confidence":...})
    Robust to dict/bytes/str; prefers JSON inside <answer>...</answer>; else last JSON blob.
    """

    def _as_text(v):
        if v is None:
            return ""
        if isinstance(v, (bytes, bytearray)):
            try:
                return v.decode("utf-8", errors="ignore")
            except Exception:
                return str(v)
        if isinstance(v, dict):
            cand = v.get("answer") or v.get("text") or v.get("content") or v
            return (
                json.dumps(cand, ensure_ascii=False)
                if not isinstance(cand, str)
                else cand
            )
        return str(v)

    s = _as_text(x)

    # Prefer content inside <answer>...</answer>
    m = re.search(r"<answer>\s*(\{.*?\}|\[.*?\])\s*</answer>", s, re.S)
    blob = m.group(1) if m else None

    # Fallback: take last JSON object/array in the text
    if not blob:
        parts = re.findall(r"(\{.*?\}|\[.*?\])", s, re.S)
        blob = parts[-1] if parts else None

    if not blob:
        return []  # default for S1 call sites

    try:
        js = json.loads(blob)
    except Exception:
        blob2 = re.sub(r",\s*([\}\]])", r"\1", blob)
        try:
            js = json.loads(blob2)
        except Exception:
            return []

    return js


def _safe_clip(s: str, a: int, b: int):
    a = max(0, int(a))
    b = min(len(s), int(b))
    return a, max(a, b)


def _window_bounds(a: int, b: int, L: int, win: int):
    lo = max(0, min(a, b) - win)
    hi = min(L, max(a, b) + win)
    return lo, hi


def _try_local_snap(text: str, start: int, end: int, echo: str, win: int = 16):
    """
    If echo doesn't match text[start:end], search a small window around (start,end)
    for an exact echo, else return original (start,end).
    """
    if not echo:
        return start, end
    L = len(text)
    lo, hi = _window_bounds(start, end, L, win)
    window = text[lo:hi]
    i = window.find(echo)
    if i >= 0:
        s = lo + i
        return s, s + len(echo)
    return start, end


def validate_and_repair_s1_spans(
    items: list[dict], text: str, *, win: int = 16, use_tokens: bool = True
):
    """
    Hybrid repair:
      1) Require ints for start/end and clip to bounds.
      2) If optional "text" present and mismatches, try a local re-align within ±win chars.
      3) Optionally snap to token boundaries (only *after* local re-align).
      4) Drop spans < 3 chars after trimming whitespace.
    Returns canonical [{"label","start","end"}].
    """
    out = []
    L = len(text)

    # Optional token helpers
    try:
        from starter.prompt_sweep_joint import _tokenize_eval, _snap_to_tokens
    except Exception:
        _tokenize_eval = _snap_to_tokens = None
        use_tokens = False

    for m in items or []:
        lab = (m.get("label") or m.get("type") or "").strip()
        if lab not in ("Actor", "Action", "Effect", "Victim", "Evidence"):
            continue

        # ints + clip
        try:
            s = int(m.get("start"))
            e = int(m.get("end"))
        except Exception:
            continue
        s, e = _safe_clip(text, s, e)
        if e <= s:
            continue

        echo = m.get("text")
        # quick mismatch check
        if isinstance(echo, str):
            cur = text[s:e]
            if cur != echo:
                # attempt small-window relocate
                s2, e2 = _try_local_snap(text, s, e, echo, win=win)
                s2, e2 = _safe_clip(text, s2, e2)
                if e2 > s2:
                    s, e = s2, e2  # only adopt if found

        # Optional token snapping (lightweight & local)
        if use_tokens and _tokenize_eval and _snap_to_tokens:
            toks = _tokenize_eval(text)
            snapped = _snap_to_tokens({"label": lab, "start": s, "end": e}, toks)
            if snapped:
                s, e = int(snapped["start"]), int(snapped["end"])
                s, e = _safe_clip(text, s, e)

        # drop tiny spans after trim
        # (compute trimmed span length with surrounding whitespace removed)
        trimmed = text[s:e].strip()
        if len(trimmed) < 3:
            continue

        out.append({"label": lab, "start": s, "end": e})

    return out


# ---- Utilities (shared) ----
def to_s2_marker(m: dict, txt: str) -> dict:
    """
    Normalize an S1-style span (label/start/end[/text]) into the S2 schema:
      {"type","startIndex","endIndex","text"}
    - Clips to bounds
    - Recomputes 'text' slice from offsets (ignores any echoed text)
    """
    s = int(m.get("start", m.get("startIndex", 0)))
    e = int(m.get("end", m.get("endIndex", s)))
    s = max(0, min(s, len(txt)))
    e = max(s, min(e, len(txt)))
    return {
        "type": (m.get("type") or m.get("label")),
        "startIndex": s,
        "endIndex": e,
        "text": txt[s:e],
    }
