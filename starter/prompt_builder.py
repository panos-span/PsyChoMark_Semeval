#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, re
from pathlib import Path
from typing import Optional, List, Dict


# --------- Artifact IO ----------
def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


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


# ---------- Few-shot utilities (S1) ----------
def _clip_span_to_text(span: dict, text: str) -> Optional[dict]:
    try:
        s = int(span.get("start", span.get("startIndex")))
        e = int(span.get("end", span.get("endIndex")))
        s = max(0, min(s, len(text)))
        e = max(s, min(e, len(text)))
        lab = (span.get("label") or span.get("type") or "").strip()
        if lab not in ("Actor", "Action", "Effect", "Victim", "Evidence"):
            return None
        if e - s < 3:
            return None
        return {"label": lab, "start": s, "end": e, "text": text[s:e]}
    except Exception:
        return None


def _coerce_fewshot_s1(ex: dict) -> Optional[dict]:
    """Accepts {'text','spans'} or {'text','answer'} (list/dict). Returns {'text','spans':[...]} or None."""
    if not isinstance(ex, dict):
        return None
    text = (ex.get("text") or "").strip()
    if not text:
        return None
    spans = ex.get("spans")
    if spans is None:
        ans = ex.get("answer")
        if isinstance(ans, dict) and "spans" in ans:
            spans = ans.get("spans")
        else:
            spans = ans
    out = []
    if isinstance(spans, list):
        for m in spans:
            m2 = _clip_span_to_text(m, text)
            if m2:
                out.append(m2)
    return {"text": text, "spans": out}


def build_s1_verifier_prompts(
    *, text: str, candidate_spans: List[Dict]
) -> tuple[str, str]:
    """
    Returns (system, user) prompts. Model should output ONLY:
      {"keep":[int,int,...], "reject":[int,int,...]}
    where indices refer to candidate_spans order.
    Criteria:
      - text slice at [start:end] MUST match candidate "text" if provided (± trivial whitespace).
      - Label validity: apply Evidence gate (URL/quote+attribution/source OR numbers+units+source),
                        Action-Effect split heuristic (verbs vs purpose clause),
                        drop spans < 3 chars.
    """
    sys = """
<role>You are a strict validator for span extraction.</role>
<rules>
  1) Offsets are 0-indexed, end-exclusive; slice must equal candidate "text" if present.
  2) Evidence requires URL/domain OR quoted+attributed source OR numeric facts+units+source.
  3) Action vs Effect: verb head = Action; purpose/result (to/in order to/so that …) = Effect.
  4) Drop spans shorter than 3 chars or obvious role mismatch.
  5) Output ONLY one JSON object: {"keep":[...], "reject":[...]}.
</rules>
""".strip()
    user = json.dumps(
        {
            "text": text,
            "candidates": candidate_spans,
            "output_format": {"keep": [0], "reject": [1]},
        },
        ensure_ascii=False,
    )
    # We intentionally give a JSON-shaped user payload; your runner can wrap it with <task> if desired.
    return sys, f"<task>\n{user}\n</task>"


# TODO: Check the span extraction method if it is valid or if we should ask the LLM to handle it
# --------- S1 builders ----------
def build_s1_system(priors: dict, conflicts: list[list[str]], use_cot: bool) -> str:
    # format priors once
    priors_str = ""
    for lab, d in (priors or {}).items():
        q90 = d.get("q90_len")
        pos = d.get("mode_pos")
        if q90 is not None and pos is not None:
            priors_str += f"- {lab}: typical length ≤ {q90:.0f} chars; often starts near {pos*100:.0f}% of text.\n"

    conflict_pairs = (
        ", ".join([f"{p[0]}–{p[1]}" for p in conflicts])
        if conflicts
        else "Action-Effect; Actor-Victim"
    )

    workflow_block = ""
    if use_cot:
        workflow_block = """<workflow>
<thinking>
  <pass_a_candidate_scan>
    - Actor: candidate agent nouns/phrases.
    - Action: candidate core verb phrases.
    - Effect: candidate goal/purpose/outcome phrases.
    - Victim: candidate harmed/targeted entities.
    - Evidence: candidate citations/quotes/links/attributions.
  </pass_a_candidate_scan>
  <pass_b_verbatim_extraction_and_filtering>
    - For each candidate from Pass A, extract the exact, verbatim text string. Do not determine character offsets yet.
    - Apply boundary rules to the text itself: ensure the extracted string is tight (no leading/trailing whitespace or punctuation).
    - Apply quality rules to the text itself: drop any extracted string that is less than 3 characters long.
  </pass_b_verbatim_extraction_and_filtering>
  <pass_c_index_mapping_and_finalization>
    - For each valid verbatim text string from Pass B:
        1. Locate this exact string within the original <text_to_analyze>.
        2. Determine its precise `start` and `end` character offsets.
    - Resolve Overlaps: Now that you have precise spans, resolve conflicts (esp. Action vs Effect) by choosing the minimal, role-true span. Actor vs Victim uses the smallest role-specific mention.
    - Use Priors: Use statistical priors (length/position) as tie-breakers only if ambiguity remains after all other rules.
    - Final check: if none remain, answer.
  </pass_c_index_mapping_and_finalization>
</thinking>
</workflow>"""

    return f"""<role>
You are a precision-focused annotator for SemEval-2026 PsyCoMark Task 10 (Subtask 1).
Extract psycholinguistic markers with exact character offsets.
</role>

<marker_definitions>
- Actor: agent portrayed as initiating/controlling events.
- Action: deliberate verb phrase describing what is done (exclude outcomes/goals).
- Effect: consequence/goal/purpose (often NP or purpose clause).
- Victim: harmed/targeted entity.
- Evidence: explicit support (links, quotes, numbers, named sources, “according to…”).
</marker_definitions>

<rules>
  <output_format>
    Return ONLY a JSON array inside <answer>.
    Each element:
    [
        {{"label":"Actor|Action|Effect|Victim|Evidence","start": <int>,"end": <int>,"text":"<optional verbatim substring>"}}
    ]

    Rules:
    - "text" must be copied verbatim from <text_to_analyze>.
    - "start" and "end" are mandatory integers (0-indexed, end-exclusive).
    - No prose, no extra keys, no trailing commas.
  </output_format>

  <boundaries>
    - Spans are measured over the raw TEXT (no normalization).
    - Keep token-tight; include prepositions/particles only if integral (“set up”, “cover up”, “in charge of”).
  </boundaries>
  
  <evidence_gate>
    Only label as Evidence if the substring contains (a) a URL/domain, OR (b) quotation marks with an attribution verb (e.g., “said”, “according to”) AND a concrete source name, OR (c) numeric facts with units AND a source (“report”, “Reuters”, “CDC”).
    Phrases like “some say”, “it has been said”, “according to X” with no URL/quote/source are NOT Evidence.
  </evidence_gate>
  
  <action_effect_split>
    If a verb phrase has a purpose/result introduced by “to/so that/in order to”, label the verb head as Action and the purpose/result as Effect. Prefer the minimal verb head as Action.
  </action_effect_split>

  <overlap_policy>
    - Ambiguous pairs: {conflict_pairs}.
    - Action vs Effect: split verb (Action) from purpose/result (Effect); allow minimal overlap only if unavoidable.
    - Evidence may overlap others if it is part of a quotation/citation.
    - Action vs Effect: verb head = Action; purpose/result introduced by “to”, “in order to”, “so that”, “for” = Effect.
  </overlap_policy>

  <priors>
{priors_str if priors_str else "- (no priors provided)\n"}
  </priors>

  <negative_case>If no markers are present, output [] in <answer>.</negative_case>
  <forbidden_output>Nothing outside <answer>.</forbidden_output>
</rules>

{workflow_block}"""


def build_s1_user(
    text_input: str, s1_fewshots: list[dict], include_cot: bool = True
) -> str:
    # --- normalize & guardrail the few-shots ---
    raw = []
    for ex in s1_fewshots or []:
        coerced = _coerce_fewshot_s1(ex)
        if coerced:
            raw.append(coerced)

    # Split into positive (has spans) vs negative ([]); prefer positives, cap negatives
    pos = [e for e in raw if (e.get("spans") or [])]
    neg = [e for e in raw if not (e.get("spans") or [])]

    # Label coverage heuristic: try to keep a mix; simple greedy cover
    want = 8
    kept: List[dict] = []
    have = set()
    for e in pos:
        labs = {m["label"] for m in e["spans"]}
        if not labs.issubset(have):
            kept.append(e)
            have |= labs
        if len(kept) >= want:
            break
    # Fill remaining with other positives
    if len(kept) < want:
        for e in pos:
            if e not in kept:
                kept.append(e)
                if len(kept) >= want:
                    break
    # Allow at most 2 negatives (avoid “all-[]” priming)
    if len(kept) < want and neg:
        kept.extend(neg[: min(2, want - len(kept))])
    kept = kept[:want]

    # Render examples
    ex_blocks = []
    for ex in kept:
        spans = ex.get("spans", [])
        ex_blocks.append(
            "<example>\n"
            "<text>\n" + ex.get("text", "") + "\n</text>\n"
            "<answer>\n" + json.dumps(spans, ensure_ascii=False) + "\n</answer>\n"
            "</example>"
        )
    examples = (
        "<examples>\n" + "\n".join(ex_blocks) + "\n</examples>\n\n" if ex_blocks else ""
    )

    tail = (
        "Provide your reasoning in <thinking> (kept private), then output ONLY the JSON array in <answer>."
        if include_cot
        else "Provide ONLY the final JSON in <answer>."
    )

    return f"""{examples}<task>
<text_to_analyze>
{text_input}
</text_to_analyze>
{tail}
</task>""".strip()


import json
import re


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
    policy_text: str | None = None,  # accepted for backward compat (DEPRECATED)
    include_cot: bool = False,
    boundary_note: str | None = None,  # accepted for backward compat (DEPRECATED)
    prompt_arts: dict | None = None,  # accepted for backward compat (DEPRECATED)
    allow_cant_tell: bool = False,
) -> str:
    """
    S2 SYSTEM PROMPT (streamlined):
      - No S1 artifacts (boundary/conflicts/priors) — they are ignored by design.
      - No probabilities: output is {label, rationale}.
      - Crisp, enforceable CoT workflow.
    NOTE: policy_text/boundary_note/prompt_arts are deprecated and ignored to avoid polluting S2.
    """

    labels_desc = (
        "- conspiracy: The text endorses a hidden, harmful plot by a powerful actor and exhibits multiple hallmarks.\n"
        "- non: The text does not endorse conspiratorial framing (e.g., neutral reporting, mocking, debunking).\n"
        + (
            "- cant_tell: The text is too ambiguous to classify reliably.\n"
            if allow_cant_tell
            else ""
        )
    )

    workflow_block = ""
    if include_cot:
        workflow_block = """
<workflow>
  <thinking>
    Step 1 — Stance: Is the author endorsing conspiratorial framing, or merely reporting/mocking/debunking?
    Step 2 — Hallmarks:
      • Roles: “us vs. them” (in-group vs. powerful malevolent out-group).
      • Causality: intentional secret action presented as the driver of events (not chance/complexity).
      • Epistemic stance: self-sealing logic (counter-evidence as cover-up), cui bono insinuations, clichés.
      • Affect: strong negative emotion and extreme consequences.
    Step 3 — Decision: Apply the <analytical_framework> and choose a label.
  </thinking>
  <answer>JSON only</answer>
</workflow>""".strip()

    # Keep your playbook content; it’s useful, compact, and not S1-specific.
    return f"""
<role>
You are an expert social scientist specializing in online discourse. Classify the text using psycholinguistic hallmarks of conspiratorial rhetoric.
</role>

<label_definitions>
{labels_desc.strip()}
</label_definitions>

<analytical_framework>
1) Manichean worldview (“Us vs. Them”): a virtuous in-group (“we”, “the people”) vs. a powerful malevolent out-group (“they”, “the elite”, “deep state”).
2) Teleological causality: events explained as deliberate secret actions; chance/complexity is minimized.
3) Self-sealing epistemology: counter-evidence reframed as a cover-up; lack of evidence presented as proof of secrecy.
4) Heightened affect and extreme stakes: fear/anger/urgency, catastrophic outcomes.
5) Endorsement test: the author must advocate or endorse the conspiratorial frame; mere reporting/mocking/debunking is “non”.
</analytical_framework>

{playbook_block()}

<output_format>
Return ONLY one JSON object inside <answer>:
{{"label":"{('conspiracy|non|cant_tell' if allow_cant_tell else 'conspiracy|non')}", "rationale":"1-2 sentences naming decisive cues"}}
</output_format>
{workflow_block}
""".strip()


def build_s2_user(
    *,
    text_input: str,
    s1_output: list | None,
    s2_fewshots: list | None = None,
    include_cot: bool = False,
    allow_cant_tell: bool = False,
) -> str:
    """
    S2 USER PROMPT (streamlined):
      - Compact few-shots that show {label, rationale} only.
      - No probabilities.
      - Uses S1 markers as evidence.
    """

    # ------- Few-shot block (normalize markers using each example's text) -------
    ex_block = ""
    if s2_fewshots:
        parts = []
        for ex in s2_fewshots:
            doc_text = ex.get("text", "")
            lbl = (
                ex.get("label") or (ex.get("gold") or {}).get("label") or "non"
            ).lower()
            valid = {"conspiracy", "non"} | (
                {"cant_tell"} if allow_cant_tell else set()
            )
            if lbl not in valid:
                lbl = "non"

            mk_norm = []
            for m in ex.get("markers") or []:
                try:
                    mk_norm.append(to_s2_marker(m, doc_text))
                except Exception:
                    continue

            gold = {
                "label": lbl,
                "rationale": ex.get("rationale", "concise example rationale."),
            }

            parts.append(
                "<example>\n"
                "<text>\n"
                + doc_text
                + "\n</text>\n"
                + (
                    "<extracted_markers>\n"
                    + json.dumps(mk_norm, ensure_ascii=False)
                    + "\n</extracted_markers>\n"
                    if mk_norm
                    else ""
                )
                + "<answer>\n"
                + json.dumps(gold, ensure_ascii=False)
                + "\n</answer>\n"
                "</example>"
            )
        ex_block = "<examples>\n" + "\n\n".join(parts) + "\n</examples>\n\n"

    # ------- Normalize live S1 markers against the CURRENT doc text -------
    s1_norm = []
    for m in s1_output or []:
        try:
            s1_norm.append(to_s2_marker(m, text_input))
        except Exception:
            continue

    task_tail = (
        "Provide brief reasoning in <thinking> then the final JSON in <answer>."
        if include_cot
        else "Provide ONLY the final JSON in <answer>."
    )

    return f"""{ex_block}<task>
<text_to_analyze>
{text_input}
</text_to_analyze>
<extracted_markers>
{json.dumps(s1_norm, ensure_ascii=False)}
</extracted_markers>
Instructions: Use the markers as evidence; ambiguity without hidden-plot framing should lean "non".
{task_tail}
</task>""".strip()


# --------- Utilities ----------
_SMART = {
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "–": "-",
    "—": "-",
}


def _normalize_quotes(s: str) -> str:
    for k, v in _SMART.items():
        s = s.replace(k, v)
    return s


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


import re


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
