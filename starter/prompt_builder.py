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


# ---------- Public adapters (easy entry points) ----------
def build_s1_prompts_adapter(
    *,
    text: str,
    prompt_arts: dict,  # output of load_artifacts(...)
    s1_fewshot_list: list,  # <-- FIX: Accept the pre-selected list
    tech: str = "fs_cot",  # e.g., "fs_cot" or "fs"
    shots: int = 8,
) -> tuple[str, str]:
    """
    Builds S1 (system,user) prompts from a pre-selected few-shot list.
    """
    use_cot = "cot" in tech
    priors = prompt_arts.get("s1_priors", {})
    conflicts = prompt_arts.get("s1_conflicts", [])
    # s1_few = (fewshot_bank.get("s1") or [])[:shots] # <-- REMOVED

    sys_prompt = build_s1_system(priors=priors, conflicts=conflicts, use_cot=use_cot)
    user_prompt = build_s1_user(
        text_input=text,
        s1_fewshots=s1_fewshot_list,  # <-- FIX: Use the balanced list
        include_cot=use_cot,
        want=shots,  # <-- FIX: 'shots' is the final cap
        victim_min=1,
        conflict_min=1,
        per_example_span_cap=4,
        max_text_chars=1200,
    )
    return sys_prompt, user_prompt


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
def build_s1_system(
    priors: dict[str, float] | None = None,
    conflicts: list[tuple[str, str]] | None = None,
    use_cot: bool = True,
) -> str:
    priors_str = json.dumps(priors or {}, ensure_ascii=False, indent=2)
    conflict_pairs_str = json.dumps(conflicts or [], ensure_ascii=False)

    workflow_block = ""
    if use_cot:
        workflow_block = """<workflow>
1. <thinking>
   - **Step 1 (Role-by-Role Scan):** Systematically scan the text for potential spans for each role: Actor, Victim, Action, Effect.
   - **Step 2 (Evidence Gate):** Scan *specifically* for 'Evidence' candidates. Keep ONLY those that strictly match the <evidence_gate> rule (URL, attributed quote, or numeric facts with a named source). Discard all other potential 'Evidence' spans.
   - **Step 3 (Action/Effect Split):** Review all 'Action' and 'Effect' candidates. Ensure 'Action' is the core verb phrase (what is done) and 'Effect' is the purpose/outcome (often starting with "to", "so that", "in order to"). Adjust boundaries if a span incorrectly merges both.
   - **Step 4 (Boundary & Length Check):** For all remaining candidates (Steps 1-3), ensure spans are 'token-tight' and at least 3 characters long. Trim any leading/trailing whitespace or punctuation that isn't part of the span.
   - **Step 5 (Overlap Resolution):** Resolve any spans that overlap according to the <overlap_policy>. Use <statistical_priors> as tie-breakers if ambiguity remains.
2. <answer>
   - Compile the final, validated, and non-overlapping spans into the JSON array.
</workflow>"""

    return f"""<role>
You are a precision-focused annotator for SemEval-2026 PsyCoMark S1.
Return exact character offsets for all markers.
</role>

<marker_definitions>
- Actor: agent portrayed as initiating/controlling events.
- Action: deliberate verb phrase describing what is done (exclude outcomes/goals).
- Effect: consequence/goal/purpose (include introducer like "to"/"so that").
- Victim: harmed/targeted entity.
- Evidence: explicit support (URLs, quoted with source, or numeric facts WITH named source).
</marker_definitions>

<rules>
  <output_format>
    Return ONLY a JSON array inside <answer>.
    Each element:
    [
      {{"label":"Actor|Action|Effect|Victim|Evidence","start": <int>,"end": <int>,"text":"<optional verbatim>"}}
    ]
    Notes:
    - "start" and "end" are MANDATORY; end-exclusive; computed over the RAW text.
    - "text" is OPTIONAL (auditing only); must match text[start:end] if present.
    - No prose, no extra keys, no trailing commas.
    - ABSOLUTELY NO prose or explanatory text outside the final <answer> tag. Your response must begin exactly with <answer> or <thinking>.
    - Offsets are 0-indexed and "end" is end-exclusive. For the text 'The cat', a span for 'cat' is {{"start": 4, "end": 7}}
  </output_format>

  <span_rules>
    - Keep spans token-tight; include particles only if integral (“set up”, “cover up”).
  </span_rules>

  <evidence_gate>
    Evidence ONLY if: (a) has URL/domain; OR (b) quotation with attribution verb AND concrete source; OR (c) numeric facts WITH units AND a named source.
  </evidence_gate>

  <overlap_policy>
    - Action vs Effect: split verb head (Action) from purpose/result clause (Effect); Effect must include the introducer.
    - Actor vs Victim: must not overlap; choose the smallest role-true mention.
    - Evidence may overlap others only if part of a quote/citation.
    - Ambiguous pairs: {conflict_pairs_str}
  </overlap_policy>

  <statistical_priors>
{priors_str}
  </statistical_priors>
</rules>

{workflow_block}
"""


def build_s1_user(
    text_input: str,
    s1_fewshots: list | None,
    include_cot: bool = True,
    *,
    want: int = 8,  # total few-shots to keep
    victim_min: int = 1,  # ensure at least one Victim example
    conflict_min: int = 1,  # ensure at least one Action–Effect overlap example
    neg_cap: int = 2,  # cap negatives to avoid “all-[]” priming
    per_example_span_cap: int = 4,  # reduce noisy gold to concise spans
    max_text_chars: int = 1200,  # clip long few-shot texts
) -> str:
    """
    Robust few-shot packer:
    - Accepts dict few-shots or pre-rendered <example>...</example> strings.
    - Normalizes spans to {'label','start','end'} ints.
    - Caps spans per example; dedups by text.
    - Ensures at least one Victim and one Action-Effect conflict example.
    - Limits negatives to avoid 'all-[]' priming.
    """
    rendered_blocks: List[str] = []
    raw_structured: List[dict] = []

    # --- Normalize incoming few-shots ---
    for ex in s1_fewshots or []:
        if isinstance(ex, str) and _is_example_xml(ex):
            # already rendered example; collect as-is
            rendered_blocks.append(ex.strip())
            continue

        # Expect a dict-like with text + spans/answer/markers
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

    # Dedup structurals by text
    raw_structured = _dedup_by_text(raw_structured)

    # Split positives (has spans) vs negatives ([]), cap per-example spans
    pos = []
    for e in raw_structured:
        ss = (
            _cap_spans_per_example(e["spans"], k=per_example_span_cap)
            if e["spans"]
            else []
        )
        pos.append({**e, "spans": ss})
    pos_has = [e for e in pos if e["spans"]]
    neg = [e for e in pos if not e["spans"]]

    # --- Greedy coverage: try to cover diverse labels early ---
    kept: List[dict] = []
    have_labels = set()
    for e in pos_has:
        labs = {m["label"] for m in e["spans"]}
        if not labs.issubset(have_labels):
            kept.append(e)
            have_labels |= labs
        if len(kept) >= want:
            break

    # Top-up with other positives
    if len(kept) < want:
        for e in pos_has:
            if e in kept:
                continue
            kept.append(e)
            if len(kept) >= want:
                break

    # Inject up to neg_cap negatives (but only if we still have room)
    if len(kept) < want and neg:
        room = min(neg_cap, want - len(kept))
        kept.extend(neg[:room])

    # --- Guarantees: Victim presence + at least one Action–Effect conflict ---
    def ensure_victim(items: List[dict]) -> List[dict]:
        if any(any(m["label"] == "Victim" for m in it["spans"]) for it in items):
            return items
        # find first positive with Victim
        for e in pos_has:
            if any(m["label"] == "Victim" for m in e["spans"]) and e not in items:
                items = ([e] + items)[:want]  # prepend; trim
                break
        return items

    def ensure_ae_conflict(items: List[dict]) -> List[dict]:
        if any(_has_ae_conflict(it["spans"]) for it in items):
            return items
        for e in pos_has:
            if _has_ae_conflict(e["spans"]) and e not in items:
                items = ([e] + items)[:want]
                break
        return items

    kept = ensure_victim(kept) if victim_min > 0 else kept
    kept = ensure_ae_conflict(kept) if conflict_min > 0 else kept

    # Finally trim to desired count
    kept = kept[:want]

    # --- Render structured few-shots into <example> blocks ---
    for ex in kept:
        spans = ex.get("spans", [])
        txt = ex.get("text", "")
        # escape only what the XML wrapper needs; JSON spans carry verbatim slices
        block = (
            "<example>\n"
            "<text>\n" + html.escape(txt) + "\n</text>\n"
            "<answer>\n" + json.dumps(spans, ensure_ascii=False) + "\n</answer>\n"
            "</example>"
        )
        rendered_blocks.append(block)

    examples = (
        "<examples>\n" + "\n".join(rendered_blocks) + "\n</examples>\n\n"
        if rendered_blocks
        else ""
    )

    tail = (
        "Provide your reasoning in <thinking> (kept private), then output ONLY the JSON array in <answer>."
        if include_cot
        else "Provide ONLY the final JSON in <answer>."
    )

    return f"""{examples}<task>
<text_to_analyze>
{_clip_text(text_input, max_chars=max_text_chars)}
</text_to_analyze>
{tail}
</task>""".strip()


# ----------------- S1 Verifier prompts -----------------


def build_s1_verify_system() -> str:
    return (
        "You are a careful span validator for PsyCoMark (S1).\n"
        "Given a post and candidate spans, keep ONLY spans that are:\n"
        " - well-formed (non-empty, not just stopwords/punctuation),\n"
        " - correctly typed (Actor, Action, Effect, Evidence, Victim),\n"
        " - text-exact substrings of the post,\n"
        " - non-duplicate (merge exact dupes),\n"
        " - Action/Effect must be eventive phrases; Evidence must look like a citation/date/measure or source cue."
        "- Evidence MUST meet the Evidence Gate: (a) has URL/domain; OR (b) quotation with attribution verb AND concrete source name; OR (c) numeric facts WITH units AND a named source. Otherwise reject."
    )


def build_s1_verify_user(*, text: str, candidates: list[dict]) -> str:
    # Expect candidates with keys: type, startIndex, endIndex, text
    blob = {
        "text": text,
        "candidates": [
            {
                "type": m.get("type"),
                "startIndex": int(m.get("startIndex", -1)),
                "endIndex": int(m.get("endIndex", -1)),
                "text": m.get("text", ""),
            }
            for m in (candidates or [])
        ],
        "return_format": {
            "kept": [
                {
                    "type": "Label",
                    "startIndex": 0,
                    "endIndex": 0,
                    "text": "exact substring",
                }
            ]
        },
    }
    return (
        "<task>\n"
        "<instructions>Return ONLY JSON inside &lt;answer&gt; exactly matching the schema in return_format. "
        "Drop anything invalid or uncertain.</instructions>\n"
        "<input>\n"
        f"{json.dumps(blob, ensure_ascii=False)}\n"
        "</input>\n"
        '<answer>{"kept": []}</answer>\n'
        "</task>"
    )


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
    Step 1 — Endorsement Test: Does the author endorse the conspiratorial frame? Or are they reporting, mocking, or debunking it? (If not endorsing, the label is 'non'.)
    Step 2 — Hallmarks (if endorsing): Identify hallmarks: “us vs. them” roles, secret/intentional causality, self-sealing logic.
    Step 3 — Decision: Synthesize and choose the final label.
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
{{"label":"{('conspiracy|non|cant_tell' if allow_cant_tell else 'conspiracy|non')}", "rationale":"1-2 sentences. **Must cite specific hallmarks** from the <analytical_framework>, e.g., 'Endorses 'us vs. them' framing and self-sealing logic.' or 'Neutral reporting, lacks endorsement.'"}}
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
    S2 USER PROMPT (XML-structured):
      - Clear separation of sections via tags.
      - Few-shots read from bank: ex["answer"] -> {label, rationale}.
      - S1 markers provided as evidence in <extracted_markers>.
      - Strict output format in <format> and <answer_schema>.
    """
    import json

    # -------- Few-shot block (normalize each example's markers to its text) --------
    examples_xml = ""
    if s2_fewshots:
        ex_parts = []
        valid_labels = {"conspiracy", "non"} | (
            {"cant_tell"} if allow_cant_tell else set()
        )

        for ex in s2_fewshots:
            doc_text = ex.get("text") or ""
            # -- read label/rationale from bank’s answer object, with fallbacks
            ans = ex.get("answer") if isinstance(ex.get("answer"), dict) else {}
            raw_lbl = (
                (ans.get("label") if isinstance(ans, dict) else None)
                or ex.get("label")
                or (ex.get("gold") or {}).get("label")
                or "non"
            )
            lbl = (raw_lbl or "non").strip().lower()
            if lbl not in valid_labels:
                lbl = "non"
            rat = (
                (ans.get("rationale") if isinstance(ans, dict) else "")
                or ex.get("rationale")
                or ""
            )

            # Normalize any example markers against THIS example text
            mk_norm = []
            for m in ex.get("markers") or []:
                try:
                    mk_norm.append(to_s2_marker(m, doc_text))
                except Exception:
                    continue

            ex_obj = {"label": lbl, "rationale": rat}

            ex_xml = [
                "<example>",
                "<text>",
                doc_text,
                "</text>",
            ]
            if mk_norm:
                ex_xml += [
                    "<extracted_markers>",
                    json.dumps(mk_norm, ensure_ascii=False),
                    "</extracted_markers>",
                ]
            ex_xml += [
                "<answer>",
                json.dumps(ex_obj, ensure_ascii=False),
                "</answer>",
                "</example>",
            ]
            ex_parts.append("\n".join(ex_xml))

        examples_xml = "<examples>\n" + "\n\n".join(ex_parts) + "\n</examples>\n"

    # -------- Current doc S1 markers normalized to CURRENT text --------
    s1_norm = []
    for m in s1_output or []:
        try:
            s1_norm.append(to_s2_marker(m, text_input))
        except Exception:
            continue

    # -------- CoT control & output spec --------
    if include_cot:
        thinking_instr = "Write a brief reasoning in <thinking> (1-3 bullets). Then output ONLY the final JSON in <answer>."
        thinking_tag_hint = "<thinking>(optional brief notes)</thinking>\n"
    else:
        thinking_instr = (
            "Do NOT include chain-of-thought. Output ONLY the final JSON in <answer>."
        )
        thinking_tag_hint = ""

    # -------- Allowed labels --------
    allowed_labels = ["conspiracy", "non"] + (["cant_tell"] if allow_cant_tell else [])

    # -------- Build the final XML prompt --------
    prompt = f"""<role>
You are a precision-focused annotator for SemEval-2026 PsyCoMark (Subtask 2).
Your job: classify a Reddit post as "conspiracy" vs "non"{' vs "cant_tell"' if allow_cant_tell else ''}.
</role>

<instructions>
1) Use markers from Subtask 1 (S1) as evidence when helpful.
2) Label as <conspiracy> when the post explicitly endorses secret coordination/cover-ups by powerful actors, self-sealing logic, or "us vs them" villainization with intentionality.
3) Label as <non> when it's newsy/reporting, neutral discussion, or lacks endorsement of hidden-plot intent. Mere mention without endorsement is "non".
4) Label 'non' even if the post discusses a conspiracy, as long as the author's stance is neutral, reporting, mocking, or debunking.
{('5) Use <cant_tell> only if evidence is insufficient or text is too ambiguous.' if allow_cant_tell else '')}
6) {thinking_instr}
</instructions>

<answer_schema>
Return JSON with keys:
{{
  "label": "{' | '.join(allowed_labels)}",
  "rationale": "1-2 concise sentences naming the decisive cues (e.g., roles, secret causality, self-sealing logic, affect, endorsement vs reporting)."
}}
</answer_schema>

<format>
{thinking_tag_hint}<answer>{{"label": "...", "rationale": "..."}}</answer>
</format>

{examples_xml}<task>
<text_to_analyze>
{text_input}
</text_to_analyze>

<extracted_markers>
{json.dumps(s1_norm, ensure_ascii=False)}
</extracted_markers>

<hints>
- Prioritize explicit endorsement of a hidden, intentional plot (vs. mere speculation or quoting news).
- Reporting style cues (sources, quotes, stats) usually imply "non" unless the author endorses conspiracy.
- Strong moral-emotional language + agentive "they/elite/deep state" may indicate endorsement, but check intent.
</hints>

<output_requirements>
- The only machine-readable output must be the JSON inside <answer>.
- Do NOT wrap the JSON in code fences.
- Keys must be exactly "label" and "rationale".
- Label must be one of: {', '.join(allowed_labels)}.
</output_requirements>
</task>"""
    return prompt.strip()


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


_PURPOSE_STARTS = ["to ", "in order to ", "so that ", "for "]


def _is_example_xml(s: str) -> bool:
    return isinstance(s, str) and "<example>" in s and "</example>" in s


def _norm_span(s: dict) -> dict | None:
    """Map {'type', 'startIndex','endIndex'} or {'label','start','end'} → {'label','start','end'} with ints."""
    if not isinstance(s, dict):
        return None
    lab = (s.get("label") or s.get("type") or "").strip()
    if not lab:
        return None
    st = s.get("start", s.get("startIndex"))
    en = s.get("end", s.get("endIndex"))
    try:
        st = int(st)
        en = int(en)
    except Exception:
        return None
    if en <= st:
        return None
    return {"label": lab, "start": st, "end": en}


def _cap_spans_per_example(spans: List[dict], k: int = 4) -> List[dict]:
    """Keep the most informative small set; prioritize role importance then brevity."""
    role_w = {"Evidence": 4, "Action": 3, "Actor": 2, "Effect": 1, "Victim": 1}
    uniq = {(m["label"], m["start"], m["end"]): m for m in spans}.values()
    ranked = sorted(
        uniq,
        key=lambda m: (-role_w.get(m["label"], 0), (m["end"] - m["start"]), m["start"]),
    )
    return ranked[:k]


def _has_ae_conflict(spans: List[dict]) -> bool:
    """Any Action-Effect overlap? (strict char overlap)"""
    # spans assumed normalized
    acts = [m for m in spans if m["label"] == "Action"]
    effs = [m for m in spans if m["label"] == "Effect"]
    for a in acts:
        for e in effs:
            if max(a["start"], e["start"]) < min(a["end"], e["end"]):
                return True
    return False


def _clip_text(text: str, max_chars: int = 1200) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    # soft clip at sentence boundary if possible
    cut = text[:max_chars]
    m = re.search(r"[.!?]\s+\S*$", cut)
    return (cut if not m else cut[: m.start() + 1]).rstrip()


def _dedup_by_text(items: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for e in items:
        key = (e.get("text") or "").strip()
        if key and key not in seen:
            seen.add(key)
            out.append(e)
    return out
