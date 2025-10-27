#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, re
from pathlib import Path
from typing import Any, Dict, List

from starter.prompt_sweep_joint import  (
    _render_boundary_block, _render_conflicts_block, _render_priors_block
)


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
    <malevolent_actor>
      <features>Vague 3rd-person plural (“they”), abstract collectives (“the elite”, “globalists”, “deep state”), hyper-competence/secret coordination.</features>
      <function>Constructs omnipresent enemy; vilifies out-group.</function>
      <cues>they; the elite; globalists; deep state; big pharma</cues>
    </malevolent_actor>
    <victim_us>
      <features>First-person plural (“we/us”), competitive victimhood, moral-emotional framing.</features>
      <function>Fosters in-group solidarity; moral high ground.</function>
      <cues>we the people; our way of life; we suffer</cues>
    </victim_us>
    <savior_campaigner>
      <features>Authoritative “I”, claims of privileged knowledge.</features>
      <function>Legitimizes authority; promises rescue.</function>
      <cues>I alone can fix; I know the truth</cues>
    </savior_campaigner>
  </narrative_roles>

  <causality_intent>
    <action_language>
      <features>Verbs of secrecy/control/hostility; linear causation.</features>
      <function>Eliminates randomness; centers intentional harm.</function>
      <cues>plot; scheme; infiltrate; engineer; manipulate; cover up; weaponize</cues>
    </action_language>
    <effect_language>
      <features>Extreme stakes; high negative affect.</features>
      <function>Maximizes threat/urgency; fuels engagement.</function>
      <cues>total control; tyranny; enslavement; destruction; depopulation</cues>
    </effect_language>
  </causality_intent>

  <epistemic_stance>
    <rhetoric_of_evidence>
      <features>Dismiss counter-evidence as disinformation/cover-up; cui bono; loaded language; thought-terminating clichés.</features>
      <function>Self-sealing logic; hermeneutics of suspicion.</function>
      <cues>who benefits; follow the money; do your own research; connect the dots</cues>
    </rhetoric_of_evidence>
    <certainty_doubt_paradox>
      <features>Absolute certainty for claim + radical skepticism of institutions; “faith in intuition”.</features>
      <function>Epistemic closure and in-group privilege.</function>
      <cues>truth is obvious; facts are clear; media is lying; don’t trust experts</cues>
    </certainty_doubt_paradox>
  </epistemic_stance>

  <shortcuts_and_pitfalls>
    <ts_001>Avoid keyword-only flags; context determines if a text is endorsing vs. reporting/debunking.</ts_001>
    <ts_002>Evidence dismissal must show the self-sealing move (counter-evidence ⇒ proof of cover-up).</ts_002>
  </shortcuts_and_pitfalls>
</psycomark_playbook>

"""


# --------- S1 builders ----------
def build_s1_system(priors: Dict[str, Any], conflicts: List[List[str]], use_cot: bool) -> str:
    """
    Builds the streamlined and de-duplicated system prompt for S1 (Marker Extraction).
    This replaces your current complex system prompt.
    """
    priors_str = ""
    # Consolidate all priors into a single, clear list from your artifacts
    for label, data in priors.items():
        q90 = data.get("q90_len")
        mode_pos = data.get("mode_pos")
        if q90 is not None and mode_pos is not None:
            priors_str += f"- **{label}**: Typical span length ≤ {q90:.0f} chars; often starts near {mode_pos*100:.0f}% of the text.\n"

    conflict_pairs_str = (
        ", ".join([f"{p}-{p[1]}" for p in conflicts])
        if conflicts
        else "Action-Effect and Actor-Victim"
    )
    
    workflow_block = ""
    if use_cot:
        workflow_block = f"""<workflow>
1) In your <thinking> block, execute exactly two passes using the following XML sections.

<pass_a_candidate_scan>
- Actor: identify candidate agent nouns/phrases.
- Action: identify candidate core verb phrases.
- Effect: identify candidate goal or outcome phrases.
- Victim: identify candidate harmed/targeted entities.
- Evidence: identify candidate citations, quotes, links, or explicit attributions.
</pass_a_candidate_scan>

<pass_b_validation_and_refinement>
- Apply boundary rules: substrings must be tight (no leading/trailing whitespace or punctuation).
- Apply overlap_policy: resolve overlaps (especially Action vs Effect) by splitting or choosing the minimal appropriate span; Actor vs Victim should use the smallest role-specific mention.
- Apply statistical_priors: use the priors on length/position as tie-breakers for ambiguous cases.
- Filter for quality: discard substrings shorter than 3 characters or speculative-only content without explicit wording.
- Final check: if nothing remains, confirm the final answer will be [].
</pass_b_validation_and_refinement>

2) After your reasoning, output ONLY the final JSON array inside <answer>.
</workflow>"""

    return f"""<role>
You are a precision-focused annotator for SemEval-2026 PsyCoMark Task 10, Subtask 1. Extract markers exactly as specified.
</role>

<task_definition>
Extract all markers for: Actor, Action, Effect, Victim, Evidence. Return STRICT JSON ONLY inside <answer>.
</task_definition>

<marker_definitions>
- **Actor**: The agent portrayed as initiating or controlling events.
  - **Rhetorical Function**: To construct a powerful, malevolent, and often vaguely defined out-group ("Them").
  - **Linguistic Signals**: Look for depersonalized third-person plural pronouns (e.g., "they") and abstract collective nouns (e.g., "the elite", "globalists", "Big Pharma").

- **Victim**: The entity targeted or harmed by the Action.
  - **Rhetorical Function**: To construct a morally righteous and persecuted in-group ("Us") and foster a sense of shared identity and grievance.
  - **Linguistic Signals**: Look for inclusive first-person plural pronouns (e.g., "we", "us") and language of collective or exclusive victimhood (e.g., "the people", "patriots", "our way of life").

- **Action**: The deliberate verb phrase describing the conspirators' activity.
  - **Rhetorical Function**: To frame events as the result of intentional, malicious agency, rejecting the role of chance or complexity.
  - **Linguistic Signals**: Look for verbs implying secrecy, control, and hostility (e.g., "plotting", "scheming", "concealing", "engineering") and threat-based framing that calls for a response.

- **Effect**: The consequence, goal, or purpose of the Action.
  - **Rhetorical Function**: To maximize the perceived threat and generate a powerful negative emotional response (e.g., anger, anxiety) in the audience.
  - **Linguistic Signals**: Look for language with high negative emotional valence and themes of power, death, control, and destruction.

- **Evidence**: The justification for the conspiratorial claim.
  - **Rhetorical Function**: To create an appearance of legitimacy while insulating the narrative from falsification.
  - **Linguistic Signals**: Beyond explicit citations, look for rhetorical devices like:
    - **Self-Sealing Logic**: Dismissing counter-evidence as "disinformation" or part of the cover-up.
    - **Rhetorical Questions**: Using questions like "Who benefits?" (*cui bono*) or "I'm just asking questions" to imply guilt without providing proof.
    - **Thought-Terminating Clichés**: Using phrases like "do your own research" or "it is what it is" to shut down critical thinking.
</marker_definitions>

<rules>
<output_format>
Return a JSON array inside <answer> ONLY. Each element:
{{"label":"Actor|Action|Effect|Victim|Evidence","text":"<exact substring from TEXT>","start":<optional int>}}

Notes:
- "text" must be copied verbatim from <text_to_analyze>.
- "start" is optional (a hint). Do NOT include "end"; it will be computed downstream.
- No prose, no extra keys, no trailing commas.
</output_format>

<span_boundaries>
- Substrings are measured over raw characters of <text_to_analyze>.
- Do not output offsets other than the optional "start".
- Keep substrings tight; include particles/prepositions only if integral (e.g., "set up", "cover up", "in charge of").
</span_boundaries>

<overlap_policy>
- Ambiguous pairs: {conflict_pairs_str}.
- Action vs Effect: split verb phrase (Action) from purpose/result (Effect); allow minimal overlap only if unavoidable.
- Actor vs Victim: if the same surface form appears in different roles, pick the smallest role-specific mention for each.
- Evidence may overlap others when it is a quotation or citation.
</overlap_policy>

<offset_scope>
- Offsets are computed exactly over <text_to_analyze> by the evaluator; do not normalize quotes/whitespace.
</offset_scope>

<span_length_limits>
- Minimum substring length: 3 characters (after trimming).
- Maximum substring length: 90; Evidence may reach 120 if it is a single explicit citation/quote.
</span_length_limits>

<evidence_quality>
- Prefer explicit sources: URLs, quotations, “according to …”, named reports.
- Avoid purely hedged claims without sources.
</evidence_quality>

<statistical_priors>
Use as tie-breakers when ambiguous:
{priors_str}
If Action and Effect overlap heavily (IoU ≥ 0.6), prefer the label whose start position is closer to its prior.
</statistical_priors>

<negative_case>
- If no markers are present, output [] in <answer>.
</negative_case>

<forbidden_output>
- Do not output anything outside <answer>.
- Inside <answer>, only keys "label","text","start" are allowed. No comments, NaN/inf, or XML echoes.
</forbidden_output>
</rules>

{workflow_block}
"""


def build_s1_user(
    text_input: str, s1_fewshots: List[Dict[str, Any]], include_cot: bool = True
) -> str:
    # format few-shots compactly
    ex_blocks = []
    for ex in (s1_fewshots or [])[:8]:
        spans = ex.get("spans", [])
        spans_json = json.dumps(spans, ensure_ascii=False)
        ex_blocks.append(
            f"<example>\n<text>\n{ex.get('text','')}\n</text>\n<answer>\n{spans_json}\n</answer>\n</example>"
        )
    examples = (
        "<examples>\n" + "\n".join(ex_blocks) + "\n</examples>" if ex_blocks else ""
    )
    cot_line = (
        "Provide your reasoning in <thinking> (kept private), then output ONLY the JSON array described in <output_format> inside <answer>."
        if include_cot
        else "Provide ONLY the final JSON in <answer>."
    )
    return f"""
{examples}

<task>
<text_to_analyze>
{text_input}
</text_to_analyze>
{cot_line}
</task>
""".strip()


import json
import re

def build_s2_system(*, policy_text: str | None = None, include_cot: bool = False,
                    boundary_note: str | None = None, prompt_arts: dict | None = None) -> str:
    # Optional helper renders (reuse your existing ones if present)
    boundary_block = ""
    conflicts_block = ""
    priors_block = ""
    if prompt_arts:
        b = _render_boundary_block(prompt_arts)
        c = _render_conflicts_block(prompt_arts)
        p = _render_priors_block(prompt_arts)
        if b: boundary_block = f"\n<boundary_guidance>\n{b}\n</boundary_guidance>"
        if c: conflicts_block = f"\n<conflicts>\n{c}\n</conflicts>"
        if p: priors_block = f"\n<priors>\n{p}\n</priors>"

    policy_block = f"\n<policy>\n{policy_text}\n</policy>" if policy_text else ""
    workflow_block = ""
    if include_cot:
        workflow_block = """<workflow>
1. You will be given the original text and a JSON list of psycholinguistic markers extracted from it.
2. First, analyze the provided markers as evidence inside a `<thinking>` block. Specifically, evaluate them against the `<hallmarks_of_conspiracy_narratives>`.
   - Does the density and type of 'Actor' and 'Victim' markers establish a strong 'us vs. them' narrative?
   - Do the 'Action' and 'Effect' markers create a causal story driven by malicious intent, rather than chance?
   - Do the 'Evidence' markers show signs of a self-sealing or non-falsifiable epistemic style?
3. Based on this structured analysis, make a final classification.
4. Provide your final answer as a single JSON object inside an `<answer>` block: {"label": "...", "rationale": "..."}
</workflow>"""

    return f"""<role>
You are an expert social scientist specializing in the analysis of online discourse. Your task is to classify a text by evaluating its narrative structure against established psycholinguistic patterns of conspiratorial rhetoric.
</role>
<label_definitions>
- **conspiracy**: The text alleges a secret plot by a powerful group that is harmful or illegal. It exhibits several of the hallmarks below.
- **non**: The text does not contain conspiratorial allegations and lacks the key narrative hallmarks.
- **cant_tell**: The text is too ambiguous or lacks sufficient information to make a clear determination.
</label_definitions>

<hallmarks_of_conspiracy_narratives>
Conspiratorial texts are not just factually wrong; they follow a specific narrative and rhetorical structure. Use these hallmarks to guide your classification:
1.  **Manichean Worldview ("Us vs. Them"):** The narrative frames events as a struggle between a virtuous in-group ("we", "the people") and a malevolent, powerful out-group ("they", "the elites"). A high density of Actor and Victim markers is a strong signal.
2.  **Teleological Causality (Nothing is by Accident):** Events are explained as the direct result of intentional, secret actions by the malevolent actors, rejecting the role of chance, complexity, or incompetence.
3.  **Self-Sealing Epistemology (Unfalsifiable Logic):** The narrative is immune to counter-evidence. Evidence against the theory is re-framed as proof of the cover-up's effectiveness. A lack of evidence is proof of the conspiracy's secrecy.
4.  **Emotionally Charged Language:** The narrative relies on powerful negative emotions like anger, anxiety, and fear to create a sense of existential threat and urgency.
</hallmarks_of_conspiracy_narratives>

  <endorsement_test>
    - Endorsing/advocating conspiratorial framing ⇒ consider "conspiracy".
    - Reporting, mocking, or debunking (neutral/critical stance) ⇒ "non".
  </endorsement_test>

  <marker_signals>
    - Role Framing: Malevolent Actor (“they”, “elite”, “deep state”) + Victim (“we/us”).
    - Intentionality: Action verbs of secrecy/control + extreme Effect outcomes.
    - Epistemic Closure: self-sealing logic (counter-evidence ⇒ “cover-up”), thought-terminating clichés.
  </marker_signals>

  <calibration>
    - Strong explicit endorsement of hidden-plot + multiple signals (Actor+Action+Effect OR self-sealing) ⇒ p_conspiracy ≥ 0.85.
    - Mixed/ambiguous signals without endorsement ⇒ 0.40 ≤ p_conspiracy ≤ 0.60.
    - Neutral reporting/debunking; absence of hidden-agent framing ⇒ p_non ≥ 0.80.
    - Ensure p_conspiracy + p_non = 1.0.
  </calibration>

  <rationale_policy>
    - ≤2 sentences; name the decisive cues (e.g., “self-sealing logic, ‘they’ + agenda”).
    - Do NOT reveal chain-of-thought beyond brief cues.
  </rationale_policy>

  <forbidden_output>
    - Nothing outside <answer>.
  </forbidden_output>

{policy_block}{boundary_block}{conflicts_block}{priors_block}
<output_format>
Return ONLY a single JSON object inside <answer>:
{{"label":"conspiracy|non","p_conspiracy":0.xx,"p_non":0.xx,"rationale":"<=2 sentences"}}
Ensure probabilities sum to 1.0 and label = argmax.
</output_format>{workflow_block}"""

def build_s2_user(*, text_input: str, s1_output: list | None,
                  s2_fewshots: list | None = None, include_cot: bool = False) -> str:
    ex_block = ""
    if s2_fewshots:
        # few-shot XML (compact)
        parts = []
        for ex in s2_fewshots:
            t = ex.get("text","")
            lbl = (ex.get("label") or (ex.get("gold") or {}).get("label") or "non").lower()
            lbl = "conspiracy" if lbl == "conspiracy" else "non"
            mks = ex.get("markers") or []
            mk_norm = []
            for m in mks:
                lab = (m.get("type") or m.get("label") or "").strip()
                s = m.get("startIndex", m.get("start"))
                e = m.get("endIndex", m.get("end"))
                try:
                    s, e = int(s), int(e)
                except: 
                    continue
                if lab and e > s:
                    mk_norm.append({"type": lab, "startIndex": s, "endIndex": e})
            gold = {"label": lbl, "p_conspiracy": 0.8 if lbl=="conspiracy" else 0.2,
                    "p_non": 0.2 if lbl=="conspiracy" else 0.8,
                    "rationale": "concise example rationale."}
            parts.append(
                "<example>\n"
                "<text>\n" + t + "\n</text>\n" +
                ("<extracted_markers>\n" + json.dumps(mk_norm, ensure_ascii=False) + "\n</extracted_markers>\n" if mk_norm else "") +
                "<answer>\n" + json.dumps(gold, ensure_ascii=False) + "\n</answer>\n"
                "</example>"
            )
        ex_block = "<examples>\n" + "\n\n".join(parts) + "\n</examples>\n\n"

    markers_json = json.dumps(s1_output or [], ensure_ascii=False)

    task_tail = ("Provide brief reasoning in <thinking> then the final JSON in <answer>."
                 if include_cot else
                 "Provide ONLY the final JSON in <answer>.")

    return f"""{ex_block}<task>
<text_to_analyze>
{text_input}
</text_to_analyze>
<extracted_markers>
{markers_json}
</extracted_markers>
Instructions: Use the markers as evidence; ambiguity without hidden-plot framing should lean "non".
{task_tail}
</task>"""



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


def extract_answer_json(text: str):
    """Extract JSON from inside <answer>...</answer>. Falls back to last JSON-like block."""
    if not isinstance(text, str):
        return None
    s = _normalize_quotes(text)
    m = re.search(r"<answer>\s*(\[.*?\]|\{.*?\})\s*</answer>", s, re.S)
    blob = m.group(1) if m else None
    if not blob:
        # fallback: last JSON-looking block
        cand = re.findall(r"(\[.*\]|\{.*\})", s, re.S)
        blob = cand[-1] if cand else None
    if not blob:
        return None
    # simple cleanup
    blob = blob.strip().strip("`")
    try:
        return json.loads(blob)
    except Exception:
        # crude repairs
        blob = re.sub(r",\s*([}\]])", r"\1", blob)  # trailing commas
        blob = blob.replace("'", '"')  # single -> double
        try:
            return json.loads(blob)
        except Exception:
            return None
