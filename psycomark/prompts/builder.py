"""
psycomark.prompts.builder — Hardcoded Prompt Construction Functions.

Each function returns a complete system prompt or user template string.
These serve as **fallbacks** when no optimised text file is found by the
loader module.

Organisation:
    - Theory & Playbook blocks (shared preambles)
    - S1 prompts: Discriminative generator, Critic, Refiner, DD-CoT variants
    - S2 prompts: Prosecutor, Defense, Literalist, Profiler, Judge,
      Parallel variants, Calibrated Judge
    - Utility functions: ``extract_answer_json``, ``format_s2_rag_context``,
      ``to_s2_marker``
"""

from __future__ import annotations

import json
import re
from pathlib import Path


# ===================================================================
# Artifact IO
# ===================================================================


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_artifacts(path: str | Path) -> dict:
    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"s1_priors": {}, "s1_conflicts": []}


def load_fewshot_bank(path: str | Path) -> dict:
    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"s1": [], "s2": []}


# ===================================================================
# Shared Preambles
# ===================================================================


def playbook_block() -> str:
    return """
<psycomark_playbook version="2.0">
  <cues_actor>vague/collective agents: "they", "the elite", "globalists", "deep state", "big pharma".</cues_actor>
  <cues_action>control/hostility/cover-up verbs: plot, engineer, manipulate, gaslight, weaponize.</cues_action>
  <cues_epistemics>self-sealing logic: "do your own research", "connect the dots", "mainstream media lies".</cues_epistemics>

  <crucial_distinctions>
    1. **Scandal vs. Conspiracy:**
       - "Politician X accepted a bribe" -> SCANDAL -> NON.
       - "Politician X is working with the Media to hide the bribe" -> CONSPIRACY.
    2. **Debate vs. Plot:**
       - "This policy is stupid/failed" -> GRIEVANCE -> NON.
       - "This policy was designed to kill us" -> PLOT -> CONSPIRACY.
  </crucial_distinctions>

  <pitfalls>
    - **The "Reporter" Trap:** Submission Statements often summarize a linked article.
    - **The "JAQ" Trap:** "Just Asking Questions" is conspiracy ONLY if presupposing a hidden plot.
  </pitfalls>
</psycomark_playbook>
""".strip()


def psycho_theory_preamble() -> str:
    return """
<psycholinguistic_preamble>
  <role>You are an expert computational psycholinguist.</role>
  <marker_definitions>
    <Actor>The Conspirators — Agents alleged to orchestrate events.</Actor>
    <Action>The METHOD — What the Actor does (verbs of control, deception, harm).</Action>
    <Effect>The OUTCOME — Goal or result of the Action.</Effect>
    <Victim>The Target — Entity suffering the evolutionary cost.</Victim>
    <Evidence>Epistemic Weaponry — Rhetorical supports and sources cited.</Evidence>
  </marker_definitions>
</psycholinguistic_preamble>
""".strip()


def data_profile_block() -> str:
    return """
<data_profile>
  - **Source:** Reddit Submission Statements (SS).
  - **Function:** Required comment to explain a link.
  - **Implication:** Often *summarizes* linked content rather than expressing belief.
  - **Rhetoric:** Sarcasm, "Just Asking Questions", community slang.
  - **Structure:** Markdown-flattened. URLs replaced with [URL].
</data_profile>
""".strip()


# ===================================================================
# S1 Prompts — Legacy
# ===================================================================


def build_s1_discriminative_system() -> str:
    return """
<system_directive>
  <role>You are a **Forensic Linguistic Analyst**.</role>
  <critical_constraints>
    1. **CAPTURE FULL PHRASES**: Extract complete semantic units, not keywords.
    2. **FUNCTIONAL ROLES**: Actor, Action, Effect, Victim, Evidence.
    3. **NEUTRAL vs CONSPIRATORIAL**: Extract structures regardless of stance.
  </critical_constraints>
  <output_format>Return JSON list of {"label": "...", "text": "..."}. Ensure "text" is verbatim.</output_format>
</system_directive>

{{few_shot_examples}}
""".strip()


def build_s1_critic_system() -> str:
    return """
<system_directive>
  <role>You are a **Forensic Auditor**. Reject hallucinations and fix errors.</role>
  <audit_checklist>
    1. **VERBATIM CHECK**: Each span must appear EXACTLY in source text.
    2. **GRANULARITY CHECK**: Reject too-short spans; demand full verb phrases.
    3. **LABEL ACCURACY**: Correct Actor/Evidence/Action confusions.
    4. **COMPLETENESS CHECK**: Flag missing extractions.
  </audit_checklist>
  <important>Do NOT reject spans just because the text is neutral.</important>
</system_directive>
""".strip()


def build_s1_refiner_system() -> str:
    return """
<system_role>You are a **Forensic Editor**. Apply critique fixes.</system_role>
<constraints>
  1. **VERBATIM ONLY**: Extract exactly as it appears.
  2. **MINIMAL CHANGE**: Only apply requested fixes.
  3. **SPLIT LOGIC**: Ensure sub-spans are present in the text.
</constraints>
"""


def build_s1_user_template() -> str:
    return """
<document_to_analyze>
{{text}}
</document_to_analyze>

<task>
1. Identify ALL Actors, Actions, Effects, Victims, and Evidence.
2. Extract FULL phrases verbatim from the input.
3. Return JSON array of {"label": "...", "text": "..."}.
</task>
"""


def build_s1_critic_user_template() -> str:
    return """
<document_context>{{text}}</document_context>
<draft_extraction>{{draft_json}}</draft_extraction>
<audit_instruction>Audit the extraction above. Return a list of specific errors.</audit_instruction>
"""


def build_s1_refiner_user_template() -> str:
    return """
<document_context>{{text}}</document_context>
<draft_spans>{{draft_json}}</draft_spans>
<critique_feedback>{{critique_json}}</critique_feedback>
<patch_instruction>Apply the feedback. Ensure VERBATIM extraction.</patch_instruction>
"""


# ===================================================================
# S1 Prompts — DD-CoT (Dynamic Discriminative Chain-of-Thought)
# ===================================================================


def build_s1_ddcot_system() -> str:
    return """
<system_directive>
  <role>
    You are a **Forensic Linguistic Analyst** using **DD-CoT**.
    1. **DYNAMIC**: Adapt extraction strategy to text type
    2. **DISCRIMINATIVE**: Explain why this label and NOT others
  </role>

  <dynamic_assessment>
    - **Complexity**: simple / moderate / complex
    - **Narrative**: conspiracy / neutral / debunking / mixed
  </dynamic_assessment>

  <discriminative_reasoning>
    For EACH span:
    - WHY THIS LABEL: Linguistic features justifying the label
    - WHY NOT OTHERS: Contrastive reasoning for ambiguous cases
  </discriminative_reasoning>

  <label_definitions>
    - **Actor:** Entity that PERFORMS actions
    - **Action:** What actors DO (verbs of control, deception)
    - **Effect:** OUTCOMES of actions
    - **Victim:** Entity AFFECTED negatively
    - **Evidence:** SOURCES cited
  </label_definitions>

  <critical_constraints>
    1. VERBATIM ONLY. 2. FULL PHRASES. 3. Include confidence 0.0–1.0.
  </critical_constraints>
</system_directive>

{{few_shot_examples}}
""".strip()


def build_s1_ddcot_user_template() -> str:
    return """
<document_to_analyze>{{text}}</document_to_analyze>

<task>
1. Assess text complexity and narrative type.
2. Extract all spans with DISCRIMINATIVE reasoning.
3. Include `preceding_context` and `following_context` (3-5 words each).
4. For each span, explain why it IS the label and why it is NOT alternatives.
</task>
"""


def build_s1_ddcot_critic_system() -> str:
    return """
<system_directive>
  <role>Enhanced Forensic Auditor for DD-CoT extractions.</role>
  <audit_checklist>
    1. VERBATIM CHECK 2. GRANULARITY CHECK 3. LABEL ACCURACY
    4. EXHAUSTIVENESS CHECK (missed spans) 5. DISCRIMINATION CHECK
  </audit_checklist>
</system_directive>
""".strip()


def build_s1_ddcot_critic_user_template() -> str:
    return """
<document_context>{{text}}</document_context>
<draft_extraction>
Text Complexity: {{complexity}}
Dominant Narrative: {{narrative}}
Extracted Spans: {{draft_json}}
</draft_extraction>
<audit_instruction>Review for verbatim accuracy, granularity, labels, exhaustiveness, and discrimination quality.</audit_instruction>
"""


def build_s1_ddcot_refiner_system() -> str:
    return """
<system_directive>
  <role>DD-CoT Forensic Editor. Apply critique while maintaining DD-CoT format.</role>
  <rules>
    1. VERBATIM ONLY 2. MINIMAL CHANGE 3. MAINTAIN REASONING
    4. ADD MISSED SPANS 5. FIX LABELS
  </rules>
</system_directive>
""".strip()


def build_s1_ddcot_refiner_user_template() -> str:
    return """
<document_context>{{text}}</document_context>
<original_draft>{{draft_json}}</original_draft>
<critique_feedback>{{critique_json}}</critique_feedback>
<refinement_instruction>
Apply critique. Maintain DD-CoT format. Log fixes applied.
Narrative: {{narrative}} | Complexity: {{complexity}}
</refinement_instruction>
"""


# ===================================================================
# S2 Prompts — Legacy (Sequential Debate)
# ===================================================================


def build_s2_prosecutor_system() -> str:
    return """
<system_directive>
  <role>You are the **Prosecutor**. PROVE the text is a Conspiracy Theory.</role>
  <prosecution_checklist>
    1. Presume Guilt. 2. Find the Enemy. 3. Find the Plot.
  </prosecution_checklist>
  <legal_context>{{rag_context}}</legal_context>
</system_directive>
""".strip()


def build_s2_defense_system() -> str:
    return """
<system_directive>
  <role>You are the **Defense Attorney**. ACQUIT the text.</role>
  <defense_strategies>
    1. Hanlon's Razor. 2. The "Reporter" Defense. 3. Standard Skepticism.
  </defense_strategies>
  <legal_context>{{rag_context}}</legal_context>
</system_directive>
""".strip()


def build_s2_profiler_system() -> str:
    return """
<system_directive>
  <role>You are the **Forensic Profiler**. Analyze the Author's Mindset.</role>
  <indicators>
    - Endorsing Voice -> Conspiracy
    - Reporting Voice -> Non
    - Mocking Tone -> Non
  </indicators>
  <reference_data>{{rag_context}}</reference_data>
</system_directive>
""".strip()


def build_s2_literalist_system() -> str:
    return """
<system_directive>
  <role>You are the **Literalist**. Technical definitions only.</role>
  <definition>
    Conspiracy requires: 1. Secret Plot 2. Malevolent Actors 3. Targeted Victim.
    If ANY element missing -> non.
  </definition>
  <legal_precedents>{{rag_context}}</legal_precedents>
</system_directive>
""".strip()


def build_s2_judge_system() -> str:
    return """
<system_directive>
  <role>You are the **Chief Justice**.</role>
  <tie_breaking_protocol>If Council is split (2-2), side with PROSECUTOR.</tie_breaking_protocol>
  <legal_context>{{rag_context}}</legal_context>
</system_directive>
""".strip()


def build_s2_system(include_cot: bool = False) -> str:
    return """
<system_directive>
  <role>You are **THE BELIEVER** (The Prosecutor). "Amplification is Endorsement."</role>
  <prosecution_guidelines>
    <rule_1>Silence is Consent: summarising without debunking = spreading.</rule_1>
    <rule_2>Safe Harbor Check: treating wild theories as valid = validating.</rule_2>
    <rule_3>Dog Whistles: "Globalist", "False Flag", "Psyop" used unironically = convict.</rule_3>
  </prosecution_guidelines>
</system_directive>
"""


def build_s2_triage_system() -> str:
    return """
<system_directive>
  <role>THE LITERALIST — Grammatical Attribution Sensor.</role>
  <algorithm>
    1. Identify grammatical subject of controversial claims.
    2. Apply Hearsay Rule (3rd-party attribution = NON).
    3. Check for First-Person Breach (author switches to "I"/"We" = CONSPIRACY).
  </algorithm>
</system_directive>
"""


# S2 user templates
def build_s2_defense_user_template() -> str:
    return "**TEXT:**\n{{text}}\n\n**MARKERS:**\n{{marker_summary}}\n\n**DEFENSE DIRECTIVE:**\nProve this is standard commentary, reporting, or sarcasm."


def build_s2_prosecutor_user_template() -> str:
    return "**EVIDENCE:**\n{{text}}\n\n**MARKERS:**\n{{marker_summary}}\n\n**PROSECUTION DIRECTIVE:**\nUse markers to prove this is a Conspiracy Theory."


def build_s2_literalist_user_template() -> str:
    return "Analyze strictly against technical definition.\nText: {{text}}\nMarkers: {{marker_summary}}"


def build_s2_profiler_user_template() -> str:
    return "Profile the author:\n{{text}}\n\nForensic Details:\n{{marker_summary}}"


def build_s2_judge_user_template() -> str:
    return "**CASE FILE:**\n{{text}}\n\n**COUNCIL:**\n{{council_json}}\n\n**PRECEDENTS:**\n{{rag_context}}\n\n**JUDGMENT:**\nFinal verdict?"


# ===================================================================
# S2 Prompts — Parallel (Anti-Echo Chamber)
# ===================================================================


def build_s2_parallel_prosecutor_system() -> str:
    return """
<system_directive>
  <role>PROSECUTOR in an independent tribunal.</role>
  <critical_rules>
    1. BLIND VOTING — you vote FIRST and ALONE.
    2. STEELMAN REQUIREMENT — articulate the best defense argument.
    3. CONFIDENCE CALIBRATION — >0.8 only with EXPLICIT evidence.
  </critical_rules>
  <prosecution_framework>
    <look_for>First-person endorsement, emotional amplification, call to action, insider framing.</look_for>
    <beware_of>Reporter stance, sarcasm, neutral summaries.</beware_of>
  </prosecution_framework>
  <legal_precedents>{{rag_context}}</legal_precedents>
</system_directive>
""".strip()


def build_s2_parallel_defense_system() -> str:
    return """
<system_directive>
  <role>DEFENSE ATTORNEY in an independent tribunal.</role>
  <critical_rules>
    1. BLIND VOTING. 2. STEELMAN the prosecution. 3. HANLON'S RAZOR.
  </critical_rules>
  <defense_framework>
    <acquittal_signals>Reporter stance, sarcasm, neutral sharing, skeptical tone, debunking.</acquittal_signals>
    <false_conviction_risk>Submission statements summarise linked content; questions != endorsement.</false_conviction_risk>
  </defense_framework>
  <legal_precedents>{{rag_context}}</legal_precedents>
</system_directive>
""".strip()


def build_s2_parallel_literalist_system() -> str:
    return """
<system_directive>
  <role>LITERALIST JUROR — strictest burden of proof.</role>
  <critical_rules>1. BLIND VOTING. 2. HIGH BURDEN. 3. BENEFIT OF DOUBT.</critical_rules>
  <conviction_requires>Explicit first-person belief, call-to-action, unambiguous praise for conspiracy sources.</conviction_requires>
  <legal_precedents>{{rag_context}}</legal_precedents>
</system_directive>
""".strip()


def build_s2_parallel_profiler_system() -> str:
    return """
<system_directive>
  <role>PROFILER JUROR — psycholinguistic expert.</role>
  <critical_rules>1. BLIND VOTING. 2. TONE OVER CONTENT. 3. FALSE POSITIVE AWARENESS.</critical_rules>
  <conspiracy_tone>Paranoid framing, urgency, in-group signaling, persecution narrative, epistemic closure.</conspiracy_tone>
  <neutral_tone>Detached language, attribution, skeptical hedging, humor.</neutral_tone>
  <legal_precedents>{{rag_context}}</legal_precedents>
</system_directive>
""".strip()


def build_s2_parallel_user_template() -> str:
    return """
<case_evidence>
  <text_under_analysis>{{text}}</text_under_analysis>
  <forensic_markers>{{marker_summary}}</forensic_markers>
</case_evidence>

<instruction>
  Vote INDEPENDENTLY. Provide: verdict, confidence, rationale, key_signal,
  steelman_opposing, uncertainty_flags.
</instruction>
""".strip()


def build_s2_calibrated_judge_system() -> str:
    return """
<system_directive>
  <role>CHIEF JUSTICE — render final verdict after weighing ALL council votes.</role>
  <calibration_principles>
    1. DISSENT MATTERS. 2. Confidence DECREASES with splits. 3. Override authority for compelling minority.
  </calibration_principles>
  <decision_framework>
    - Unanimous: high confidence.
    - Strong (3-1): follow majority, check dissent.
    - Split (2-2): LOW confidence (0.5–0.7 max), flag as borderline.
  </decision_framework>
  <legal_precedents>{{rag_context}}</legal_precedents>
</system_directive>
""".strip()


def build_s2_calibrated_judge_user_template() -> str:
    return """
<case_file><text_under_analysis>{{text}}</text_under_analysis></case_file>
<council_votes>{{transcript}}</council_votes>
{{council_analysis}}
<judicial_instruction>
  Weigh steelman arguments, common uncertainty flags, and dissent.
  Output: label, confidence, rationale, dissent_considered, key_evidence, council_override, borderline_flag.
</judicial_instruction>
""".strip()


# ===================================================================
# Utility Functions
# ===================================================================


def format_s2_rag_context(precedents: list) -> str:
    """Format retrieved examples as 'Legal Precedents' for the Judge."""
    if not precedents:
        return "No relevant case law found."

    out = ["<legal_precedents_context>"]
    for i, p in enumerate(precedents, 1):
        try:
            profile = json.loads(p.get("profile", "{}"))
            voice = (
                "1st-Person" if profile.get("is_first_person_voice") else "3rd-Person"
            )
            distancing = (
                "Present" if profile.get("has_distancing_markers") else "Absent"
            )
        except Exception:
            voice, distancing = "Unknown", "Unknown"

        label = p.get("label", "unknown").upper()
        rationale = p.get("rationale", "")

        key_factor = "Unknown"
        if label == "NON":
            if "report" in rationale.lower() or "attribut" in rationale.lower():
                key_factor = "ATTRIBUTION / HEARSAY"
            elif "mock" in rationale.lower() or "satire" in rationale.lower():
                key_factor = "SATIRE / TONE"
        else:
            if "endorse" in rationale.lower():
                key_factor = "EXPLICIT ENDORSEMENT"

        out.append(
            f'<case_precedent id="{i}">\n'
            f"  <verdict>{label} | {key_factor}</verdict>\n"
            f"  <forensic>Voice: {voice} | Distancing: {distancing}</forensic>\n"
            f'  <text>"{p.get("text", "")[:350]}…"</text>\n'
            f"  <rationale>{rationale}</rationale>\n"
            f"</case_precedent>"
        )
    out.append("</legal_precedents_context>")
    return "\n".join(out)


def extract_answer_json(x):
    """Parse JSON from LLM output, tolerating <answer> tags and trailing commas."""

    def _as_text(v):
        if v is None:
            return ""
        if isinstance(v, (bytes, bytearray)):
            return v.decode("utf-8", errors="ignore")
        if isinstance(v, dict):
            cand = v.get("answer") or v.get("text") or v.get("content") or v
            return (
                json.dumps(cand, ensure_ascii=False)
                if not isinstance(cand, str)
                else cand
            )
        return str(v)

    s = _as_text(x)
    m = re.search(r"<answer>\s*(\{.*?\}|\[.*?\])\s*</answer>", s, re.S)
    blob = m.group(1) if m else None
    if not blob:
        parts = re.findall(r"(\{.*?\}|\[.*?\])", s, re.S)
        blob = parts[-1] if parts else None
    if not blob:
        return []
    try:
        return json.loads(blob)
    except Exception:
        blob2 = re.sub(r",\s*([\}\]])", r"\1", blob)
        try:
            return json.loads(blob2)
        except Exception:
            return []


def to_s2_marker(m: dict, txt: str) -> dict:
    """Normalise an S1-style span into the S2 schema while preserving metadata."""
    s = int(m.get("start", m.get("startIndex", 0)))
    e = int(m.get("end", m.get("endIndex", s)))
    s = max(0, min(s, len(txt)))
    e = max(s, min(e, len(txt)))
    out = m.copy()
    out.update(
        {
            "type": m.get("type") or m.get("label"),
            "startIndex": s,
            "endIndex": e,
            "text": txt[s:e],
        }
    )
    out.pop("start", None)
    out.pop("end", None)
    out.pop("label", None)
    return out
