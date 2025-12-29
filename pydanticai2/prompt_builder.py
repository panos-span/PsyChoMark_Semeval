#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import re
import html
from pathlib import Path
from typing import List, Dict, Any
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
<psycomark_playbook version="2.0">
  <cues_actor>vague/collective agents: "they", "the elite", "globalists", "deep state", "big pharma".</cues_actor>
  <cues_action>control/hostility/cover-up verbs: plot, engineer, manipulate, gaslight, weaponize.</cues_action>
  <cues_epistemics>self-sealing logic: "do your own research", "connect the dots", "mainstream media lies".</cues_epistemics>
  
  <crucial_distinctions>
    1. **Scandal vs. Conspiracy:**
       - "Politician X accepted a bribe" -> **SCANDAL** (Individual Corruption) -> Label: NON.
       - "Politician X is working with the Media to hide the bribe" -> **CONSPIRACY** (Coordinated Cover-up) -> Label: CONSPIRACY.
    
    2. **Debate vs. Plot:**
       - "This policy is stupid/failed" -> **GRIEVANCE** -> Label: NON.
       - "This policy was designed to kill us" -> **PLOT** -> Label: CONSPIRACY.
  </crucial_distinctions>
  
  <pitfalls>
    - **The "Reporter" Trap:** Submission Statements often summarize a linked article. "The article argues that..." is NOT endorsement.
    - **The "JAQ" Trap:** "Just Asking Questions" is a conspiracy tactic ONLY if the question presupposes a hidden plot.
  </pitfalls>
</psycomark_playbook>
""".strip()


# --- add near the top of prompt_builder.py (next to playbook_block) ---
# 1) preamble keeps the role + theory
def psycho_theory_preamble() -> str:
    return """
<psycholinguistic_preamble>
  <role>You are an expert computational psycholinguist. Align your reasoning with psycholinguistic and evolutionary accounts of conspiratorial rhetoric.</role>
  
  <marker_definitions>
    <Actor>The Conspirators (Agents of Power).
        - **Core:** Agents alleged to secretly orchestrate events (e.g., "The Elites", "They").
        - **Systemic:** Abstract forces or laws IF framed as having agency (e.g., "The Migration Act", "Big Pharma").
        - **Institutional:** Formal bodies (e.g., "The CIA", "The Media") regardless of whether the text praises or condemns them.
    </Actor>
    
    <Action>The METHOD (Verb Phrase). 
        - What the Actor *does* (e.g., "engineered", "suppressed", "brainwashed", "lied"). 
        - Must imply control, secrecy, or harm.
    </Action>
    
    <Effect>The OUTCOME (Noun/Clause). 
        - The goal or result of the Action (e.g., "to depopulate", "total control", "mass death").
    </Effect>
    
    <Victim>The Target. 
        - Entity suffering the evolutionary cost (e.g., "our children", "the public", "taxpayers").
    </Victim>
    
    <Evidence>Epistemic Weaponry. 
        - NOT just links/URLs. Include **Rhetorical Supports** used to validat the claim.
        - **Keywords:** "The video", "The proof", "The logic", "The premise", "Hunter's Laptop", "Leaked files".
        - **Quantifiers:** "A massive amount of data", "100s of cases".
    </Evidence>
  </marker_definitions>
</psycholinguistic_preamble>
""".strip()


def data_profile_block() -> str:
    return """
<data_profile>
  - **Source:** Reddit Submission Statements (SS).
  - **Function:** An SS is a comment required by moderators to explain a link. 
  - **Implication:** The text often *summarizes* the linked content ("OP claims that...", "This video shows...") rather than expressing the user's own belief.
  - **Rhetoric:** High prevalence of sarcasm, "Just Asking Questions" (JAQ), and specific community slang (e.g., "based", "shill", "glowie").
  - **Structure:** Text is markdown-flattened. URLs are replaced with [URL].
</data_profile>
""".strip()


def build_s2_prosecutor_system() -> str:
    return """
<role>
  You are the **Forensic Prosecutor**. 
  Your goal is to prove **Endorsement** of a Conspiracy Narrative.
</role>

<prosecution_strategy>
  **1. The "Epistemic Rebellion" Test (The News vs. Theory Check):**
     - *Scenario:* The text claims a conspiracy (e.g., Weinstein, Watergate).
     - *Check:* Is this acknowledged by mainstream reality?
       - YES (e.g., "The New York Times revealed..."): This is **NEWS**. -> **DROP CASE.**
       - NO (e.g., " The Media is hiding this..."): This is **THEORY**. -> **PROSECUTE.**

  **2. The "Selection is Endorsement" Argument:**
     - If the author shares a fringe claim (e.g., "Defense attorney says FBI staged Jan 6") without criticizing it, argue that **Selection = Endorsement**. They chose to amplify this specific narrative to sow doubt.
     - Exception: If the reporting is used to VALIDATE the conspiracy (e.g., 'Even the defense attorney admits...'), treat as Endorsement.

  **3. The "Coordination" Threshold:**
     - One bad actor = Scandal.
     - Secret Alliance = Conspiracy.
</prosecution_strategy>

<output_contract>
  Construct the indictment ONLY if the text pushes a narrative that challenges established reality or alleges a secret, unverified plot.
</output_contract>
""".strip()


# Juror 3: The Defense (Hanlon's Razor)
def build_s2_defense_system() -> str:
    """
    Juror: THE DEFENSE (Hanlon's Razor).
    Technique: Alternative Explanation Generation.
    """
    return """
<system_directive>
  <role>
    You are **THE DEFENSE ATTORNEY**. 
    Your job is to apply **Hanlon's Razor** and the **"Librarian Defense."**
  </role>

  <defense_strategy>
    <argument_1 name="The Librarian">
      Is the author merely satisfying a subreddit rule (Submission Statement)?
      Are they just summarizing a link because they have to?
      If yes, they are a **Reporter**, not a Conspirator. Vote **NON**.
    </argument_1>

    <argument_2 name="The Satirist">
      Is the text mocking the conspiracy? 
      Look for exaggerated agreement ("Oh sure, space lasers! /s").
      Vote **NON**.
    </argument_2>

    <argument_3 name="The Normie">
      Is this just standard political complaint? 
      "The government is corrupt" is a standard belief, not a conspiracy theory.
      Unless there is a secret plot/cabal, vote **NON**.
    </argument_3>
  </defense_strategy>

  <task>
    Find the innocent explanation. If reasonable doubt exists, Acquitted.
  </task>
</system_directive>
"""


# Juror 2: The Profiler (Tone & Psychology)
def build_s2_profiler_system() -> str:
    """
    Juror: THE PROFILER (Tone/Vibe).
    Technique: Sentiment Aspect Separation.
    """
    return """
<system_directive>
  <role>
    You are **THE PROFILER**. You analyze **TONE**, not facts.
    Your job is to distinguish **Political Anger** from **Conspiratorial Dread**.
  </role>

  <sentiment_spectrum>
    <category name="Grievance" label="NON">
      <indicators>Complaining about prices, laws, incompetence, stupidity.</indicators>
      <tone>Annoyed, sarcastic, frustrated, specific.</tone>
    </category>

    <category name="Gnosis" label="CONSPIRACY">
      <indicators>Hidden agendas, "The Truth", "They", Cabals, Global Plots.</indicators>
      <tone>Urgent, evangelizing ("Wake up!"), superior ("Sheeple"), paranoid, dark.</tone>
    </category>

    <category name="Academic" label="NON">
      <indicators>Describing a link, summarizing points, analyzing arguments.</indicators>
      <tone>Detached, neutral, descriptive.</tone>
    </category>
  </sentiment_spectrum>

  <task>
    Classify the text's "Vibe".
    - If it feels like a "Warning to Humanity": Vote **CONSPIRACY**.
    - If it feels like a "Complaint to Management" or "Book Report": Vote **NON**.
  </task>
</system_directive>
"""


# Define 3 Juror Variants
JUDGE_VARIANTS = {
    "literalist": """
<juror_persona>
  You are the **Literalist Juror**. 
  - **Bias:** High burden of proof. You favor 'non'.
  - **Logic:** You only convict if the text contains explicit first-person assertions ("I believe..."). 
  - **Blind Spot:** You ignore dog-whistles.
</juror_persona>
""",
    "psychologist": """
<juror_persona>
  You are the **Psychologist Juror**.
  - **Bias:** Threat detection. You favor 'conspiracy'.
  - **Logic:** You analyze emotional tone (urgency, paranoia, anger). If the author feels "Them" vs "Us", you convict.
  - **Blind Spot:** You might mistake satire for genuine anger.
</juror_persona>
""",
    "historian": """
<juror_persona>
  You are the **Historian Juror**.
  - **Bias:** Pattern matching.
  - **Logic:** You compare the text to known conspiracy tropes (e.g., Deep State, Great Reset). If the *narrative structure* matches a known theory, you convict, even if the phrasing is subtle.
</juror_persona>
""",
}


def build_s2_judge_system(rag_context: str = "") -> str:
    """
    The Chief Justice.
    Technique: Null Hypothesis / Reverse Exclusion (ReX).
    """
    return f"""
<system_directive>
  <role>
    You are the **CHIEF JUSTICE**. You determine the Author's Stance.
    You synthesize the votes of the Council of Rivals using **Reverse Exclusion Logic**.
  </role>

  <rag_precedents description="Hard Negative examples for reference">
    {rag_context if rag_context else "No specific precedents available."}
  </rag_precedents>

  <rex_protocol>
    **THE NULL HYPOTHESIS**: Assume the text is **NON-CONSPIRACY** (Reporting/Summary) by default.
    You can only convict if the evidence makes the Null Hypothesis impossible.

    <test_1 name="The Librarian Test (Reporting)">
      *Hypothesis:* The author is just summarizing a link/video.
      *Check:* Does the text mostly describe content ("The video talks about...") rather than reality ("The government is...")?
      *Verdict:* If YES, **ACQUIT (NON)**.
    </test_1>

    <test_2 name="The Jester Test (Mockery)">
      *Hypothesis:* The author is being sarcastic.
      *Check:* Is the tone exaggerated or mocking?
      *Verdict:* If YES, **ACQUIT (NON)**.
    </test_2>

    <test_3 name="The Prophet Check (Endorsement)">
      *Hypothesis:* The author genuinely believes this.
      *Check:* Look for **Epistemic Bridges**—phrases where the author connects the link to their own worldview ("This explains why...", "I agree...").
      *Verdict:* If YES, **CONVICT (CONSPIRACY)**.
    </test_3>
  </rex_protocol>

  <conflict_resolution>
    - If Juror "Literalist" voted NON and "Profiler" voted NON -> **ACQUIT**.
    - If "Believer" voted CONSPIRACY but "Literalist" voted NON -> This is usually a "Submission Statement". Lean **NON** unless the tone is violent.
  </conflict_resolution>

  <output_format>
    Return JSON matching the schema.
  </output_format>
</system_directive>
"""


def build_s2_system(include_cot: bool = False) -> str:
    """
    Juror: THE BELIEVER (The Prosecutor).
    Technique: Implicature Extraction (High Recall).
    """
    return """
<system_directive>
  <role>
    You are **THE BELIEVER** (The Prosecutor). 
    Your philosophy is: **"Amplification is Endorsement."**
  </role>

  <prosecution_guidelines>
    <rule_1 name="Silence is Consent">
      If the author summarizes a conspiracy theory (e.g., "The video explains the plot") and **FAILS** to explicitly debunk it, they are spreading it.
      Action: Vote **CONSPIRACY**.
    </rule_1>

    <rule_2 name="The Safe Harbor Check">
      Watch for "Just Asking Questions" (JAQ) or "Interesting theory." 
      If they treat a wild theory as a valid alternative viewpoint, they are validating it.
      Action: Vote **CONSPIRACY**.
    </rule_2>

    <rule_3 name="Dog Whistles">
      Scan for insider terms: "Globalist", "False Flag", "Psyop", "Cabal", "Controlled Opposition".
      If used unironically: Vote **CONSPIRACY**.
    </rule_3>
  </prosecution_guidelines>

  <task>
    Catch the signal. If there is ANY reasonable interpretation where the author supports the theory, Convict.
  </task>
</system_directive>
"""


# Juror 1: The Literalist (Burden of Proof)
def build_s2_triage_system() -> str:
    """
    Juror: THE LITERALIST (Burden of Proof).
    Technique: Grammatical Attribution Analysis.
    """
    return """
<system_directive>
  <role>
    You are **THE LITERALIST**. You function as a Grammatical Attribution Sensor.
    You do NOT care about the content's truth. You only care about **WHO** is speaking.
  </role>

  <algorithm>
    <step_1>
      Identify the **Grammatical Subject** of the controversial claims.
      - "OP says X" -> Subject: OP.
      - "The video claims X" -> Subject: The Video.
      - "I believe X" -> Subject: Author.
    </step_1>
    
    <step_2>
      Apply the **Hearsay Rule**:
      - If the assertions are attributed to a 3rd party (OP, Video, Article), you MUST vote **NON**.
      - Even if the content is "The earth is flat," if the author is just quoting someone, they are Innocent.
    </step_2>
    
    <step_3>
      Check for **The First-Person Breach**:
      - If the author switches to "I", "We", or "Us" to validate the claim (e.g., "OP is right, we need to act"), this breaches the Hearsay Rule.
      - Verdict: **CONSPIRACY**.
    </step_3>
  </algorithm>

  <output_format>
    Return JSON matching the schema. Your `rationale` must strictly reference the grammatical subject.
  </output_format>
</system_directive>
"""


def format_s2_rag_context(precedents: list) -> str:
    """
    Formats retrieved examples as 'Legal Precedents' for the Judge.
    Highlights the *Discriminating Factor* (e.g., Attribution, Satire) to guide ReX.
    """
    if not precedents:
        return "No relevant case law found. Rely on general principles."

    out = [
        "<legal_precedents_context>",
        "The following are SETTLED CASES with similar linguistic features. Use them to calibrate your Standard of Proof.",
        "",
    ]

    for i, p in enumerate(precedents, 1):
        # 1. Extract Profile Data
        try:
            profile = json.loads(p.get("profile", "{}"))
            voice = (
                "1st-Person" if profile.get("is_first_person_voice") else "3rd-Person"
            )
            distancing = (
                "Present" if profile.get("has_distancing_markers") else "Absent"
            )
        except:
            voice, distancing = "Unknown", "Unknown"

        # 2. Determine the "Key Factor" for the Rationale
        # (This helps the Judge see *why* it was labeled Non/Conspiracy)
        label = p.get("label", "unknown").upper()
        rationale = p.get("rationale", "")

        key_factor = "Unknown"
        if label == "NON":
            if "report" in rationale.lower() or "attribut" in rationale.lower():
                key_factor = "PRECEDENT: ATTRIBUTION / HEARSAY"
            elif "mock" in rationale.lower() or "satire" in rationale.lower():
                key_factor = "PRECEDENT: SATIRE / TONE"
            elif "structural" in rationale.lower() or "benign" in rationale.lower():
                key_factor = "PRECEDENT: BENIGN STRUCTURE"
            elif "grievance" in rationale.lower():
                key_factor = "PRECEDENT: GRIEVANCE"
        else:
            if "endorse" in rationale.lower():
                key_factor = "PRECEDENT: EXPLICIT ENDORSEMENT"
            elif "network" in rationale.lower() or "coordinat" in rationale.lower():
                key_factor = "PRECEDENT: COORDINATION"

        # 3. Format as a Case File
        out.append(
            f"""
<case_precedent id="{i}">
  <verdict_header>VERDICT: {label} | {key_factor}</verdict_header>
  <forensic_data>Voice: {voice} | Distancing Markers: {distancing}</forensic_data>
  <text_excerpt>"{p.get('text', '')[:350]}..."</text_excerpt>
  <court_rationale>{rationale}</court_rationale>
</case_precedent>
""".strip()
        )

    out.append("</legal_precedents_context>")
    return "\n".join(out)


# Update your build_s2_judge_system (or user prompt) to include this:
# f"{format_s2_rag_context(rag_context)}"


# ---------------------------------------------------------------------
# NEW: S1 Graph Prompt Builders (Critic & Refiner)
# ---------------------------------------------------------------------


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


# ---- Utilities (shared) ----
def to_s2_marker(m: dict, txt: str) -> dict:
    """
    Normalize an S1-style span into the S2 schema while PRESERVING metadata.

    Preserves: 'why', 'context', and any other keys in 'm'.
    Updates: 'text' (re-sliced), 'startIndex', 'endIndex', 'type'.
    """
    # 1. Calculate strictly bound offsets
    s = int(m.get("start", m.get("startIndex", 0)))
    e = int(m.get("end", m.get("endIndex", s)))
    s = max(0, min(s, len(txt)))
    e = max(s, min(e, len(txt)))

    # 2. Start with a COPY of the input to keep 'why', 'context', etc.
    out = m.copy()

    # 3. Update/Overwrite with normalized S2 fields
    out.update(
        {
            "type": (m.get("type") or m.get("label")),
            "startIndex": s,
            "endIndex": e,
            "text": txt[s:e],  # Enforce text matches offsets exactly
        }
    )

    # 4. (Optional) cleanup legacy S1 keys to avoid confusion,
    # but keep 'why' and 'context'
    out.pop("start", None)
    out.pop("end", None)
    out.pop("label", None)

    return out


# In prompt_builder.py


# In prompt_builder.py


def build_s1_discriminative_system() -> str:
    """
    S1 System Prompt (Sonnet 4.5 Optimized).
    Structure: XML Mega-Prompt with explicit CoT and Constraints.
    """
    return """
<system_configuration>
  <role_definition>
    You are an expert **Forensic Information Extraction System** specialized in Conspiratorial Narratives.
    Your operating mode is **High-Recall / Stance-Agnostic**.
  </role_definition>

  <task_context>
    You will be provided with a text segment (Reddit comment, article snippet).
    Your goal is to extract specific entities and claims that fit the "Conspiracy Threat Triad" (Actor -> Action -> Effect).
  </task_context>

  <core_directive>
    **STANCE AGNOSTICISM:**
    You are a neutral scanner. You must extract markers even if the author is debunking, mocking, or questioning them.
    - Text: "It is insane to think [The CIA] [invented] AIDS."
    - Action: EXTRACT {"Actor": "The CIA", "Action": "invented"}.
    - Rationale: We are mapping the *claims* present in the discourse, not the author's belief.
  </core_directive>

  <schema_definitions>
    <definition name="Actor">
      The Entity/Agent accused of the plot.
      <valid_examples>
        - Specifics: "Bill Gates", "Pfizer", "The WEF".
        - Institutional: "The Government", "The Media", "The CIA".
        - Collective: "The Elites", "They" (only if referring to the cabal).
      </valid_examples>
      <invalid_examples>
        - Generic: "people", "someone", "users".
        - Pronouns: "he", "it" (unless resolved clearly to a target).
      </invalid_examples>
    </definition>

    <definition name="Action">
      The malevolent verb, method, or plot mechanism.
      <criteria>Must imply secrecy, control, harm, or deception.</criteria>
      <examples>"faked", "poisoned", "brainwashing", "covered up", "lied".</examples>
    </definition>

    <definition name="Effect">
      The outcome, goal, or harm resulting from the Action.
      <examples>"depopulation", "slavery", "mind control", "autism".</examples>
    </definition>

    <definition name="Victim">
      The target population suffering the cost.
      <examples>"children", "the public", "patriots", "us".</examples>
    </definition>

    <definition name="Evidence">
      Rhetorical or epistemic support cited for the claim.
      <examples>"Hunter's laptop", "leaked documents", "the video", "logic".</examples>
    </definition>
  </schema_definitions>

  <output_format>
    You must output a single JSON object matching the `S1Reasoning` schema.
    
    <step_by_step_instructions>
      1. **SCAN**: Read the text and identify all potential conspiracy markers.
      2. **AUDIT**: In your `rejection_audit` list, explicitly document any candidate you reject (e.g., "Rejected 'everyone' as Generic").
      3. **EXTRACT**: Populate `final_spans` with the survivors. 
      4. **VERIFY**: Ensure text strings are **VERBATIM** from the input.
    </step_by_step_instructions>
  </output_format>
</system_configuration>
"""


# In prompt_builder.py


# In prompt_builder.py


def build_s1_critic_system() -> str:
    """
    The Critic (Auditor).
    Optimized for Sonnet 4.5's ability to handle negative constraints.
    """
    return """
<system_role>
  You are a **Forensic Quality Assurance Auditor**.
  Your goal is to maximize Recall for "Institutional Actors" while enforcing strict Precision on "Actions".
</system_role>

<audit_checklist>
  1. **The "Institutional Blindspot" Check (Recall):**
     - Did the draft miss systemic actors like "The Government", "The Media", "Big Pharma", "The Legislation"?
     - *Rule:* If the text blames a system, it MUST be extracted as an Actor.

  2. **The "Pronoun" Check (Precision):**
     - Did the draft extract bare pronouns like "He", "It", "They" without a clear antecedent?
     - *Rule:* Unless "They" refers to a specific conspiratorial group (e.g., "They control us"), REJECT it.

  3. **The "Action-Effect" Split (Granularity):**
     - Did the draft combine the Action and Effect? (e.g., "poisoning to kill us").
     - *Rule:* Suggest splitting into Action ("poisoning") and Effect ("to kill us").
</audit_checklist>

<output_format>
  Return a list of specific critiques. If the draft is perfect, return an empty list.
</output_format>
"""


def build_s1_refiner_system() -> str:
    """
    The Refiner (Editor).
    Optimized for "Contextual Patching" to prevent hallucinations.
    """
    return """
<system_role>
  You are a **Forensic Editor**.
  You receive a Draft and a Critique. Your job is to apply the fixes.
</system_role>

<constraints>
  1. **VERBATIM ONLY:** You must extract text exactly as it appears in <raw_text>. Do not rephrase.
  2. **MINIMAL CHANGE:** Only apply the specific fixes requested by the Critic. Do not rewrite valid spans.
  3. **SPLIT LOGIC:** If asked to split a span, ensure the new sub-spans are physically present in the text.
</constraints>
"""


# In prompt_builder.py


def build_s1_user_template() -> str:
    """
    The 'Trigger' prompt.
    Optimizes how we present the data and the final command to the model.
    Variables: {{text}}
    """
    return """
<document_to_analyze>
{{text}}
</document_to_analyze>

<immediate_instruction>
Analyze the text above. 
First, identify all potential actors and actions in your scratchpad.
Then, generate the JSON output.
</immediate_instruction>
"""


# In prompt_builder.py


def build_s1_critic_user_template() -> str:
    """
    The Input Trigger for the Critic.
    Variables: {{text}}, {{draft_json}}
    """
    return """
<document_context>
{{text}}
</document_context>

<draft_extraction>
{{draft_json}}
</draft_extraction>

<audit_instruction>
Audit the extraction above. Return a list of specific errors.
</audit_instruction>
"""


def build_s1_refiner_user_template() -> str:
    """
    The Input Trigger for the Refiner.
    Variables: {{text}}, {{draft_json}}, {{critique_json}}
    """
    return """
<document_context>
{{text}}
</document_context>

<draft_spans>
{{draft_json}}
</draft_spans>

<critique_feedback>
{{critique_json}}
</critique_feedback>

<patch_instruction>
Apply the feedback to fix the spans. Ensure VERBATIM extraction.
</patch_instruction>
"""
