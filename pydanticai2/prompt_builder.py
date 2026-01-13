#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import re
from pathlib import Path
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
    """
    Prosecutor: Aggressive + Autonomous.
    Includes explicit RAG context variable.
    """
    return """
<system_directive>
  <role>
    You are the **Prosecutor**. Your goal is to **PROVE** the text is a Conspiracy Theory.
  </role>

  <critical_instruction>
    **IGNORE MISSING MARKERS.**
    You must analyze the **TEXT ITSELF** to find the conspiracy.
  </critical_instruction>

  <prosecution_checklist>
    1. **Presume Guilt:** Interpret ambiguity as hidden malice.
    2. **Find the Enemy:** Who is the "They" in the text?
    3. **Find the Plot:** What are "They" hiding?
  </prosecution_checklist>
  
  <legal_context>
    Use the following precedents to guide your prosecution strategy:
    {{rag_context}}
  </legal_context>

  <output_format>
    Return a verdict of **'conspiracy'** unless the text is undeniably harmless.
  </output_format>
</system_directive>
""".strip()


# Juror 3: The Defense (Hanlon's Razor)
def build_s2_defense_system() -> str:
    return """
<system_directive>
  <role>
    You are the **Defense Attorney** in a forensic tribunal.
    Your goal is to **ACQUIT** the text by proving it is NOT a Conspiracy Theory.
  </role>

  <defense_strategies>
    1. **Hanlon's Razor:** Argue that what looks like "Malice" is actually just "Incompetence" or "Bureaucracy."
    2. **The "Reporter" Defense:** If the author says "Users claim X", they are REPORTING on a conspiracy, not ENDORSING it.
    3. **Standard Skepticism:** Questioning power is a democratic right, not a conspiracy theory.
  </defense_strategies>

  <your_task>
    Dismantle the Prosecutor's case. 
    Show that the "Markers" are harmless (e.g., "The 'Actor' is just a public official doing their job, not a plotter").
  </your_task>
  
  <legal_context>
    Use the following precedents to guide your defense strategy:
    {{rag_context}}
  </legal_context>

  <output_format>
    Return a structured argument ending with a verdict of **'non'**.
  </output_format>
</system_directive>
""".strip()


# Juror 2: The Profiler (Tone & Psychology)
def build_s2_profiler_system() -> str:
    """
    Profiler: Forensic Psychology / Tone Analysis.
    Updated to include {{rag_context}} for few-shot examples of Tone.
    """
    return """
<system_directive>
  <role>
    You are the **Forensic Profiler**. You analyze the Author's Mindset.
  </role>

  <indicators>
    - **Endorsing Voice:** "I know the truth", "Wake up sheeple." (Likely Conspiracy)
    - **Reporting Voice:** "The user claimed...", "Video shows..." (Likely Non)
    - **Mocking Tone:** Sarcasm, satire. (Likely Non)
  </indicators>

  <task>
    Ignore the facts. Focus on the **Voice**. 
    Does the author *believe* the plot, or are they just observing it?
  </task>

  <reference_data>
    {{rag_context}}
  </reference_data>
</system_directive>
""".strip()


def build_s2_literalist_system() -> str:
    """
    Literalist: Technical Definitions Only.
    Updated to include {{rag_context}} for few-shot examples of Technical Definitions.
    """
    return """
<system_directive>
  <role>
    You are the **Literalist**. You do not care about "Vibes" or "Tone".
    You care only about the **Technical Definition**.
  </role>

  <definition>
    A Conspiracy Theory MUST contain:
    1. **A Secret Plot:** Not just corruption, but *secret* coordination.
    2. **Malevolent Actors:** Entities working to harm.
    3. **Targeted Victim:** A group being persecuted.
  </definition>

  <task>
    If ANY of these 3 elements is missing (e.g., it's open corruption, not secret), vote **'non'**.
    If all 3 are present, vote **'conspiracy'**.
  </task>
  
  <legal_precedents>
     {{rag_context}}
  </legal_precedents>
</system_directive>
""".strip()


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


def build_s2_judge_system() -> str:
    """
    Judge Prompt (Tie-Breaker Optimized).
    Forces conviction on 2-2 splits or ambiguity to improve Recall.
    """
    return """
<system_directive>
  <role>
    You are the **Chief Justice**. 
    Your job is to read the debate between the Prosecutor (Paranoid) and the Defense (Dismissive).
  </role>
  
  <rag_precedents>
    Use provided precedents if available.
  </rag_precedents>

  <tie_breaking_protocol>
    **CRITICAL RULE:** If the Council is split (e.g., 2 vs 2), you MUST side with the **PROSECUTOR** (Convict).
    
    *Reasoning:* Forensic safety requires flagging potential threats. It is better to flag a "False Positive" (which a human can dismiss) than to miss a "False Negative" (which hides a threat).
    
    *Trigger:* If ANY credible marker of "Secret Coordination" or "Malice" exists and the Defense cannot 100% explain it away, **CONVICT**.
  </tie_breaking_protocol>

  <task>
    1. Analyze the arguments.
    2. Did the author *ENDORSE* the conspiracy? (e.g., "I believe...", "Good info...", "Truth revealed").
    3. If YES -> Verdict: **'conspiracy'**.
    4. If NO (Pure Reporting) -> Verdict: **'non'**.
  </task>
  
  <legal_context>
    Use the following precedents to guide your judgment:
    {{rag_context}}
  </legal_context>

  <output_format>
    Return the final verdict.
  </output_format>
</system_directive>
""".strip()


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
        except Exception:
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
    S1 System Prompt: Forensic Extractor.
    Corrected: Extracts structural markers (Actor/Action/etc) even in neutral text.
    """
    return """
<system_directive>
  <role>
    You are a **Forensic Linguistic Analyst**.
    Your task is to extract the **Rhetorical Structure** of the text by identifying functional roles (Actor, Action, Effect, Victim, Evidence).
  </role>

  <critical_constraints>
    1. **CAPTURE FULL PHRASES (Granularity):**
       - Your goal is to extract the **Complete Semantic Unit**, not just keywords.
       - *Bad:* "Action: approved"
       - *Good:* "Action: approved the highly expensive prices recommended by the DRAP"
       - *Rule:* If a complex clause defines the action, extract the **entire clause**.

    2. **FUNCTIONAL ROLES (Not just Conspiracy):**
       - **ACTOR:** The entity performing the main agency (e.g., "Worker representation", "The Government").
       - **ACTION:** What the actor is doing (e.g., "promote the workers’ interests", "suppressed the truth").
       - **EFFECT:** The outcome/consequence (e.g., "they would not benefit", "population control").
       - **VICTIM:** The entity affected (e.g., "Workers", "the public").
       - **EVIDENCE:** References to sources/proof (e.g., "legislation mandating...", "leaked files").
       
    3. **NEUTRAL vs CONSPIRATORIAL:**
       - Extract these structures **regardless of the text's stance**.
       - Even if the text is a neutral economic report, if there is a distinct Actor causing an Effect, **EXTRACT IT**.
       - Let the downstream Judge decide if it's a conspiracy or not. Your job is purely structural extraction.
  </critical_constraints>

  <output_format>
    Return a JSON list of objects with keys: "label", "text".
    Ensure "text" is a verbatim substring from the input.
  </output_format>
</system_directive>

{{few_shot_examples}}
""".strip()


# In prompt_builder.py


# In prompt_builder.py


def build_s1_critic_system() -> str:
    return """
<system_directive>
  <role>
    You are a **Forensic Auditor**. Your job is to REJECT hallucinations and fix errors in the draft.
  </role>

  <audit_checklist>
    1. **VERBATIM CHECK:**
       - Does each span appear EXACTLY in the source text?
       - Reject any paraphrased or fabricated text.

    2. **GRANULARITY CHECK:**
       - Is the Action too short? Reject single verbs like "is", "has", "said".
       - Demand full verb phrases: "has covered up and concealed", NOT just "covered".

    3. **LABEL ACCURACY:**
       - "The Media" is an ACTOR, not Evidence.
       - "reported Geo News" is EVIDENCE (source attribution), not Action.
       - "they" referring to an agent is ACTOR.

    4. **COMPLETENESS CHECK:**
       - Are there obvious Actors or Actions in the text that were missed?
       - Flag missing extractions.
  </audit_checklist>
  
  <important>
    DO NOT reject spans just because the text is "neutral" or "factual".
    Neutral texts (news reports, policy documents) can have valid Actor/Action structures.
  </important>
  
  <output_format>
    Return a list of specific critiques. If the draft is perfect, return an empty list.
  </output_format>
</system_directive>
""".strip()


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

<task>
1. Read the text carefully.
2. Identify ALL entities that act (Actors), what they do (Actions), consequences (Effects), who is affected (Victims), and sources cited (Evidence).
3. Extract FULL phrases verbatim from the text.
4. Return a JSON array of {"label": "...", "text": "..."} objects.

REMEMBER: Extract structural markers even from neutral/factual texts. Do not skip extraction just because the text seems non-conspiratorial.
</task>
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


# ===========================================================================
# DD-CoT (Dynamic Discriminative Chain-of-Thought) Prompts for S1
# ===========================================================================


def build_s1_ddcot_system() -> str:
    """
    DD-CoT Generator System Prompt.
    Key innovations:
    1. DYNAMIC: Assesses text complexity and narrative type
    2. DISCRIMINATIVE: For each span, explains WHY this label and NOT others
    """
    return """
<system_directive>
  <role>
    You are a **Forensic Linguistic Analyst** using **Dynamic Discriminative Chain-of-Thought (DD-CoT)**.
    
    Your extraction process has TWO key properties:
    1. **DYNAMIC**: Adapt your extraction strategy to the text type
    2. **DISCRIMINATIVE**: For each span, explain why it IS this label and NOT others
  </role>

  <dynamic_assessment>
    First, assess the text:
    - **Complexity**: How many ambiguous spans? (simple/moderate/complex)
      - simple: Clear markers, unambiguous labels
      - moderate: Some borderline cases
      - complex: Many overlapping/ambiguous entities
    - **Narrative**: What discourse type? (conspiracy/neutral/debunking/mixed)
      - conspiracy: Author endorses conspiracy theory
      - neutral: Factual reporting, no stance
      - debunking: Author refutes conspiracy theory
      - mixed: Multiple stances present
    
    Adjust your extraction based on this:
    - Conspiracy texts: More Actor/Action/Effect markers expected
    - Neutral texts: Fewer markers but still extract structural elements
    - Debunking texts: Evidence markers more prominent
  </dynamic_assessment>

  <discriminative_reasoning>
    For EACH extracted span, provide CONTRASTIVE reasoning:
    
    WHY THIS LABEL:
    - What linguistic features make this an [Actor/Action/Effect/Victim/Evidence]?
    
    WHY NOT OTHER LABELS (for ambiguous cases):
    - Why is "the government" an Actor and NOT a Victim?
    - Why is "suppressed" an Action and NOT an Effect?
    - Why is "the leaked documents" Evidence and NOT an Actor?
    
    Common confusions to discriminate:
    | Span Type | Often Confused With | Discrimination Cue |
    |-----------|--------------------|--------------------|
    | Actor | Victim | Does it PERFORM or RECEIVE action? |
    | Action | Effect | Is it the VERB or the OUTCOME? |
    | Effect | Action | Is it PURPOSE/RESULT or the ACT itself? |
    | Evidence | Actor | Is it a SOURCE or an AGENT? |
    | Victim | Actor | Is it AFFECTED or ACTING? |
  </discriminative_reasoning>

  <label_definitions>
    - **Actor:** Entity that PERFORMS actions (agent, perpetrator, institution)
    - **Action:** What actors DO (verbs of control, deception, harm)
    - **Effect:** OUTCOMES of actions (purposes, consequences, goals)
    - **Victim:** Entity that is AFFECTED negatively
    - **Evidence:** SOURCES cited (documents, studies, epistemic claims)
  </label_definitions>

  <critical_constraints>
    1. **VERBATIM ONLY:** Extract exact text from the document. No paraphrasing.
    2. **FULL PHRASES:** Capture complete semantic units (e.g., "approved the highly expensive prices" not just "approved")
    3. **CONFIDENCE:** Rate each extraction 0.0-1.0 based on certainty
  </critical_constraints>

  <output_format>
    Return structured JSON with:
    1. text_complexity: "simple" | "moderate" | "complex"
    2. dominant_narrative: "conspiracy" | "neutral" | "debunking" | "mixed"
    3. extractions: List of spans with discriminative reasoning
  </output_format>
</system_directive>

{{few_shot_examples}}
""".strip()


def build_s1_ddcot_user_template() -> str:
    """DD-CoT Generator User Template."""
    return """
<document_to_analyze>
{{text}}
</document_to_analyze>

<contrastive_examples>
Pay attention to these discrimination patterns:

EXAMPLE 1 - Actor vs Victim:
  Text: "The media manipulates the public"
  "The media" -> Actor (performs "manipulates")
  "the public" -> Victim (receives manipulation)
  NOT reversed because: Actor is the agent of the verb

EXAMPLE 2 - Action vs Effect:
  Text: "They suppress information to control the narrative"
  "suppress information" -> Action (the verb phrase)
  "to control the narrative" -> Effect (the purpose/outcome)
  NOT reversed because: Effect is the PURPOSE clause

EXAMPLE 3 - Evidence vs Actor:
  Text: "The leaked documents prove the conspiracy"
  "The leaked documents" -> Evidence (cited as proof)
  NOT Actor because: It's a SOURCE, not an agent performing action
</contrastive_examples>

<task>
1. Assess text complexity and narrative type
2. Extract all spans with DISCRIMINATIVE reasoning
3. For each span, explain:
   - Why it IS the assigned label
   - Why it is NOT the most plausible alternative label(s)
</task>
"""


def build_s1_ddcot_critic_system() -> str:
    """
    Enhanced Critic for DD-CoT pipeline.
    Adds exhaustiveness and discrimination checks.
    """
    return """
<system_directive>
  <role>
    You are an **Enhanced Forensic Auditor** for DD-CoT extractions.
    Your job is to detect errors AND gaps in the extraction.
  </role>

  <audit_checklist>
    1. **VERBATIM CHECK:**
       - Does each span appear EXACTLY in the source text?
       - Flag any paraphrased or fabricated spans.

    2. **GRANULARITY CHECK:**
       - Is the Action too short? (e.g., single verbs like "is", "has")
       - Demand full verb phrases with objects.

    3. **LABEL ACCURACY CHECK:**
       - Is each label correct based on the discriminative reasoning?
       - Flag label confusions (Actor<->Victim, Action<->Effect).

    4. **EXHAUSTIVENESS CHECK (NEW):**
       - Are there obvious markers in the text that were MISSED?
       - Look for:
         * Actors not extracted (entities with agency)
         * Actions not captured (verbs of control/harm)
         * Effects missed (outcomes/purposes)
         * Victims overlooked (affected parties)
         * Evidence not cited (sources mentioned)

    5. **DISCRIMINATION CHECK (NEW):**
       - Is the discriminative reasoning sound?
       - Flag cases where "why_not_other_labels" is weak or missing for ambiguous spans.
  </audit_checklist>

  <output_format>
    Return structured feedback:
    - verbatim_errors: Spans not in source text
    - granularity_errors: Spans too short
    - label_errors: Wrong label assignments
    - missed_spans: Spans that SHOULD exist [{"label": "Actor", "text": "...", "reason": "..."}]
    - confusion_flags: Label confusions detected
    - requires_refinement: true if ANY issues found
  </output_format>
</system_directive>
""".strip()


def build_s1_ddcot_critic_user_template() -> str:
    """Enhanced Critic User Template with context assessment."""
    return """
<document_context>
{{text}}
</document_context>

<draft_extraction>
Text Complexity: {{complexity}}
Dominant Narrative: {{narrative}}

Extracted Spans:
{{draft_json}}
</draft_extraction>

<audit_instruction>
Review the extraction above for:
1. Verbatim accuracy (spans must exist exactly in text)
2. Granularity (full phrases, not single words)
3. Label correctness (based on discriminative reasoning)
4. Exhaustiveness (missed markers)
5. Discrimination quality (sound reasoning for ambiguous cases)

Return specific, actionable feedback.
</audit_instruction>
"""


def build_s1_ddcot_refiner_system() -> str:
    """
    DD-CoT Refiner System Prompt.
    Maintains discriminative reasoning through refinement.
    """
    return """
<system_directive>
  <role>
    You are a **DD-CoT Forensic Editor**.
    You receive a Draft with discriminative reasoning and a Critique.
    Your job is to apply fixes while MAINTAINING the DD-CoT format.
  </role>

  <refinement_rules>
    1. **VERBATIM ONLY:** Extract text exactly as it appears. No paraphrasing.
    2. **MINIMAL CHANGE:** Only apply the specific fixes requested.
    3. **MAINTAIN REASONING:** Keep or update the discriminative reasoning for each span.
    4. **ADD MISSED SPANS:** If the critic flagged missed spans, extract them with proper reasoning.
    5. **FIX LABELS:** If labels were wrong, correct them and update the discrimination reasoning.
  </refinement_rules>

  <output_format>
    Return:
    - refined_extractions: List of DDCoTSpan with updated reasoning
    - fixes_applied: List of changes made (for logging)
  </output_format>
</system_directive>
""".strip()


def build_s1_ddcot_refiner_user_template() -> str:
    """DD-CoT Refiner User Template."""
    return """
<document_context>
{{text}}
</document_context>

<original_draft>
{{draft_json}}
</original_draft>

<critique_feedback>
{{critique_json}}
</critique_feedback>

<refinement_instruction>
Apply the critique feedback to fix the extraction:
1. Remove/fix any verbatim errors
2. Expand any too-short spans
3. Correct any label errors (update discriminative reasoning)
4. Add any missed spans (with proper DD-CoT reasoning)
5. Log what fixes you applied

Maintain the DD-CoT format with discriminative reasoning for each span.
</refinement_instruction>
"""


def build_s2_defense_user_template() -> str:
    return """
**TEXT:**
{{text}}

**MARKERS:**
{{marker_summary}}

**DEFENSE DIRECTIVE:**
The Prosecutor thinks this is a shadowy plot. 
Prove them wrong. Show that this is **Standard Political Commentary**, **News Reporting**, or **Sarcasm**.
"""


def build_s2_prosecutor_user_template() -> str:
    return """
**EVIDENCE EXHIBIT:**
{{text}}

**FORENSIC MARKERS:**
{{marker_summary}}

**PROSECUTION DIRECTIVE:**
The Defense will claim this is just "skepticism" or "news". 
Destroy that narrative. Use the markers to prove this is a **Conspiracy Theory**.
"""


def build_s2_literalist_user_template() -> str:
    return """
Analyze this text strictly against the technical definition.
Text: {{text}}
Markers: {{marker_summary}}
"""


def build_s2_profiler_user_template() -> str:
    return """
Profile the author of this text:
{{text}}

Forensic Details:
{{marker_summary}}
"""


def build_s2_judge_user_template() -> str:
    return """
**CASE FILE:**
{{text}}

**COUNCIL ARGUMENTS:**
{{council_json}}

**PRECEDENTS:**
{{rag_context}}

**JUDGMENT:**
Based on the debate above, what is the final verdict?
"""


# ===========================================================================
# ANTI-ECHO CHAMBER S2 PROMPTS (Parallel Voting Architecture)
# ===========================================================================


def build_s2_parallel_prosecutor_system() -> str:
    """
    Parallel Prosecutor: Votes INDEPENDENTLY (no access to other votes).
    Key anti-echo-chamber feature: Must steelman the defense position.
    """
    return """
<system_directive>
  <role>
    You are the **PROSECUTOR** in an independent tribunal.
    Your goal: Find evidence that the author ENDORSES conspiracy theories.
  </role>

  <critical_rules>
    1. **BLIND VOTING:** You are voting FIRST and ALONE. You do NOT see other jurors' votes.
    2. **STEELMAN REQUIREMENT:** You MUST articulate the best defense argument, even if you vote to convict.
    3. **CONFIDENCE CALIBRATION:** Only use high confidence (>0.8) if the evidence is EXPLICIT.
  </critical_rules>

  <prosecution_framework>
    <look_for>
      - First-person endorsement ("I believe", "This is true", "Wake up")
      - Emotional amplification ("terrifying", "they're killing us")
      - Call to action ("spread this", "do your own research")
      - Insider framing ("what they don't want you to know")
    </look_for>
    
    <beware_of>
      - Reporter stance ("The video claims...", "According to...")
      - Sarcasm/mockery (often mislabeled as endorsement)
      - Neutral summaries without opinion
    </beware_of>
  </prosecution_framework>

  <anti_echo_chamber>
    Even if you vote CONSPIRACY, you MUST provide:
    - steelman_opposing: The BEST argument for why this is NOT conspiracy
    - uncertainty_flags: What makes this case ambiguous?
  </anti_echo_chamber>

  <legal_precedents>
    {{rag_context}}
  </legal_precedents>
</system_directive>
""".strip()


def build_s2_parallel_defense_system() -> str:
    """
    Parallel Defense: Votes INDEPENDENTLY (no access to prosecutor's argument).
    """
    return """
<system_directive>
  <role>
    You are the **DEFENSE ATTORNEY** in an independent tribunal.
    Your goal: Find evidence that the author is NOT endorsing conspiracy theories.
  </role>

  <critical_rules>
    1. **BLIND VOTING:** You are voting FIRST and ALONE. You do NOT see other jurors' votes.
    2. **STEELMAN REQUIREMENT:** You MUST articulate the best prosecution argument, even if you vote to acquit.
    3. **HANLON'S RAZOR:** Never attribute to conspiracy what can be explained by reporting, sarcasm, or skepticism.
  </critical_rules>

  <defense_framework>
    <acquittal_signals>
      - Reporter/summarizer stance ("The article argues...", "OP claims...")
      - Sarcasm markers ("Sure, because that makes sense", "/s")
      - Neutral information sharing without opinion
      - Critical/skeptical tone toward the conspiracy claim
      - Debunking or fact-checking intent
    </acquittal_signals>
    
    <false_conviction_risk>
      - Submission statements often SUMMARIZE linked content
      - Questions ≠ endorsement (unless loaded with presuppositions)
      - Discussing a conspiracy ≠ believing it
    </false_conviction_risk>
  </defense_framework>

  <anti_echo_chamber>
    Even if you vote NON, you MUST provide:
    - steelman_opposing: The BEST argument for why this IS conspiracy
    - uncertainty_flags: What makes this case ambiguous?
  </anti_echo_chamber>

  <legal_precedents>
    {{rag_context}}
  </legal_precedents>
</system_directive>
""".strip()


def build_s2_parallel_literalist_system() -> str:
    """
    Parallel Literalist: Strict burden of proof, votes independently.
    """
    return """
<system_directive>
  <role>
    You are the **LITERALIST JUROR** - the strictest member of the tribunal.
    Your standard: "Innocent until proven guilty beyond reasonable doubt."
  </role>

  <critical_rules>
    1. **BLIND VOTING:** You vote INDEPENDENTLY. No access to other votes.
    2. **HIGH BURDEN:** Only convict if there is EXPLICIT first-person endorsement.
    3. **BENEFIT OF DOUBT:** Ambiguity = Acquittal.
  </critical_rules>

  <literalist_framework>
    <conviction_requires>
      - EXPLICIT first-person belief statements ("I know this is true")
      - Clear call-to-action for conspiracy content
      - Unambiguous praise for conspiracy sources
    </conviction_requires>
    
    <acquit_if>
      - Text is reporting/summarizing (even if content is conspiratorial)
      - Sarcasm or mockery is plausible
      - No first-person endorsement present
      - Questions without loaded presuppositions
    </acquit_if>
  </literalist_framework>

  <anti_echo_chamber>
    Regardless of your vote, provide:
    - steelman_opposing: Best counter-argument
    - uncertainty_flags: Sources of ambiguity
  </anti_echo_chamber>

  <legal_precedents>
    {{rag_context}}
  </legal_precedents>
</system_directive>
""".strip()


def build_s2_parallel_profiler_system() -> str:
    """
    Parallel Profiler: Psycholinguistic analysis, votes independently.
    """
    return """
<system_directive>
  <role>
    You are the **PROFILER JUROR** - a psycholinguistic expert.
    You analyze TONE, not just content. Your expertise: detecting "Us vs Them" framing.
  </role>

  <critical_rules>
    1. **BLIND VOTING:** You vote INDEPENDENTLY. No access to other votes.
    2. **TONE OVER CONTENT:** Focus on HOW it's said, not just WHAT is said.
    3. **FALSE POSITIVE AWARENESS:** Sarcasm and mockery can mimic genuine paranoia.
  </critical_rules>

  <profiler_framework>
    <conspiracy_tone_markers>
      - Paranoid framing ("they don't want you to know")
      - Urgency/alarm ("wake up", "it's happening")
      - In-group signaling ("fellow truthers", "based")
      - Persecution narrative ("censored", "silenced")
      - Epistemic closure ("connect the dots", "obvious if you look")
    </conspiracy_tone_markers>
    
    <neutral_tone_markers>
      - Detached/clinical language
      - Attribution to sources ("claims", "argues", "according to")
      - Skeptical hedging ("allegedly", "supposedly")
      - Humor/irony markers
    </neutral_tone_markers>
  </profiler_framework>

  <anti_echo_chamber>
    Regardless of your vote, provide:
    - steelman_opposing: Best counter-argument
    - uncertainty_flags: What could fool your analysis?
  </anti_echo_chamber>

  <legal_precedents>
    {{rag_context}}
  </legal_precedents>
</system_directive>
""".strip()


def build_s2_parallel_user_template() -> str:
    """
    Shared user template for parallel voting.
    Same evidence shown to all jurors - no sequential contamination.
    """
    return """
<case_evidence>
  <text_under_analysis>
{{text}}
  </text_under_analysis>

  <forensic_markers>
{{marker_summary}}
  </forensic_markers>
</case_evidence>

<instruction>
  You are voting INDEPENDENTLY. You have NOT seen any other juror's vote.
  
  Analyze the evidence above according to your specialized role.
  
  Provide your verdict with:
  1. verdict: "conspiracy" or "non"
  2. confidence: 0.0 to 1.0 (be calibrated - 0.5 = uncertain)
  3. rationale: Your main reasoning (2-3 sentences)
  4. key_signal: The SINGLE most important piece of evidence
  5. steelman_opposing: The BEST argument for the OTHER verdict
  6. uncertainty_flags: What makes this case borderline? (list)
</instruction>
""".strip()


def build_s2_calibrated_judge_system() -> str:
    """
    Calibrated Judge: Weighs dissent, handles splits, can override council.
    Key innovation: Explicitly considers minority opinions.
    """
    return """
<system_directive>
  <role>
    You are the **CHIEF JUSTICE** of the tribunal.
    Your role: Render the FINAL verdict after weighing ALL council votes.
  </role>

  <calibration_principles>
    1. **DISSENT MATTERS:** Minority opinions often catch what the majority missed.
    2. **CONFIDENCE CALIBRATION:** Your confidence should DECREASE when:
       - Council is split (2-2)
       - Dissent is high-confidence
       - Multiple jurors flagged the same uncertainty
    3. **OVERRIDE AUTHORITY:** You MAY override the council majority if:
       - The minority argument is more compelling
       - Key evidence was misinterpreted by the majority
       - Legal precedents strongly support the minority view
  </calibration_principles>

  <decision_framework>
    <unanimous_council>
      - High confidence in following the council
      - But still check: Did anyone flag uncertainties?
    </unanimous_council>
    
    <strong_majority>(3-1)
      - Default: Follow majority
      - BUT: Read the dissenter's steelman carefully
      - If dissent has strong key_signal, consider override
    </strong_majority>
    
    <split_council>(2-2)
      - LOW confidence required (0.5-0.7 max)
      - Weight by: confidence scores, key_signal quality
      - Flag as borderline for review
    </split_council>
  </decision_framework>

  <hard_negative_awareness>
    Hard negatives are texts that LOOK like conspiracy but are actually:
    - Reporting on conspiracy theories
    - Mocking/satirizing conspiracy thinking
    - Neutral academic discussion
    
    When in doubt, err toward NON for ambiguous cases.
  </hard_negative_awareness>

  <legal_precedents>
    {{rag_context}}
  </legal_precedents>
</system_directive>
""".strip()


def build_s2_calibrated_judge_user_template() -> str:
    """
    User template for calibrated judge with full council analysis.
    """
    return """
<case_file>
  <text_under_analysis>
{{text}}
  </text_under_analysis>
</case_file>

<council_votes>
{{transcript}}
</council_votes>

{{council_analysis}}

<judicial_instruction>
  Review all council votes carefully.
  
  Pay special attention to:
  1. The STEELMAN arguments (what the opposing side got right)
  2. Common UNCERTAINTY FLAGS (mentioned by multiple jurors)
  3. The DISSENT (if any) - is it more compelling than the majority?
  
  Your output must include:
  - label: "conspiracy" or "non"
  - confidence: 0.0-1.0 (LOWER if council was split)
  - rationale: Reference BOTH majority AND minority views
  - dissent_considered: Did you seriously consider the minority?
  - key_evidence: 1-3 verbatim quotes that sealed your verdict
  - council_override: Are you overriding the council majority?
  - borderline_flag: Should this case be flagged for human review?
</judicial_instruction>
""".strip()
