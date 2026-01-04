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
