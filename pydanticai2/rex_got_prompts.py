#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rex_got_prompts.py — ReX-GoT (Reverse Exclusion Graph of Thought) Prompt Templates

Implements the proper ReX-GoT architecture for S2 classification:
- Graph of Thought: Parallel analysis paths with explicit dependencies
- Reverse Exclusion: "Innocent until proven guilty" with enumerated exclusion criteria

Author: PsyCoMark Team
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ===========================================================================
# ReX-GoT Schemas
# ===========================================================================


class ExclusionNode(BaseModel):
    """A single node in the exclusion graph."""
    criterion: str = Field(description="The exclusion criterion being evaluated")
    triggered: bool = Field(description="True if this exclusion applies (text is NON)")
    evidence: str = Field(description="Specific text evidence supporting this evaluation")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this evaluation")


class ReXGoTAnalysis(BaseModel):
    """
    Complete ReX-GoT analysis output.
    The text is classified as 'non' if ANY exclusion node triggers with high confidence.
    """
    # Graph Node 1: Attribution Analysis
    attribution_exclusion: ExclusionNode = Field(
        description="Does the author attribute claims to a third party? (If yes → NON)"
    )
    
    # Graph Node 2: Tone Analysis  
    tone_exclusion: ExclusionNode = Field(
        description="Is the author mocking/satirizing the conspiracy? (If yes → NON)"
    )
    
    # Graph Node 3: Structural Completeness
    structure_exclusion: ExclusionNode = Field(
        description="Is the conspiracy structure incomplete (missing Actor+Action+Victim)? (If yes → NON)"
    )
    
    # Graph Node 4: Explicit Debunking
    debunking_exclusion: ExclusionNode = Field(
        description="Does the author explicitly debunk or reject the theory? (If yes → NON)"
    )
    
    # Synthesis
    any_exclusion_triggered: bool = Field(
        description="True if ANY exclusion criterion was triggered with confidence > 0.7"
    )
    
    final_verdict: Literal["conspiracy", "non"] = Field(
        description="'non' if any_exclusion_triggered, else 'conspiracy'"
    )
    
    reasoning_chain: str = Field(
        description="Brief explanation of the decision path through the graph"
    )


# ===========================================================================
# ReX-GoT System Prompts
# ===========================================================================


def build_rex_got_system() -> str:
    """
    ReX-GoT (Reverse Exclusion Graph of Thought) System Prompt.
    
    This implements proper ReX-GoT: 
    - Enumerate all exclusion criteria as graph nodes
    - Evaluate each independently
    - Synthesize: ANY trigger → acquit (non)
    """
    return """
<system_directive>
  <role>
    You are a **Forensic Analyst** using the **ReX-GoT** (Reverse Exclusion Graph of Thought) framework.
    Your job is to determine if the text ENDORSES a conspiracy theory.
  </role>

  <rex_got_framework>
    **PRINCIPLE:** "Innocent until proven guilty."
    
    **METHOD:** Evaluate the text against a GRAPH of EXCLUSION CRITERIA.
    If **ANY** exclusion criterion triggers, the verdict is **'non'** (not a conspiracy endorsement).
    
    The text is only labeled **'conspiracy'** if ALL exclusion paths FAIL.
  </rex_got_framework>

  <exclusion_graph>
    Evaluate these 4 independent nodes:
    
    **NODE 1: ATTRIBUTION EXCLUSION**
    - Question: Does the author attribute the claims to a third party?
    - Trigger Phrases: "OP claims...", "The video shows...", "According to...", "Users say..."
    - If triggered → Verdict: NON (the author is reporting, not endorsing)
    
    **NODE 2: TONE EXCLUSION**
    - Question: Is the author's tone mocking, sarcastic, or satirical?
    - Trigger Phrases: Exaggerated language, quotation marks for emphasis ("truth"), eye-roll indicators
    - If triggered → Verdict: NON (the author is ridiculing the theory)
    
    **NODE 3: STRUCTURAL EXCLUSION**
    - Question: Is the conspiracy structure incomplete?
    - Required Elements: (1) Secret Plot, (2) Malevolent Actor, (3) Targeted Victim
    - If ANY element is missing → Verdict: NON (not a complete conspiracy narrative)
    
    **NODE 4: DEBUNKING EXCLUSION**
    - Question: Does the author explicitly reject or debunk the theory?
    - Trigger Phrases: "This is false", "Debunked", "There's no evidence", "Conspiracy theorists claim..."
    - If triggered → Verdict: NON (the author opposes the theory)
  </exclusion_graph>

  <decision_logic>
    ```
    IF (attribution_exclusion.triggered AND confidence > 0.7) → RETURN 'non'
    ELIF (tone_exclusion.triggered AND confidence > 0.7) → RETURN 'non'
    ELIF (structure_exclusion.triggered AND confidence > 0.7) → RETURN 'non'
    ELIF (debunking_exclusion.triggered AND confidence > 0.7) → RETURN 'non'
    ELSE → RETURN 'conspiracy'
    ```
  </decision_logic>

  <output_format>
    Return a structured analysis evaluating each graph node independently, then synthesize.
  </output_format>
</system_directive>
""".strip()


def build_rex_got_user_template() -> str:
    """
    ReX-GoT User Prompt Template.
    Variables: {{text}}, {{marker_summary}}, {{rag_context}}
    """
    return """
<case_file>
  <evidence_text>
{{text}}
  </evidence_text>

  <forensic_markers>
{{marker_summary}}
  </forensic_markers>

  <legal_precedents>
{{rag_context}}
  </legal_precedents>
</case_file>

<analysis_instruction>
Apply the ReX-GoT framework:
1. Evaluate each EXCLUSION NODE independently
2. Cite specific text evidence for each evaluation
3. Synthesize: If ANY exclusion triggers → verdict is 'non'
4. Only convict ('conspiracy') if ALL exclusions fail
</analysis_instruction>
"""


# ===========================================================================
# Enhanced S1: Reverse Extraction Prompts
# ===========================================================================


class S1CandidateSpan(BaseModel):
    """A candidate span before filtering."""
    label: Literal["Actor", "Action", "Effect", "Victim", "Evidence"]
    text: str = Field(description="Verbatim text from the document")
    keep: bool = Field(description="True if this span should be kept, False if excluded")
    exclusion_reason: Optional[str] = Field(
        default=None, 
        description="If keep=False, why was this span excluded?"
    )


class S1ReverseExtraction(BaseModel):
    """
    Reverse Extraction schema: Over-extract then prune.
    This prevents the "miss" problem by ensuring exhaustive initial extraction.
    """
    # Step 1: Exhaustive extraction (over-extract)
    all_candidates: List[S1CandidateSpan] = Field(
        description="ALL potential markers found in text. Include borderline cases."
    )
    
    # Step 2: Pruning rationale
    exclusion_summary: str = Field(
        description="Brief summary of why excluded spans were removed (e.g., 'Removed pronouns without referents')"
    )
    
    # Step 3: Final output
    final_spans: List[S1CandidateSpan] = Field(
        description="Only spans where keep=True, representing the final extraction"
    )


def build_s1_reverse_extraction_system() -> str:
    """
    S1 System Prompt using Reverse Extraction principle.
    Over-extract first, then apply negative constraints.
    """
    return """
<system_directive>
  <role>
    You are a **Forensic Extraction Analyst** using **Reverse Extraction**.
    Your principle: "Better to over-extract and prune than to miss."
  </role>

  <method>
    **PHASE 1: EXHAUSTIVE EXTRACTION**
    - Extract ALL potential structural markers, including borderline cases
    - Include spans even if you're uncertain about their label
    - DO NOT self-censor based on text neutrality
    
    **PHASE 2: NEGATIVE CONSTRAINT FILTERING**
    Apply these exclusion rules to mark spans as keep=False:
    
    | Exclusion Rule | Description | Example |
    |----------------|-------------|---------|
    | PRONOUN_ONLY | Bare pronouns without resolution | "they", "it" alone |
    | TOO_GENERIC | Common words with no specificity | "the problem", "things" |
    | AUTHOR_OPINION | Author's personal stance, not a marker | "I think", "in my view" |
    | INCOMPLETE_PHRASE | Truncated or incomplete extraction | "and the" |
    | WRONG_LABEL | Correct text, wrong category | "The Media" labeled as Evidence |
  </method>

  <marker_definitions>
    - **Actor:** Agents performing actions (includes institutions, vague collectives like "they")
    - **Action:** What actors DO (verbs/verb phrases implying control/secrecy/harm)
    - **Effect:** Outcomes/consequences of actions
    - **Victim:** Entities suffering from actions
    - **Evidence:** Sources, proofs, epistemic claims cited
  </marker_definitions>

  <critical_reminder>
    Extract from NEUTRAL texts too! 
    A news report about "Government passed policy affecting workers" has valid Actor/Action/Victim.
    Do NOT skip extraction just because text seems factual.
  </critical_reminder>

  <output_format>
    Return the full candidate list with keep/exclusion annotations, then the filtered final spans.
  </output_format>
</system_directive>

{{few_shot_examples}}
""".strip()


def build_s1_reverse_extraction_user_template() -> str:
    """
    User template for Reverse Extraction.
    Variables: {{text}}
    """
    return """
<document_to_analyze>
{{text}}
</document_to_analyze>

<extraction_task>
**PHASE 1:** Extract ALL candidate markers (over-extract).
**PHASE 2:** Apply exclusion rules to filter candidates.
**OUTPUT:** Return both the full candidate list and the pruned final_spans.

IMPORTANT: Even if the text is neutral/factual, extract its structural markers.
</extraction_task>
"""


# ===========================================================================
# Graph-Structured Debate for S2 (Alternative to Sequential)
# ===========================================================================


def build_got_council_system() -> str:
    """
    Graph of Thought Council System.
    Instead of sequential debate, runs PARALLEL analysis paths.
    """
    return """
<system_directive>
  <role>
    You are a **Multi-Path Analyst** using Graph of Thought (GoT).
    You will evaluate the text through PARALLEL analytical lenses, then synthesize.
  </role>

  <parallel_analysis_paths>
    Run these 4 analyses INDEPENDENTLY (do not let one bias another):
    
    **PATH A: SEMANTIC ANALYSIS**
    - Focus: What claims are being made?
    - Output: List of explicit/implicit assertions
    
    **PATH B: PRAGMATIC ANALYSIS**  
    - Focus: What is the author DOING with this text?
    - Output: Speech act classification (reporting, endorsing, mocking, questioning)
    
    **PATH C: STRUCTURAL ANALYSIS**
    - Focus: Does it have conspiracy structure?
    - Output: Presence/absence of Actor+Action+Victim+Secrecy
    
    **PATH D: TONE ANALYSIS**
    - Focus: Author's emotional stance
    - Output: Urgency level, in-group/out-group markers, certainty language
  </parallel_analysis_paths>

  <synthesis>
    Combine paths using this logic:
    - If PATH B = "reporting" AND no first-person endorsement → 'non'
    - If PATH B = "mocking" → 'non'
    - If PATH C = incomplete structure → 'non'
    - If PATH B = "endorsing" AND PATH C = complete AND PATH D = urgent → 'conspiracy'
  </synthesis>
</system_directive>
""".strip()


# ===========================================================================
# Utility: Prompt Selection Helper
# ===========================================================================


TECHNIQUE_MAP = {
    "s1": {
        "default": ("self_consistency", "Current ensemble approach"),
        "reverse_extraction": ("rex", "Over-extract then prune - better recall"),
        "cot": ("chain_of_thought", "Step-by-step reasoning"),
    },
    "s2": {
        "default": ("multi_persona_debate", "Current prosecutor/defense approach"),
        "rex_got": ("rex_got", "Reverse Exclusion Graph of Thought - optimal for binary classification"),
        "got_parallel": ("graph_of_thought", "Parallel analysis paths with synthesis"),
    }
}


def get_recommended_technique(task: Literal["s1", "s2"]) -> str:
    """Returns the recommended prompting technique for a task."""
    if task == "s1":
        return """
**Recommended for S1 (Span Extraction): REVERSE EXTRACTION**

Why: 
- Prevents "miss" errors by over-extracting first
- Uses explicit negative constraints to prune
- Aligns with competition's recall-weighted F1 metric

Current ensemble (k=3 voting) is good for PRECISION.
Add Reverse Extraction for better RECALL.
"""
    else:
        return """
**Recommended for S2 (Classification): ReX-GoT**

Why:
- Systematic exclusion criteria prevent false positives
- Graph structure ensures all relevant factors are considered
- "Innocent until proven guilty" aligns with the Reporter/Hard Negative problem

Your current debate structure is partially ReX-GoT.
Full implementation adds:
1. Explicit exclusion nodes
2. Parallel (not sequential) evaluation
3. Confidence-weighted synthesis
"""


if __name__ == "__main__":
    print("=== S1 Recommendation ===")
    print(get_recommended_technique("s1"))
    print("\n=== S2 Recommendation ===")
    print(get_recommended_technique("s2"))
