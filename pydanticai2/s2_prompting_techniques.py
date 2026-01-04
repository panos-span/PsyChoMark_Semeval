#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s2_prompting_techniques.py — Comprehensive Evaluation & Implementation of 
                              Prompting Techniques for S2 (Conspiracy Classification)

This file provides:
1. Detailed analysis of each prompting technique for binary classification
2. Recommended implementations with Pydantic schemas
3. Performance trade-off analysis

Task: S2 - Classify text as 'conspiracy' or 'non' (conspiracy endorsement detection)
Metric: Binary F1 (conspiracy class is positive)
Challenge: Hard negatives (reporting ABOUT conspiracies ≠ endorsing them)
"""

from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


# ===========================================================================
# S2 TASK CHARACTERISTICS
# ===========================================================================
"""
S2 CHALLENGE ANALYSIS:

BINARY CLASSIFICATION: 'conspiracy' vs 'non'

KEY DIFFICULTIES:
1. **Reporter Problem**: Text about conspiracies vs endorsing them
   - "Users claim the government is hiding aliens" → NON (reporting)
   - "The government IS hiding aliens!" → CONSPIRACY (endorsing)

2. **Incomplete Narratives**: Some texts mention plots without full structure
   - Need Actor + Action + Victim + Secrecy for true conspiracy

3. **Sarcasm/Satire**: Mocking conspiracy theories ≠ endorsing them
   - "Sure, and 5G causes COVID too" → NON (sarcastic)

4. **Epistemic Language**: "Some say..." vs "I know..." changes meaning

CURRENT IMPLEMENTATION:
- Sequential Debate: Prosecutor → Defense → Profiler → Literalist
- Each persona has a bias (conviction vs acquittal)
- Judge synthesizes votes

PROBLEM: Sequential → Later agents influenced by earlier ones
SOLUTION: Parallel analysis + explicit exclusion criteria
"""


# ===========================================================================
# TECHNIQUE EVALUATION MATRIX FOR S2
# ===========================================================================
"""
┌──────────────────────────────────────┬────────┬────────┬─────────┬─────────┬──────────┐
│ Technique                            │ F1     │ Precis │ Recall  │ Tokens  │ OVERALL  │
├──────────────────────────────────────┼────────┼────────┼─────────┼─────────┼──────────┤
│ Standard Prompting                   │  ★★★   │  ★★★   │  ★★★    │ ★★★★★   │  ★★★     │
│ Chain-of-Thought (CoT)               │ ★★★★   │ ★★★★   │ ★★★★    │ ★★★★    │ ★★★★     │
│ Multi-Persona Debate (Current)       │ ★★★★   │ ★★★★   │ ★★★★    │  ★★     │ ★★★★     │
│ Tree-of-Thought (ToT)                │ ★★★★   │ ★★★★★  │ ★★★★    │   ★     │  ★★★     │
│ Divergent CoT (DCoT)                 │ ★★★★   │ ★★★★   │ ★★★★★   │  ★★★    │ ★★★★     │
│ **DD-CoT (Dynamic Discriminative)**  │ ★★★★★  │ ★★★★★  │ ★★★★    │  ★★★    │ ★★★★★    │
│ **ReX-GoT (Reverse Exclusion GoT)**  │ ★★★★★  │ ★★★★★  │ ★★★★    │  ★★★    │ ★★★★★    │
│ Graph of Thought (GoT - Parallel)    │ ★★★★★  │ ★★★★★  │ ★★★★    │  ★★     │ ★★★★★    │
│ Self-Consistency (k=N)               │ ★★★★   │ ★★★★★  │ ★★★★    │   ★     │  ★★★     │
│ Contrastive Chain-of-Thought         │ ★★★★★  │ ★★★★★  │ ★★★★    │  ★★★    │ ★★★★★    │
│ MCTS-Based Reasoning                 │ ★★★★★  │ ★★★★★  │ ★★★★    │   ★     │  ★★★     │
└──────────────────────────────────────┴────────┴────────┴─────────┴─────────┴──────────┘

★ = Poor, ★★★ = Average, ★★★★★ = Excellent

RECOMMENDATIONS FOR S2:
  🥇 DD-CoT (Dynamic Discriminative) - Best for reporter/endorser discrimination
  🥈 ReX-GoT (Reverse Exclusion) - Systematic exclusion criteria
  🥉 Contrastive CoT - Explicit "conspiracy vs non" reasoning
  
Combined Optimal: DD-CoT + ReX-GoT Hybrid
"""


# ===========================================================================
# 1. DD-CoT (Dynamic Discriminative CoT) - TOP RECOMMENDATION
# ===========================================================================
"""
DD-CoT ANALYSIS FOR S2:

WHY DD-CoT IS OPTIMAL FOR S2:

The core S2 challenge is DISCRIMINATION between:
- Reporter ("Users claim X") vs Endorser ("I know X is true")
- Complete conspiracy vs Incomplete mention
- Sincere belief vs Sarcastic mockery

DD-CoT forces the model to explain:
✅ "This IS conspiracy BECAUSE..."
✅ "This is NOT just reporting BECAUSE..."
✅ "This is NOT sarcasm BECAUSE..."

PROS:
✅ Directly addresses Reporter Problem with contrastive reasoning
✅ Dynamic exemplar selection adapts to text type
✅ Calibrated decisions via explicit exclusion reasoning
✅ Single call (efficient)

CONS:
- Requires curated contrastive examples
- Model must understand nuanced linguistic distinctions

VERDICT: ★★★★★ TOP RECOMMENDATION FOR S2
"""


class S2DDCoTConfusion(BaseModel):
    """Common confusions for S2 that DD-CoT should discriminate."""
    confusion_type: Literal[
        "reporter_vs_endorser",
        "sarcasm_vs_sincere", 
        "incomplete_vs_complete",
        "questioning_vs_asserting"
    ]
    why_this_not_that: str = Field(
        description="Explicit reasoning for why text IS this category and NOT the other"
    )


class S2DDCoTAnalysis(BaseModel):
    """
    DD-CoT analysis for S2 classification.
    The key innovation: explicit discrimination between conspiracy and non.
    """
    # Dynamic context assessment
    text_complexity: Literal["simple", "moderate", "complex"] = Field(
        description="How ambiguous is the classification?"
    )
    primary_challenge: Literal[
        "reporter_attribution",
        "sarcasm_detection", 
        "incomplete_structure",
        "epistemic_hedging",
        "clear_case"
    ] = Field(
        description="What makes this text challenging to classify?"
    )
    
    # Evidence for CONSPIRACY
    conspiracy_evidence: List[str] = Field(
        description="Specific text markers suggesting conspiracy endorsement"
    )
    
    # Evidence for NON-CONSPIRACY  
    non_evidence: List[str] = Field(
        description="Specific text markers suggesting this is NOT conspiracy endorsement"
    )
    
    # Discriminative reasoning (the key innovation)
    discriminations: List[S2DDCoTConfusion] = Field(
        description="For each relevant confusion type, explain the discrimination"
    )
    
    # Final verdict with confidence
    verdict: Literal["conspiracy", "non"]
    confidence: float = Field(ge=0.0, le=1.0)
    verdict_reasoning: str = Field(
        description="One-sentence explanation of final decision"
    )


def build_s2_ddcot_system() -> str:
    """
    DD-CoT System Prompt for S2 Classification.
    Emphasizes discriminative reasoning between conspiracy and non.
    """
    return """
<system_directive>
  <role>
    You are a **Forensic Discourse Analyst** using **Dynamic Discriminative Chain-of-Thought**.
    Your task is to classify text as 'conspiracy' (endorses conspiracy) or 'non' (does not endorse).
  </role>

  <core_principle>
    For every classification, you must DISCRIMINATE between the label you choose and the alternative.
    It's not enough to say "This IS conspiracy." 
    You must also say "This is NOT just reporting/sarcasm/incomplete BECAUSE..."
  </core_principle>

  <discrimination_framework>
    **CONFUSION 1: REPORTER vs ENDORSER**
    - Reporter: "Users claim the government is hiding X" → NON
    - Endorser: "The government IS hiding X" → CONSPIRACY
    - Discrimination Cue: Attribution phrases ("according to", "OP says", "the video claims")
    
    **CONFUSION 2: SARCASM vs SINCERE**
    - Sarcasm: "Sure, and 5G causes COVID too" → NON
    - Sincere: "5G is designed to weaken us" → CONSPIRACY
    - Discrimination Cue: Eye-roll indicators, exaggeration, quotation marks
    
    **CONFUSION 3: INCOMPLETE vs COMPLETE**
    - Incomplete: "Big Pharma is corrupt" → NON (no secret plot specified)
    - Complete: "Big Pharma created COVID to sell vaccines" → CONSPIRACY
    - Discrimination Cue: Presence of Actor + Secret Action + Harmful Effect + Victim
    
    **CONFUSION 4: QUESTIONING vs ASSERTING**
    - Questioning: "What if the election was rigged?" → NON (genuine question)
    - Asserting: "The election WAS rigged, wake up!" → CONSPIRACY (assertion)
    - Discrimination Cue: "Just asking questions" vs declarative statements with certainty
  </discrimination_framework>

  <conspiracy_requirements>
    A text is CONSPIRACY only if it:
    1. Contains a secret plot (hidden action by actors)
    2. Has a malevolent actor (the conspirators)
    3. Identifies victims (those harmed by the plot)
    4. The AUTHOR endorses or believes the narrative (not just reporting it)
  </conspiracy_requirements>

  <output_format>
    1. Assess text complexity and primary challenge
    2. List evidence FOR conspiracy
    3. List evidence FOR non-conspiracy
    4. For each relevant confusion type, provide explicit discrimination reasoning
    5. Give final verdict with confidence and reasoning
  </output_format>
</system_directive>
""".strip()


def build_s2_ddcot_user_template() -> str:
    """User template for DD-CoT S2 classification."""
    return """
<document_to_classify>
{{text}}
</document_to_classify>

<contextual_markers>
{{marker_summary}}
</contextual_markers>

<dynamic_examples>
Based on this text's characteristics, here are relevant examples:

{{few_shot_examples}}
</dynamic_examples>

<task>
Classify this text as 'conspiracy' or 'non' using DISCRIMINATIVE reasoning.

For your classification, you MUST explain:
1. Why it IS your chosen label
2. Why it is NOT the alternative label
3. Address any relevant confusions (reporter vs endorser, sarcasm vs sincere, etc.)
</task>
"""


# ===========================================================================
# 2. ReX-GoT (Reverse Exclusion Graph of Thought) - ALSO TOP TIER
# ===========================================================================
"""
ReX-GoT ANALYSIS FOR S2:

Already implemented in rex_got_prompts.py, but here's the analysis:

PRINCIPLE: "Innocent until proven guilty"
- Start with assumption: conspiracy
- Apply exclusion criteria in parallel
- If ANY exclusion triggers → acquit (non)

EXCLUSION NODES:
1. Attribution Exclusion: Author attributes claims to third party
2. Tone Exclusion: Author is mocking/satirizing  
3. Structure Exclusion: Incomplete conspiracy structure
4. Debunking Exclusion: Author explicitly rejects the theory

PROS:
✅ Systematic exclusion prevents false positives
✅ Parallel evaluation prevents ordering bias
✅ Explicit criteria make decisions transparent
✅ Aligns with "reporter problem" challenge

CONS:
- Requires threshold tuning (confidence > 0.7)
- May miss edge cases not covered by exclusion criteria

VERDICT: ★★★★★ TOP RECOMMENDATION FOR S2
- Complements DD-CoT: DD-CoT discriminates, ReX-GoT systematically excludes
"""


# Schema already in rex_got_prompts.py, adding hybrid schema here:


class HybridDDCoTReXGoT(BaseModel):
    """
    Hybrid approach: DD-CoT discrimination + ReX-GoT exclusion.
    Best of both worlds.
    """
    # Phase 1: DD-CoT Analysis
    ddcot_analysis: S2DDCoTAnalysis
    
    # Phase 2: ReX-GoT Exclusion Check (only if DD-CoT says conspiracy)
    exclusion_check: Optional["ReXGoTExclusionCheck"] = Field(
        default=None,
        description="If DD-CoT says conspiracy, verify with ReX-GoT exclusion criteria"
    )
    
    # Final synthesis
    final_verdict: Literal["conspiracy", "non"]
    agreement: bool = Field(
        description="True if DD-CoT and ReX-GoT agree"
    )
    resolution_note: Optional[str] = Field(
        default=None,
        description="If they disagree, explain how conflict was resolved"
    )


class ReXGoTExclusionCheck(BaseModel):
    """Simplified ReX-GoT check for hybrid approach."""
    attribution_excluded: bool = Field(
        description="Does author attribute claims to third party?"
    )
    tone_excluded: bool = Field(
        description="Is author mocking/satirizing?"
    )
    structure_excluded: bool = Field(
        description="Is conspiracy structure incomplete?"
    )
    debunking_excluded: bool = Field(
        description="Does author explicitly debunk?"
    )
    any_exclusion: bool = Field(
        description="True if ANY exclusion criterion triggered"
    )


HybridDDCoTReXGoT.model_rebuild()


def build_s2_hybrid_system() -> str:
    """
    Hybrid DD-CoT + ReX-GoT System Prompt.
    Two-phase approach for maximum accuracy.
    """
    return """
<system_directive>
  <role>
    You are a **Forensic Classification Expert** using a HYBRID approach:
    **Phase 1: DD-CoT** - Discriminative reasoning
    **Phase 2: ReX-GoT** - Systematic exclusion verification
  </role>

  <phase_1_ddcot>
    First, apply DISCRIMINATIVE Chain-of-Thought:
    
    1. Gather evidence FOR conspiracy
    2. Gather evidence FOR non-conspiracy
    3. For each ambiguity (reporter/endorser, sarcasm/sincere, etc.), discriminate
    4. Make preliminary verdict
    
    If preliminary verdict is 'non', proceed to final output.
    If preliminary verdict is 'conspiracy', proceed to Phase 2.
  </phase_1_ddcot>

  <phase_2_rexgot>
    If DD-CoT says 'conspiracy', VERIFY with exclusion criteria:
    
    **EXCLUSION 1: ATTRIBUTION** 
    - Does author attribute to third party? ("Users claim...", "The video shows...")
    - If YES → Override to 'non'
    
    **EXCLUSION 2: TONE**
    - Is author mocking/satirizing? (Sarcasm indicators, quotation marks)
    - If YES → Override to 'non'
    
    **EXCLUSION 3: STRUCTURE**
    - Is conspiracy structure incomplete? (Missing Actor/Action/Victim)
    - If YES → Override to 'non'
    
    **EXCLUSION 4: DEBUNKING**
    - Does author explicitly reject the theory? ("This is false", "Debunked")
    - If YES → Override to 'non'
    
    If NO exclusion triggers → Confirm 'conspiracy'
  </phase_2_rexgot>

  <conflict_resolution>
    If DD-CoT and ReX-GoT disagree:
    - ReX-GoT exclusion takes precedence (false positive prevention)
    - Document the disagreement and resolution reasoning
  </conflict_resolution>
</system_directive>
""".strip()


# ===========================================================================
# 3. CONTRASTIVE CHAIN-OF-THOUGHT
# ===========================================================================
"""
Contrastive CoT ANALYSIS FOR S2:

APPROACH:
- For each classification, generate reasoning for BOTH labels
- Compare the strength of arguments
- Choose the label with stronger reasoning

SIMILAR TO DD-CoT but less structured:
- DD-CoT: Explicit confusion types + discrimination rules
- Contrastive: Freeform reasoning for both sides

PROS:
✅ Forces consideration of both possibilities
✅ Reduces confirmation bias
✅ Good for edge cases

CONS:
- Less structured than DD-CoT
- May lead to verbose output

VERDICT: ★★★★ GOOD (DD-CoT is better for S2 specifically)
"""


class ContrastiveS2Analysis(BaseModel):
    """Contrastive CoT: Reason for both labels, then choose."""
    
    # Argument for CONSPIRACY
    case_for_conspiracy: str = Field(
        description="Build the strongest case that this IS conspiracy endorsement"
    )
    conspiracy_strength: float = Field(ge=0.0, le=1.0)
    
    # Argument for NON-CONSPIRACY
    case_for_non: str = Field(
        description="Build the strongest case that this is NOT conspiracy endorsement"
    )
    non_strength: float = Field(ge=0.0, le=1.0)
    
    # Decision
    verdict: Literal["conspiracy", "non"]
    decisive_factor: str = Field(
        description="The single most important factor that tipped the decision"
    )


def build_s2_contrastive_system() -> str:
    """Contrastive CoT System Prompt for S2."""
    return """
<system_directive>
  <role>
    You are a **Balanced Analyst** using **Contrastive Chain-of-Thought**.
    You will argue BOTH sides before deciding.
  </role>

  <method>
    **STEP 1: Build Case for CONSPIRACY**
    - Assume this IS conspiracy endorsement
    - Find ALL evidence supporting this interpretation
    - Rate the strength of this case (0.0 - 1.0)
    
    **STEP 2: Build Case for NON-CONSPIRACY**
    - Assume this is NOT conspiracy endorsement
    - Find ALL evidence supporting this interpretation
    - Rate the strength of this case (0.0 - 1.0)
    
    **STEP 3: Compare and Decide**
    - Which case is stronger?
    - What is the DECISIVE factor?
    - Make final verdict
  </method>

  <key_question>
    Does the AUTHOR endorse the conspiracy, or are they just mentioning/reporting/mocking it?
  </key_question>
</system_directive>
""".strip()


# ===========================================================================
# 4. GRAPH OF THOUGHT (PARALLEL ANALYSIS)
# ===========================================================================
"""
GoT (Graph of Thought) ANALYSIS FOR S2:

APPROACH:
- Run PARALLEL analysis paths (not sequential)
- Each path focuses on different aspect
- Synthesize at the end

DIFFERS FROM CURRENT DEBATE:
- Current: Prosecutor → Defense → Profiler → Literalist (SEQUENTIAL)
- GoT: All 4 run INDEPENDENTLY, then synthesize

WHY PARALLEL IS BETTER:
- No ordering bias (defense doesn't just react to prosecutor)
- Each path has fresh perspective
- Synthesis is principled (not just voting)

PROS:
✅ Eliminates ordering effects
✅ More comprehensive coverage
✅ Principled synthesis

CONS:
- Harder to implement in single call
- May need multiple calls for true independence

VERDICT: ★★★★★ EXCELLENT - but requires architectural change
"""


class GoTPath(BaseModel):
    """A single analysis path in Graph of Thought."""
    path_name: str
    focus: str
    findings: List[str]
    path_verdict: Literal["conspiracy", "non", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0)


class S2GraphOfThought(BaseModel):
    """Graph of Thought: Parallel analysis paths with synthesis."""
    
    # Path A: Semantic Analysis
    semantic_path: GoTPath = Field(
        description="What claims are being made? (Focus on content)"
    )
    
    # Path B: Pragmatic Analysis
    pragmatic_path: GoTPath = Field(
        description="What is the author DOING? (reporting, endorsing, mocking)"
    )
    
    # Path C: Structural Analysis
    structural_path: GoTPath = Field(
        description="Does it have conspiracy structure? (Actor+Action+Victim+Secrecy)"
    )
    
    # Path D: Epistemic Analysis
    epistemic_path: GoTPath = Field(
        description="What certainty level? (hedged, asserted, questioned)"
    )
    
    # Synthesis
    path_votes: Dict[str, Literal["conspiracy", "non", "uncertain"]]
    synthesis_logic: str = Field(
        description="How paths were combined to reach final verdict"
    )
    final_verdict: Literal["conspiracy", "non"]


def build_s2_got_system() -> str:
    """Graph of Thought System Prompt for S2."""
    return """
<system_directive>
  <role>
    You are a **Multi-Path Analyst** using **Graph of Thought**.
    You will evaluate the text through 4 PARALLEL lenses, then synthesize.
  </role>

  <parallel_paths>
    Run these analyses INDEPENDENTLY (do not let one bias another):
    
    **PATH A: SEMANTIC**
    - What explicit/implicit claims are made?
    - What is the content about?
    
    **PATH B: PRAGMATIC**
    - What SPEECH ACT is this? (reporting, endorsing, questioning, mocking)
    - What is the author's communicative intent?
    
    **PATH C: STRUCTURAL**
    - Does it have conspiracy structure?
    - Check: Secret Plot + Malevolent Actor + Victim + Secrecy
    
    **PATH D: EPISTEMIC**
    - What certainty level is expressed?
    - Is it hedged ("some say") or asserted ("I know")?
  </parallel_paths>

  <synthesis_rules>
    Combine paths using this priority:
    
    1. If PATH B = "reporting" + no first-person endorsement → 'non'
    2. If PATH B = "mocking" → 'non'
    3. If PATH C = incomplete structure → 'non'
    4. If PATH D = hedged/questioning only → 'non'
    5. If PATH B = "endorsing" AND PATH C = complete → 'conspiracy'
    
    PATH B (pragmatic) has highest priority - it determines AUTHOR STANCE.
  </synthesis_rules>
</system_directive>
""".strip()


# ===========================================================================
# 5. TREE OF THOUGHT (ToT) - ANALYSIS ONLY
# ===========================================================================
"""
ToT ANALYSIS FOR S2:

APPROACH:
- Explore multiple interpretation branches
- Evaluate each branch
- Backtrack if needed, select best path

FOR S2:
- Branch 1: Assume conspiracy, explore evidence
- Branch 2: Assume non, explore evidence
- Evaluate which branch has stronger support

PROS:
✅ Comprehensive exploration
✅ Can backtrack from wrong paths
✅ Good for very ambiguous cases

CONS:
- Very token-expensive
- Binary classification usually doesn't need tree exploration
- Overkill for most S2 samples

VERDICT: ★★★ MODERATE - Use only for highly ambiguous cases
- For standard S2, DD-CoT/ReX-GoT are more efficient
"""


# ===========================================================================
# 6. MULTI-PERSONA DEBATE (CURRENT APPROACH)
# ===========================================================================
"""
Current S2 Implementation:
- Prosecutor: Bias toward 'conspiracy'
- Defense: Bias toward 'non'
- Profiler: Analyzes author stance
- Literalist: Checks structural requirements
- Judge: Synthesizes votes

STRENGTHS:
✅ Multiple perspectives
✅ Built-in adversarial check
✅ Each persona has clear role

WEAKNESSES:
- Sequential: Later agents influenced by earlier ones
- 4+ LLM calls per sample (expensive)
- Voting may not capture nuance

IMPROVEMENT OPPORTUNITY:
→ Make parallel (GoT) instead of sequential
→ Add DD-CoT discrimination to each persona
→ Use ReX-GoT exclusion as final check
"""


# ===========================================================================
# 7. SELF-CONSISTENCY FOR S2
# ===========================================================================
"""
Self-Consistency ANALYSIS FOR S2:

APPROACH:
- Run k independent classification calls
- Majority vote

FOR S2:
- k=3: Run 3 classifications, take majority
- Reduces variance from single call

PROS:
✅ Robust to random variation
✅ Simple to implement

CONS:
- k × token cost
- Doesn't fix systematic biases
- All k calls may make same error

VERDICT: ★★★ MODERATE
- Good as add-on, not primary technique
- Combine with DD-CoT: k=3 DD-CoT calls, vote on result
"""


# ===========================================================================
# FINAL RECOMMENDATION: OPTIMAL S2 PIPELINE
# ===========================================================================
"""
RECOMMENDED S2 PIPELINE (Ranked by effectiveness):

┌─────────────────────────────────────────────────────────────────────────┐
│                     OPTIMAL S2 ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. RAG RETRIEVAL (Few-shots) - DYNAMIC SELECTION                       │
│     └── Select examples based on text characteristics                   │
│     └── Include contrastive pairs (reporter vs endorser examples)       │
│                                                                         │
│  2. S1 MARKER EXTRACTION (Optional - provides context)                  │
│     └── Actor/Action/Effect/Victim/Evidence markers                     │
│     └── Helps structural completeness check                             │
│                                                                         │
│  3. DD-CoT CLASSIFICATION (Primary Decision)                            │
│     └── Dynamic assessment (text complexity, primary challenge)         │
│     └── Evidence for conspiracy + Evidence for non                      │
│     └── Discriminative reasoning (reporter/endorser, sarcasm/sincere)   │
│     └── Preliminary verdict                                             │
│                                                                         │
│  4. ReX-GoT VERIFICATION (If DD-CoT says 'conspiracy')                  │
│     └── Attribution exclusion check                                     │
│     └── Tone exclusion check                                            │
│     └── Structure exclusion check                                       │
│     └── Debunking exclusion check                                       │
│     └── If ANY exclusion → Override to 'non'                            │
│                                                                         │
│  5. FINAL OUTPUT                                                        │
│     └── Verdict with confidence                                         │
│     └── Reasoning chain                                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

TOKEN COMPARISON:
- Current (4 personas + judge): 5 LLM calls
- Optimized (DD-CoT + ReX-GoT): 2 LLM calls (or 1 combined call)
- Savings: 60% fewer calls, BETTER quality

ALTERNATIVE: Single-Call Hybrid (Most Efficient)
- Combine DD-CoT + ReX-GoT in ONE structured output
- Model does both analysis phases in single call
- See HybridDDCoTReXGoT schema above
"""


# ===========================================================================
# UTILITY: Technique Selection Helper
# ===========================================================================

def get_s2_technique_recommendation(
    false_positive_rate: float = 0.0,  # Labeling 'non' as 'conspiracy'
    false_negative_rate: float = 0.0,  # Missing actual conspiracies
    reporter_confusion_rate: float = 0.0,  # Misclassifying reporters
    token_budget: str = "medium",  # low, medium, high
) -> str:
    """
    Returns the recommended S2 technique based on current performance.
    """
    
    if reporter_confusion_rate > 0.2:
        # Major reporter problem
        return "DD-CoT (discriminate reporter vs endorser) + Attribution Exclusion"
    
    if false_positive_rate > 0.3:
        # Too many false positives
        return "ReX-GoT (systematic exclusion) - triggers prevent false convictions"
    
    if false_negative_rate > 0.3:
        # Missing actual conspiracies
        return "DD-CoT with conviction bias + light ReX-GoT verification"
    
    if token_budget == "low":
        return "Single-call Hybrid (DD-CoT + ReX-GoT combined)"
    
    if token_budget == "high":
        return "Full GoT (4 parallel paths) + DD-CoT synthesis + ReX-GoT verification"
    
    # Balanced case
    return "DD-CoT + ReX-GoT Hybrid (2 calls or 1 combined)"


# ===========================================================================
# COMPARISON: DD-CoT vs ReX-GoT vs Current Debate
# ===========================================================================
"""
┌─────────────────────────┬────────────────────────────────────────────────────┐
│ Aspect                  │ DD-CoT          │ ReX-GoT         │ Current Debate │
├─────────────────────────┼─────────────────┼─────────────────┼────────────────┤
│ Focus                   │ Discrimination  │ Exclusion       │ Adversarial    │
│ Key Question            │ "Why IS and     │ "Is there any   │ "Who wins the  │
│                         │  NOT?"          │  exclusion?"    │  argument?"    │
├─────────────────────────┼─────────────────┼─────────────────┼────────────────┤
│ Best For                │ Ambiguous cases │ False positive  │ General use    │
│                         │ with confusions │ prevention      │                │
├─────────────────────────┼─────────────────┼─────────────────┼────────────────┤
│ Reporter Problem        │ ★★★★★ (direct) │ ★★★★★ (excl.)  │ ★★★ (indirect) │
│ Sarcasm Detection       │ ★★★★★ (direct) │ ★★★★ (tone)    │ ★★★ (profiler) │
│ Structure Check         │ ★★★★ (implicit)│ ★★★★★ (explicit)│ ★★★★ (literal) │
├─────────────────────────┼─────────────────┼─────────────────┼────────────────┤
│ Token Efficiency        │ ★★★★ (1 call)  │ ★★★★ (1 call)  │ ★★ (5 calls)   │
│ Transparency            │ ★★★★★          │ ★★★★★          │ ★★★            │
├─────────────────────────┼─────────────────┼─────────────────┼────────────────┤
│ OVERALL                 │ ★★★★★          │ ★★★★★          │ ★★★★           │
└─────────────────────────┴─────────────────┴─────────────────┴────────────────┘

RECOMMENDATION: Use DD-CoT + ReX-GoT Hybrid
- DD-CoT provides discriminative reasoning (WHY is/isn't)
- ReX-GoT provides safety net (systematic exclusion)
- Together: Best accuracy + interpretability
"""


if __name__ == "__main__":
    print("=" * 70)
    print("S2 PROMPTING TECHNIQUE RECOMMENDATIONS")
    print("=" * 70)
    
    # Example usage
    recommendation = get_s2_technique_recommendation(
        false_positive_rate=0.15,
        false_negative_rate=0.10,
        reporter_confusion_rate=0.20,  # Common problem
        token_budget="medium",
    )
    
    print(f"\nGiven your current performance:")
    print(f"  - False Positive Rate: 15%")
    print(f"  - False Negative Rate: 10%")
    print(f"  - Reporter Confusion: 20%")
    print(f"\nRECOMMENDED: {recommendation}")
    
    print("\n" + "=" * 70)
    print("TECHNIQUE SUMMARY")
    print("=" * 70)
    print("""
    🥇 DD-CoT (Dynamic Discriminative) ★★★★★
       - Best for: Reporter vs Endorser discrimination
       - Key: "WHY IS conspiracy AND WHY NOT just reporting?"
       - Addresses the hardest S2 challenge directly
       
    🥈 ReX-GoT (Reverse Exclusion) ★★★★★
       - Best for: False positive prevention
       - Key: 4 exclusion criteria, ANY trigger → 'non'
       - Systematic safety net
       
    🥉 Contrastive CoT ★★★★★
       - Best for: Edge cases
       - Key: Argue both sides, compare strength
       
    4. Graph of Thought (Parallel) ★★★★★
       - Best for: Comprehensive analysis
       - Key: 4 parallel paths, principled synthesis
       - Fixes ordering bias of current debate
       
    5. Multi-Persona Debate (Current) ★★★★
       - Good but: Sequential ordering causes bias
       - Improvement: Make parallel (GoT)
       
    6. Self-Consistency ★★★
       - Good as add-on, not primary technique
       - k=3 DD-CoT calls with voting
       
    7. Tree of Thought ★★★
       - Overkill for binary classification
       - Use only for highly ambiguous cases

    ─────────────────────────────────────────────────────────────────
    KEY INSIGHT: DD-CoT vs ReX-GoT
    ─────────────────────────────────────────────────────────────────
    • DD-CoT: "This IS conspiracy BECAUSE X, NOT just reporting BECAUSE Y"
    • ReX-GoT: "Check exclusion 1... 2... 3... 4... None triggered → conspiracy"
    
    BEST APPROACH: HYBRID
    1. DD-CoT makes preliminary decision with discrimination
    2. If 'conspiracy', ReX-GoT verifies with exclusion criteria
    3. Any exclusion → Override to 'non' (false positive prevention)
    """)
