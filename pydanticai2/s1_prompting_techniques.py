#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s1_prompting_techniques.py — Comprehensive Evaluation & Implementation of 
                              Prompting Techniques for S1 (Span Extraction)

This file provides:
1. Detailed analysis of each prompting technique for NER/span extraction
2. Recommended implementations with Pydantic schemas
3. Performance trade-off analysis

Task: S1 - Extract psycholinguistic markers (Actor, Action, Effect, Victim, Evidence)
Metric: Token-IoU based F1 (overlap ≥ 0.5 threshold)
Challenge: High recall needed (don't miss spans) + High precision (don't hallucinate)
"""

from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


# ===========================================================================
# TECHNIQUE EVALUATION MATRIX
# ===========================================================================
"""
┌──────────────────────────────┬────────┬────────┬─────────┬─────────┬──────────┐
│ Technique                    │ Recall │ Precis │ Tokens  │ Latency │ OVERALL  │
├──────────────────────────────┼────────┼────────┼─────────┼─────────┼──────────┤
│ Standard Prompting           │   ★★   │  ★★★   │  ★★★★★  │ ★★★★★   │   ★★★    │
│ Chain-of-Thought (CoT)       │  ★★★   │  ★★★   │  ★★★    │  ★★★    │  ★★★     │
│ Self-Consistency (SC)        │  ★★★★  │  ★★★★  │   ★★    │   ★★    │  ★★★★    │
│ Tree-of-Thought (ToT)        │ ★★★★★  │  ★★★   │    ★    │    ★    │  ★★★     │
│ Divergent CoT (DCoT)         │ ★★★★★  │ ★★★★   │   ★★    │   ★★    │ ★★★★★    │
│ **DD-CoT (Dynamic Discrim)** │ ★★★★★  │ ★★★★★  │  ★★★    │  ★★★    │ ★★★★★    │
│ Self-Refine                  │  ★★★★  │ ★★★★★  │   ★★    │   ★★    │  ★★★★    │
│ Reverse Extraction           │ ★★★★★  │  ★★★★  │  ★★★    │  ★★★    │ ★★★★★    │
│ Label-Guided Extraction      │  ★★★★  │ ★★★★★  │  ★★★    │  ★★★    │  ★★★★    │
│ Skeleton-of-Thought          │  ★★★   │ ★★★★   │  ★★★★   │  ★★★★   │  ★★★     │
│ Least-to-Most                │ ★★★★★  │  ★★★   │   ★★    │   ★★    │  ★★★★    │
└──────────────────────────────┴────────┴────────┴─────────┴─────────┴──────────┘

★ = Poor, ★★★ = Average, ★★★★★ = Excellent

RECOMMENDATION FOR S1: 
  🥇 DD-CoT (Dynamic Discriminative) - Best for NER with contrast reasoning
  🥈 Divergent CoT (DCoT) - Best recall/precision balance  
  🥉 Reverse Extraction - Explicit over-extract → prune

Combined Optimal: DD-CoT + Self-Refine + Ensemble Voting
"""


# ===========================================================================
# 1. TREE-OF-THOUGHT (ToT) - Comprehensive Analysis
# ===========================================================================
"""
ToT ANALYSIS FOR S1:

PROS:
- Explores multiple interpretation branches
- Can backtrack when labeling seems wrong
- Good for ambiguous spans

CONS:
- Very token-expensive (exponential branching)
- Latency too high for batch processing
- NER is not naturally "tree-shaped" - spans are mostly independent
- Overkill for straightforward extractions

VERDICT: ⚠️ NOT RECOMMENDED for S1
- ToT is designed for problems with dependent decision chains (puzzles, math)
- Span extraction has mostly independent entities
- Cost/benefit ratio is poor

WHEN TO USE ToT:
- Complex nested structures (e.g., parsing code)
- When one span label depends on another's interpretation
- Single high-value document analysis (not batch)
"""


class ToTNode(BaseModel):
    """A single node in the Tree of Thought."""
    thought: str = Field(description="Current reasoning step")
    candidates: List[Dict[str, str]] = Field(description="Candidate spans at this node")
    evaluation: float = Field(ge=0.0, le=1.0, description="Self-evaluated quality score")
    should_backtrack: bool = Field(description="True if this branch should be abandoned")


class ToTExtraction(BaseModel):
    """Tree of Thought for span extraction - NOT RECOMMENDED."""
    exploration_tree: List[ToTNode] = Field(
        description="Branching exploration of possible extractions"
    )
    best_path: List[int] = Field(description="Indices of nodes in the optimal path")
    final_spans: List[Dict[str, str]] = Field(description="Final extracted spans")


# ===========================================================================
# 2. DYNAMIC DISCRIMINATIVE CHAIN-OF-THOUGHT (DD-CoT) - TOP RECOMMENDATION
# ===========================================================================
"""
DD-CoT ANALYSIS FOR S1:

WHAT IS DD-CoT?
DD-CoT combines three powerful ideas:
1. DYNAMIC: Exemplar selection adapts to input characteristics
2. DISCRIMINATIVE: Reasoning explains WHY something IS vs IS NOT a marker
3. CHAIN-OF-THOUGHT: Step-by-step reasoning for each extraction

KEY INSIGHT: Traditional CoT says "This IS an Actor because..."
             DD-CoT says "This IS an Actor because X, and NOT an Action because Y"

PROS:
✅ Discriminative reasoning reduces label confusion (Actor vs Victim)
✅ Dynamic exemplar selection → right few-shots for each text type
✅ Contrastive examples teach boundary cases explicitly
✅ Aligns perfectly with your RAG few-shot retrieval system
✅ Better calibration (model learns what to EXCLUDE)

CONS:
- Requires curated contrastive examples
- Slightly more prompt engineering
- Need good negative examples

VERDICT: ★★★★★ TOP RECOMMENDATION for S1
- Your RAG already does dynamic few-shot selection
- Adding discriminative reasoning will reduce label errors
- Perfect for ambiguous cases (is "they" an Actor or just a pronoun?)

WHY DD-CoT BEATS DCoT FOR S1:
- DCoT: "Here are 3 different extractions" (diverse but may all be wrong)
- DD-CoT: "Here's WHY this IS X and NOT Y" (calibrated decisions)
- For NER with 5 overlapping labels, discrimination is more valuable than diversity
"""


class DDCoTContrastiveExample(BaseModel):
    """A contrastive example for DD-CoT training."""
    text_fragment: str = Field(description="The ambiguous text fragment")
    correct_label: Optional[SpanLabel] = Field(description="The correct label (None if not a marker)")
    incorrect_labels: List[SpanLabel] = Field(description="Labels it could be confused with")
    discrimination_reasoning: str = Field(
        description="Why it IS the correct label and NOT the incorrect ones"
    )


class DDCoTSpanExtraction(BaseModel):
    """
    Dynamic Discriminative CoT: Each span includes contrastive reasoning.
    """
    # Dynamic context assessment
    text_complexity: Literal["simple", "moderate", "complex"] = Field(
        description="How ambiguous is this text?"
    )
    dominant_narrative: Literal["conspiracy", "neutral", "debunking", "mixed"] = Field(
        description="What type of discourse is this?"
    )
    
    # Discriminative extraction
    extractions: List["DDCoTSpan"] = Field(
        description="Spans with discriminative reasoning"
    )


class DDCoTSpan(BaseModel):
    """A single span with discriminative reasoning."""
    text: str = Field(description="Verbatim span from document")
    label: SpanLabel = Field(description="Assigned label")
    
    # Discriminative reasoning (the key innovation)
    why_this_label: str = Field(
        description="Why this span IS this label type"
    )
    why_not_other_labels: Dict[str, str] = Field(
        description="For each plausible alternative label, why it's NOT that"
    )
    
    confidence: float = Field(ge=0.0, le=1.0)


# Update forward reference
DDCoTSpanExtraction.model_rebuild()


def build_ddcot_system() -> str:
    """
    Dynamic Discriminative CoT System Prompt.
    Combines dynamic exemplar context with contrastive reasoning.
    """
    return """
<system_directive>
  <role>
    You are a **Forensic Linguistic Analyst** using **Dynamic Discriminative Chain-of-Thought**.
    
    Your extraction process has TWO key properties:
    1. **DYNAMIC**: Adapt your extraction strategy to the text type
    2. **DISCRIMINATIVE**: For each span, explain why it IS this label and NOT others
  </role>

  <dynamic_assessment>
    First, assess the text:
    - **Complexity**: How many ambiguous spans? (simple/moderate/complex)
    - **Narrative**: What discourse type? (conspiracy/neutral/debunking/mixed)
    
    Adjust your extraction based on this:
    - Conspiracy texts → More Actor/Action/Effect markers expected
    - Neutral texts → Fewer markers but still extract what's present
    - Debunking texts → Evidence markers more prominent
  </dynamic_assessment>

  <discriminative_reasoning>
    For EACH extracted span, provide CONTRASTIVE reasoning:
    
    ✅ WHY THIS LABEL:
    - What linguistic features make this an [Actor/Action/Effect/Victim/Evidence]?
    
    ❌ WHY NOT OTHER LABELS:
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

  <output_format>
    Return structured output with:
    1. Dynamic assessment (complexity, narrative type)
    2. Extractions with discriminative reasoning for each span
  </output_format>
</system_directive>
""".strip()


def build_ddcot_user_template() -> str:
    """User template with dynamic few-shot injection."""
    return """
<document_to_analyze>
{{text}}
</document_to_analyze>

<dynamic_fewshots>
Based on this text's characteristics, here are relevant examples:

{{few_shot_examples}}
</dynamic_fewshots>

<contrastive_examples>
Pay attention to these discrimination patterns:

EXAMPLE 1 - Actor vs Victim:
  Text: "The media manipulates the public"
  "The media" → Actor (performs "manipulates")
  "the public" → Victim (receives manipulation)
  NOT reversed because: Actor is the agent of the verb

EXAMPLE 2 - Action vs Effect:
  Text: "They suppress information to control the narrative"
  "suppress information" → Action (the verb phrase)
  "to control the narrative" → Effect (the purpose/outcome)
  NOT reversed because: Effect is the PURPOSE clause

EXAMPLE 3 - Evidence vs Actor:
  Text: "The leaked documents prove the conspiracy"
  "The leaked documents" → Evidence (cited as proof)
  NOT Actor because: It's a SOURCE, not an agent performing action
</contrastive_examples>

<task>
Extract all spans with DISCRIMINATIVE reasoning.
For each span, explain:
1. Why it IS the assigned label
2. Why it is NOT the most plausible alternative label(s)
</task>
"""


# ===========================================================================
# 3. DIVERGENT CHAIN-OF-THOUGHT (DCoT) - ALSO RECOMMENDED
# ===========================================================================
"""
DCoT ANALYSIS FOR S1:

PROS:
✅ Generates multiple PARALLEL reasoning paths in a single call
✅ Each path may find different spans (maximizes coverage)
✅ More token-efficient than k=3 separate calls (Self-Consistency)
✅ Natural fit for NER: "What if I interpret 'they' as Actor vs Evidence?"
✅ Diversity without ensemble overhead

CONS:
- Requires careful prompt engineering to ensure divergence
- Aggregation logic needed (similar to voting)
- Model may collapse paths into similar answers

VERDICT: ★★★★★ HIGHLY RECOMMENDED for S1
- Replaces Self-Consistency with single-call efficiency
- Generates diverse candidates in one LLM call
- Pair with voting for final selection

IMPLEMENTATION:
- Ask model to generate 3 DIFFERENT interpretations
- Each interpretation extracts spans independently
- Aggregate with 2/3 majority voting (like current ensemble)
"""


class DCoTInterpretation(BaseModel):
    """A single divergent interpretation of the text."""
    interpretation_id: Literal["conservative", "moderate", "aggressive"]
    interpretation_rationale: str = Field(
        description="How this interpretation differs from others"
    )
    spans: List[Dict[str, str]] = Field(
        description="Spans extracted under this interpretation"
    )


class DCoTExtraction(BaseModel):
    """
    Divergent CoT: Generate 3 parallel interpretations in ONE call.
    More efficient than k=3 ensemble while maintaining diversity.
    """
    # Three parallel reasoning paths
    conservative: DCoTInterpretation = Field(
        description="Conservative: Only extract absolutely certain spans. Minimize false positives."
    )
    moderate: DCoTInterpretation = Field(
        description="Moderate: Balanced extraction. Include probable spans."
    )
    aggressive: DCoTInterpretation = Field(
        description="Aggressive: Extract all possible spans. Maximize recall."
    )
    
    # Aggregation
    consensus_spans: List[Dict[str, str]] = Field(
        description="Spans that appear in at least 2 of the 3 interpretations"
    )


def build_dcot_system() -> str:
    """
    Divergent CoT System Prompt for S1.
    Generates 3 parallel interpretations with different extraction strategies.
    """
    return """
<system_directive>
  <role>
    You are a **Forensic Linguistic Analyst** using **Divergent Chain-of-Thought**.
    You will generate THREE parallel interpretations of the text, each with a different extraction strategy.
  </role>

  <divergent_strategy>
    Generate these 3 DIFFERENT interpretations:
    
    **INTERPRETATION 1: CONSERVATIVE**
    - Philosophy: "When in doubt, leave it out"
    - Only extract spans you are 100% confident about
    - Avoid borderline cases
    - Prioritize precision over recall
    
    **INTERPRETATION 2: MODERATE**  
    - Philosophy: "Balance is key"
    - Extract spans with >70% confidence
    - Include contextually supported markers
    - Balance precision and recall
    
    **INTERPRETATION 3: AGGRESSIVE**
    - Philosophy: "Better to over-extract than miss"
    - Extract ALL potential markers
    - Include borderline and implicit cases
    - Prioritize recall over precision
  </divergent_strategy>

  <marker_definitions>
    - **Actor:** Entities performing actions (agents, institutions, collectives)
    - **Action:** What actors DO (verbs implying control/secrecy/harm)
    - **Effect:** Outcomes/consequences of actions
    - **Victim:** Entities affected negatively
    - **Evidence:** Sources, proofs, epistemic claims
  </marker_definitions>

  <aggregation_rule>
    After generating all 3 interpretations, identify **consensus_spans**:
    - A span appears in consensus if it exists in ≥2 interpretations
    - Normalize matching: "The CIA" = "CIA" = "the cia"
    - If labels differ, prefer the most common label
  </aggregation_rule>

  <output_format>
    Return structured output with all 3 interpretations and the consensus.
  </output_format>
</system_directive>
""".strip()


def build_dcot_user_template() -> str:
    """User template for DCoT extraction."""
    return """
<document_to_analyze>
{{text}}
</document_to_analyze>

<task>
Generate THREE divergent interpretations (Conservative, Moderate, Aggressive).
Each interpretation should independently extract spans.
Then compute the consensus (spans appearing in ≥2 interpretations).

IMPORTANT: Make the interpretations genuinely DIFFERENT. 
The aggressive interpretation should find spans that conservative misses.
</task>

{{few_shot_examples}}
"""


# ===========================================================================
# 3. SELF-REFINE (Currently Partially Implemented)
# ===========================================================================
"""
Self-Refine ANALYSIS FOR S1:

CURRENT IMPLEMENTATION:
- Generator → Critic → Refiner chain
- Critic checks verbatim, granularity, labels
- Refiner applies fixes

PROS:
✅ Catches hallucinations (Critic checks verbatim)
✅ Fixes granularity issues ("approved" → "approved the highly expensive prices")
✅ Corrects label errors

CONS:
- 3 LLM calls per sample (token cost)
- Critic may miss systematic errors
- Refiner may introduce new errors

VERDICT: ★★★★ KEEP but OPTIMIZE

OPTIMIZATION:
- Add "EXHAUSTIVENESS" check to Critic
- Critic should flag MISSING spans, not just wrong ones
- This addresses the recall problem
"""


class EnhancedS1Critique(BaseModel):
    """Enhanced Critic schema with exhaustiveness check."""
    
    # Existing checks
    verbatim_errors: List[str] = Field(
        description="Spans not found exactly in text"
    )
    granularity_errors: List[str] = Field(
        description="Spans that are too short or incomplete"
    )
    label_errors: List[str] = Field(
        description="Spans with wrong label assignments"
    )
    
    # NEW: Exhaustiveness check
    missed_spans: List[Dict[str, str]] = Field(
        description="Spans that SHOULD have been extracted but weren't. Include label and approximate text."
    )
    
    # Decision
    requires_refinement: bool


def build_enhanced_critic_system() -> str:
    """Enhanced Critic with exhaustiveness checking."""
    return """
<system_directive>
  <role>
    You are a **Forensic Auditor** with TWO responsibilities:
    1. QUALITY: Check for errors in extracted spans
    2. EXHAUSTIVENESS: Check for MISSING spans
  </role>

  <quality_checklist>
    1. **VERBATIM:** Is each span exactly in the source text?
    2. **GRANULARITY:** Are Actions full verb phrases, not just single verbs?
    3. **LABELS:** Is each span correctly labeled?
  </quality_checklist>

  <exhaustiveness_checklist>
    READ THE TEXT AGAIN. Ask yourself:
    
    1. **ACTORS:** Are there entities mentioned that perform actions? Did we extract them ALL?
       - Look for: "The government", "They", "Big Pharma", institutions, collectives
       
    2. **ACTIONS:** What are these actors DOING? Did we capture all verbs?
       - Look for: verbs of control, deception, harm
       
    3. **EFFECTS:** What are the CONSEQUENCES mentioned? 
       - Look for: "to cause X", "resulting in Y", "for the purpose of Z"
       
    4. **VICTIMS:** Who is affected?
       - Look for: "the people", "citizens", "workers", "children"
       
    5. **EVIDENCE:** What sources or proofs are cited?
       - Look for: "the video", "studies show", "leaked documents"
  </exhaustiveness_checklist>

  <output_format>
    Return specific errors AND missing spans.
    If the draft is perfect and complete, return empty lists.
  </output_format>
</system_directive>
""".strip()


# ===========================================================================
# 4. REVERSE EXTRACTION (Over-Extract → Prune)
# ===========================================================================
"""
Reverse Extraction ANALYSIS FOR S1:

PRINCIPLE: "It's easier to remove excess than to add missing"

APPROACH:
1. PHASE 1: Extract ALL possible candidates (over-extract)
2. PHASE 2: Apply negative constraints to prune

PROS:
✅ Directly addresses recall problem
✅ Explicit exclusion reasoning (transparent)
✅ Aligns with competition's overlap-based metric

CONS:
- May generate more tokens for candidate list
- Requires good exclusion criteria

VERDICT: ★★★★★ HIGHLY RECOMMENDED
- Best for maximizing recall
- Combine with DCoT for best results
"""


class SpanLabel(str, Enum):
    Actor = "Actor"
    Action = "Action"
    Effect = "Effect"
    Victim = "Victim"
    Evidence = "Evidence"


class CandidateSpan(BaseModel):
    """A span candidate with keep/exclude annotation."""
    label: SpanLabel
    text: str = Field(description="Verbatim text from document")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence")
    keep: bool = Field(description="True to keep, False to exclude")
    exclusion_reason: Optional[str] = Field(
        default=None,
        description="If keep=False, why? E.g., 'pronoun_only', 'too_generic', 'author_opinion'"
    )


class ReverseExtraction(BaseModel):
    """
    Reverse Extraction: Over-extract then prune with explicit reasoning.
    """
    # Phase 1: All candidates
    all_candidates: List[CandidateSpan] = Field(
        description="ALL potential spans including borderline cases"
    )
    
    # Phase 2: Exclusion summary
    exclusion_summary: str = Field(
        description="Summary of pruning decisions"
    )
    
    # Final output
    final_spans: List[CandidateSpan] = Field(
        description="Only spans where keep=True"
    )


def build_reverse_extraction_system() -> str:
    """System prompt for Reverse Extraction."""
    return """
<system_directive>
  <role>
    You are a **Forensic Extractor** using the **Reverse Extraction** method.
    Principle: "Over-extract, then prune with explicit reasoning."
  </role>

  <phase_1_overextract>
    **GOAL:** Extract ALL potential markers. Include borderline cases.
    
    For each marker type, ask:
    - ACTOR: Who/what performs actions? Include vague references ("they", "the elite")
    - ACTION: What verbs indicate activity? Include all candidates
    - EFFECT: What outcomes are mentioned? Include implicit consequences
    - VICTIM: Who is affected? Include implied victims
    - EVIDENCE: What sources are cited? Include rhetorical appeals
    
    **BIAS:** When uncertain, INCLUDE the span (we'll prune later)
  </phase_1_overextract>

  <phase_2_prune>
    **GOAL:** Remove invalid candidates with EXPLICIT reasoning.
    
    Apply these EXCLUSION RULES:
    
    | Code | Rule | Example |
    |------|------|---------|
    | PRONOUN_ONLY | Bare pronoun without context | "it", "they" (alone) |
    | TOO_GENERIC | Generic term, no specificity | "the problem", "things" |
    | AUTHOR_OPINION | Author's stance, not marker | "I think", "in my view" |
    | NOT_VERBATIM | Text not exactly in source | Paraphrased content |
    | WRONG_LABEL | Correct text, wrong category | "The Media" as Evidence |
    | INCOMPLETE | Truncated phrase | "and the", partial verb |
    
    For each candidate, mark:
    - keep=True → Include in final output
    - keep=False + exclusion_reason → Explain why excluded
  </phase_2_prune>

  <output_format>
    Return:
    1. all_candidates: Complete list with annotations
    2. exclusion_summary: Brief summary of pruning decisions  
    3. final_spans: Only keep=True spans
  </output_format>
</system_directive>

{{few_shot_examples}}
""".strip()


# ===========================================================================
# 5. LABEL-GUIDED EXTRACTION (One Label at a Time)
# ===========================================================================
"""
Label-Guided ANALYSIS FOR S1:

APPROACH:
- Extract one label type at a time (5 passes: Actor, Action, Effect, Victim, Evidence)
- Model focuses on single category per pass

PROS:
✅ Reduces cognitive load on model
✅ Better precision per category
✅ Can use label-specific prompts

CONS:
- 5x LLM calls (token expensive)
- May miss label interactions (Actor performs Action)
- Harder to maintain context across passes

VERDICT: ★★★★ GOOD but token-heavy
- Use when specific labels have low F1
- Good for debugging which labels are problematic
"""


class LabelGuidedExtraction(BaseModel):
    """Extract spans for a SINGLE label type."""
    target_label: SpanLabel = Field(description="The label being extracted")
    spans: List[Dict[str, str]] = Field(description="All spans of this label type")
    confidence_notes: str = Field(description="Notes on extraction confidence")


def build_label_guided_system(label: SpanLabel) -> str:
    """Generate label-specific system prompt."""
    
    label_guidance = {
        SpanLabel.Actor: """
    **ACTOR EXTRACTION GUIDE**
    You are extracting ACTORS: entities that perform actions.
    
    INCLUDE:
    - Named entities (CIA, Government, Media)
    - Collective references (They, The Elite, Big Pharma)
    - Institutions with agency (The System, The Establishment)
    - Abstract entities acting as agents (The Law, The Policy)
    
    EXCLUDE:
    - Bare pronouns without referent ("it" alone)
    - Generic nouns ("the thing", "stuff")
""",
        SpanLabel.Action: """
    **ACTION EXTRACTION GUIDE**
    You are extracting ACTIONS: what actors DO.
    
    INCLUDE:
    - Verb phrases indicating control (manipulate, engineer, suppress)
    - Phrases with objects (suppressed the truth, engineered the crisis)
    - Passive constructions (was covered up, being hidden)
    
    EXCLUDE:
    - Single copula verbs (is, was, are) without predicate
    - State verbs (exists, remains) unless with malicious intent
""",
        SpanLabel.Effect: """
    **EFFECT EXTRACTION GUIDE**
    You are extracting EFFECTS: outcomes and consequences.
    
    INCLUDE:
    - Purpose clauses (to control the population, for profit)
    - Result phrases (causing deaths, leading to...)
    - Goal statements (ultimate goal is...)
    
    EXCLUDE:
    - Abstract concepts without outcome framing
""",
        SpanLabel.Victim: """
    **VICTIM EXTRACTION GUIDE**
    You are extracting VICTIMS: entities negatively affected.
    
    INCLUDE:
    - Group references (the people, citizens, workers)
    - Specific victim categories (children, the elderly)
    - Pronoun references to affected groups (us, we)
    
    EXCLUDE:
    - Actors who are also victims (handle as Actor)
""",
        SpanLabel.Evidence: """
    **EVIDENCE EXTRACTION GUIDE**
    You are extracting EVIDENCE: sources and proofs cited.
    
    INCLUDE:
    - Document references (the video, leaked files, studies)
    - Epistemic claims (the proof, the truth, evidence shows)
    - Source attributions (according to X, reports say)
    
    EXCLUDE:
    - URLs or [URL] placeholders
    - Generic "something" references
"""
    }
    
    return f"""
<system_directive>
  <role>
    You are a **Specialized Extractor** for ONE label type: {label.value}
  </role>

  {label_guidance.get(label, "")}

  <output_format>
    Return ONLY {label.value} spans. Do not extract other label types.
    Each span must be a verbatim substring from the text.
  </output_format>
</system_directive>
""".strip()


# ===========================================================================
# 6. LEAST-TO-MOST (Decomposition)
# ===========================================================================
"""
Least-to-Most ANALYSIS FOR S1:

APPROACH:
- Break text into smaller chunks/sentences
- Extract from each chunk independently
- Merge results

PROS:
✅ Handles long documents well
✅ Less likely to miss spans in the middle
✅ Parallelizable

CONS:
- Loses cross-sentence context
- May extract partial spans at chunk boundaries
- Merging logic needed

VERDICT: ★★★★ GOOD for long texts
- Use when documents exceed context window comfort zone
- Combine with span boundary snapping
"""


class ChunkExtraction(BaseModel):
    """Extraction from a single chunk."""
    chunk_id: int
    chunk_text: str
    spans: List[Dict[str, str]]


class LeastToMostExtraction(BaseModel):
    """Least-to-Most: Extract from chunks then merge."""
    chunks: List[ChunkExtraction]
    merged_spans: List[Dict[str, str]] = Field(
        description="De-duplicated and boundary-adjusted spans"
    )


# ===========================================================================
# 7. SKELETON-OF-THOUGHT (Outline → Fill)
# ===========================================================================
"""
Skeleton-of-Thought ANALYSIS FOR S1:

APPROACH:
- First pass: Identify LOCATIONS where spans exist
- Second pass: Extract exact text for each location

PROS:
✅ Fast first pass (just mark positions)
✅ Good for understanding document structure
✅ Reduces hallucination (position-first)

CONS:
- Two passes required
- Position marking is not natural for LLMs

VERDICT: ★★★ MODERATE
- Novel approach but not proven for NER
- Could work with character position marking
"""


class SpanSkeleton(BaseModel):
    """Skeleton: Just identify WHERE spans are."""
    label: SpanLabel
    approximate_location: str = Field(
        description="Quote 5-10 words around the span"
    )


class SkeletonOfThoughtExtraction(BaseModel):
    """Skeleton-of-Thought: Position first, then extract."""
    skeleton: List[SpanSkeleton] = Field(
        description="Rough locations of spans"
    )
    final_spans: List[Dict[str, str]] = Field(
        description="Exact verbatim spans from skeleton positions"
    )


# ===========================================================================
# FINAL RECOMMENDATION: OPTIMAL S1 PIPELINE
# ===========================================================================
"""
RECOMMENDED S1 PIPELINE (Ranked by effectiveness):

┌─────────────────────────────────────────────────────────────────────────┐
│                     OPTIMAL S1 ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. RAG Retrieval (Few-shots) - DYNAMIC SELECTION                       │
│     └── Stratified: Based on text complexity & narrative type           │
│     └── Include contrastive examples (IS/IS-NOT pairs)                  │
│                                                                         │
│  2. DD-CoT GENERATION (Discriminative Reasoning)                        │
│     └── Dynamic assessment (complexity, narrative type)                 │
│     └── For each span: WHY this label, WHY NOT alternatives             │
│                                                                         │
│  3. ENHANCED CRITIC (Exhaustiveness + Discrimination Check)             │
│     └── Quality errors + MISSING spans flagged                          │
│     └── Label confusion check (Actor↔Victim, Action↔Effect)             │
│                                                                         │
│  4. REFINER (Targeted Fixes)                                            │
│     └── Apply critic feedback with discrimination reasoning             │
│                                                                         │
│  5. STRUCTURE VERIFIER                                                  │
│     └── Map text → (start, end) indices                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

ALTERNATIVE: DD-CoT + DCoT HYBRID
┌─────────────────────────────────────────────────────────────────────────┐
│  1. DCoT: Generate 3 interpretations (Conservative/Moderate/Aggressive) │
│  2. DD-CoT: For consensus spans, add discriminative reasoning           │
│  3. Critic + Refiner                                                    │
└─────────────────────────────────────────────────────────────────────────┘

TOKEN COMPARISON:
- Current (k=3 ensemble): 3 × (Gen + Critic + Refiner) = 9 calls
- DD-CoT: 1 × DD-CoT + 1 × Critic + 1 × Refiner = 3 calls
- Savings: 66% fewer LLM calls, BETTER quality via discrimination

WHEN TO USE EACH TECHNIQUE:

| Scenario | Recommended Technique |
|----------|----------------------|
| High label confusion (Actor↔Victim) | DD-CoT (discriminative) |
| Low recall, need more spans | DCoT (divergent, aggressive interpretation) |
| Long documents | Least-to-Most + DD-CoT per chunk |
| Debugging extraction logic | Reverse Extraction (explicit pruning) |
| Balanced production use | DD-CoT + Enhanced Critic |
"""


# ===========================================================================
# UTILITY: Technique Selection Helper
# ===========================================================================

def get_s1_technique_recommendation(
    avg_recall: float = 0.5,
    avg_precision: float = 0.5,
    label_confusion_rate: float = 0.0,  # NEW: How often are labels swapped?
    token_budget: str = "medium",  # low, medium, high
    document_length: str = "short"  # short, medium, long
) -> str:
    """
    Returns the recommended S1 technique based on current performance and constraints.
    """
    
    if label_confusion_rate > 0.2:
        # High label confusion → DD-CoT is essential
        return "DD-CoT (Dynamic Discriminative) - discrimination reduces label swaps"
    
    if avg_recall < 0.3:
        # Severe recall problem
        if token_budget == "high":
            return "DCoT (Aggressive) + DD-CoT refinement"
        else:
            return "DD-CoT with recall bias + Enhanced Critic"
    
    if avg_precision < 0.3:
        # Severe precision problem
        return "DD-CoT (Conservative) + Strong Critic + Self-Refine"
    
    if document_length == "long":
        return "Least-to-Most (Chunking) + DD-CoT per chunk"
    
    if token_budget == "low":
        return "Single-pass DD-CoT (no ensemble)"
    
    # Balanced case
    return "DD-CoT + Enhanced Critic + Self-Refine"


if __name__ == "__main__":
    print("=" * 60)
    print("S1 PROMPTING TECHNIQUE RECOMMENDATIONS")
    print("=" * 60)
    
    # Example usage
    recommendation = get_s1_technique_recommendation(
        avg_recall=0.08,  # Your current recall is very low
        avg_precision=0.12,
        label_confusion_rate=0.15,  # Estimated Actor↔Victim confusion
        token_budget="medium",
        document_length="short"
    )
    
    print(f"\nGiven your current performance (Recall: 0.08, Precision: 0.12):")
    print(f"RECOMMENDED: {recommendation}")
    
    print("\n" + "=" * 60)
    print("TECHNIQUE SUMMARY")
    print("=" * 60)
    print("""
    🥇 DD-CoT (Dynamic Discriminative) ★★★★★
       - Best for: Label discrimination, reducing confusion
       - Key: "WHY this label AND WHY NOT others"
       - Aligns with RAG dynamic few-shot selection
       
    🥈 Divergent CoT (DCoT) ★★★★★
       - Best for: Maximizing recall via diversity
       - 3 interpretations in ONE call
       
    🥉 Reverse Extraction ★★★★★
       - Best for: Explicit over-extract → prune
       - Good for debugging
       
    4. Enhanced Self-Refine ★★★★
       - Best for: Precision improvement
       - Add exhaustiveness + discrimination check
       
    5. Label-Guided ★★★★
       - Best for: Debugging specific label issues
       - One label at a time (5 passes)
       
    6. Tree-of-Thought ★★★
       - Best for: High-value single documents
       - Too expensive for batch (NOT recommended)
    
    ─────────────────────────────────────────────────
    KEY INSIGHT: DD-CoT vs DCoT
    ─────────────────────────────────────────────────
    • DCoT generates DIVERSE extractions (more recall)
    • DD-CoT generates DISCRIMINATED extractions (less label confusion)
    
    For S1 with 5 overlapping label types, DD-CoT is preferred
    because Actor↔Victim and Action↔Effect confusions hurt F1.
    """)
