"""
psycomark.schemas.s1 — Pydantic Schemas for Subtask 1 (Marker Span Extraction).

Defines the structured output contracts for the DD-CoT Self-Refine pipeline:
    - DDCoTSpan: A single span with discriminative reasoning
    - DDCoTExtraction: Generator output (text assessment + spans)
    - EnhancedS1Critique: Multi-dimensional quality audit
    - DDCoTRefinement: Corrected extraction with change log
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Core Enums
# ---------------------------------------------------------------------------


class S1Label(str, Enum):
    """Five psycholinguistic marker categories for conspiracy rhetoric."""

    Actor = "Actor"  # The conspiratorial agent
    Action = "Action"  # The malicious mechanism
    Effect = "Effect"  # The intended outcome
    Victim = "Victim"  # The target of harm
    Evidence = "Evidence"  # Epistemic supports for the claim


# ---------------------------------------------------------------------------
# Span Models
# ---------------------------------------------------------------------------


class S1Span(BaseModel):
    """
    Atomic unit of span extraction.

    ``start`` / ``end`` offsets are computed by the deterministic Verifier,
    not by the LLM.
    """

    label: S1Label

    # Context anchors to disambiguate duplicate spans
    preceding_context: Optional[str] = Field(
        None, description="3-5 words immediately BEFORE the span."
    )
    following_context: Optional[str] = Field(
        None, description="3-5 words immediately AFTER the span."
    )

    text: str = Field(..., description="Exact verbatim substring from the source text.")

    start: Optional[int] = None
    end: Optional[int] = None
    why: Optional[str] = None


class DDCoTSpan(BaseModel):
    """
    Span with **Dynamic Discriminative Chain-of-Thought** reasoning.

    Key innovation: requires explicit articulation of both inclusion
    (``why_this_label``) and exclusion (``why_not_other_labels``) criteria,
    directly addressing label confusion between semantically adjacent
    categories (Actor <-> Victim, Action <-> Effect).
    """

    text: str = Field(..., description="Verbatim span from the document.")
    label: S1Label = Field(
        description="Assigned label (Actor|Action|Effect|Victim|Evidence)."
    )

    # Context anchors
    preceding_context: Optional[str] = Field(
        None, description="3-5 words immediately BEFORE the span."
    )
    following_context: Optional[str] = Field(
        None, description="3-5 words immediately AFTER the span."
    )

    # Verifier-computed offsets
    start: Optional[int] = None
    end: Optional[int] = None

    # Discriminative reasoning (the core DD-CoT innovation)
    why_this_label: str = Field(
        description="Why this span IS this label type (1-2 sentences)."
    )
    action_nucleus: Optional[str] = Field(
        None,
        description="For ACTION labels only: the main verb (e.g., 'rigged').",
    )
    why_not_other_labels: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "For each plausible alternative label, why it is NOT that. "
            "E.g., {'Victim': 'NOT Victim because it performs the action'}."
        ),
    )

    confidence: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Extraction confidence."
    )

    @field_validator("why_not_other_labels", mode="before")
    @classmethod
    def normalize_why_not(cls, v: Any) -> Dict[str, str]:
        """Handle LLM returning a plain string instead of a dict."""
        if isinstance(v, str):
            return {"Alternative": v}
        if v is None:
            return {}
        return v


# ---------------------------------------------------------------------------
# Generator Output
# ---------------------------------------------------------------------------


class DDCoTExtraction(BaseModel):
    """
    DD-CoT Generator output: dynamic context assessment + discriminative
    span extraction.
    """

    text_complexity: Literal["simple", "moderate", "complex"] = Field(
        description="How ambiguous is this text?"
    )
    dominant_narrative: Literal["conspiracy", "neutral", "debunking", "mixed"] = Field(
        description="Dominant discourse type — calibrates extraction strategy."
    )
    extractions: List[DDCoTSpan] = Field(
        description="Extracted spans with discriminative reasoning."
    )


# ---------------------------------------------------------------------------
# Critic Output
# ---------------------------------------------------------------------------


class CritiqueError(BaseModel):
    """A single error identified during critique."""

    span_text: str
    error_type: str  # "verbatim" | "granularity" | "label" | "missed"
    description: str


class EnhancedS1Critique(BaseModel):
    """
    Enhanced Critic output for the DD-CoT pipeline.

    Performs multi-dimensional quality auditing:
        1. Verbatim validation (anti-hallucination)
        2. Granularity check (not too short / too long)
        3. Label consistency (discrimination heuristics)
        4. Exhaustiveness audit (missed spans)
    """

    verbatim_errors: List[str] = Field(
        default_factory=list,
        description="Spans not appearing verbatim in the source text.",
    )
    granularity_errors: List[str] = Field(
        default_factory=list,
        description="Spans that are too short (single-word Actions) or too long.",
    )
    label_errors: List[Union[str, Dict[str, str]]] = Field(
        default_factory=list,
        description="Wrong label assignments.",
    )
    missed_spans: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Spans that should have been extracted but were not.",
    )
    confusion_flags: List[str] = Field(
        default_factory=list,
        description="Label confusions detected (e.g., 'Actor<->Victim on the people').",
    )
    requires_refinement: bool = Field(
        description="True if ANY errors detected and refinement needed."
    )


# ---------------------------------------------------------------------------
# Refiner Output
# ---------------------------------------------------------------------------


class DDCoTRefinement(BaseModel):
    """Corrected extraction preserving discriminative reasoning."""

    refined_extractions: List[DDCoTSpan] = Field(
        description="Corrected list of spans with discriminative reasoning."
    )
    fixes_applied: List[str] = Field(
        default_factory=list,
        description="Change log for debugging.",
    )


# ---------------------------------------------------------------------------
# Legacy Models (backward compatibility)
# ---------------------------------------------------------------------------


class S1Reasoning(BaseModel):
    """Legacy CoT output. Use ``DDCoTExtraction`` for new work."""

    text_type: str = Field(
        description="Brief classification: 'conspiracy_claim', 'neutral_report', etc."
    )
    reasoning: str = Field(description="1-2 sentences explaining extraction strategy.")
    final_spans: List[S1Span] = Field(
        description="Extracted markers (verbatim substrings)."
    )


class S1Critique(BaseModel):
    """Legacy critique model. Use ``EnhancedS1Critique`` for DD-CoT."""

    critiques: List[str] = Field(description="Specific errors found.")
    requires_refinement: bool = Field(description="True if changes needed.")


class S1Refinement(BaseModel):
    """Legacy refinement output. Use ``DDCoTRefinement`` for DD-CoT."""

    final_spans: List[S1Span] = Field(description="Corrected spans.")


# ---------------------------------------------------------------------------
# Pattern Recognition Models (optimized extraction)
# ---------------------------------------------------------------------------


class S1PatternSpan(BaseModel):
    """Optimized atomic span for the pattern-recognition generator."""

    text: str = Field(..., description="Verbatim atomic span from the text.")
    label: Literal["Actor", "Action", "Evidence", "Victim", "Effect"] = Field(
        ..., description="Functional role of the span."
    )
    why_this_label: str = Field(
        ..., description="Explanation of malicious intent or identity found."
    )
    why_not_other_labels: str = Field(
        ..., description="Contrast against natural forces, reporting frames, etc."
    )
    confidence: float = Field(default=1.0)


class S1PatternExtraction(BaseModel):
    """Top-level response for the S1 Pattern Generator."""

    text_complexity: Literal["simple", "moderate", "complex"]
    dominant_narrative: Literal["conspiracy", "debunking", "neutral", "mixed"]
    extractions: List[S1PatternSpan] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Runtime Dependencies
# ---------------------------------------------------------------------------


class S1Deps(BaseModel):
    """Runtime dependency injection for S1 agents (text + few-shots)."""

    model_config = ConfigDict(extra="ignore")

    text: str = Field(..., alias="raw_text")
    doc_id: Optional[str] = None
    few_shots: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
