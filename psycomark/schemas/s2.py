"""
psycomark.schemas.s2 — Pydantic Schemas for Subtask 2 (Endorsement Classification).

Defines the structured output contracts for the Anti-Echo Chamber pipeline:
    - S2Output: Final binary verdict (conspiracy | non)
    - EnhancedS2Vote: Independent juror vote with anti-echo safeguards
    - ParallelCouncilOutput: Aggregated parallel council results
    - CalibratedJudgeOutput: Dissent-aware final adjudication
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Core Enums
# ---------------------------------------------------------------------------


class S2Juror(str, Enum):
    """Four adversarial personas composing the parallel council."""

    LITERALIST = "Literalist"  # Strict burden-of-proof (acquits hearsay)
    BELIEVER = "Believer"  # High recall / Prosecutor (flags implicit support)
    PROFILER = "Profiler"  # Psycholinguistic analysis (flags Us-vs-Them tone)
    DEFENSE = "Defense"  # Hanlon's Razor (flags incompetence vs. malice)


# ---------------------------------------------------------------------------
# Runtime Dependencies
# ---------------------------------------------------------------------------


class S2Deps(BaseModel):
    """Runtime dependency injection for S2 agents."""

    model_config = ConfigDict(extra="ignore")

    raw_text: str
    doc_id: Optional[str] = None

    # Forensic evidence from S1
    s1_markers: List[Dict[str, Any]] = Field(default_factory=list)
    marker_summary: Optional[str] = None

    # Contextual signals (e.g., ``{"subreddit": "conspiracy"}``)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Retrieved legal precedents for the Judge
    rag_context: str = Field(default="")


# ---------------------------------------------------------------------------
# Final Verdict
# ---------------------------------------------------------------------------


class S2Output(BaseModel):
    """
    Final binary classification verdict.

    ``conspiracy`` = the author endorses or promotes a conspiratorial worldview.
    ``non`` = reporting, debunking, satire, or neutral discussion.
    """

    label: Literal["conspiracy", "non"] = Field(
        ..., description="Final classification."
    )
    rationale: str = Field(
        ..., description="State-of-mind explanation: endorsement vs. summary."
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Certainty score.")
    key_evidence: List[str] = Field(
        default_factory=list,
        description="1-3 verbatim substrings proving the author's stance.",
    )


# ---------------------------------------------------------------------------
# Voting Models (Sequential Council — Legacy)
# ---------------------------------------------------------------------------


class S2Vote(BaseModel):
    """Individual juror vote (legacy sequential council)."""

    juror: S2Juror
    verdict: Literal["conspiracy", "non"]
    confidence: float
    rationale: str = Field(..., description="One-sentence explanation.")


class S2CouncilOutput(BaseModel):
    """Aggregated output from the sequential council phase."""

    votes: List[S2Vote]
    tally: Dict[str, int]
    weighted_score: float = Field(
        default=0.0,
        description="Confidence-weighted score: positive=conspiracy, negative=non.",
    )
    debate_summary: str = Field(
        default="",
        description="Summary of prosecutor/defense arguments.",
    )


# ---------------------------------------------------------------------------
# Anti-Echo Chamber Models (Parallel Council)
# ---------------------------------------------------------------------------


class BlindVote(BaseModel):
    """
    Independent vote with NO access to other jurors' decisions.

    Anti-echo guarantee: each juror must also articulate the best
    counter-argument (``alternative_interpretation``).
    """

    verdict: Literal["conspiracy", "non"] = Field(
        description="Independent verdict based ONLY on the evidence."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Certainty: 0.5 = coin flip, 1.0 = certain.",
    )
    key_signal: str = Field(
        description="The SINGLE most important signal driving the decision."
    )
    alternative_interpretation: str = Field(
        description="How could the OTHER side interpret this text?"
    )


class EnhancedS2Vote(BaseModel):
    """Enhanced vote with anti-echo-chamber safeguards."""

    juror: S2Juror
    verdict: Literal["conspiracy", "non"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(description="Main reasoning (2-3 sentences).")

    # Anti-echo fields
    key_signal: str = Field(
        default="", description="Single most decisive piece of evidence."
    )
    steelman_opposing: str = Field(
        default="",
        description="Best argument FOR the opposing verdict.",
    )
    uncertainty_flags: List[str] = Field(
        default_factory=list,
        description="What makes this case borderline?",
    )


class ParallelCouncilOutput(BaseModel):
    """
    Output from parallel (non-sequential) council voting.

    All four jurors vote independently (no information leakage),
    preventing the echo-chamber effect observed in sequential debate.
    """

    votes: List[EnhancedS2Vote]
    tally: Dict[str, int]

    # Per-verdict confidence averages
    conspiracy_confidence_avg: float = Field(default=0.0)
    non_confidence_avg: float = Field(default=0.0)
    weighted_score: float = Field(
        default=0.0,
        description="Aggregate: positive = conspiracy, negative = non.",
    )

    # Dissent analysis (fed to the Calibrated Judge)
    dissent_strength: float = Field(
        default=0.0,
        description="Strength of minority opinion (0 = unanimous, 0.5 = split).",
    )
    consensus_level: Literal["unanimous", "strong", "split", "chaotic"] = Field(
        default="split",
        description="unanimous=4-0, strong=3-1, split=2-2, chaotic=abstentions.",
    )
    common_uncertainty_flags: List[str] = Field(
        default_factory=list,
        description="Flags mentioned by 2+ jurors.",
    )


# ---------------------------------------------------------------------------
# Calibrated Judge
# ---------------------------------------------------------------------------


class CalibratedJudgeOutput(BaseModel):
    """
    Dissent-aware final adjudication output.

    Confidence damping rules:
        - Unanimous verdict: confidence = 0.95
        - 3-1 split: confidence capped at 0.80
        - 2-2 deadlock: confidence capped at 0.65, defaults to 'non'
    """

    label: Literal["conspiracy", "non"] = Field(
        description="Final verdict after weighing all evidence."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Calibrated confidence (lower when council split).",
    )
    rationale: str = Field(
        description="Explanation referencing council votes and dissent.",
    )

    # Calibration fields
    dissent_considered: bool = Field(
        default=False,
        description="Did the Judge explicitly consider minority opinions?",
    )
    key_evidence: List[str] = Field(
        default_factory=list,
        description="1-3 verbatim quotes that sealed the verdict.",
    )
    council_override: bool = Field(
        default=False,
        description="Did the Judge override the council majority?",
    )
    borderline_flag: bool = Field(
        default=False,
        description="Hard case that should be flagged for review.",
    )

