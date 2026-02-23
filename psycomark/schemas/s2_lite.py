"""
psycomark.schemas.s2_lite — Lite Pydantic Schemas for S2 (Local Models).

Minimal schemas optimized for small open-source models (e.g. Qwen3-8B)
that struggle with complex tool-calling output schemas.

Compared to the full EnhancedS2Vote/CalibratedJudgeOutput:
    - 3 fields per vote instead of 7
    - No steelman/uncertainty/dissent fields
    - No key_evidence lists
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LiteVote(BaseModel):
    """Minimal juror vote for local models with limited tool-calling fidelity."""

    verdict: Literal["conspiracy", "non"] = Field(
        description="Independent verdict based on the evidence."
    )
    confidence: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Certainty score."
    )
    rationale: str = Field(
        default="", description="One-sentence explanation."
    )


class LiteJudgeOutput(BaseModel):
    """Minimal judge output for local models."""

    label: Literal["conspiracy", "non"] = Field(
        description="Final verdict."
    )
    confidence: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Calibrated confidence."
    )
    rationale: str = Field(
        default="", description="Explanation of the verdict."
    )
