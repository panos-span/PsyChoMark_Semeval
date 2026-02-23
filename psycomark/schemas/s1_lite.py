"""
psycomark.schemas.s1_lite — Lite Pydantic Schemas for S1 (Local Models).

Minimal schemas optimized for small open-source models (e.g. Qwen3-8B)
that struggle with complex tool-calling output schemas.

Compared to the full DDCoTSpan/DDCoTExtraction:
    - 3 fields per span instead of 10
    - Uses Literal instead of Enum for labels (better tool-call compat)
    - No nested dicts (why_not_other_labels)
    - No optional context anchors
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class LiteSpan(BaseModel):
    """Minimal span for local models with limited tool-calling fidelity."""

    text: str = Field(..., description="Verbatim span from the document.")
    label: Literal["Actor", "Action", "Effect", "Victim", "Evidence"] = Field(
        ..., description="Functional role of the span."
    )
    why: str = Field(default="", description="Brief reason for the label.")
    start: Optional[int] = None
    end: Optional[int] = None


class LiteExtraction(BaseModel):
    """Simplified extraction output for local models."""

    extractions: List[LiteSpan] = Field(default_factory=list)
