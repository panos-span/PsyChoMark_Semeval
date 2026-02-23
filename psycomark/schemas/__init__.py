"""Pydantic schemas for Subtask 1 (Marker Span Extraction) and Subtask 2 (Classification)."""

from psycomark.schemas.s1 import (
    CritiqueError,
    DDCoTExtraction,
    DDCoTRefinement,
    DDCoTSpan,
    EnhancedS1Critique,
    S1Critique,
    S1Deps,
    S1Label,
    S1PatternExtraction,
    S1PatternSpan,
    S1Reasoning,
    S1Refinement,
    S1Span,
)
from psycomark.schemas.s2 import (
    BlindVote,
    CalibratedJudgeOutput,
    EnhancedS2Vote,
    ParallelCouncilOutput,
    S2CouncilOutput,
    S2Deps,
    S2Juror,
    S2Output,
    S2Vote,
)

__all__ = [
    # S1
    "S1Label",
    "S1Span",
    "S1Deps",
    "DDCoTSpan",
    "DDCoTExtraction",
    "EnhancedS1Critique",
    "DDCoTRefinement",
    "S1Reasoning",
    "S1Critique",
    "S1Refinement",
    "S1PatternSpan",
    "S1PatternExtraction",
    "CritiqueError",
    # S2
    "S2Deps",
    "S2Output",
    "S2Juror",
    "S2Vote",
    "S2CouncilOutput",
    "BlindVote",
    "EnhancedS2Vote",
    "ParallelCouncilOutput",
    "CalibratedJudgeOutput",
]
