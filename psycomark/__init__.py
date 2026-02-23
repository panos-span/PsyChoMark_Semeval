"""
PsyCoMark: Agentic LLM Architecture for Psycholinguistic Conspiracy Marker
Extraction and Endorsement Detection.

SemEval-2025 Task 10 — ACL Submission

Architecture:
    S1 (Marker Span Extraction):
        DD-CoT Generator → Enhanced Critic → Refiner → Deterministic Verifier

    S2 (Endorsement Classification):
        Forensic Profiler → Parallel Council (4 jurors) → Calibrated Judge

Key Contributions:
    1. Dynamic Discriminative Chain-of-Thought (DD-CoT)
    2. Self-Refine pipeline (Generator → Critic → Refiner)
    3. Anti-Echo Chamber parallel council for robust classification
    4. Contrastive Retrieval-Augmented Generation with hard negatives
"""

__version__ = "1.0.0"
__author__ = "PsyCoMark Team"
