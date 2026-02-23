"""
psycomark.evaluation — Evaluation Metrics.

Provides:
    - ``S1Evaluator``: Macro Overlap F1-Score (IoU ≥ 0.5, 5 fixed labels)
    - ``normalize_label``: S2 conspiracy / non normalisation
"""

from psycomark.evaluation.metrics import S1Evaluator, normalize_label

__all__ = ["S1Evaluator", "normalize_label"]
