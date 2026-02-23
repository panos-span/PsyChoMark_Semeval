"""
psycomark.evaluation.metrics — S1 and S2 Evaluation Metrics.

S1: Macro Overlap F1-Score
    - IoU threshold: 0.5 (character-level intersection-over-union)
    - 5 fixed categories: Actor, Action, Effect, Evidence, Victim
    - Greedy matching per category; macro-averaged across all 5

S2: Binary classification
    - ``normalize_label`` maps raw strings to ``conspiracy`` / ``non``
    - Final metric computed externally via ``sklearn.metrics.f1_score``
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# S2 Label Normalisation
# ---------------------------------------------------------------------------


def normalize_label(label: Any) -> str:
    """Normalise an S2 label string to 'conspiracy' or 'non'."""
    s = str(label).lower().strip().replace(".", "")
    if s in ("yes", "true", "conspiracy", "conspiracy theory", "1"):
        return "conspiracy"
    if s in ("no", "false", "non", "not conspiracy", "0"):
        return "non"
    return "ambiguous"


# ---------------------------------------------------------------------------
# S1 Evaluator
# ---------------------------------------------------------------------------


class S1Evaluator:
    """
    Strict implementation of the SemEval-2025 Task 10 *Macro Overlap F1-Score*.

    Parameters
    ----------
    iou_threshold : float
        Minimum IoU for a predicted span to count as a true positive (default 0.5).
    """

    SCHEMA_LABELS = {"Actor", "Action", "Effect", "Evidence", "Victim"}

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.tp: Dict[str, int] = {k: 0 for k in self.SCHEMA_LABELS}
        self.fp: Dict[str, int] = {k: 0 for k in self.SCHEMA_LABELS}
        self.fn: Dict[str, int] = {k: 0 for k in self.SCHEMA_LABELS}

    # ----- helpers -----

    @staticmethod
    def compute_iou(span_a: dict, span_b: dict) -> float:
        """Character-level Intersection-over-Union."""
        s_a, e_a = span_a["start"], span_a["end"]
        s_b, e_b = span_b["start"], span_b["end"]

        inter_s = max(s_a, s_b)
        inter_e = min(e_a, e_b)
        if inter_e <= inter_s:
            return 0.0

        inter = inter_e - inter_s
        union = (e_a - s_a) + (e_b - s_b) - inter
        return inter / union if union > 0 else 0.0

    def normalize_span(self, s: dict) -> Optional[dict]:
        """Normalise a span dict to ``{start, end, type}``."""
        start = s.get("startIndex") or s.get("start")
        end = s.get("endIndex") or s.get("end")
        label = s.get("type") or s.get("label")

        if start is None or end is None or not label:
            return None

        clean = str(label).strip().capitalize()
        if clean not in self.SCHEMA_LABELS:
            return None

        return {"start": int(start), "end": int(end), "type": clean}

    # ----- core -----

    def update(self, predictions: List[Dict], ground_truth: List[Dict]):
        """Accumulate TP / FP / FN for one document."""
        pred_map: Dict[str, list] = defaultdict(list)
        gt_map: Dict[str, list] = defaultdict(list)

        for p in predictions:
            norm = self.normalize_span(p)
            if norm:
                pred_map[norm["type"]].append(norm)

        for g in ground_truth:
            norm = self.normalize_span(g)
            if norm:
                gt_map[norm["type"]].append(norm)

        for label in self.SCHEMA_LABELS:
            preds = pred_map[label]
            golds = gt_map[label]
            matched: set[int] = set()

            for g in golds:
                best_iou, best_idx = 0.0, -1
                for i, p in enumerate(preds):
                    if i in matched:
                        continue
                    iou = self.compute_iou(p, g)
                    if iou >= self.iou_threshold and iou > best_iou:
                        best_iou, best_idx = iou, i

                if best_idx != -1:
                    self.tp[label] += 1
                    matched.add(best_idx)
                else:
                    self.fn[label] += 1

            self.fp[label] += len(preds) - len(matched)

    def get_macro_f1(self) -> Dict[str, float]:
        """Return per-label F1 scores and unweighted macro average."""
        f1_scores: list[float] = []
        metrics: Dict[str, float] = {}

        for label in self.SCHEMA_LABELS:
            tp, fp, fn = self.tp[label], self.fp[label], self.fn[label]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            metrics[f"{label}_f1"] = f1
            f1_scores.append(f1)

        metrics["macro_f1"] = sum(f1_scores) / 5.0
        return metrics

    def reset(self):
        """Reset all counters."""
        for k in self.SCHEMA_LABELS:
            self.tp[k] = self.fp[k] = self.fn[k] = 0
