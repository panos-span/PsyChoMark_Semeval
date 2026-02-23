#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psycomark.run_lite — Lite Pipeline Runner for Local Models.

Separate from run.py — uses simplified schemas and concurrent processing.

Usage:
    python -m psycomark.run_lite --data dev_rehydrated.jsonl
    python -m psycomark.run_lite --data dev_rehydrated.jsonl --concurrency 3 --timeout 600
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


# ---------------------------------------------------------------------------
# I/O Helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read all JSON objects from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(path: str, record: dict):
    """Append a single JSON record to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Lite Document Processor
# ---------------------------------------------------------------------------


async def process_document_lite(
    row: Dict[str, Any],
    task: str,
    timeout: int = 600,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Process a single document through the lite S1 and/or S2 pipelines.

    Returns:
        (s1_record, s2_record) — either may be None depending on ``task``.
    """
    from psycomark.agents.s1_agents_lite import run_s1_lite
    from psycomark.agents.s2_agents_lite import (
        run_s2_lite_council,
        run_s2_lite_judge,
    )

    doc_id = row.get("id") or row.get("_id", "unknown")
    text = row["text"]

    # Support both nested and flat JSONL layouts
    meta = row.get("metadata", {})
    if not meta:
        meta = {}
        if "conspiracy" in row and row["conspiracy"] is not None:
            meta["label"] = row["conspiracy"]
        if "markers" in row and row["markers"] is not None:
            meta["gt_markers"] = row["markers"]
        if "subreddit" in row:
            meta["subreddit"] = row["subreddit"]

    if len(text) < 10:
        return None, None

    s1_record = None
    s1_spans_for_s2: list[dict] = []
    s1_dossier = ""

    # ---- S1 Extraction (Lite) ----
    if task in ("s1", "both"):
        try:
            s1_spans = await asyncio.wait_for(
                run_s1_lite(text, metadata=meta),
                timeout=timeout,
            )

            markers = [
                {
                    "type": s.label.value if hasattr(s.label, "value") else str(s.label),
                    "text": s.text,
                    "startIndex": s.start or 0,
                    "endIndex": s.end or len(s.text),
                }
                for s in s1_spans
            ]

            s1_record = {"_id": doc_id, "markers": markers}
            s1_spans_for_s2 = markers
            s1_dossier = (
                f"Found {len(markers)} markers: "
                + ", ".join(f'{m["type"]}="{m["text"][:30]}"' for m in markers[:5])
                if markers
                else "No markers found."
            )

        except asyncio.TimeoutError:
            logger.warning(f"[{doc_id}] S1 timeout")
            s1_record = {"_id": doc_id, "markers": [], "error": "Timeout"}
        except Exception as e:
            logger.error(f"[{doc_id}] S1 error: {e}")
            s1_record = {"_id": doc_id, "markers": [], "error": str(e)}

    # ---- S2 Classification (Lite) ----
    s2_record = None
    if task in ("s2", "both"):
        try:
            # Lite council (2 jurors)
            council = await asyncio.wait_for(
                run_s2_lite_council(
                    text=text,
                    s1_spans=s1_spans_for_s2,
                    marker_summary=s1_dossier,
                    metadata=meta,
                ),
                timeout=timeout,
            )

            # Lite judge
            judge_output = await asyncio.wait_for(
                run_s2_lite_judge(
                    text=text,
                    council_result=council,
                    doc_id=doc_id,
                    metadata=meta,
                ),
                timeout=timeout,
            )

            s2_record = {
                "_id": doc_id,
                "conspiracy": (
                    "Yes" if judge_output.label.lower() == "conspiracy" else "No"
                ),
                "confidence": judge_output.confidence,
                "rationale": judge_output.rationale,
            }

        except asyncio.TimeoutError:
            logger.warning(f"[{doc_id}] S2 timeout")
            s2_record = {"_id": doc_id, "conspiracy": "No", "error": "Timeout"}
        except Exception as e:
            logger.error(f"[{doc_id}] S2 error: {e}")
            s2_record = {"_id": doc_id, "conspiracy": "No", "error": str(e)}

    return s1_record, s2_record


# ---------------------------------------------------------------------------
# Main (Concurrent)
# ---------------------------------------------------------------------------


async def main_async():
    parser = argparse.ArgumentParser(description="PsyCoMark Lite Inference Runner")
    parser.add_argument("--data", required=True, help="Path to input JSONL")
    parser.add_argument("--task", default="both", choices=["s1", "s2", "both"])
    parser.add_argument("--s1-out", default="submission_s1_lite.jsonl")
    parser.add_argument("--s2-out", default="submission_s2_lite.jsonl")
    parser.add_argument("--concurrency", type=int, default=3, help="Max concurrent docs")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per step (sec)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    data = load_jsonl(args.data)
    logger.info(f"Loaded {len(data)} documents from {args.data}")
    logger.info(f"Concurrency={args.concurrency}, Timeout={args.timeout}s")

    # Resume logic
    done_ids: set[str] = set()
    if args.resume:
        for path in (args.s1_out, args.s2_out):
            if Path(path).exists():
                for rec in load_jsonl(path):
                    done_ids.add(rec.get("_id", ""))
        logger.info(f"Resuming: {len(done_ids)} already done")

    # Filter to pending documents
    pending = []
    for i, row in enumerate(data):
        doc_id = row.get("id") or row.get("_id") or row.get("doc_id") or f"doc_{i}"
        if doc_id not in done_ids:
            pending.append((i, row))

    logger.info(f"Processing {len(pending)} documents")

    # Evaluation state (thread-safe via single-writer)
    from psycomark.evaluation.metrics import S1Evaluator, normalize_label

    s1_eval = S1Evaluator()
    s2_y_true, s2_y_pred = [], []
    completed = 0

    # Semaphore for concurrency control
    sem = asyncio.Semaphore(args.concurrency)

    async def _process_one(idx: int, row: Dict) -> None:
        nonlocal completed
        doc_id = row.get("id") or row.get("_id") or row.get("doc_id") or f"doc_{idx}"

        async with sem:
            s1_res, s2_res = await process_document_lite(
                row, args.task, timeout=args.timeout
            )

        # Write results (sequential to avoid file corruption)
        if s1_res:
            append_jsonl(args.s1_out, s1_res)
            # Support both nested and flat JSONL for gt_markers
            meta = row.get("metadata", {})
            gt_markers = meta.get("gt_markers") or meta.get("markers") or row.get("markers")
            if gt_markers:
                s1_eval.update(s1_res.get("markers", []), gt_markers)

        if s2_res:
            append_jsonl(args.s2_out, s2_res)
            meta = row.get("metadata", {})
            raw_gt = meta.get("label") or row.get("label") or row.get("conspiracy")

            
            # Only evaluate if we have a valid ground truth
            if raw_gt is not None:
                gt_label = normalize_label(str(raw_gt))
                pred_label = normalize_label(s2_res.get("conspiracy", "No"))
                
                if gt_label != "ambiguous":
                    s2_y_true.append(gt_label)
                    s2_y_pred.append(pred_label)

        completed += 1
        
        # Safe metric calculation
        s1_f1 = s1_eval.get_macro_f1()["macro_f1"]
        s2_f1 = 0.0
        if s2_y_true:
            s2_acc = (
                sum(1 for t, p in zip(s2_y_true, s2_y_pred) if t == p)
                / len(s2_y_true)
            )
            try:
                from sklearn.metrics import f1_score
                s2_f1 = f1_score(s2_y_true, s2_y_pred, average="macro")
            except ImportError:
                pass
        else:
            s2_acc = 0.0

        if doc_id.startswith("doc_"):
            logger.warning(f"Could not find ID for row keys: {list(row.keys())}")

        logger.info(
            f"[{completed}/{len(pending)}] {doc_id} | "
            f"S1 F1: {s1_f1:.1%} | S2 Acc: {s2_acc:.1%} | S2 F1: {s2_f1:.1%}"
        )

    # Launch all tasks concurrently (semaphore controls parallelism)
    tasks = [_process_one(idx, row) for idx, row in pending]
    await asyncio.gather(*tasks, return_exceptions=True)

    # Final summary
    logger.info("=" * 60)
    logger.info("FINAL RESULTS")
    logger.info(f"  S1 Macro Overlap F1: {s1_eval.get_macro_f1()}")
    if s2_y_true:
        try:
            from sklearn.metrics import f1_score, classification_report

            logger.info(
                f"  S2 Weighted F1: {f1_score(s2_y_true, s2_y_pred, average='weighted'):.4f}"
            )
            logger.info(f"\n{classification_report(s2_y_true, s2_y_pred)}")
        except ImportError:
            s2_acc = sum(1 for t, p in zip(s2_y_true, s2_y_pred) if t == p) / max(
                len(s2_y_true), 1
            )
            logger.info(f"  S2 Accuracy: {s2_acc:.4f} (install sklearn for F1)")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
