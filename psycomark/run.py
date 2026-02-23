#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psycomark.run — CLI Entry Point for PsyCoMark Inference.

Usage:
    python -m psycomark.run --task both --data dev_rehydrated.jsonl

This script orchestrates the full S1 → S2 pipeline:
    1. Load data (JSONL)
    2. Optional: initialise RAG collections (ChromaDB)
    3. Process documents through LangGraph workflows
    4. Write JSONL submission files
    5. Print live evaluation metrics
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from psycomark.config import safe_agent_run
from psycomark.evaluation.metrics import S1Evaluator, normalize_label
from psycomark.graphs.s1_graph import s1_graph, S1DDCoTGraphState
from psycomark.graphs.s2_graph import s2_graph, S2ParallelGraphState


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
# Document Processor
# ---------------------------------------------------------------------------


async def process_document(
    row: Dict[str, Any],
    task: str,
    rag_collections: Dict[str, Any],
    rerank: bool = True,
    s2_temp: float = 0.4,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Process a single document through the S1 and/or S2 pipelines.

    Returns:
        (s1_record, s2_record) — either may be None depending on ``task``.
    """
    from psycomark.agents.s2_agents import synthesize_dossier
    from psycomark.rag.retrieval import (
        retrieve_fewshots,
        retrieve_stratified_s1_reranked,
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

    # ---- S1 Extraction ----
    if task in ("s1", "both"):
        try:
            s1_rag = rag_collections.get("s1")
            few_shots = []
            if s1_rag:
                few_shots = (
                    retrieve_stratified_s1_reranked(s1_rag, text)
                    if rerank
                    else retrieve_fewshots(s1_rag, text, k=6)
                )

            s1_state: S1DDCoTGraphState = {
                "doc_id": doc_id,
                "text": text,
                "few_shots": few_shots,
                "metadata": meta,
                "text_complexity": "",
                "dominant_narrative": "",
                "draft_extractions": [],
                "critique": None,
                "requires_refinement": False,
                "refined_extractions": [],
                "final_spans": [],
                "token_usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
            }

            final = await asyncio.wait_for(s1_graph.ainvoke(s1_state), timeout=200)
            verified = final.get("final_spans", [])
            complexity = final.get("text_complexity", "unknown")
            narrative = final.get("dominant_narrative", "unknown")

            markers = [
                {
                    "type": s["label"],
                    "startIndex": s["start"],
                    "endIndex": s["end"],
                    "text": s["text"],
                }
                for s in verified
            ]

            s1_record = {"_id": doc_id, "markers": markers}
            s1_spans_for_s2 = markers
            s1_dossier = synthesize_dossier(
                markers, complexity=complexity, narrative=narrative
            )

        except asyncio.TimeoutError:
            logger.warning(f"[{doc_id}] S1 timeout")
            s1_record = {"_id": doc_id, "markers": [], "error": "Timeout"}
        except Exception as e:
            logger.error(f"[{doc_id}] S1 error: {e}")
            s1_record = {"_id": doc_id, "markers": [], "error": str(e)}

    # ---- S2 Classification ----
    s2_record = None
    if task in ("s2", "both"):
        gt_raw = str(meta.get("label", "")).lower().strip()
        if gt_raw in ("cant_tell", "can't tell", "ambiguous"):
            return s1_record, None

        try:
            s2_rag = rag_collections.get("s2")
            rag_context = ""
            if s2_rag:
                precs = retrieve_fewshots(
                    s2_rag, text, k=4, filters={"is_hard_negative": True}
                )
                if precs:
                    rag_context = json.dumps(precs, indent=2, ensure_ascii=False)

            s2_state: S2ParallelGraphState = {
                "doc_id": doc_id,
                "text": text,
                "s1_markers": s1_spans_for_s2,
                "s1_spans": None,
                "marker_summary": s1_dossier or "No markers.",
                "rag_context": rag_context,
                "juror_temperature": s2_temp,
                "metadata": meta,
                "forensic_stats": {},
                "council_output": None,
                "calibrated_output": None,
                "final_output": None,
                "token_usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
            }

            final = await asyncio.wait_for(s2_graph.ainvoke(s2_state), timeout=200)
            s2_res = final.get("final_output")

            if s2_res:
                s2_record = {
                    "_id": doc_id,
                    "conspiracy": (
                        "Yes" if s2_res.label.lower() == "conspiracy" else "No"
                    ),
                    "confidence": s2_res.confidence,
                    "rationale": s2_res.rationale,
                }
            else:
                s2_record = {"_id": doc_id, "conspiracy": "No", "error": "None result"}

        except asyncio.TimeoutError:
            logger.warning(f"[{doc_id}] S2 timeout")
            s2_record = {"_id": doc_id, "conspiracy": "No", "error": "Timeout"}
        except Exception as e:
            logger.error(f"[{doc_id}] S2 error: {e}")
            s2_record = {"_id": doc_id, "conspiracy": "No", "error": str(e)}

    return s1_record, s2_record


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main_async():
    parser = argparse.ArgumentParser(description="PsyCoMark Inference Runner")
    parser.add_argument("--data", required=True, help="Path to input JSONL")
    parser.add_argument("--task", default="both", choices=["s1", "s2", "both"])
    parser.add_argument("--s1-out", default="submission_s1.jsonl")
    parser.add_argument("--s2-out", default="submission_s2.jsonl")
    parser.add_argument("--s1-rag", default=None, help="ChromaDB path for S1")
    parser.add_argument("--s1-rag-name", default="s1_fewshots")
    parser.add_argument("--s2-rag", default=None, help="ChromaDB path for S2")
    parser.add_argument("--s2-rag-name", default="s2_precedents")
    parser.add_argument("--rerank", action="store_true", default=True)
    parser.add_argument("--s2-temp", type=float, default=0.4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    data = load_jsonl(args.data)
    logger.info(f"Loaded {len(data)} documents from {args.data}")

    # Init RAG
    rag_collections: Dict[str, Any] = {}
    if args.s1_rag:
        from psycomark.rag.retrieval import get_rag_collection

        rag_collections["s1"] = get_rag_collection(args.s1_rag, args.s1_rag_name)
    if args.s2_rag:
        from psycomark.rag.retrieval import get_rag_collection

        rag_collections["s2"] = get_rag_collection(args.s2_rag, args.s2_rag_name)

    # Resume logic
    done_ids: set[str] = set()
    if args.resume:
        for path in (args.s1_out, args.s2_out):
            if Path(path).exists():
                for rec in load_jsonl(path):
                    done_ids.add(rec.get("_id", ""))
        logger.info(f"Resuming: {len(done_ids)} already done")

    s1_eval = S1Evaluator()
    s2_y_true, s2_y_pred = [], []

    for i, row in enumerate(data):
        doc_id = row.get("id") or row.get("_id", f"doc_{i}")
        if doc_id in done_ids:
            continue

        s1_res, s2_res = await process_document(
            row,
            args.task,
            rag_collections,
            rerank=args.rerank,
            s2_temp=args.s2_temp,
        )

        if s1_res:
            append_jsonl(args.s1_out, s1_res)
            gt_markers = row.get("metadata", {}).get("gt_markers")
            if gt_markers:
                s1_eval.update(s1_res.get("markers", []), gt_markers)

        if s2_res:
            append_jsonl(args.s2_out, s2_res)
            gt_label = normalize_label(row.get("metadata", {}).get("label", ""))
            pred_label = normalize_label(s2_res.get("conspiracy", "No"))
            if gt_label != "ambiguous":
                s2_y_true.append(gt_label)
                s2_y_pred.append(pred_label)

        # Progress
        s1_f1 = s1_eval.get_macro_f1()["macro_f1"]
        s2_acc = sum(1 for t, p in zip(s2_y_true, s2_y_pred) if t == p) / max(
            len(s2_y_true), 1
        )
        logger.info(
            f"[{i+1}/{len(data)}] {doc_id} | S1 F1: {s1_f1:.1%} | S2 Acc: {s2_acc:.1%}"
        )

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
            logger.info(f"  S2 Accuracy: {s2_acc:.4f} (install sklearn for F1)")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
