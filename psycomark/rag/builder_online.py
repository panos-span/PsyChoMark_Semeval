#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psycomark.rag.builder_online

Online RAG artifact builder (Contrastive ICL edition), ported and cleaned from
`src/psycomark/eda/rag_builder_online.py`.

Builds two ChromaDB collections:
  - `s1_patterns`: balanced positive + negative examples with teacher rationales
  - `s2_precedents`: legal-style precedents prioritising hard negatives

Key design points:
  - Uses `psycomark.config.LLM` (no sys.path hacks)
  - Uses OpenAI embeddings (`text-embedding-3-small`)
  - MMR selection implemented with pure NumPy (no sklearn dependency)
  - Stable IDs via SHA1(text)

Usage:
  python -m psycomark.rag.builder_online \
    --s1-input data/clean_v2/train_clean_s1.jsonl \
    --s2-input data/clean_v2/train_clean_s2.jsonl \
    --output-dir data/rag_openai_contrastive \
    --max-docs 500
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, cast

from loguru import logger
import numpy as np
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelSettings

from psycomark.config import LLM
from psycomark.prompts.builder import psycho_theory_preamble


DEFAULT_CONCURRENCY_LIMIT = 5
EMBED_DIMS = 1536


# ---------------------------------------------------------------------------
# Optional tqdm progress
# ---------------------------------------------------------------------------

def _tqdm_asyncio(iterable, desc: str):
    try:
        from tqdm.asyncio import tqdm_asyncio

        return tqdm_asyncio(iterable, desc=desc)
    except Exception:
        return iterable


# ---------------------------------------------------------------------------
# Embedding Function
# ---------------------------------------------------------------------------

def get_openai_embedding_function():
    from chromadb.utils import embedding_functions

    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        model_name="text-embedding-3-small",
    )


def batched_embed(documents: List[str], embedder_fn, batch_size: int = 50) -> np.ndarray:
    """Embed a large list of documents by chunking to avoid token/request limits."""
    all_embeddings: list[list[float]] = []
    total = len(documents)

    if total > 500:
        logger.info(f"Batching {total} documents for embedding…")

    for i in range(0, total, batch_size):
        batch = documents[i : i + batch_size]
        try:
            batch_embs = embedder_fn(batch)
            all_embeddings.extend(batch_embs)
        except Exception as e:
            logger.warning(f"Batch {i}-{i+batch_size} failed: {e}. Retrying size=1…")
            for doc in batch:
                try:
                    all_embeddings.append(embedder_fn([doc])[0])
                except Exception as inner_e:
                    logger.error(f"Skipping malformed doc: {inner_e}")
                    all_embeddings.append([0.0] * EMBED_DIMS)

    return np.array(all_embeddings, dtype=float)


def _cosine_sim_matrix(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between each row of A and vector b (both assumed float)."""
    # Normalise rows and vector
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return A_norm @ b_norm


def perform_mmr_selection(
    documents: List[str],
    embedder_fn,
    k: int,
    lambda_param: float = 0.6,
) -> List[int]:
    """Maximal Marginal Relevance selection (pure NumPy)."""
    if not documents:
        return []
    if k >= len(documents):
        return list(range(len(documents)))

    try:
        embeddings = batched_embed(documents, embedder_fn, batch_size=50)
    except Exception as e:
        logger.warning(f"MMR embedding failure ({e}); falling back to random.")
        return random.sample(range(len(documents)), k)

    if len(embeddings) == 0:
        return []

    # Normalise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-10)

    query_vec = np.mean(embeddings, axis=0)
    sim_to_query = _cosine_sim_matrix(embeddings, query_vec)

    selected: list[int] = []
    candidate_mask = np.ones(len(documents), dtype=bool)
    max_sim_to_selected = np.full(len(documents), -1.0, dtype=float)

    for _ in range(k):
        scores = (lambda_param * sim_to_query) - ((1 - lambda_param) * max_sim_to_selected)
        scores[~candidate_mask] = -float("inf")

        best_idx = int(np.argmax(scores))
        selected.append(best_idx)
        candidate_mask[best_idx] = False

        # Update redundancy using dot product since embeddings are unit-normalised
        new_vec = embeddings[best_idx]
        new_sims = embeddings @ new_vec
        max_sim_to_selected = np.maximum(max_sim_to_selected, new_sims)

    return selected


def _stable_id_from_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


# ---------------------------------------------------------------------------
# S1 Teacher — contrastive instructor
# ---------------------------------------------------------------------------

class S1ContrastiveOutput(BaseModel):
    analysis_type: Literal["positive_match", "negative_contrast"] = Field(
        ..., description="Is this a conspiracy example or a neutral contrast?"
    )
    rationale: str = Field(
        ..., description="Concise explanation of why spans are/aren't markers."
    )


_S1_TEACHER_SYSTEM = f"""
{psycho_theory_preamble()}

# ROLE
You are a **Forensic Instructor** creating training examples for the S1 span extractor.
Explain the ground truth with contrast between conspiracy markers and benign language.

## CASE A: SPANS EXIST
Explain WHY these spans imply agency/intent/hidden malice.

## CASE B: NO SPANS
Explain WHY the text is benign despite potential triggers (reporting, policy, natural force).

# OUTPUT
Return JSON with `analysis_type` and a 2-sentence `rationale`.
""".strip()


_s1_teacher_agent = Agent(
    LLM,
    output_type=S1ContrastiveOutput,
    system_prompt=_S1_TEACHER_SYSTEM,
)


async def generate_s1_contrastive_rationale(doc_text: str, spans: List[dict], sem: asyncio.Semaphore) -> Dict[str, Any]:
    async with sem:
        try:
            spans_str = json.dumps(spans, ensure_ascii=False) if spans else "[] (NO MARKERS)"
            prompt = (
                f"TEXT: {doc_text}\n"
                f"GROUND TRUTH SPANS: {spans_str}\n\n"
                "Explain this ground truth using the contrastive rules."
            )
            res = await _s1_teacher_agent.run(prompt, model_settings=ModelSettings(temperature=0.0))
            return {
                "text": doc_text,
                "spans": spans,
                "rationale": res.output.rationale,
                "type": res.output.analysis_type,
            }
        except Exception:
            return {
                "text": doc_text,
                "spans": spans,
                "rationale": "Processing failed.",
                "type": "negative_contrast",
            }


# ---------------------------------------------------------------------------
# S2 Teacher — precedent setter
# ---------------------------------------------------------------------------

class S2PrecedentOutput(BaseModel):
    verdict_header: str = Field(..., description="e.g. 'ACQUITTAL: Reporter Defense'")
    legal_rationale: str = Field(..., description="3-sentence precedent explaining the verdict")


_S2_TEACHER_SYSTEM = f"""
{psycho_theory_preamble()}

# ROLE
You are a **Supreme Court Justice** writing precedents for a case law database (RAG).

# TASK
Write a binding precedent for the verdict (CONSPIRACY vs NON), focusing on endorsement vs reporting.

# OUTPUT
Return JSON with `verdict_header` and `legal_rationale`.
""".strip()


_s2_teacher_agent = Agent(
    LLM,
    output_type=S2PrecedentOutput,
    system_prompt=_S2_TEACHER_SYSTEM,
)


async def generate_s2_precedent(doc: Dict[str, Any], sem: asyncio.Semaphore) -> Dict[str, Any]:
    async with sem:
        try:
            prompt = (
                f"TEXT: {doc['text']}\n"
                f"VERDICT: {str(doc['label']).upper()}\n"
                f"MARKERS FOUND: {len(doc.get('markers', []))}\n\n"
                "Write the legal precedent."
            )
            res = await _s2_teacher_agent.run(prompt, model_settings=ModelSettings(temperature=0.0))
            doc = dict(doc)
            doc["rationale"] = f"{res.output.verdict_header}\n{res.output.legal_rationale}"
            return doc
        except Exception:
            doc = dict(doc)
            doc["rationale"] = "Precedent generation failed."
            return doc


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

async def run_s1_pipeline(
    input_path: str,
    output_dir: str,
    target_docs: int,
    concurrency: int = DEFAULT_CONCURRENCY_LIMIT,
):
    import chromadb

    logger.info("--- Building S1 Pattern Bank (balanced by marker type) ---")
    client = chromadb.PersistentClient(path=output_dir)
    col = client.get_or_create_collection(
        "s1_patterns",
        embedding_function=cast(Any, get_openai_embedding_function()),
    )

    raw_docs: list[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                if "text" in d:
                    raw_docs.append(d)
            except Exception:
                continue

    all_positives = [d for d in raw_docs if d.get("spans") or d.get("markers")]
    negatives = [d for d in raw_docs if not (d.get("spans") or d.get("markers"))]

    rare_labels = {"Evidence", "Victim"}

    def has_rare_marker(doc: dict) -> bool:
        spans = doc.get("spans") or doc.get("markers") or []
        for s in spans:
            lbl = s.get("label") or s.get("type")
            if lbl in rare_labels:
                return True
        return False

    rare_positives = [d for d in all_positives if has_rare_marker(d)]
    common_positives = [d for d in all_positives if d not in rare_positives]

    k_pos_total = target_docs // 2
    k_rare = min(len(rare_positives), int(k_pos_total * 0.6))
    k_common = max(0, k_pos_total - k_rare)

    logger.info(f"Selection quotas: {k_rare} rare + {k_common} common + {k_pos_total} negatives")

    ef = get_openai_embedding_function()

    def embed_text(texts: list[str]):
        return ef(texts)

    sel_rare_idx = perform_mmr_selection([d["text"] for d in rare_positives], embed_text, k_rare)
    sel_common_idx = perform_mmr_selection([d["text"] for d in common_positives], embed_text, k_common)
    sel_neg_idx = perform_mmr_selection([d["text"] for d in negatives], embed_text, k_pos_total)

    selection = (
        [rare_positives[i] for i in sel_rare_idx]
        + [common_positives[i] for i in sel_common_idx]
        + [negatives[i] for i in sel_neg_idx]
    )
    random.shuffle(selection)

    sem = asyncio.Semaphore(concurrency)
    logger.info(f"Enriching {len(selection)} docs with S1 teacher rationales…")

    tasks = []
    for d in selection:
        spans = d.get("spans") or d.get("markers") or []
        tasks.append(generate_s1_contrastive_rationale(d["text"], spans, sem))

    enriched = []
    for res in _tqdm_asyncio(await asyncio.gather(*tasks), desc="S1 Teacher"):
        enriched.append(res)

    def check_flag(spans: list[dict], target_lbls: list[str]) -> str:
        for s in spans:
            if (s.get("label") or s.get("type")) in target_lbls:
                return "True"
        return "False"

    ids = [_stable_id_from_text(d["text"]) for d in enriched]
    docs = [d["text"] for d in enriched]
    metadatas = []
    for d in enriched:
        spans = d.get("spans") or []
        metadatas.append(
            {
                "rationale": d.get("rationale", ""),
                "spans_json": json.dumps(spans, ensure_ascii=False),
                "type": d.get("type", ""),
                "has_evidence": check_flag(spans, ["Evidence"]),
                "has_victim": check_flag(spans, ["Victim"]),
                "has_effect": check_flag(spans, ["Effect"]),
            }
        )

    for i in range(0, len(ids), 50):
        col.add(ids=ids[i : i + 50], documents=docs[i : i + 50], metadatas=metadatas[i : i + 50])

    logger.success(f"S1 Done. Indexed {len(ids)} patterns into '{output_dir}'.")


async def run_s2_pipeline(
    s1_path: Optional[str],
    s2_path: str,
    output_dir: str,
    target_docs: int,
    concurrency: int = DEFAULT_CONCURRENCY_LIMIT,
):
    import chromadb

    logger.info("--- Building S2 Precedent Bank (legal) ---")
    client = chromadb.PersistentClient(path=output_dir)
    col = client.get_or_create_collection(
        "s2_precedents",
        embedding_function=cast(Any, get_openai_embedding_function()),
    )

    s1_map: dict[str, list] = {}
    if s1_path and Path(s1_path).exists():
        logger.info(f"Loading S1 markers from {s1_path}…")
        with open(s1_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    did = str(d.get("doc_id") or d.get("_id") or "")
                    spans = d.get("spans") or d.get("markers") or []
                    if did:
                        s1_map[did] = spans
                except Exception:
                    continue
    elif s1_path:
        logger.warning(f"S1 path not found: {s1_path}. Hard-negative mining will be limited.")

    raw_docs: list[dict] = []
    with open(s2_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                if d.get("label") not in ("conspiracy", "non"):
                    continue
                did = str(d.get("doc_id") or d.get("_id") or "")
                d["doc_id"] = did
                d["markers"] = s1_map.get(did, d.get("markers", []))
                raw_docs.append(d)
            except Exception:
                continue

    hard_negatives = [d for d in raw_docs if d["label"] == "non" and len(d.get("markers", [])) > 0]
    others = [d for d in raw_docs if d not in hard_negatives]

    k_hard = min(len(hard_negatives), target_docs // 2)
    k_other = max(0, target_docs - k_hard)

    ef = get_openai_embedding_function()

    def embed_text(texts: list[str]):
        return ef(texts)

    logger.info(f"Selecting {k_hard} hard negatives + {k_other} others (MMR)…")
    sel_hard_idx = perform_mmr_selection([d["text"] for d in hard_negatives], embed_text, k_hard)
    sel_other_idx = perform_mmr_selection([d["text"] for d in others], embed_text, k_other)

    selection = [hard_negatives[i] for i in sel_hard_idx] + [others[i] for i in sel_other_idx]

    sem = asyncio.Semaphore(concurrency)
    logger.info(f"Enriching {len(selection)} docs with S2 legal precedents…")

    tasks = [generate_s2_precedent(d, sem) for d in selection]
    enriched = []
    for res in _tqdm_asyncio(await asyncio.gather(*tasks), desc="S2 Teacher"):
        enriched.append(res)

    ids = [d.get("doc_id") or _stable_id_from_text(d["text"]) for d in enriched]
    docs = [d["text"] for d in enriched]
    metas = []
    for d in enriched:
        marker_count = len(d.get("markers", []))
        is_hard_negative = (d.get("label") == "non" and marker_count > 0)
        metas.append(
            {
                "rationale": d.get("rationale", ""),
                "label": d.get("label", ""),
                "marker_count": marker_count,
                "is_hard_negative": str(is_hard_negative),
            }
        )

    for i in range(0, len(ids), 50):
        col.add(ids=ids[i : i + 50], documents=docs[i : i + 50], metadatas=metas[i : i + 50])

    logger.success(f"S2 Done. Indexed {len(ids)} precedents into '{output_dir}'.")


def main(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-input", default="data/clean_v2/train_clean_s1.jsonl")
    parser.add_argument("--s2-input", default="data/clean_v2/train_clean_s2.jsonl")
    parser.add_argument("--output-dir", default="data/rag_openai_contrastive")
    parser.add_argument("--max-docs", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY_LIMIT)
    args = parser.parse_args(argv)

    asyncio.run(
        run_s1_pipeline(
            args.s1_input,
            args.output_dir,
            args.max_docs,
            concurrency=args.concurrency,
        )
    )
    asyncio.run(
        run_s2_pipeline(
            args.s1_input,
            args.s2_input,
            args.output_dir,
            args.max_docs,
            concurrency=args.concurrency,
        )
    )


if __name__ == "__main__":
    main()
