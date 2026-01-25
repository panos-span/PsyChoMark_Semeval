#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_builder_online.py

Optimized Artifact Builder for PsyCoMark RAG (Contrastive ICL Edition).
Replaces AWS Batch with Pydantic-AI Online Inference.

Features:
- **S1 Pattern Bank:** Indexes both Positive AND Negative examples to teach contrast.
- **S2 Precedent Bank:** Prioritizes "Hard Negatives" (Reporter Traps) for legal defense.
- **Markdown Prompts:** Optimized for modern reasoning models.
"""

import asyncio
import json
import os
import argparse
import sys
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Literal
from collections import defaultdict

# --- Environment & Imports ---
sys.path.append(str(Path(__file__).resolve().parents[3]))

from loguru import logger
from tqdm.asyncio import tqdm_asyncio
import chromadb
from chromadb.utils import embedding_functions
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelSettings

# Reuse your existing configuration
from pydanticai2.psycomark_agents import LLM
from pydanticai2.prompt_builder import psycho_theory_preamble

CONCURRENCY_LIMIT = 5

# ==============================================================================
# 1. OPTIMIZED SELECTION LOGIC (MMR + STRATIFICATION)
# ==============================================================================


def perform_mmr_selection(
    documents: List[str], embedder_fn, k: int, lambda_param: float = 0.6
) -> List[int]:
    """
    Maximal Marginal Relevance (MMR) Selection.
    Ensures we don't index 500 duplicates of the same topic.
    """
    if k >= len(documents):
        return list(range(len(documents)))

    try:
        from sklearn.metrics.pairwise import cosine_similarity

        embeddings = np.array(embedder_fn(documents))
    except Exception as e:
        logger.warning(f"Embedding failed ({e}), falling back to random.")
        return random.sample(range(len(documents)), k)

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-10)

    # Global Centroid
    query_vec = np.mean(embeddings, axis=0).reshape(1, -1)
    sim_to_query = cosine_similarity(embeddings, query_vec).flatten()

    selected_indices = []
    candidate_mask = np.ones(len(documents), dtype=bool)
    max_sim_to_selected = np.full(len(documents), -1.0)

    for _ in range(k):
        # Score = Relevance - (1-Lambda)*Redundancy
        current_scores = (lambda_param * sim_to_query) - (
            (1 - lambda_param) * max_sim_to_selected
        )
        current_scores[~candidate_mask] = -float("inf")

        best_idx = np.argmax(current_scores)
        selected_indices.append(best_idx)
        candidate_mask[best_idx] = False

        # Update Redundancy
        new_vec = embeddings[best_idx].reshape(1, -1)
        new_sims = cosine_similarity(embeddings, new_vec).flatten()
        max_sim_to_selected = np.maximum(max_sim_to_selected, new_sims)

    return selected_indices


# ==============================================================================
# 2. S1 TEACHER: CONTRASTIVE INSTRUCTOR
# ==============================================================================


class S1ContrastiveOutput(BaseModel):
    analysis_type: Literal["positive_match", "negative_contrast"] = Field(
        ..., description="Is this a conspiracy example or a neutral contrast?"
    )
    rationale: str = Field(
        ...,
        description="The logic to be indexed. Must explain WHY it is or isn't a marker.",
    )


# OPTIMIZED MARKDOWN PROMPT FOR S1
s1_teacher_system = f"""
{psycho_theory_preamble()}

# ROLE
You are a **Forensic Instructor** creating a training dataset for the S1 Conspiracy Extractor.
Your goal is to explain the **Ground Truth** to a student model, focusing on the boundary between "Conspiracy" and "Normalcy".

# INSTRUCTIONS

## CASE A: SPANS EXIST (Positive Example)
If the input JSON contains spans, explain **why** these specific words imply secret malice.
* **The Intent Rule:** "Extracted 'engineered' because it implies intentional design, unlike 'mutated'."
* **The Hidden Agency Rule:** "Extracted 'The Cabal' because it refers to a shadow actor, unlike 'The Government'."

## CASE B: NO SPANS (Negative Example / Hard Negative)
If the input JSON is empty `[]`, explain **why** this text is benign despite potential triggers.
* **The Natural Force Rule:** "Rejected because 'virus'/'earthquake' is presented as a natural phenomenon, not a weapon."
* **The Reporter Rule:** "Rejected because the author is merely reporting ('Reuters says') without endorsing the claim."
* **The Policy Rule:** "Rejected because 'tax increase' is open public policy, not a secret plot."

# OUTPUT FORMAT
Return a JSON object with:
* `analysis_type`: "positive_match" or "negative_contrast"
* `rationale`: A concise, 2-sentence explanation.
"""

s1_teacher_agent = Agent(
    LLM, output_type=S1ContrastiveOutput, system_prompt=s1_teacher_system
)


async def generate_s1_contrastive_rationale(
    doc_text: str, spans: List[dict], sem: asyncio.Semaphore
) -> Dict[str, Any]:
    async with sem:
        try:
            spans_str = json.dumps(spans) if spans else "[] (NO MARKERS)"
            prompt = (
                f"TEXT: {doc_text}\n"
                f"GROUND TRUTH SPANS: {spans_str}\n\n"
                "Explain this ground truth using the Contrastive Rules."
            )
            res = await s1_teacher_agent.run(
                prompt, model_settings=ModelSettings(temperature=0.0)
            )
            return {
                "text": doc_text,
                "spans": spans,
                "rationale": res.output.rationale,
                "type": res.output.analysis_type,
            }
        except Exception as e:
            return {
                "text": doc_text,
                "spans": spans,
                "rationale": "Processing failed.",
                "type": "negative_contrast",
            }


# ==============================================================================
# 3. S2 TEACHER: PRECEDENT SETTER
# ==============================================================================


class S2PrecedentOutput(BaseModel):
    verdict_header: str = Field(
        ..., description="e.g. 'ACQUITTAL: The Reporter Defense'"
    )
    legal_rationale: str = Field(
        ..., description="3-sentence legal precedent explaining the verdict."
    )


# OPTIMIZED MARKDOWN PROMPT FOR S2
s2_teacher_system = f"""
{psycho_theory_preamble()}

# ROLE
You are a **Supreme Court Justice** writing Legal Precedents for a Case Law Database (RAG).

# TASK
Analyze the Text and the Label (Guilty/Not Guilty) to write a binding precedent.

## FOR 'NON' (ACQUITTAL)
Use these specific defense templates:
* **The Reporter Defense:** "Acquitted because the text attributes claims ('Users say', 'Reuters reports') without endorsement."
* **Hanlon's Razor:** "Acquitted because the text describes incompetence, bureaucracy, or greed, not coordinated malice."
* **The Natural Force Defense:** "Acquitted because the 'threat' is a natural event (pandemic, weather) with no alleged human plotter."
* **The Satire Defense:** "Acquitted because the tone is mocking or sarcastic."

## FOR 'CONSPIRACY' (CONVICTION)
Use these specific conviction templates:
* **The Design Rule:** "Convicted because the text claims the event was *engineered*, *staged*, or *faked*."
* **The Cabal Rule:** "Convicted because it alleges a secret alliance ('They', 'Elites') working against the public interest."
* **The Truth Claim:** "Convicted because the author asserts privileged knowledge ('The truth is...') regarding a cover-up."
"""

s2_teacher_agent = Agent(
    LLM, output_type=S2PrecedentOutput, system_prompt=s2_teacher_system
)


async def generate_s2_precedent(
    doc: Dict[str, Any], sem: asyncio.Semaphore
) -> Dict[str, Any]:
    async with sem:
        try:
            prompt = (
                f"TEXT: {doc['text']}\n"
                f"VERDICT: {doc['label'].upper()}\n"
                f"MARKERS FOUND: {len(doc.get('markers', []))}\n\n"
                "Write the Legal Precedent."
            )
            res = await s2_teacher_agent.run(
                prompt, model_settings=ModelSettings(temperature=0.0)
            )
            doc["rationale"] = (
                f"{res.output.verdict_header}\n{res.output.legal_rationale}"
            )
            return doc
        except Exception:
            doc["rationale"] = "Precedent generation failed."
            return doc


# ==============================================================================
# 4. PIPELINE EXECUTION
# ==============================================================================


def get_ef():
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        model_name="text-embedding-3-small",
    )


async def run_s1_pipeline(input_path: str, output_dir: str, target_docs: int):
    logger.info("--- Building S1 Pattern Bank (Contrastive) ---")
    client = chromadb.PersistentClient(path=output_dir)
    col = client.get_or_create_collection("s1_patterns", embedding_function=get_ef())

    # 1. Load Data
    raw_docs = []
    with open(input_path) as f:
        for line in f:
            try:
                d = json.loads(line)
                if "text" in d:
                    raw_docs.append(d)
            except:
                continue

    # 2. Stratified Selection (50/50 Split)
    # We explicitly want Negatives to teach the model what NOT to do.
    positives = [d for d in raw_docs if d.get("spans") or d.get("markers")]
    negatives = [d for d in raw_docs if not (d.get("spans") or d.get("markers"))]

    k_split = target_docs // 2

    # Simple embedding wrapper for MMR
    def embed_text(texts):
        return get_ef()(texts)

    logger.info(f"Selecting {k_split} Positives and {k_split} Negatives via MMR...")

    sel_pos_idx = perform_mmr_selection(
        [d["text"] for d in positives], embed_text, k_split
    )
    sel_neg_idx = perform_mmr_selection(
        [d["text"] for d in negatives], embed_text, k_split
    )

    selection = [positives[i] for i in sel_pos_idx] + [
        negatives[i] for i in sel_neg_idx
    ]
    random.shuffle(selection)

    # 3. Enrich with Teacher
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    logger.info(f"Enriching {len(selection)} docs with Contrastive Rationales...")

    tasks = [
        generate_s1_contrastive_rationale(d["text"], d.get("spans", []), sem)
        for d in selection
    ]
    enriched = []

    for res in tqdm_asyncio(await asyncio.gather(*tasks), desc="S1 Teacher"):
        enriched.append(res)

    # 4. Index
    ids = [str(hash(d["text"])) for d in enriched]  # Simple hash ID for uniqueness
    docs = [d["text"] for d in enriched]
    metas = [
        {
            "rationale": d["rationale"],
            "spans_json": json.dumps(d["spans"]),
            "type": d["type"],
        }
        for d in enriched
    ]

    for i in range(0, len(ids), 50):
        col.add(
            ids=ids[i : i + 50], documents=docs[i : i + 50], metadatas=metas[i : i + 50]
        )

    logger.success(f"S1 Done. Indexed {len(ids)} patterns.")


async def run_s2_pipeline(
    s1_path: str, s2_path: str, output_dir: str, target_docs: int
):
    logger.info("--- Building S2 Precedent Bank (Legal) ---")
    client = chromadb.PersistentClient(path=output_dir)
    col = client.get_or_create_collection("s2_precedents", embedding_function=get_ef())

    # 1. Load S2 Data
    raw_docs = []
    with open(s2_path) as f:
        for line in f:
            try:
                d = json.loads(line)
                if d.get("label") in ["conspiracy", "non"]:
                    raw_docs.append(d)
            except:
                continue

    # 2. "Hard Negative" Mining
    # We want neutral texts that HAVE markers (Confusion cases)
    hard_negatives = [
        d for d in raw_docs if d["label"] == "non" and len(d.get("markers", [])) > 0
    ]
    others = [d for d in raw_docs if d not in hard_negatives]

    # Prioritize Hard Negatives (50% of index if possible)
    k_hard = min(len(hard_negatives), target_docs // 2)
    k_other = target_docs - k_hard

    def embed_text(texts):
        return get_ef()(texts)

    logger.info("Selecting Hard Negatives & Precedents via MMR...")

    sel_hard_idx = perform_mmr_selection(
        [d["text"] for d in hard_negatives], embed_text, k_hard
    )
    sel_other_idx = perform_mmr_selection(
        [d["text"] for d in others], embed_text, k_other
    )

    selection = [hard_negatives[i] for i in sel_hard_idx] + [
        others[i] for i in sel_other_idx
    ]

    # 3. Enrich with Teacher
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    logger.info(f"Enriching {len(selection)} docs with Legal Precedents...")

    tasks = [generate_s2_precedent(d, sem) for d in selection]
    enriched = []

    for res in tqdm_asyncio(await asyncio.gather(*tasks), desc="S2 Teacher"):
        enriched.append(res)

    # 4. Index
    ids = [str(d.get("doc_id") or hash(d["text"])) for d in enriched]
    docs = [d["text"] for d in enriched]
    metas = [
        {
            "rationale": d["rationale"],
            "label": d["label"],
            "marker_count": len(d.get("markers", [])),
        }
        for d in enriched
    ]

    for i in range(0, len(ids), 50):
        col.add(
            ids=ids[i : i + 50], documents=docs[i : i + 50], metadatas=metas[i : i + 50]
        )

    logger.success(f"S2 Done. Indexed {len(ids)} precedents.")


# --- Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1-input", default="data/clean/train_clean_s1.jsonl")
    parser.add_argument("--s2-input", default="data/clean/train_clean_s2.jsonl")
    parser.add_argument("--output-dir", default="data/rag_optimized")
    parser.add_argument("--max-docs", type=int, default=1000)
    args = parser.parse_args()

    asyncio.run(run_s1_pipeline(args.s1_input, args.output_dir, args.max_docs))
    asyncio.run(
        run_s2_pipeline(args.s1_input, args.s2_input, args.output_dir, args.max_docs)
    )
