"""
psycomark.rag.retrieval — RAG Retrieval & Reranking Pipeline.

Components:
    1. **Bi-encoder** (ChromaDB): ``text-embedding-3-small`` (1 536 d)
    2. **Cross-encoder reranker**: ``BAAI/bge-reranker-v2-m3``
    3. **MMR selection**: Maximal Marginal Relevance for diverse shot selection
    4. **Stratified retrieval**: Balanced conspiracy / non / ambiguous examples
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from chromadb import Collection
from loguru import logger

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    cosine_similarity = None


# ---------------------------------------------------------------------------
# Cross-Encoder Singleton
# ---------------------------------------------------------------------------

_CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"
_CROSS_ENCODER = None


def get_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        try:
            from sentence_transformers import CrossEncoder
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"[Reranker] Loading cross-encoder: {_CROSS_ENCODER_MODEL}")
            _CROSS_ENCODER = CrossEncoder(_CROSS_ENCODER_MODEL, device=device)
            logger.success(f"[Reranker] Cross-encoder loaded on {device}")
        except ImportError:
            logger.warning("[Reranker] sentence-transformers not installed.")
            _CROSS_ENCODER = None
        except Exception as e:
            logger.error(f"[Reranker] Failed to load: {e}")
            _CROSS_ENCODER = None
    return _CROSS_ENCODER


# ---------------------------------------------------------------------------
# ChromaDB Collection Setup
# ---------------------------------------------------------------------------


def get_rag_collection(path: str, name: str) -> Collection:
    """Initialise a ChromaDB PersistentClient and return the named collection."""
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=path)
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )
    coll = client.get_collection(name=name, embedding_function=ef)
    logger.info(f"  - Loaded Index {name} ({coll.count()} docs)")
    return coll


# ---------------------------------------------------------------------------
# MMR Selection
# ---------------------------------------------------------------------------


def mmr_selection(
    docs: List[Dict[str, Any]],
    relevance_scores: np.ndarray,
    doc_embeddings: np.ndarray | None = None,
    top_k: int = 5,
    lambda_mult: float = 0.6,
) -> List[Dict[str, Any]]:
    """Select documents using Maximal Marginal Relevance."""
    if not docs:
        return []

    if doc_embeddings is None or len(doc_embeddings) == 0 or cosine_similarity is None:
        sorted_idx = np.argsort(relevance_scores)[::-1][:top_k]
        return [docs[i] for i in sorted_idx]

    selected: list[int] = []
    candidates = list(range(len(docs)))

    while len(selected) < top_k and candidates:
        best_score, best_idx = -float("inf"), -1
        for idx in candidates:
            rel = relevance_scores[idx]
            if not selected:
                redundancy = 0.0
            else:
                sims = cosine_similarity(
                    doc_embeddings[idx].reshape(1, -1),
                    doc_embeddings[selected],
                )
                redundancy = float(np.max(sims))
            mmr = lambda_mult * rel - (1 - lambda_mult) * redundancy
            if mmr > best_score:
                best_score, best_idx = mmr, idx
        if best_idx != -1:
            selected.append(best_idx)
            candidates.remove(best_idx)

    return [docs[i] for i in selected]


# ---------------------------------------------------------------------------
# Cross-Encoder Reranking
# ---------------------------------------------------------------------------


def rerank_documents(
    query: str,
    documents: List[Dict[str, Any]],
    top_k: int,
    text_field: str = "text",
    use_mmr: bool = True,
) -> List[Dict[str, Any]]:
    """Rerank using CrossEncoder + optional MMR diversity."""
    if not documents:
        return []

    cross_encoder = get_cross_encoder()
    if cross_encoder is None:
        return documents[:top_k]

    try:
        pairs = [(query, doc.get(text_field, "")) for doc in documents]
        raw_scores = cross_encoder.predict(pairs)

        # Min-Max normalisation (BGE outputs raw logits)
        if len(raw_scores) > 1:
            mn, mx = np.min(raw_scores), np.max(raw_scores)
            scores = (
                np.ones_like(raw_scores) * 0.5
                if (mx - mn) < 1e-9
                else (raw_scores - mn) / (mx - mn)
            )
        else:
            scores = np.array([1.0])

        # Extract embeddings for MMR
        embeddings = None
        key = next(
            (k for k in ("embeddings", "embedding") if documents[0].get(k) is not None),
            None,
        )
        if key:
            try:
                embeddings = np.array([d[key] for d in documents])
            except Exception:
                embeddings = None

        if use_mmr and embeddings is not None:
            final = mmr_selection(
                documents, scores, embeddings, top_k=top_k, lambda_mult=0.6
            )
        else:
            scored = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            final = [d for d, _ in scored[:top_k]]

        # Strip embeddings from output
        for doc in final:
            doc.pop("embeddings", None)
            doc.pop("embedding", None)

        return final

    except Exception as e:
        logger.warning(f"[Reranker] Failed: {e}. Returning original.")
        return documents[:top_k]


# ---------------------------------------------------------------------------
# Bi-Encoder Retrieval
# ---------------------------------------------------------------------------


def retrieve_fewshots(
    collection: Collection,
    query_text: str,
    k: int = 8,
    filters: Optional[dict] = None,
) -> List[dict]:
    """Retrieve k examples from a ChromaDB collection."""
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=k,
            where=filters,
            include=["metadatas", "documents", "embeddings"],
        )
        examples = []
        if results["documents"] and results["metadatas"]:
            emb_batch = results.get("embeddings", [])
            for i in range(len(results["documents"][0])):
                meta = results["metadatas"][0][i] if results["metadatas"][0] else {}
                ex = {"text": results["documents"][0][i], **meta}
                if emb_batch and len(emb_batch) > 0:
                    ex["embeddings"] = emb_batch[0][i]
                if "spans_json" in ex:
                    ex["spans"] = json.loads(ex.pop("spans_json"))
                examples.append(ex)
        return examples
    except Exception as e:
        logger.error(f"[RAG] Retrieval failed: {e}")
        return []


def retrieve_fewshots_reranked(
    collection: Collection,
    query_text: str,
    k: int = 4,
    overretrieve_factor: int = 3,
    filters: Optional[dict] = None,
) -> List[dict]:
    """Over-retrieve → cross-encoder rerank → return top-k."""
    overretrieve_k = min(k * overretrieve_factor, 50)
    candidates = retrieve_fewshots(
        collection, query_text, k=overretrieve_k, filters=filters
    )
    if len(candidates) <= k:
        return candidates
    return rerank_documents(query_text, candidates, top_k=k, text_field="text")


# ---------------------------------------------------------------------------
# Stratified Retrieval
# ---------------------------------------------------------------------------


def retrieve_stratified_s1_reranked(
    collection: Collection,
    query_text: str,
    k_total: int = 6,
    overretrieve_factor: int = 3,
) -> List[Dict]:
    """
    Balanced retrieval: 40 % Conspiracy, 40 % Non, 20 % Ambiguous.
    Falls back to 50/50 if no ambiguous examples exist.
    """
    if not collection:
        return []

    k_amb = max(1, k_total // 5)
    k_rem = k_total - k_amb
    k_pos = k_rem // 2
    k_neg = k_rem - k_pos

    pos = retrieve_fewshots_reranked(
        collection,
        query_text,
        k=k_pos,
        overretrieve_factor=overretrieve_factor,
        filters={"label": "conspiracy"},
    )
    neg = retrieve_fewshots_reranked(
        collection,
        query_text,
        k=k_neg,
        overretrieve_factor=overretrieve_factor,
        filters={"label": "non"},
    )

    ambiguous = retrieve_fewshots_reranked(
        collection,
        query_text,
        k=k_amb,
        overretrieve_factor=overretrieve_factor,
        filters={"label": "cant_tell"},
    )
    if not ambiguous:
        ambiguous = retrieve_fewshots_reranked(
            collection,
            query_text,
            k=k_amb,
            overretrieve_factor=overretrieve_factor,
            filters={"label": "ambiguous"},
        )
    if not ambiguous:
        extra = retrieve_fewshots_reranked(
            collection,
            query_text,
            k=k_amb,
            overretrieve_factor=overretrieve_factor,
            filters={"label": "non"},
        )
        neg.extend(extra)

    # Force inclusion of evidence-backed example
    force_ev = retrieve_fewshots(
        collection, query_text, k=1, filters={"has_evidence": "True"}
    )
    if force_ev and force_ev[0] not in pos and force_ev[0] not in neg:
        (pos if force_ev[0].get("label") == "conspiracy" else neg).append(force_ev[0])

    # Interleave
    stratified: list[dict] = []
    for p, n in zip(pos, neg):
        stratified.extend([p, n])
    if len(pos) > len(neg):
        stratified.extend(pos[len(neg) :])
    elif len(neg) > len(pos):
        stratified.extend(neg[len(pos) :])

    if ambiguous:
        idx = min(len(stratified), 2)
        stratified[idx:idx] = ambiguous

    return stratified[:k_total]


def retrieve_hard_negatives_reranked(
    collection: Collection,
    query_text: str,
    k: int = 4,
    overretrieve_factor: int = 4,
) -> List[dict]:
    """Retrieve hard negatives with cross-encoder reranking."""
    return retrieve_fewshots_reranked(
        collection,
        query_text,
        k=k,
        overretrieve_factor=overretrieve_factor,
        filters={"is_hard_negative": True},
    )
