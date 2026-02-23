"""
psycomark.rag — Contrastive Retrieval-Augmented Generation.

Provides:
    - ChromaDB collection initialisation (OpenAI embeddings)
    - Bi-encoder retrieval with optional metadata filters
    - Cross-encoder reranking (BAAI/bge-reranker-v2-m3) + MMR diversity
    - Stratified retrieval (conspiracy / non / ambiguous)
    - Hard-negative retrieval
"""

from psycomark.rag.retrieval import (
    get_rag_collection,
    rerank_documents,
    retrieve_fewshots,
    retrieve_fewshots_reranked,
    retrieve_hard_negatives_reranked,
    retrieve_stratified_s1_reranked,
)

__all__ = [
    "get_rag_collection",
    "retrieve_fewshots",
    "retrieve_fewshots_reranked",
    "retrieve_stratified_s1_reranked",
    "retrieve_hard_negatives_reranked",
    "rerank_documents",
]
