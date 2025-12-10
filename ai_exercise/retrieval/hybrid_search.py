"""Hybrid search combining BM25 and vector search using RRF fusion.

This module provides hybrid retrieval by combining lexical (BM25) and
semantic (vector) search results using Reciprocal Rank Fusion (RRF).
"""

from typing import Any

import chromadb

from ai_exercise.retrieval.bm25_index import BM25Index
from ai_exercise.retrieval.retrieval import RetrievedChunk


def rrf_score(rank: int, k: int = 60) -> float:
    """Calculate RRF score for a document at a given rank.

    Reciprocal Rank Fusion (RRF) is a simple yet effective method for
    combining ranked lists. The formula is: 1 / (k + rank).

    Args:
        rank: The rank of the document (1-indexed).
        k: The RRF constant (default 60, standard in literature).

    Returns:
        The RRF score for this rank.
    """
    return 1.0 / (k + rank)


class HybridSearcher:
    """Hybrid search combining BM25 and vector search with RRF fusion.

    This class retrieves results from both BM25 (lexical) and ChromaDB (vector)
    indices, then combines them using Reciprocal Rank Fusion.

    Attributes:
        collection: ChromaDB collection for vector search.
        bm25_index: BM25Index for lexical search.
        alpha: Weight for vector search scores (1-alpha for BM25).
    """

    def __init__(
        self,
        collection: chromadb.Collection,
        bm25_index: BM25Index,
        alpha: float = 0.5,
    ) -> None:
        """Initialize the hybrid searcher.

        Args:
            collection: ChromaDB collection for vector search.
            bm25_index: BM25Index for lexical search.
            alpha: Weight for vector search in RRF fusion (0-1).
                   0.5 = equal weight, higher = more vector, lower = more BM25.
        """
        self.collection = collection
        self.bm25_index = bm25_index
        self.alpha = alpha

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """Perform hybrid search combining BM25 and vector results.

        Args:
            query: The search query.
            k: Number of results to return.

        Returns:
            List of RetrievedChunk objects with combined relevance.
        """
        # Get more candidates from each source to ensure good coverage
        n_candidates = k * 3

        # 1. Vector search via ChromaDB
        vector_results = self.collection.query(
            query_texts=[query],
            n_results=n_candidates,
            include=["documents", "metadatas"],
        )

        # 2. BM25 search
        bm25_results = self.bm25_index.search(query, k=n_candidates)

        # 3. Build document metadata lookup from vector results
        doc_metadata: dict[str, dict[str, Any]] = {}
        doc_content: dict[str, str] = {}

        if vector_results["ids"] and vector_results["ids"][0]:
            for idx, doc_id in enumerate(vector_results["ids"][0]):
                metadata = (
                    vector_results["metadatas"][0][idx]
                    if vector_results["metadatas"]
                    else {}
                )
                document = (
                    vector_results["documents"][0][idx]
                    if vector_results["documents"]
                    else ""
                )
                # Use chunk_id from metadata if available, otherwise use ChromaDB ID
                chunk_id = metadata.get("chunk_id", doc_id)
                doc_metadata[chunk_id] = metadata
                doc_content[chunk_id] = document

        # 4. Compute RRF scores
        scores: dict[str, float] = {}

        # Vector search scores (alpha weight)
        if vector_results["ids"] and vector_results["ids"][0]:
            for rank, doc_id in enumerate(vector_results["ids"][0], 1):
                metadata = doc_metadata.get(doc_id, {})
                chunk_id = metadata.get("chunk_id", doc_id)
                scores[chunk_id] = self.alpha * rrf_score(rank)

        # BM25 scores (1-alpha weight)
        for rank, (doc_id, _bm25_score) in enumerate(bm25_results, 1):
            scores[doc_id] = scores.get(doc_id, 0) + (1 - self.alpha) * rrf_score(rank)

        # 5. Sort by combined score and take top-k
        sorted_ids = sorted(scores.keys(), key=lambda x: -scores[x])[:k]

        # 6. Build RetrievedChunk objects
        results: list[RetrievedChunk] = []
        for chunk_id in sorted_ids:
            # Try to get content from vector results first
            content = doc_content.get(chunk_id)
            metadata = doc_metadata.get(chunk_id, {})

            # Fall back to BM25 index if not in vector results
            if content is None:
                content = self.bm25_index.get_document(chunk_id) or ""

            results.append(
                RetrievedChunk(
                    content=content,
                    chunk_id=chunk_id,
                    metadata=metadata,
                )
            )

        return results


def get_relevant_chunks_hybrid(
    collection: chromadb.Collection,
    bm25_index: BM25Index,
    query: str,
    k: int = 5,
    alpha: float = 0.5,
) -> list[RetrievedChunk]:
    """Convenience function for hybrid search.

    Args:
        collection: ChromaDB collection for vector search.
        bm25_index: BM25Index for lexical search.
        query: The search query.
        k: Number of results to return.
        alpha: Weight for vector search (0.5 = equal weight).

    Returns:
        List of RetrievedChunk objects.
    """
    searcher = HybridSearcher(collection, bm25_index, alpha)
    return searcher.search(query, k)
