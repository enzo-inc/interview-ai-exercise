"""BM25 index for lexical search.

This module provides a BM25 index implementation using the rank_bm25 library.
BM25 excels at exact keyword matching, complementing semantic vector search.
"""

import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Simple tokenization: lowercase and split on whitespace.

    Args:
        text: The text to tokenize.

    Returns:
        List of lowercase tokens.
    """
    return text.lower().split()


class BM25Index:
    """BM25 index for lexical search over documents.

    Attributes:
        documents: List of original document texts.
        doc_ids: List of document IDs corresponding to documents.
        bm25: The BM25Okapi model from rank_bm25.
    """

    def __init__(self, documents: list[str], doc_ids: list[str]) -> None:
        """Initialize BM25 index with documents.

        Args:
            documents: List of document texts to index.
            doc_ids: List of document IDs (must match length of documents).

        Raises:
            ValueError: If documents and doc_ids have different lengths.
        """
        if len(documents) != len(doc_ids):
            raise ValueError(
                f"Documents ({len(documents)}) and doc_ids ({len(doc_ids)}) "
                "must have same length"
            )

        self.documents = documents
        self.doc_ids = doc_ids

        # Tokenize all documents
        tokenized_docs = [tokenize(doc) for doc in documents]

        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Search the index for documents matching the query.

        Args:
            query: The search query.
            k: Number of results to return.

        Returns:
            List of (doc_id, score) tuples, sorted by score descending.
        """
        # Tokenize query
        query_tokens = tokenize(query)

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        # Use negative scores for descending sort
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]

        # Return (doc_id, score) tuples
        return [(self.doc_ids[i], scores[i]) for i in top_indices]

    def get_document(self, doc_id: str) -> str | None:
        """Get a document by its ID.

        Args:
            doc_id: The document ID to look up.

        Returns:
            The document text, or None if not found.
        """
        try:
            idx = self.doc_ids.index(doc_id)
            return self.documents[idx]
        except ValueError:
            return None

    def save(self, path: Path) -> None:
        """Save the index to disk.

        Args:
            path: Path to save the index file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "documents": self.documents,
            "doc_ids": self.doc_ids,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        """Load an index from disk.

        Args:
            path: Path to the index file.

        Returns:
            Loaded BM25Index instance.

        Raises:
            FileNotFoundError: If the index file doesn't exist.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(documents=data["documents"], doc_ids=data["doc_ids"])

    def __len__(self) -> int:
        """Return the number of documents in the index."""
        return len(self.documents)
