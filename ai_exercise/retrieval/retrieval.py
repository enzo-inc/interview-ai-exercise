"""Retrieve relevant chunks from a vector store."""

from dataclasses import dataclass

import chromadb


@dataclass
class RetrievedChunk:
    """A retrieved chunk with its content, ID, and metadata."""

    content: str
    chunk_id: str
    metadata: dict


def get_relevant_chunks(
    collection: chromadb.Collection, query: str, k: int
) -> list[str]:
    """Retrieve k most relevant chunks for the query.

    Returns only the document content (for backward compatibility).
    """
    results = collection.query(query_texts=[query], n_results=k)
    return results["documents"][0]


def get_relevant_chunks_with_ids(
    collection: chromadb.Collection, query: str, k: int
) -> list[RetrievedChunk]:
    """Retrieve k most relevant chunks with their IDs and metadata.

    Args:
        collection: ChromaDB collection to query.
        query: The query text.
        k: Number of results to return.

    Returns:
        List of RetrievedChunk objects with content, chunk_id, and metadata.
    """
    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas"],
    )

    chunks = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
    ids = results["ids"][0]

    for doc, metadata, chunk_id in zip(documents, metadatas, ids):
        chunks.append(
            RetrievedChunk(
                content=doc,
                chunk_id=metadata.get("chunk_id", chunk_id),
                metadata=metadata or {},
            )
        )

    return chunks
