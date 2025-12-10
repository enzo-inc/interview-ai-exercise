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
    collection: chromadb.Collection,
    query: str,
    k: int,
    api_filter: list[str] | None = None,
) -> list[RetrievedChunk]:
    """Retrieve k most relevant chunks with their IDs and metadata.

    Args:
        collection: ChromaDB collection to query.
        query: The query text.
        k: Number of results to return.
        api_filter: Optional list of API names to filter by.
            If provided, only chunks from these APIs are returned.

    Returns:
        List of RetrievedChunk objects with content, chunk_id, and metadata.
    """
    # Build where clause for API filtering
    where = None
    if api_filter and len(api_filter) > 0:
        if len(api_filter) == 1:
            where = {"api_name": api_filter[0]}
        else:
            where = {"api_name": {"$in": api_filter}}

    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas"],
        where=where,
    )

    chunks = []
    documents = results["documents"][0]
    metadatas = (
        results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
    )
    ids = results["ids"][0]

    for doc, metadata, chunk_id in zip(documents, metadatas, ids, strict=False):
        chunks.append(
            RetrievedChunk(
                content=doc,
                chunk_id=metadata.get("chunk_id", chunk_id),
                metadata=metadata or {},
            )
        )

    return chunks
