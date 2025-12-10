"""LLM-based re-ranking of retrieved chunks.

This module provides re-ranking functionality using an LLM to refine
the top-K results from initial retrieval, improving precision.
"""

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from ai_exercise.retrieval.retrieval import RetrievedChunk


class RerankResult(BaseModel):
    """Structured output for re-ranking results."""

    ranked_indices: list[int] = Field(
        description="List of chunk indices in order of relevance (most relevant first)"
    )


async def rerank_chunks(
    client: AsyncOpenAI,
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int = 5,
    model: str = "gpt-5-mini-2025-08-07",
) -> list[RetrievedChunk]:
    """Re-rank retrieved chunks using an LLM.

    Takes a list of candidate chunks and uses an LLM to re-order them
    by relevance to the query. This can significantly improve precision
    by using the LLM's deeper understanding of relevance.

    Args:
        client: AsyncOpenAI client instance.
        query: The user's query.
        chunks: List of RetrievedChunk objects to re-rank.
        top_k: Number of top results to return after re-ranking.
        model: Model to use for re-ranking (default: gpt-5-mini-2025-08-07).

    Returns:
        List of RetrievedChunk objects re-ordered by relevance,
        limited to top_k results.
    """
    if not chunks:
        return []

    # If we have fewer chunks than requested, just return them all
    if len(chunks) <= top_k:
        return chunks

    # Format chunks for the prompt
    chunks_text = _format_chunks_for_prompt(chunks)

    system_prompt = """You are a relevance ranking expert. Given a query and a list of document chunks from API documentation, rank the chunks by their relevance to answering the query.

Rules:
1. Rank chunks that directly answer the query highest
2. Consider both exact matches and semantic relevance
3. Chunks with specific endpoint/schema information are more valuable than general descriptions
4. Return indices of ALL chunks in order of relevance (most relevant first)
5. Be precise - the order matters for retrieval quality

Return the indices as a list of integers, e.g., [2, 0, 4, 1, 3] means chunk 2 is most relevant, then chunk 0, etc."""

    user_prompt = f"""Query: {query}

Chunks to rank:
{chunks_text}

Rank all {len(chunks)} chunks by relevance to the query."""

    response = await client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=RerankResult,
    )

    result = response.choices[0].message.parsed

    # Handle parse failure - return original order
    if result is None:
        return chunks[:top_k]

    # Validate and apply ranking
    ranked_chunks = _apply_ranking(chunks, result.ranked_indices, top_k)

    return ranked_chunks


def _format_chunks_for_prompt(chunks: list[RetrievedChunk]) -> str:
    """Format chunks for inclusion in the re-ranking prompt.

    Args:
        chunks: List of chunks to format.

    Returns:
        Formatted string with numbered chunks.
    """
    formatted_parts = []

    for idx, chunk in enumerate(chunks):
        # Include metadata context if available
        metadata_str = ""
        if chunk.metadata:
            api_name = chunk.metadata.get("api_name", "unknown")
            method = chunk.metadata.get("method", "")
            path = chunk.metadata.get("path", "")
            if method and path:
                metadata_str = f" [API: {api_name}, {method} {path}]"
            else:
                metadata_str = f" [API: {api_name}]"

        # Truncate very long chunks to avoid token limits
        content = chunk.content
        max_chunk_length = 1500
        if len(content) > max_chunk_length:
            content = content[:max_chunk_length] + "..."

        formatted_parts.append(f"[{idx}]{metadata_str}\n{content}")

    return "\n\n---\n\n".join(formatted_parts)


def _apply_ranking(
    chunks: list[RetrievedChunk],
    ranked_indices: list[int],
    top_k: int,
) -> list[RetrievedChunk]:
    """Apply the ranking to get top-k chunks.

    Handles edge cases like invalid indices or incomplete rankings.

    Args:
        chunks: Original list of chunks.
        ranked_indices: Indices in ranked order from LLM.
        top_k: Number of results to return.

    Returns:
        List of chunks in ranked order, limited to top_k.
    """
    # Filter to valid indices only
    valid_indices = [idx for idx in ranked_indices if 0 <= idx < len(chunks)]

    # If we don't have enough valid indices, add missing ones at the end
    seen = set(valid_indices)
    for idx in range(len(chunks)):
        if idx not in seen:
            valid_indices.append(idx)

    # Apply ranking and limit to top_k
    ranked_chunks = [chunks[idx] for idx in valid_indices[:top_k]]

    return ranked_chunks
