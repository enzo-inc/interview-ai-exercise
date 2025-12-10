"""FastAPI app creation, main API routes."""

from pathlib import Path

from fastapi import FastAPI

from ai_exercise.configs.base import CONFIGS, get_config
from ai_exercise.constants import SETTINGS, chroma_client, openai_client
from ai_exercise.llm.completions import create_prompt, get_completion
from ai_exercise.llm.embeddings import openai_ef
from ai_exercise.loading.document_loader import (
    add_documents,
    load_all_specs,
    split_docs,
)
from ai_exercise.models import (
    ChatOutput,
    ChatQuery,
    HealthRouteOutput,
    LoadDocumentsOutput,
    SourceInfo,
)
from ai_exercise.retrieval.bm25_index import BM25Index
from ai_exercise.retrieval.hybrid_search import get_relevant_chunks_hybrid
from ai_exercise.retrieval.retrieval import get_relevant_chunks_with_ids
from ai_exercise.retrieval.vector_store import create_collection

app = FastAPI()

config = get_config(SETTINGS.config_name)

# Track current collection (mutable at runtime)
current_collection_name = SETTINGS.collection_name
collection = create_collection(chroma_client, openai_ef, current_collection_name)

# BM25 index storage
BM25_INDEX_DIR = Path(".bm25_index")
current_bm25_index: BM25Index | None = None


def get_bm25_index_path(collection_name: str) -> Path:
    """Get the path for a BM25 index file."""
    return BM25_INDEX_DIR / f"{collection_name}.pkl"


def load_bm25_index_if_exists(collection_name: str) -> BM25Index | None:
    """Load BM25 index from disk if it exists."""
    path = get_bm25_index_path(collection_name)
    if path.exists():
        try:
            return BM25Index.load(path)
        except Exception as e:
            print(f"Warning: Failed to load BM25 index: {e}")
    return None


# Try to load existing BM25 index for current collection
current_bm25_index = load_bm25_index_if_exists(current_collection_name)


@app.get("/health")
def health_check_route() -> HealthRouteOutput:
    """Health check route to check that the API is up."""
    return HealthRouteOutput(status="ok")


@app.get("/configs")
def list_configs() -> dict:
    """List all available configurations."""
    # Import configs module to ensure all configs are registered
    import ai_exercise.configs  # noqa: F401

    return {
        "configs": list(CONFIGS.keys()),
        "current": config.name,
    }


@app.post("/configs/{name}/select")
def select_config(name: str) -> dict:
    """Switch to a different configuration and its matching collection.

    Automatically selects the collection named '{config_name}_vector_index'.
    Also loads the BM25 index if it exists for hybrid search.
    """
    global config, collection, current_collection_name, current_bm25_index

    # Switch config
    config = get_config(name)

    # Auto-select matching collection
    matching_collection = f"{name}_vector_index"
    collection = create_collection(chroma_client, openai_ef, matching_collection)
    current_collection_name = matching_collection

    # Load BM25 index if it exists
    current_bm25_index = load_bm25_index_if_exists(matching_collection)

    return {
        "status": "ok",
        "config": name,
        "collection": matching_collection,
        "bm25_index_loaded": current_bm25_index is not None,
    }


@app.get("/config")
def config_route() -> dict:
    """Get the current system configuration."""
    return {
        "name": config.name,
        "description": config.description,
        "use_smart_chunking": config.use_smart_chunking,
        "use_hybrid_search": config.use_hybrid_search,
        "use_metadata_filtering": config.use_metadata_filtering,
        "use_reranking": config.use_reranking,
        "use_unknown_detection": config.use_unknown_detection,
    }


@app.get("/collections")
def list_collections() -> dict:
    """List all available collections and current selection."""
    # In ChromaDB v0.6.0+, list_collections() returns names directly as strings
    collections = chroma_client.list_collections()
    return {
        "collections": collections,
        "current": current_collection_name,
    }


@app.post("/collections/{name}/select")
def select_collection(name: str) -> dict:
    """Switch to a different collection."""
    global collection, current_collection_name
    collection = create_collection(chroma_client, openai_ef, name)
    current_collection_name = name
    return {"status": "ok", "collection": name}


@app.get("/load")
async def load_docs_route(
    collection_name: str | None = None,
    use_smart_chunking: bool | None = None,
) -> LoadDocumentsOutput:
    """Route to load all 7 OpenAPI specs into vector store and BM25 index.

    Args:
        collection_name: Optional collection name to load into. If not specified,
                        loads into the current collection.
        use_smart_chunking: Override config's smart chunking setting. If not specified,
                           uses the current config's use_smart_chunking value.
    """
    global current_bm25_index

    # Determine target collection
    target_name = collection_name or current_collection_name
    target_collection = create_collection(chroma_client, openai_ef, target_name)

    # Determine chunking strategy
    smart_chunking = (
        use_smart_chunking
        if use_smart_chunking is not None
        else config.use_smart_chunking
    )

    print(f"Loading specs with smart_chunking={smart_chunking}")

    # Load all specs with api_name metadata
    documents = load_all_specs(use_smart_chunking=smart_chunking)

    # split docs that are too long
    documents = split_docs(documents)

    # load documents into vector store
    add_documents(target_collection, documents)

    # Build BM25 index
    doc_texts = [doc.page_content for doc in documents]
    doc_ids = [
        doc.metadata.get("chunk_id", f"doc_{i}")
        for i, doc in enumerate(documents)
    ]
    bm25_index = BM25Index(doc_texts, doc_ids)

    # Save BM25 index
    bm25_path = get_bm25_index_path(target_name)
    bm25_index.save(bm25_path)
    print(f"BM25 index saved to {bm25_path} with {len(bm25_index)} documents")

    # Update current BM25 index if loading into current collection
    if target_name == current_collection_name:
        current_bm25_index = bm25_index

    # check the number of documents in the collection
    print(
        f"Number of documents in collection '{target_name}': "
        f"{target_collection.count()}"
    )

    return LoadDocumentsOutput(status="ok")


@app.post("/chat")
def chat_route(chat_query: ChatQuery) -> ChatOutput:
    """Chat route to chat with the API."""
    # Get relevant chunks - use hybrid search if enabled and BM25 index is available
    if config.use_hybrid_search and current_bm25_index is not None:
        relevant_chunks = get_relevant_chunks_hybrid(
            collection=collection,
            bm25_index=current_bm25_index,
            query=chat_query.query,
            k=SETTINGS.k_neighbors,
        )
        print(
            f"Using hybrid search (BM25 + Vector) for query: "
            f"{chat_query.query[:50]}..."
        )
    else:
        relevant_chunks = get_relevant_chunks_with_ids(
            collection=collection, query=chat_query.query, k=SETTINGS.k_neighbors
        )
        if config.use_hybrid_search and current_bm25_index is None:
            print(
                "Warning: Hybrid search enabled but BM25 index not loaded. "
                "Using vector only."
            )

    # Extract content for prompt and source info for response
    context = [chunk.content for chunk in relevant_chunks]

    # Build source info from metadata, deduplicating by
    # (api_name, source, resource_name)
    seen_sources = set()
    sources = []
    for chunk in relevant_chunks:
        meta = chunk.metadata
        key = (
            meta.get("api_name", "unknown"),
            meta.get("source", "unknown"),
            meta.get("resource_name", "unknown"),
        )
        if key not in seen_sources:
            seen_sources.add(key)
            sources.append(
                SourceInfo(
                    api_name=key[0],
                    source_type=key[1],
                    resource_name=key[2],
                )
            )

    # Create prompt with context
    prompt = create_prompt(query=chat_query.query, context=context)

    print(f"Prompt: {prompt}")

    # Get completion from LLM
    result = get_completion(
        client=openai_client,
        prompt=prompt,
        model=SETTINGS.openai_model,
    )

    return ChatOutput(message=result, sources=sources)


@app.get("/bm25/status")
def bm25_status() -> dict:
    """Get BM25 index status for current collection.

    Returns:
        Dict with BM25 index status including whether it exists and document count.
    """
    has_index = current_bm25_index is not None
    return {
        "has_index": has_index,
        "collection_name": current_collection_name,
        "document_count": len(current_bm25_index) if has_index else 0,
        "index_path": str(get_bm25_index_path(current_collection_name)),
        "hybrid_search_enabled": config.use_hybrid_search,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=80, reload=True)
