"""Script to load documents directly into vector store."""

import sys
from pathlib import Path

import ai_exercise.configs  # noqa: F401 - Register all configs
from ai_exercise.configs.base import get_config
from ai_exercise.constants import chroma_client
from ai_exercise.llm.embeddings import openai_ef
from ai_exercise.loading.document_loader import (
    add_documents,
    load_all_specs,
    split_docs,
)
from ai_exercise.retrieval.bm25_index import BM25Index
from ai_exercise.retrieval.vector_store import create_collection

BM25_INDEX_DIR = Path(".bm25_index")


def get_bm25_index_path(collection_name: str) -> Path:
    """Get the path for a BM25 index file."""
    BM25_INDEX_DIR.mkdir(exist_ok=True)
    return BM25_INDEX_DIR / f"{collection_name}.pkl"


def load_for_config(config_name: str) -> None:
    """Load all OpenAPI specs for a specific config.

    Args:
        config_name: Name of the config (e.g., 'c0', 'c1', 'c2')
    """
    config = get_config(config_name)
    collection_name = f"{config_name}_vector_index"

    print(f"\nLoading data for config: {config_name}")
    print(f"Description: {config.description}")
    print(f"Smart chunking: {config.use_smart_chunking}")
    print(f"Collection: {collection_name}")

    # Create or get collection
    collection = create_collection(chroma_client, openai_ef, collection_name)

    # Check current count and clear if needed
    current_count = collection.count()
    if current_count > 0:
        print(f"Collection already has {current_count} documents. Clearing...")
        chroma_client.delete_collection(collection_name)
        collection = create_collection(chroma_client, openai_ef, collection_name)

    # Load all specs with config's chunking strategy
    print("Fetching OpenAPI specs...")
    documents = load_all_specs(use_smart_chunking=config.use_smart_chunking)
    print(f"Loaded {len(documents)} documents from specs")

    # Split docs that are too long
    print("Splitting documents...")
    documents = split_docs(documents)
    print(f"After splitting: {len(documents)} documents")

    # Load documents into vector store
    print("Adding documents to vector store...")
    add_documents(collection, documents)

    # Build and save BM25 index
    print("Building BM25 index...")
    doc_texts = [doc.page_content for doc in documents]
    doc_ids = [
        doc.metadata.get("chunk_id", f"doc_{i}") for i, doc in enumerate(documents)
    ]
    bm25_index = BM25Index(doc_texts, doc_ids)
    bm25_path = get_bm25_index_path(collection_name)
    bm25_index.save(bm25_path)
    print(f"BM25 index saved to {bm25_path}")

    # Verify
    final_count = collection.count()
    print(f"Final document count: {final_count}")
    print(f"Done loading {config_name}!")


def main() -> None:
    """Load documents for specified config or default collection."""
    # Check for config argument
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
        load_for_config(config_name)
    else:
        # Default behavior: load for c0
        print(
            "No config specified. Usage: "
            "python -m ai_exercise.loading.loader <config>"
        )
        print("Example: python -m ai_exercise.loading.loader c0")
        print("\nLoading for c0 by default...")
        load_for_config("c0")


if __name__ == "__main__":
    main()
