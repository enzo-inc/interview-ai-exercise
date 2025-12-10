"""Script to load documents directly into vector store."""

from ai_exercise.constants import SETTINGS, chroma_client
from ai_exercise.llm.embeddings import openai_ef
from ai_exercise.loading.document_loader import (
    add_documents,
    load_all_specs,
    split_docs,
)
from ai_exercise.retrieval.vector_store import create_collection


def main() -> None:
    """Load all OpenAPI specs into the vector store."""
    print("Loading all 7 OpenAPI specs...")

    # Create collection
    collection = create_collection(
        chroma_client, openai_ef, SETTINGS.collection_name
    )

    # Check current count
    current_count = collection.count()
    if current_count > 0:
        print(f"Collection already has {current_count} documents.")
        print("Clearing collection...")
        # Clear existing documents by recreating the collection
        chroma_client.delete_collection(SETTINGS.collection_name)
        collection = create_collection(
            chroma_client, openai_ef, SETTINGS.collection_name
        )

    # Load all specs with api_name metadata
    print("Fetching OpenAPI specs...")
    documents = load_all_specs()
    print(f"Loaded {len(documents)} documents from specs")

    # Split docs that are too long
    print("Splitting documents...")
    documents = split_docs(documents)
    print(f"After splitting: {len(documents)} documents")

    # Load documents into vector store
    print("Adding documents to vector store...")
    add_documents(collection, documents)

    # Verify
    final_count = collection.count()
    print(f"Final document count: {final_count}")
    print("Done!")


if __name__ == "__main__":
    main()
