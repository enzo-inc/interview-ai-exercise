"""Document loader for the RAG example."""

import json
from typing import Any

import chromadb
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai_exercise.constants import OPENAPI_SPECS, SETTINGS
from ai_exercise.loading.chunk_json import chunk_data_with_ids
from ai_exercise.loading.smart_chunker import build_smart_chunks
from ai_exercise.loading.structural_ids import (
    generate_structural_id,
    get_structural_id_for_component,
    get_structural_ids_for_path,
)
from ai_exercise.models import Document


def get_json_data(url: str) -> dict[str, Any]:
    """Fetch JSON data from a URL.

    Args:
        url: The URL to fetch JSON data from.

    Returns:
        The parsed JSON data as a dictionary.
    """
    response = requests.get(url)
    json_data = response.json()
    response.raise_for_status()

    return json_data


def document_json_array_with_ids(
    chunks_with_ids: list[tuple[str, str, dict[str, Any]]],
    source: str,
    api_name: str,
) -> list[Document]:
    """Convert (chunk_id, original_key, chunk_data) tuples to Documents.

    Args:
        chunks_with_ids: List of (chunk_id, original_key, chunk_data) tuples.
        source: The source type (paths, webhooks, or components).
        api_name: The name of the API this chunk belongs to.

    Returns:
        List of Document objects with metadata including chunk_id, resource_name,
        and covers (list of structural IDs this chunk contains).
    """
    documents = []
    for chunk_id, original_key, chunk_data in chunks_with_ids:
        # Compute structural IDs this chunk covers
        covers: list[str] = []

        if source == "paths":
            # For paths, chunk_data is {"/path": {methods...}}
            # Extract all HTTP methods and generate structural IDs
            path_item = chunk_data.get(original_key, {})
            covers = get_structural_ids_for_path(api_name, original_key, path_item)
        elif source == "components":
            # For components, it's a single schema
            covers = [get_structural_id_for_component(api_name, original_key)]
        elif source == "webhooks":
            # For webhooks, generate a webhook structural ID
            covers = [generate_structural_id(api_name, "webhooks", original_key)]

        documents.append(
            Document(
                page_content=json.dumps(chunk_data),
                metadata={
                    "source": source,
                    "api_name": api_name,
                    "chunk_id": chunk_id,
                    "resource_name": original_key,
                    "covers": json.dumps(covers),  # Serialize list as JSON string
                },
            )
        )

    return documents


def build_docs(data: dict[str, Any], api_name: str) -> list[Document]:
    """Chunk and convert the JSON data into a list of Document objects.

    Args:
        data: The JSON data to chunk and convert.
        api_name: The name of the API this data belongs to.

    Returns:
        List of Document objects with metadata including chunk IDs.
    """
    docs = []
    for attribute in ["paths", "webhooks", "components"]:
        chunks_with_ids = chunk_data_with_ids(data, attribute, api_name)
        docs.extend(document_json_array_with_ids(chunks_with_ids, attribute, api_name))
    return docs


def load_all_specs(use_smart_chunking: bool = False) -> list[Document]:
    """Load all OpenAPI specs with api_name metadata.

    Iterates through all specs defined in OPENAPI_SPECS,
    fetches them, chunks them, and returns combined documents.

    Args:
        use_smart_chunking: If True, use endpoint-centric chunking with $ref
            resolution. If False, use naive JSON key-based chunking.

    Returns:
        List of Document objects from all API specs.
    """
    all_docs: list[Document] = []
    for api_name, url in OPENAPI_SPECS.items():
        json_data = get_json_data(url)
        if use_smart_chunking:
            docs = build_smart_chunks(json_data, api_name=api_name)
        else:
            docs = build_docs(json_data, api_name=api_name)
        all_docs.extend(docs)
    return all_docs


def split_docs(docs_array: list[Document]) -> list[Document]:
    """Some may still be too long, so we split them.

    Preserves chunk_id in metadata, appending a sub-index for split chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["}],", "},", "}", "]", " ", ""], chunk_size=SETTINGS.chunk_size
    )

    # Split documents and preserve chunk IDs
    split_documents = []
    for doc in docs_array:
        splits = splitter.split_documents([doc])
        base_chunk_id = doc.metadata.get("chunk_id", "unknown")

        if len(splits) == 1:
            # No splitting needed, keep original chunk_id
            split_documents.append(splits[0])
        else:
            # Multiple splits, append sub-index to chunk_id
            for idx, split_doc in enumerate(splits):
                split_doc.metadata["chunk_id"] = f"{base_chunk_id}_part{idx}"
                split_doc.metadata["parent_chunk_id"] = base_chunk_id
                split_documents.append(split_doc)

    return split_documents


def add_documents(
    collection: chromadb.Collection, docs: list[Document], batch_size: int = 100
) -> None:
    """Add documents to the collection in batches.

    Uses chunk_id from metadata as the ChromaDB document ID for deterministic
    retrieval matching.

    Args:
        collection: ChromaDB collection to add to.
        docs: List of documents to add.
        batch_size: Number of documents per batch to avoid API token limits.
    """
    total = len(docs)
    for i in range(0, total, batch_size):
        batch = docs[i : i + batch_size]
        # Use chunk_id from metadata as ChromaDB ID
        ids = [
            doc.metadata.get("chunk_id", f"doc_{i + j}")
            for j, doc in enumerate(batch)
        ]
        collection.add(
            documents=[doc.page_content for doc in batch],
            metadatas=[doc.metadata or {} for doc in batch],
            ids=ids,
        )
        print(f"Added batch {i // batch_size + 1} ({i + len(batch)}/{total})")
