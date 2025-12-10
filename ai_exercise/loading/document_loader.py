"""Document loader for the RAG example."""

import json
from typing import Any

import chromadb
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai_exercise.constants import OPENAPI_SPECS, SETTINGS
from ai_exercise.loading.chunk_json import chunk_data
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


def document_json_array(
    data: list[dict[str, Any]], source: str, api_name: str
) -> list[Document]:
    """Converts an array of JSON chunks into a list of Document objects.

    Args:
        data: List of JSON chunks to convert.
        source: The source type (paths, webhooks, or components).
        api_name: The name of the API this chunk belongs to.

    Returns:
        List of Document objects with metadata.
    """
    return [
        Document(
            page_content=json.dumps(item),
            metadata={"source": source, "api_name": api_name},
        )
        for item in data
    ]


def build_docs(data: dict[str, Any], api_name: str) -> list[Document]:
    """Chunk (badly) and convert the JSON data into a list of Document objects.

    Args:
        data: The JSON data to chunk and convert.
        api_name: The name of the API this data belongs to.

    Returns:
        List of Document objects with metadata.
    """
    docs = []
    for attribute in ["paths", "webhooks", "components"]:
        chunks = chunk_data(data, attribute)
        docs.extend(document_json_array(chunks, attribute, api_name))
    return docs


def load_all_specs() -> list[Document]:
    """Load all OpenAPI specs with api_name metadata.

    Iterates through all specs defined in OPENAPI_SPECS,
    fetches them, chunks them, and returns combined documents.

    Returns:
        List of Document objects from all API specs.
    """
    all_docs: list[Document] = []
    for api_name, url in OPENAPI_SPECS.items():
        json_data = get_json_data(url)
        docs = build_docs(json_data, api_name=api_name)
        all_docs.extend(docs)
    return all_docs


def split_docs(docs_array: list[Document]) -> list[Document]:
    """Some may still be too long, so we split them."""
    splitter = RecursiveCharacterTextSplitter(
        separators=["}],", "},", "}", "]", " ", ""], chunk_size=SETTINGS.chunk_size
    )
    return splitter.split_documents(docs_array)


def add_documents(
    collection: chromadb.Collection, docs: list[Document], batch_size: int = 100
) -> None:
    """Add documents to the collection in batches.

    Args:
        collection: ChromaDB collection to add to.
        docs: List of documents to add.
        batch_size: Number of documents per batch to avoid API token limits.
    """
    total = len(docs)
    for i in range(0, total, batch_size):
        batch = docs[i : i + batch_size]
        collection.add(
            documents=[doc.page_content for doc in batch],
            metadatas=[doc.metadata or {} for doc in batch],
            ids=[f"doc_{j}" for j in range(i, i + len(batch))],
        )
        print(f"Added batch {i // batch_size + 1} ({i + len(batch)}/{total})")
