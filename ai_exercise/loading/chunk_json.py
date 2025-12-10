"""Basic function to chunk JSON data by key."""

import re
from typing import Any


def normalize_chunk_id(raw_id: str) -> str:
    """Normalize a chunk ID to be safe for use as a ChromaDB ID.

    Converts paths like '/unified/hris/employees/{id}' to 'unified_hris_employees_id'
    """
    # Remove leading/trailing slashes
    normalized = raw_id.strip("/")
    # Replace path parameters {param} with just param
    normalized = re.sub(r"\{(\w+)\}", r"\1", normalized)
    # Replace non-alphanumeric chars with underscores
    normalized = re.sub(r"[^a-zA-Z0-9]", "_", normalized)
    # Collapse multiple underscores
    normalized = re.sub(r"_+", "_", normalized)
    # Remove leading/trailing underscores
    normalized = normalized.strip("_")
    return normalized.lower()


def generate_chunk_id(api_name: str, source: str, key: str) -> str:
    """Generate a deterministic chunk ID.

    Format: {api_name}_{source}_{normalized_key}
    Example: hris_paths_unified_hris_employees

    Args:
        api_name: Name of the API (e.g., 'hris', 'ats')
        source: Type of chunk (e.g., 'paths', 'components', 'webhooks')
        key: The specific path or schema name

    Returns:
        A deterministic, normalized chunk ID.
    """
    normalized_key = normalize_chunk_id(key)
    return f"{api_name}_{source}_{normalized_key}"


def chunk_data(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """Chunk JSON data by key, extracting each sub-key as a separate chunk.

    Args:
        data: The full JSON data (e.g., OpenAPI spec).
        key: The top-level key to chunk (e.g., 'paths', 'components').

    Returns:
        List of dicts, each containing one sub-key and its data.
    """
    info = data.get(key, {})
    return [{sub_key: sub_info} for sub_key, sub_info in info.items()]


def chunk_data_with_ids(
    data: dict[str, Any], key: str, api_name: str
) -> list[tuple[str, str, dict[str, Any]]]:
    """Chunk JSON data by key, returning (chunk_id, original_key, chunk_data) tuples.

    Args:
        data: The full JSON data (e.g., OpenAPI spec).
        key: The top-level key to chunk (e.g., 'paths', 'components').
        api_name: Name of the API for ID generation.

    Returns:
        List of (chunk_id, original_key, chunk_data) tuples.
    """
    info = data.get(key, {})
    return [
        (generate_chunk_id(api_name, key, sub_key), sub_key, {sub_key: sub_info})
        for sub_key, sub_info in info.items()
    ]
