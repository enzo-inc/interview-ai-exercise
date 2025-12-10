"""Structural ID generation for config-agnostic evaluation.

Structural IDs are derived from the JSON structure itself, not from chunking
strategies. This allows evaluation to work consistently across different
chunking approaches (naive, smart, etc.).

Examples:
    - "stackone.paths./connect_sessions.post" (endpoint)
    - "stackone.components.ConnectSessionCreate" (schema)
"""


def generate_structural_id(
    api_name: str,
    section: str,
    key: str,
    method: str | None = None,
) -> str:
    """Generate a stable structural ID from JSON path.

    Args:
        api_name: Name of the API (e.g., 'stackone', 'hris')
        section: Section of the spec ('paths', 'components', 'webhooks')
        key: The specific path or schema name
        method: HTTP method for paths (e.g., 'get', 'post')

    Returns:
        A stable structural ID like "stackone.paths./connect_sessions.post"

    Examples:
        >>> generate_structural_id("stackone", "paths", "/connect_sessions", "post")
        'stackone.paths./connect_sessions.post'
        >>> generate_structural_id("stackone", "components", "ConnectSessionCreate")
        'stackone.components.ConnectSessionCreate'
    """
    if method:
        return f"{api_name}.paths.{key}.{method.lower()}"
    return f"{api_name}.{section}.{key}"


def get_structural_ids_for_path(
    api_name: str,
    path: str,
    path_item: dict,
) -> list[str]:
    """Get all structural IDs for a path (all HTTP methods).

    Args:
        api_name: Name of the API
        path: The endpoint path (e.g., '/connect_sessions')
        path_item: The path item object containing methods

    Returns:
        List of structural IDs for each method defined on this path
    """
    http_methods = ["get", "post", "put", "patch", "delete", "options", "head"]
    ids = []

    for method in http_methods:
        if method in path_item:
            ids.append(generate_structural_id(api_name, "paths", path, method))

    return ids


def get_structural_id_for_component(api_name: str, schema_name: str) -> str:
    """Get structural ID for a component/schema.

    Args:
        api_name: Name of the API
        schema_name: Name of the schema

    Returns:
        Structural ID like "stackone.components.ConnectSessionCreate"
    """
    return generate_structural_id(api_name, "components", schema_name)
