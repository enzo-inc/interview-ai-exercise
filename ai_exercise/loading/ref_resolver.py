"""$ref resolution for OpenAPI specifications.

Resolves JSON $ref pointers to inline referenced schemas,
handling circular references and depth limits.
"""

from typing import Any


def resolve_ref(ref_path: str, spec: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve a single $ref pointer to its target in the spec.

    Args:
        ref_path: The $ref path (e.g., '#/components/schemas/Employee')
        spec: The full OpenAPI specification

    Returns:
        The resolved schema dict, or None if not found.
    """
    if not ref_path.startswith("#/"):
        # External refs not supported
        return None

    # Parse path like '#/components/schemas/Employee'
    parts = ref_path[2:].split("/")
    current = spec

    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None

    return current if isinstance(current, dict) else None


def resolve_refs_in_object(
    obj: Any,
    spec: dict[str, Any],
    max_depth: int = 2,
    current_depth: int = 0,
    visited: set[str] | None = None,
) -> Any:
    """Recursively resolve $ref pointers in an object.

    Args:
        obj: The object to resolve refs in (can be dict, list, or primitive)
        spec: The full OpenAPI specification for lookups
        max_depth: Maximum depth to resolve nested refs
        current_depth: Current recursion depth
        visited: Set of already visited $ref paths (to prevent cycles)

    Returns:
        The object with $refs resolved (up to max_depth).
    """
    if visited is None:
        visited = set()

    if current_depth >= max_depth:
        return obj

    if isinstance(obj, dict):
        # Check if this is a $ref object
        if "$ref" in obj and len(obj) == 1:
            ref_path = obj["$ref"]

            # Prevent circular references
            if ref_path in visited:
                return {"$ref": ref_path, "_circular": True}

            resolved = resolve_ref(ref_path, spec)
            if resolved is not None:
                # Track this ref to prevent cycles
                new_visited = visited | {ref_path}
                # Recursively resolve the resolved schema
                return resolve_refs_in_object(
                    resolved, spec, max_depth, current_depth + 1, new_visited
                )
            return obj

        # Not a $ref, recursively process all values
        return {
            key: resolve_refs_in_object(
                value, spec, max_depth, current_depth, visited
            )
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [
            resolve_refs_in_object(item, spec, max_depth, current_depth, visited)
            for item in obj
        ]

    # Primitive value, return as-is
    return obj


def get_schema_name_from_ref(ref_path: str) -> str | None:
    """Extract the schema name from a $ref path.

    Args:
        ref_path: The $ref path (e.g., '#/components/schemas/Employee')

    Returns:
        The schema name (e.g., 'Employee'), or None if invalid.
    """
    if not ref_path.startswith("#/"):
        return None

    parts = ref_path.split("/")
    return parts[-1] if parts else None
