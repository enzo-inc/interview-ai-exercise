"""Smart chunking for OpenAPI specifications.

Creates endpoint-centric chunks with $ref resolution,
formatted as prose for better embedding quality.
"""

from typing import Any

from ai_exercise.loading.chunk_json import generate_chunk_id
from ai_exercise.loading.ref_resolver import resolve_refs_in_object
from ai_exercise.loading.structural_ids import (
    generate_structural_id,
    get_structural_id_for_component,
)
from ai_exercise.models import Document


def format_parameter(param: dict[str, Any]) -> str:
    """Format a parameter as a readable string.

    Args:
        param: OpenAPI parameter object

    Returns:
        Formatted string like "- name (location, required): description"
    """
    name = param.get("name", "unknown")
    location = param.get("in", "unknown")
    required = "required" if param.get("required", False) else "optional"
    description = param.get("description", "No description")

    # Include schema type if available
    schema = param.get("schema", {})
    param_type = schema.get("type", "")
    if param_type:
        return f"- {name} ({location}, {required}, {param_type}): {description}"
    return f"- {name} ({location}, {required}): {description}"


def format_schema_properties(
    schema: dict[str, Any], indent: int = 2, max_depth: int = 2, current_depth: int = 0
) -> str:
    """Format schema properties as readable text.

    Args:
        schema: OpenAPI schema object
        indent: Number of spaces for indentation
        max_depth: Maximum nesting depth to format
        current_depth: Current depth level

    Returns:
        Formatted string describing the schema properties
    """
    if current_depth >= max_depth:
        return f"{'  ' * indent}(nested object)"

    if not isinstance(schema, dict):
        return ""

    # Handle array types
    if schema.get("type") == "array":
        items = schema.get("items", {})
        if items.get("type") == "object":
            nested_props = format_schema_properties(
                items, indent, max_depth, current_depth + 1
            )
            return f"Array of objects with properties:\n{nested_props}"
        item_type = items.get("type", "unknown")
        return f"Array of {item_type}"

    # Handle object types
    properties = schema.get("properties", {})
    if not properties:
        schema_type = schema.get("type", "object")
        return f"{'  ' * indent}({schema_type})"

    lines = []
    required_fields = set(schema.get("required", []))

    for prop_name, prop_schema in list(properties.items())[:15]:  # Limit to 15 props
        prop_type = prop_schema.get("type", "unknown")
        description = prop_schema.get("description", "")
        required_marker = " (required)" if prop_name in required_fields else ""

        if prop_type == "object" and prop_schema.get("properties"):
            nested = format_schema_properties(
                prop_schema, indent + 1, max_depth, current_depth + 1
            )
            lines.append(
                f"{'  ' * indent}- {prop_name} (object){required_marker}: "
                f"{description}\n{nested}"
            )
        elif prop_type == "array":
            items = prop_schema.get("items", {})
            item_type = items.get("type", "unknown")
            lines.append(
                f"{'  ' * indent}- {prop_name} (array of {item_type})"
                f"{required_marker}: {description}"
            )
        else:
            lines.append(
                f"{'  ' * indent}- {prop_name} ({prop_type})"
                f"{required_marker}: {description}"
            )

    if len(properties) > 15:
        lines.append(f"{'  ' * indent}... and {len(properties) - 15} more fields")

    return "\n".join(lines)


def format_request_body(
    request_body: dict[str, Any] | None, spec: dict[str, Any]
) -> str:
    """Format request body as readable text.

    Args:
        request_body: OpenAPI requestBody object
        spec: Full OpenAPI spec for $ref resolution

    Returns:
        Formatted string describing the request body
    """
    if not request_body:
        return "None"

    content = request_body.get("content", {})
    json_content = content.get("application/json", {})
    schema = json_content.get("schema", {})

    if not schema:
        return "Request body expected (no schema defined)"

    # Resolve any $refs in the schema
    resolved_schema = resolve_refs_in_object(schema, spec, max_depth=2)

    description = request_body.get("description", "")
    required = " (required)" if request_body.get("required", False) else " (optional)"

    props = format_schema_properties(resolved_schema)
    if description:
        return f"{description}{required}\n{props}"
    return f"JSON body{required}\n{props}"


def format_response(
    responses: dict[str, Any],
    spec: dict[str, Any],
    status_codes: list[str] | None = None,
) -> str:
    """Format response schemas as readable text.

    Args:
        responses: OpenAPI responses object
        spec: Full OpenAPI spec for $ref resolution
        status_codes: List of status codes to include (default: ['200', '201'])

    Returns:
        Formatted string describing the responses
    """
    if status_codes is None:
        status_codes = ["200", "201"]

    lines = []

    for code in status_codes:
        response = responses.get(code)
        if not response:
            continue

        description = response.get("description", f"Response {code}")
        content = response.get("content", {})
        json_content = content.get("application/json", {})
        schema = json_content.get("schema", {})

        if schema:
            resolved_schema = resolve_refs_in_object(schema, spec, max_depth=2)
            props = format_schema_properties(resolved_schema)
            lines.append(f"Response ({code}): {description}\n{props}")
        else:
            lines.append(f"Response ({code}): {description}")

    return "\n\n".join(lines) if lines else "No success response documented"


def get_auth_info(spec: dict[str, Any]) -> str:
    """Extract authentication info from the spec.

    Args:
        spec: Full OpenAPI specification

    Returns:
        String describing the authentication method
    """
    security_schemes = spec.get("components", {}).get("securitySchemes", {})

    if not security_schemes:
        return "Not specified"

    auth_methods = []
    for _name, scheme in security_schemes.items():
        scheme_type = scheme.get("type", "unknown")
        if scheme_type == "apiKey":
            location = scheme.get("in", "header")
            key_name = scheme.get("name", "api-key")
            auth_methods.append(f"API Key ({key_name} in {location})")
        elif scheme_type == "http":
            http_scheme = scheme.get("scheme", "basic")
            auth_methods.append(f"HTTP {http_scheme.title()}")
        elif scheme_type == "oauth2":
            auth_methods.append("OAuth 2.0")
        else:
            auth_methods.append(scheme_type.title())

    return ", ".join(auth_methods) if auth_methods else "Not specified"


def build_endpoint_chunk(
    path: str,
    method: str,
    operation: dict[str, Any],
    spec: dict[str, Any],
    api_name: str,
) -> Document:
    """Build a single endpoint-centric chunk.

    Args:
        path: The endpoint path (e.g., '/hris/employees')
        method: HTTP method (e.g., 'GET', 'POST')
        operation: The operation object from the spec
        spec: Full OpenAPI specification
        api_name: Name of the API

    Returns:
        Document with prose-formatted endpoint information
    """
    # Extract operation details
    summary = operation.get("summary", "")
    description = operation.get("description", summary)
    operation_id = operation.get("operationId", f"{method}_{path}")
    tags = operation.get("tags", [])
    parameters = operation.get("parameters", [])
    request_body = operation.get("requestBody")
    responses = operation.get("responses", {})

    # Get auth info
    auth_info = get_auth_info(spec)

    # Resolve $refs in parameters
    resolved_params = [
        resolve_refs_in_object(p, spec, max_depth=1) for p in parameters
    ]

    # Format parameters
    params_text = "None"
    if resolved_params:
        params_lines = [format_parameter(p) for p in resolved_params]
        params_text = "\n".join(params_lines)

    # Format request body and response
    request_body_text = format_request_body(request_body, spec)
    response_text = format_response(responses, spec)

    # Build prose-formatted chunk
    chunk_content = f"""API: {api_name.upper()}
Endpoint: {method.upper()} {path}
Operation ID: {operation_id}
Tags: {', '.join(tags) if tags else 'None'}

Description: {description}

Authentication: {auth_info}

Parameters:
{params_text}

Request Body: {request_body_text}

{response_text}
"""

    # Generate deterministic chunk ID
    chunk_id = generate_chunk_id(api_name, "paths", f"{method}_{path}")

    # Generate structural ID for this endpoint
    endpoint_structural_id = generate_structural_id(api_name, "paths", path, method)
    covers = [endpoint_structural_id]

    import json as json_module

    return Document(
        page_content=chunk_content,
        metadata={
            "api_name": api_name,
            "source": "paths",
            "path": path,
            "method": method.upper(),
            "operation_id": operation_id,
            "tags": ",".join(tags) if tags else "",
            "chunk_type": "endpoint",
            "chunk_id": chunk_id,
            "resource_name": path,
            "covers": json_module.dumps(covers),  # Serialize list as JSON string
        },
    )


def build_endpoint_chunks(spec: dict[str, Any], api_name: str) -> list[Document]:
    """Build endpoint-centric chunks for all endpoints in the spec.

    Args:
        spec: Full OpenAPI specification
        api_name: Name of the API

    Returns:
        List of Documents, one per endpoint (path + method combination)
    """
    documents = []
    paths = spec.get("paths", {})

    for path, path_item in paths.items():
        # Iterate through HTTP methods
        for method in ["get", "post", "put", "patch", "delete", "options", "head"]:
            operation = path_item.get(method)
            if operation:
                doc = build_endpoint_chunk(path, method, operation, spec, api_name)
                documents.append(doc)

    return documents


def build_schema_chunks(spec: dict[str, Any], api_name: str) -> list[Document]:
    """Build chunks for standalone schemas (components/schemas).

    These are useful for schema-specific questions not tied to endpoints.

    Args:
        spec: Full OpenAPI specification
        api_name: Name of the API

    Returns:
        List of Documents, one per top-level schema
    """
    documents = []
    schemas = spec.get("components", {}).get("schemas", {})

    for schema_name, schema in schemas.items():
        resolved_schema = resolve_refs_in_object(schema, spec, max_depth=2)
        description = resolved_schema.get("description", f"Schema for {schema_name}")
        props = format_schema_properties(resolved_schema)

        chunk_content = f"""API: {api_name.upper()}
Schema: {schema_name}
Type: {resolved_schema.get('type', 'object')}

Description: {description}

Properties:
{props}
"""

        chunk_id = generate_chunk_id(api_name, "components", schema_name)

        # Generate structural ID for this schema
        schema_structural_id = get_structural_id_for_component(api_name, schema_name)
        covers = [schema_structural_id]

        import json as json_module

        documents.append(
            Document(
                page_content=chunk_content,
                metadata={
                    "api_name": api_name,
                    "source": "components",
                    "schema_name": schema_name,
                    "chunk_type": "schema",
                    "chunk_id": chunk_id,
                    "resource_name": schema_name,
                    "covers": json_module.dumps(
                        covers
                    ),  # Serialize list as JSON string
                },
            )
        )

    return documents


def build_smart_chunks(spec: dict[str, Any], api_name: str) -> list[Document]:
    """Build all smart chunks for an API spec.

    Combines endpoint chunks and schema chunks.

    Args:
        spec: Full OpenAPI specification
        api_name: Name of the API

    Returns:
        List of all Documents for this spec
    """
    documents = []

    # Build endpoint chunks
    documents.extend(build_endpoint_chunks(spec, api_name))

    # Build schema chunks
    documents.extend(build_schema_chunks(spec, api_name))

    return documents
