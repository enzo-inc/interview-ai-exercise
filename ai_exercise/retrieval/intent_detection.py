"""Query intent detection for metadata filtering.

Detects which API(s) a query is asking about to enable targeted retrieval.
"""

from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from ai_exercise.constants import OPENAPI_SPECS

# Available API names
AVAILABLE_APIS = list(OPENAPI_SPECS.keys())

# API descriptions for better intent detection
API_DESCRIPTIONS = {
    "stackone": (
        "Core StackOne platform - connect sessions, accounts, connectors, "
        "proxy requests"
    ),
    "hris": (
        "Human Resources - employees, employments, time off, documents, "
        "departments, locations, benefits"
    ),
    "ats": (
        "Applicant Tracking System - candidates, applications, jobs, "
        "interviews, offers, scorecards"
    ),
    "lms": (
        "Learning Management System - courses, content, completions, "
        "assignments, categories"
    ),
    "iam": (
        "Identity & Access Management - users, roles, permissions, groups, "
        "policies"
    ),
    "crm": (
        "Customer Relationship Management - contacts, accounts, lists, "
        "campaigns"
    ),
    "marketing": (
        "Marketing automation - email templates, campaigns, push templates, "
        "content blocks"
    ),
}


class QueryIntent(BaseModel):
    """Structured output for query intent detection."""

    apis: list[
        Literal["stackone", "hris", "ats", "lms", "iam", "crm", "marketing", "all"]
    ] = Field(
        description=(
            "List of API names relevant to the query. Use 'all' if the query "
            "is general or spans all APIs."
        )
    )


async def detect_query_intent(
    client: AsyncOpenAI,
    query: str,
    model: str = "gpt-5-mini-2025-08-07",
) -> list[str]:
    """Detect which API(s) a query is asking about.

    Uses an LLM with structured outputs to analyze the query and determine
    which StackOne API(s) are most relevant. Returns a list of API names
    for metadata filtering.

    Args:
        client: AsyncOpenAI client instance.
        query: The user's query.
        model: Model to use for intent detection.
            Default: gpt-5-mini-2025-08-07 (for speed/cost).

    Returns:
        List of API names that the query is relevant to.
        Returns all APIs if intent cannot be determined or query spans multiple domains.
    """
    api_list = "\n".join(
        f"- {name}: {desc}" for name, desc in API_DESCRIPTIONS.items()
    )

    system_prompt = (
        f"""You are an API intent classifier. Analyze user queries and """
        f"""determine which StackOne API(s) they are asking about.

Available APIs:
{api_list}

Rules:
1. If the query clearly relates to one API, return just that API
2. If the query spans multiple APIs or compares them, return all relevant APIs
3. If the query is general/unclear or asks about authentication/connectors, return "all"
4. Be precise - don't include APIs that aren't directly relevant

Examples:
- "How do I list employees?" -> apis: ["hris"]
- "What endpoints handle job applications?" -> apis: ["ats"]
- "How do I create a course?" -> apis: ["lms"]
- "Compare employee and candidate endpoints" -> apis: ["hris", "ats"]
- "How do I authenticate?" -> apis: ["stackone"]
- "What APIs are available?" -> apis: ["all"]"""
    )

    response = await client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        response_format=QueryIntent,
    )

    result = response.choices[0].message.parsed

    # Handle parse failure
    if result is None:
        return AVAILABLE_APIS

    # Handle "all" case
    if "all" in result.apis:
        return AVAILABLE_APIS

    # Filter to only valid API names (excluding "all")
    valid_apis = [api for api in result.apis if api in AVAILABLE_APIS]

    # Return all if no valid APIs detected
    if not valid_apis:
        return AVAILABLE_APIS

    return valid_apis
