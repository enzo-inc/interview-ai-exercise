"""Generate synthetic evaluation dataset using LLM with structured outputs.

Uses async parallel execution with tqdm for progress tracking.
"""

import asyncio
import json
from datetime import datetime
from typing import Any

import aiohttp
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm_asyncio

from ai_exercise.constants import OPENAPI_SPECS, SETTINGS
from ai_exercise.evals.datasets import EvalQuestion, GeneratedDataset, save_eval_dataset

# Initialize async OpenAI client
async_openai_client = AsyncOpenAI(api_key=SETTINGS.openai_api_key.get_secret_value())


class QuestionBatch(BaseModel):
    """Batch of generated questions for structured output."""

    questions: list[EvalQuestion] = Field(
        description="List of generated evaluation questions"
    )


# Question category specifications
CATEGORY_SPECS: dict[str, dict[str, Any]] = {
    "factual": {
        "count_per_api": 3,
        "description": "Questions about specific facts from the API documentation",
        "examples": [
            "What is the default value for X?",
            "What format does Y use?",
            "What is the maximum limit for Z?",
        ],
    },
    "endpoint": {
        "count_per_api": 2,
        "description": "Questions about finding the right endpoint for a task",
        "examples": [
            "How do I create a new X?",
            "What endpoint lists all Y?",
            "How do I update Z?",
        ],
    },
    "schema": {
        "count_per_api": 2,
        "description": "Questions about request/response structure and fields",
        "examples": [
            "What fields are required to create X?",
            "What does the Y response contain?",
            "What are the possible values for Z?",
        ],
    },
    "auth": {
        "count_per_api": 1,
        "description": "Questions about authentication and authorization",
        "examples": [
            "How do I authenticate?",
            "What headers are required?",
            "What permissions are needed?",
        ],
    },
}


def extract_spec_summary(spec_data: dict[str, Any], api_name: str) -> str:
    """Extract a summary of key information from an API spec."""
    summary_parts = [f"API: {api_name.upper()}"]

    # Extract servers
    if "servers" in spec_data:
        servers = [s.get("url", "") for s in spec_data.get("servers", [])]
        summary_parts.append(f"Base URLs: {', '.join(servers)}")

    # Extract security schemes
    if "components" in spec_data and "securitySchemes" in spec_data["components"]:
        schemes = list(spec_data["components"]["securitySchemes"].keys())
        summary_parts.append(f"Auth schemes: {', '.join(schemes)}")

    # Extract paths (endpoints)
    if "paths" in spec_data:
        paths = list(spec_data["paths"].keys())[:20]  # Limit to first 20
        methods_by_path = []
        for path in paths:
            methods = list(spec_data["paths"][path].keys())
            methods_by_path.append(f"{path}: {', '.join(methods)}")
        summary_parts.append(f"Endpoints ({len(spec_data['paths'])} total):")
        summary_parts.extend([f"  - {p}" for p in methods_by_path[:15]])

    # Extract schemas
    if "components" in spec_data and "schemas" in spec_data["components"]:
        schemas = list(spec_data["components"]["schemas"].keys())[:15]
        summary_parts.append(f"Schemas: {', '.join(schemas)}")

    # Extract webhooks
    if "webhooks" in spec_data:
        webhooks = list(spec_data["webhooks"].keys())[:5]
        summary_parts.append(f"Webhooks: {', '.join(webhooks)}")

    return "\n".join(summary_parts)


def extract_detailed_context(
    spec_data: dict[str, Any], api_name: str, category: str
) -> str:
    """Extract detailed context relevant to a specific category."""
    context_parts = []

    if category in ["factual", "endpoint"]:
        # Include more endpoint details
        if "paths" in spec_data:
            for path, methods in list(spec_data["paths"].items())[:10]:
                for method, details in methods.items():
                    if method in ["get", "post", "put", "patch", "delete"]:
                        desc = details.get("description", details.get("summary", ""))
                        params = [
                            p.get("name", "")
                            for p in details.get("parameters", [])[:5]
                        ]
                        context_parts.append(
                            f"{method.upper()} {path}: {desc[:100]}"
                            f" Params: {params}"
                        )

    elif category == "schema":
        # Include schema details
        if "components" in spec_data and "schemas" in spec_data["components"]:
            for name, schema in list(spec_data["components"]["schemas"].items())[:10]:
                props = list(schema.get("properties", {}).keys())[:8]
                required = schema.get("required", [])[:5]
                context_parts.append(
                    f"Schema {name}: properties={props}, required={required}"
                )

    elif category == "auth":
        # Include auth details
        if "components" in spec_data and "securitySchemes" in spec_data["components"]:
            for name, scheme in spec_data["components"]["securitySchemes"].items():
                context_parts.append(f"Auth {name}: {json.dumps(scheme)}")
        if "security" in spec_data:
            context_parts.append(f"Security: {json.dumps(spec_data['security'])}")

    return "\n".join(context_parts[:20])


async def fetch_spec_async(
    session: aiohttp.ClientSession, api_name: str, url: str
) -> tuple[str, dict[str, Any]]:
    """Fetch an API spec asynchronously."""
    async with session.get(url) as response:
        response.raise_for_status()
        data = await response.json()
        return api_name, data


async def load_all_specs_async() -> dict[str, dict[str, Any]]:
    """Load all OpenAPI specs in parallel."""
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_spec_async(session, name, url)
            for name, url in OPENAPI_SPECS.items()
        ]
        results = await tqdm_asyncio.gather(
            *tasks, desc="Loading API specs", unit="spec"
        )
        return dict(results)


async def generate_questions_for_api_async(
    api_name: str,
    spec_data: dict[str, Any],
    category: str,
    count: int,
    id_prefix: str,
) -> list[EvalQuestion]:
    """Generate questions for a specific API and category using structured output."""
    spec_summary = extract_spec_summary(spec_data, api_name)
    detailed_context = extract_detailed_context(spec_data, api_name, category)
    category_spec = CATEGORY_SPECS[category]

    prompt = f"""You are generating evaluation questions for a RAG system \
that answers questions about API documentation.

API SPECIFICATION SUMMARY:
{spec_summary}

DETAILED CONTEXT:
{detailed_context}

CATEGORY: {category}
DESCRIPTION: {category_spec['description']}
EXAMPLE QUESTIONS: {category_spec['examples']}

Generate exactly {count} high-quality evaluation questions \
for the {api_name.upper()} API.

Requirements:
1. Questions must be answerable from the API documentation provided
2. Ground truth answers must be factual and based on the spec
3. Required keywords should be specific terms that must appear in a correct answer
4. Ground truth chunks should reference specific paths or schema names
5. IDs should follow format: {id_prefix}_XXX (e.g., {id_prefix}_001)
6. Vary difficulty levels (easy, medium, hard)
7. Make questions specific and unambiguous

Do NOT generate questions about:
- Pricing or SLAs
- Dashboard UI
- Information not in the API spec
"""

    response = await async_openai_client.beta.chat.completions.parse(
        model=SETTINGS.openai_model,
        messages=[{"role": "user", "content": prompt}],
        response_format=QuestionBatch,
    )

    questions = response.choices[0].message.parsed
    if questions is None:
        return []

    # Update questions with correct metadata
    result = []
    for i, q in enumerate(questions.questions):
        result.append(
            EvalQuestion(
                id=f"{id_prefix}_{i + 1:03d}",
                question=q.question,
                category=category,
                relevant_apis=[api_name],
                ground_truth_answer=q.ground_truth_answer,
                ground_truth_chunks=q.ground_truth_chunks,
                required_keywords=q.required_keywords,
                difficulty=q.difficulty,
            )
        )

    return result


async def generate_cross_api_questions_async(
    all_specs: dict[str, dict[str, Any]],
) -> list[EvalQuestion]:
    """Generate questions that span multiple APIs."""
    combined_summary = "\n\n".join(
        extract_spec_summary(spec, name) for name, spec in all_specs.items()
    )

    prompt = f"""You are generating evaluation questions for a RAG system \
about StackOne APIs.

AVAILABLE APIs AND THEIR SUMMARIES:
{combined_summary}

Generate exactly 5 cross-API questions that require knowledge \
from multiple API specifications.

Examples of cross-API questions:
- "Which APIs support webhook events?"
- "What is the common pagination pattern across all APIs?"
- "How do I transfer data from ATS to HRIS when hiring a candidate?"

Requirements:
1. Questions must span at least 2 different APIs
2. relevant_apis should list ALL APIs needed to answer
3. Ground truth should synthesize information from multiple specs
4. IDs should follow format: cross_api_XXX
"""

    response = await async_openai_client.beta.chat.completions.parse(
        model=SETTINGS.openai_model,
        messages=[{"role": "user", "content": prompt}],
        response_format=QuestionBatch,
    )

    questions = response.choices[0].message.parsed
    if questions is None:
        return []

    result = []
    for i, q in enumerate(questions.questions):
        result.append(
            EvalQuestion(
                id=f"cross_api_{i + 1:03d}",
                question=q.question,
                category="cross_api",
                relevant_apis=q.relevant_apis if q.relevant_apis else ["stackone"],
                ground_truth_answer=q.ground_truth_answer,
                ground_truth_chunks=q.ground_truth_chunks,
                required_keywords=q.required_keywords,
                difficulty=q.difficulty,
            )
        )

    return result


async def generate_out_of_scope_questions_async() -> list[EvalQuestion]:
    """Generate questions that should trigger 'I don't know' responses."""
    prompt = """Generate exactly 5 out-of-scope questions \
for a StackOne API documentation RAG system.

These questions should be about topics NOT covered in API documentation:
- Pricing and billing
- SLAs and uptime guarantees
- Dashboard/UI configuration
- Third-party integrations not in the API
- Company policies
- On-premise deployment options

The system should respond with "I don't know" or similar for these questions.

Requirements:
1. Questions should sound reasonable but be unanswerable from API docs
2. relevant_apis should be empty []
3. ground_truth_answer should indicate the information is not available
4. required_keywords should include phrases like "don't know", "not available"
5. IDs should follow format: out_of_scope_XXX
"""

    response = await async_openai_client.beta.chat.completions.parse(
        model=SETTINGS.openai_model,
        messages=[{"role": "user", "content": prompt}],
        response_format=QuestionBatch,
    )

    questions = response.choices[0].message.parsed
    if questions is None:
        return []

    result = []
    for i, q in enumerate(questions.questions):
        result.append(
            EvalQuestion(
                id=f"out_of_scope_{i + 1:03d}",
                question=q.question,
                category="out_of_scope",
                relevant_apis=[],
                ground_truth_answer=q.ground_truth_answer,
                ground_truth_chunks=[],
                required_keywords=q.required_keywords
                or ["don't know", "not available"],
                difficulty="easy",
            )
        )

    return result


async def generate_eval_dataset_async() -> GeneratedDataset:
    """Generate the complete evaluation dataset using async parallel execution."""
    print("=" * 60)
    print("Generating Synthetic Evaluation Dataset")
    print("=" * 60)

    # Load all specs in parallel
    print("\nLoading OpenAPI specifications...")
    all_specs = await load_all_specs_async()

    # Build list of all generation tasks
    generation_tasks = []

    for api_name, spec_data in all_specs.items():
        for category, spec in CATEGORY_SPECS.items():
            count = spec["count_per_api"]
            id_prefix = f"{category}_{api_name}"
            generation_tasks.append(
                generate_questions_for_api_async(
                    api_name, spec_data, category, count, id_prefix
                )
            )

    # Add cross-API and out-of-scope tasks
    generation_tasks.append(generate_cross_api_questions_async(all_specs))
    generation_tasks.append(generate_out_of_scope_questions_async())

    # Run all generation tasks in parallel with progress bar
    print(f"\nGenerating questions ({len(generation_tasks)} tasks in parallel)...")
    results = await tqdm_asyncio.gather(
        *generation_tasks,
        desc="Generating questions",
        unit="task",
    )

    # Flatten results
    all_questions: list[EvalQuestion] = []
    for result in results:
        all_questions.extend(result)

    # Create dataset with metadata
    dataset = GeneratedDataset(
        version="1.0",
        description="Synthetic evaluation dataset for StackOne RAG system",
        generation_metadata={
            "generated_at": datetime.now().isoformat(),
            "model": SETTINGS.openai_model,
            "api_specs": list(OPENAPI_SPECS.keys()),
            "total_questions": len(all_questions),
            "questions_by_category": {
                cat: len([q for q in all_questions if q.category == cat])
                for cat in [
                    "factual",
                    "endpoint",
                    "schema",
                    "auth",
                    "cross_api",
                    "out_of_scope",
                ]
            },
        },
        questions=all_questions,
    )

    return dataset


def main() -> None:
    """Main entry point for dataset generation."""
    dataset = asyncio.run(generate_eval_dataset_async())

    print(f"\nTotal questions generated: {len(dataset.questions)}")
    print("\nQuestions by category:")
    for cat, count in dataset.generation_metadata["questions_by_category"].items():
        print(f"  {cat}: {count}")

    print("\nSaving dataset...")
    save_eval_dataset(dataset)
    print("Dataset saved to ai_exercise/evals/datasets/eval_dataset.json")


if __name__ == "__main__":
    main()
