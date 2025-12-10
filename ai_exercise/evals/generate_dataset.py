"""Generate synthetic evaluation dataset using LLM with structured outputs.

Uses async parallel execution with tqdm for progress tracking.
Random sampling ensures coverage across entire API specs without context overflow.
"""

import asyncio
import json
import random
from datetime import datetime
from typing import Any

import aiohttp
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm_asyncio

from ai_exercise.constants import OPENAPI_SPECS, SETTINGS
from ai_exercise.evals.datasets import EvalQuestion, GeneratedDataset, save_eval_dataset

# Sampling configuration
MAX_PATHS_PER_SAMPLE = 15  # Sample this many paths per generation call
MAX_SCHEMAS_PER_SAMPLE = 20  # Sample this many schemas per generation call
SAMPLES_PER_API = 3  # Number of different random samples per API to ensure coverage

# Cross-API sampling configuration
CROSS_API_PATHS_PER_API = 5  # Paths from each API for cross-API questions
CROSS_API_SCHEMAS_PER_API = 5  # Schemas from each API for cross-API questions
CROSS_API_SAMPLES = 3  # Number of cross-API samples to generate questions from
CROSS_API_QUESTIONS_PER_SAMPLE = 3  # Questions to generate per cross-API sample

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


def sample_spec_content(
    spec_data: dict[str, Any],
    api_name: str,
    max_paths: int = MAX_PATHS_PER_SAMPLE,
    max_schemas: int = MAX_SCHEMAS_PER_SAMPLE,
    seed: int | None = None,
) -> dict[str, Any]:
    """Create a sampled subset of the spec for context-limited generation.

    Randomly samples paths and schemas to ensure diverse coverage
    across the entire spec without exceeding context limits.

    Args:
        spec_data: Full OpenAPI spec
        api_name: Name of the API
        max_paths: Maximum number of paths to sample
        max_schemas: Maximum number of schemas to sample
        seed: Random seed for reproducibility (None for random)

    Returns:
        A subset of the spec with sampled paths and schemas
    """
    if seed is not None:
        random.seed(seed)

    sampled = {
        "api_name": api_name,
        "info": spec_data.get("info", {}),
        "servers": spec_data.get("servers", []),
    }

    # Sample paths randomly
    all_paths = list(spec_data.get("paths", {}).items())
    if len(all_paths) > max_paths:
        sampled_paths = random.sample(all_paths, max_paths)
    else:
        sampled_paths = all_paths
    sampled["paths"] = dict(sampled_paths)
    sampled["total_paths_in_spec"] = len(all_paths)

    # Sample schemas randomly
    all_schemas = list(spec_data.get("components", {}).get("schemas", {}).items())
    if len(all_schemas) > max_schemas:
        sampled_schemas = random.sample(all_schemas, max_schemas)
    else:
        sampled_schemas = all_schemas
    sampled["schemas"] = dict(sampled_schemas)
    sampled["total_schemas_in_spec"] = len(all_schemas)

    # Always include security schemes (usually small)
    if "components" in spec_data and "securitySchemes" in spec_data["components"]:
        sampled["securitySchemes"] = spec_data["components"]["securitySchemes"]

    # Always include webhooks if present (usually small)
    if "webhooks" in spec_data:
        sampled["webhooks"] = spec_data["webhooks"]

    return sampled


def extract_spec_summary(spec_data: dict[str, Any], api_name: str) -> str:
    """Extract a compact summary for cross-API questions.

    Lists ALL endpoints and schemas by name (without full details)
    to enable cross-API question generation.
    """
    summary_parts = [f"API: {api_name.upper()}"]

    # Extract servers
    if "servers" in spec_data:
        servers = [s.get("url", "") for s in spec_data.get("servers", [])]
        summary_parts.append(f"Base URLs: {', '.join(servers)}")

    # Extract security schemes
    if "components" in spec_data and "securitySchemes" in spec_data["components"]:
        schemes = list(spec_data["components"]["securitySchemes"].keys())
        summary_parts.append(f"Auth schemes: {', '.join(schemes)}")

    # List ALL paths with methods (no truncation)
    if "paths" in spec_data:
        paths = spec_data["paths"]
        path_list = []
        for path, methods in paths.items():
            method_names = [
                m.upper() for m in methods
                if m in ["get", "post", "put", "patch", "delete"]
            ]
            if method_names:
                path_list.append(f"{path} [{', '.join(method_names)}]")
        summary_parts.append(f"Endpoints ({len(paths)} total):")
        summary_parts.extend([f"  {p}" for p in path_list])

    # List ALL schema names
    if "components" in spec_data and "schemas" in spec_data["components"]:
        schemas = list(spec_data["components"]["schemas"].keys())
        summary_parts.append(f"Schemas ({len(schemas)}): {', '.join(schemas)}")

    # List ALL webhooks
    if "webhooks" in spec_data:
        webhooks = list(spec_data["webhooks"].keys())
        summary_parts.append(f"Webhooks ({len(webhooks)}): {', '.join(webhooks)}")

    return "\n".join(summary_parts)


def format_sampled_spec_for_llm(sampled_spec: dict[str, Any]) -> str:
    """Format a sampled spec subset as detailed context for LLM.

    Provides full JSON details for the sampled paths and schemas,
    enabling accurate question and answer generation.
    """
    parts = []

    api_name = sampled_spec.get("api_name", "unknown")
    parts.append(f"=== {api_name.upper()} API (Sampled Content) ===")
    n_paths = len(sampled_spec.get("paths", {}))
    total_paths = sampled_spec.get("total_paths_in_spec", "?")
    n_schemas = len(sampled_spec.get("schemas", {}))
    total_schemas = sampled_spec.get("total_schemas_in_spec", "?")
    parts.append(
        f"Note: This is a random sample of {n_paths} paths from {total_paths} "
        f"total, and {n_schemas} schemas from {total_schemas} total.\n"
    )

    # Include server info
    if sampled_spec.get("servers"):
        parts.append("SERVERS:")
        parts.append(json.dumps(sampled_spec["servers"], indent=2))

    # Include security schemes (full detail)
    if sampled_spec.get("securitySchemes"):
        parts.append("\nSECURITY SCHEMES:")
        parts.append(json.dumps(sampled_spec["securitySchemes"], indent=2))

    # Include sampled paths with FULL details
    if sampled_spec.get("paths"):
        parts.append("\nSAMPLED ENDPOINTS (full details):")
        parts.append(json.dumps(sampled_spec["paths"], indent=2))

    # Include sampled schemas with FULL details
    if sampled_spec.get("schemas"):
        parts.append("\nSAMPLED SCHEMAS (full details):")
        parts.append(json.dumps(sampled_spec["schemas"], indent=2))

    # Include webhooks if present
    if sampled_spec.get("webhooks"):
        parts.append("\nWEBHOOKS:")
        parts.append(json.dumps(sampled_spec["webhooks"], indent=2))

    return "\n".join(parts)


def sample_multi_api_content(
    all_specs: dict[str, dict[str, Any]],
    paths_per_api: int = CROSS_API_PATHS_PER_API,
    schemas_per_api: int = CROSS_API_SCHEMAS_PER_API,
    seed: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Sample paths and schemas from ALL APIs for cross-API question generation.

    Returns a dict mapping api_name -> sampled content with full details.

    Args:
        all_specs: Dict of all OpenAPI specs keyed by API name
        paths_per_api: Number of paths to sample from each API
        schemas_per_api: Number of schemas to sample from each API
        seed: Random seed for reproducibility (None for random)

    Returns:
        Dict mapping api_name to sampled spec content
    """
    if seed is not None:
        random.seed(seed)

    sampled_apis = {}

    for api_name, spec_data in all_specs.items():
        sampled = {
            "api_name": api_name,
            "info": spec_data.get("info", {}),
        }

        # Sample paths randomly
        all_paths = list(spec_data.get("paths", {}).items())
        if len(all_paths) > paths_per_api:
            sampled_paths = random.sample(all_paths, paths_per_api)
        else:
            sampled_paths = all_paths
        sampled["paths"] = dict(sampled_paths)
        sampled["total_paths_in_spec"] = len(all_paths)

        # Sample schemas randomly
        all_schemas = list(spec_data.get("components", {}).get("schemas", {}).items())
        if len(all_schemas) > schemas_per_api:
            sampled_schemas = random.sample(all_schemas, schemas_per_api)
        else:
            sampled_schemas = all_schemas
        sampled["schemas"] = dict(sampled_schemas)
        sampled["total_schemas_in_spec"] = len(all_schemas)

        sampled_apis[api_name] = sampled

    return sampled_apis


def format_multi_api_sampled_spec_for_llm(
    sampled_apis: dict[str, dict[str, Any]]
) -> str:
    """Format sampled content from multiple APIs for LLM context.

    Provides full JSON details for sampled paths and schemas from each API,
    enabling accurate cross-API question and answer generation.

    Args:
        sampled_apis: Dict mapping api_name to sampled spec content

    Returns:
        Formatted string with all sampled API content
    """
    parts = []
    api_names = list(sampled_apis.keys())
    parts.append(f"=== SAMPLED CONTENT FROM {len(api_names)} APIs ===")
    parts.append(f"APIs included: {', '.join(api_names).upper()}\n")

    for api_name, sampled in sampled_apis.items():
        parts.append(f"\n{'=' * 40}")
        parts.append(f"API: {api_name.upper()}")
        parts.append(f"Sampled {len(sampled.get('paths', {}))} paths from "
                     f"{sampled.get('total_paths_in_spec', '?')} total")
        parts.append(f"Sampled {len(sampled.get('schemas', {}))} schemas from "
                     f"{sampled.get('total_schemas_in_spec', '?')} total")
        parts.append("=" * 40)

        # Include API info
        if sampled.get("info"):
            parts.append(f"\nAPI Info: {sampled['info'].get('title', api_name)}")
            if sampled["info"].get("description"):
                desc = sampled["info"]["description"][:200]
                parts.append(f"Description: {desc}...")

        # Include sampled paths with FULL details
        if sampled.get("paths"):
            parts.append(f"\nENDPOINTS ({api_name.upper()}):")
            parts.append(json.dumps(sampled["paths"], indent=2))

        # Include sampled schemas with FULL details
        if sampled.get("schemas"):
            parts.append(f"\nSCHEMAS ({api_name.upper()}):")
            parts.append(json.dumps(sampled["schemas"], indent=2))

    return "\n".join(parts)


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
    sample_seed: int | None = None,
) -> list[EvalQuestion]:
    """Generate questions for a specific API and category using structured output.

    Uses random sampling to select a subset of paths and schemas, ensuring:
    1. Full details are provided for sampled content (not truncated)
    2. Context size stays within LLM limits
    3. Different samples across calls enable broader coverage

    Args:
        api_name: Name of the API
        spec_data: Full OpenAPI spec
        category: Question category (factual, endpoint, schema, auth)
        count: Number of questions to generate
        id_prefix: Prefix for question IDs
        sample_seed: Random seed for reproducible sampling (None for random)
    """
    # Sample a subset of the spec with full details
    sampled_spec = sample_spec_content(spec_data, api_name, seed=sample_seed)
    spec_context = format_sampled_spec_for_llm(sampled_spec)
    category_spec = CATEGORY_SPECS[category]

    prompt = f"""You are generating evaluation questions for a RAG system \
that answers questions about API documentation.

{spec_context}

CATEGORY: {category}
DESCRIPTION: {category_spec['description']}
EXAMPLE QUESTIONS: {category_spec['examples']}

Generate exactly {count} high-quality evaluation questions \
for the {api_name.upper()} API based on the sampled content above.

CRITICAL REQUIREMENTS:
1. Questions MUST be answerable from the API content provided above
2. Ground truth answers MUST be factually accurate based on the spec details shown
3. Required keywords should be specific terms from the spec that must appear
4. Ground truth chunks should reference specific paths (e.g., "/unified/hris/employees")
   or schema names (e.g., "EmployeeResult") from the content shown
5. IDs should follow format: {id_prefix}_XXX (e.g., {id_prefix}_001)
6. Vary difficulty levels (easy, medium, hard)
7. Make questions specific and unambiguous

Do NOT generate questions about:
- Pricing or SLAs
- Dashboard UI
- Information not visible in the provided API content
- Paths or schemas that aren't in the sampled content above
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
    count: int = CROSS_API_QUESTIONS_PER_SAMPLE,
    id_prefix: str = "cross_api",
    sample_seed: int | None = None,
) -> list[EvalQuestion]:
    """Generate questions that span multiple APIs using actual sampled content.

    Instead of using summaries, samples actual endpoints and schemas from ALL APIs
    to provide detailed context for generating accurate cross-API questions.

    Args:
        all_specs: Dict of all OpenAPI specs keyed by API name
        count: Number of questions to generate
        id_prefix: Prefix for question IDs
        sample_seed: Random seed for reproducible sampling (None for random)
    """
    # Sample actual content from all APIs
    sampled_apis = sample_multi_api_content(all_specs, seed=sample_seed)
    combined_context = format_multi_api_sampled_spec_for_llm(sampled_apis)
    api_names = list(sampled_apis.keys())

    prompt = f"""You are generating evaluation questions for a RAG system \
about StackOne APIs.

{combined_context}

Generate exactly {count} cross-API questions that require knowledge \
from multiple API specifications shown above.

CRITICAL REQUIREMENTS:
1. Questions MUST be answerable using ONLY the sampled endpoints/schemas shown
2. Questions MUST require information from at least 2 different APIs
3. Ground truth answers MUST be factually accurate based on the spec details
4. Ground truth chunks should reference paths (e.g., "/unified/hris/employees")
   or schema names from MULTIPLE APIs in the content above
5. relevant_apis should list ALL APIs whose content was used to form the answer
6. IDs should follow format: {id_prefix}_XXX
7. Vary difficulty levels (easy, medium, hard)

Examples of good cross-API questions (adapt based on actual content shown):
- "What fields are common between Employee in HRIS and Candidate in ATS?"
- "How do pagination parameters differ between the HRIS and CRM APIs?"
- "Which APIs use the same authentication header format?"

Do NOT generate questions about:
- Information not visible in the sampled content above
- Paths or schemas that aren't in the sampled content
- Generic questions that could be answered from a single API
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
        # Ensure relevant_apis contains at least 2 APIs from the sampled set
        relevant = q.relevant_apis if q.relevant_apis else api_names[:2]
        if len(relevant) < 2:
            relevant = api_names[:2]  # Default to first 2 APIs if not specified

        result.append(
            EvalQuestion(
                id=f"{id_prefix}_{i + 1:03d}",
                question=q.question,
                category="cross_api",
                relevant_apis=relevant,
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
    """Generate the complete evaluation dataset using async parallel execution.

    Uses random sampling with multiple samples per API to ensure broad coverage
    across the entire spec without exceeding context limits.
    """
    print("=" * 60)
    print("Generating Synthetic Evaluation Dataset")
    print("=" * 60)
    print(f"Using {SAMPLES_PER_API} random samples per API for diverse coverage")
    print(f"Each sample: {MAX_PATHS_PER_SAMPLE} paths,{MAX_SCHEMAS_PER_SAMPLE} schemas")

    # Load all specs in parallel
    print("\nLoading OpenAPI specifications...")
    all_specs = await load_all_specs_async()

    # Print spec sizes for reference
    print("\nSpec sizes:")
    for api_name, spec_data in all_specs.items():
        n_paths = len(spec_data.get("paths", {}))
        n_schemas = len(spec_data.get("components", {}).get("schemas", {}))
        print(f"  {api_name}: {n_paths} paths, {n_schemas} schemas")

    # Build list of all generation tasks with multiple samples per API
    generation_tasks = []
    sample_counter = 0

    for api_name, spec_data in all_specs.items():
        for category, spec in CATEGORY_SPECS.items():
            # Determine questions per sample based on total count and samples
            total_count = spec["count_per_api"]

            # For categories that need broader coverage, use multiple samples
            # For auth (which is consistent across the API), use single sample
            if category == "auth":
                # Auth is usually the same across all endpoints, single sample
                id_prefix = f"{category}_{api_name}"
                generation_tasks.append(
                    generate_questions_for_api_async(
                        api_name, spec_data, category, total_count, id_prefix,
                        sample_seed=sample_counter
                    )
                )
                sample_counter += 1
            else:
                # For factual/endpoint/schema, use multiple samples for coverage
                questions_per_sample = max(1, total_count // SAMPLES_PER_API)
                for sample_idx in range(SAMPLES_PER_API):
                    id_prefix = f"{category}_{api_name}_s{sample_idx}"
                    generation_tasks.append(
                        generate_questions_for_api_async(
                            api_name,
                            spec_data,
                            category,
                            questions_per_sample,
                            id_prefix,
                            sample_seed=sample_counter,
                        )
                    )
                    sample_counter += 1

    # Add cross-API tasks with multiple samples for diverse coverage
    for sample_idx in range(CROSS_API_SAMPLES):
        id_prefix = f"cross_api_s{sample_idx}"
        generation_tasks.append(
            generate_cross_api_questions_async(
                all_specs,
                count=CROSS_API_QUESTIONS_PER_SAMPLE,
                id_prefix=id_prefix,
                sample_seed=sample_counter,
            )
        )
        sample_counter += 1

    # Add out-of-scope tasks
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

    # Deduplicate by question text (in case samples overlap)
    seen_questions = set()
    unique_questions = []
    for q in all_questions:
        if q.question not in seen_questions:
            seen_questions.add(q.question)
            unique_questions.append(q)

    print(f"\nGenerated {len(all_questions)} questions, {len(unique_questions)} unique")

    # Create dataset with metadata
    dataset = GeneratedDataset(
        version="1.0",
        description="Synthetic evaluation dataset for StackOne RAG system "
                    "(generated with random sampling for broad spec coverage)",
        generation_metadata={
            "generated_at": datetime.now().isoformat(),
            "model": SETTINGS.openai_model,
            "api_specs": list(OPENAPI_SPECS.keys()),
            "sampling_config": {
                "max_paths_per_sample": MAX_PATHS_PER_SAMPLE,
                "max_schemas_per_sample": MAX_SCHEMAS_PER_SAMPLE,
                "samples_per_api": SAMPLES_PER_API,
                "cross_api_paths_per_api": CROSS_API_PATHS_PER_API,
                "cross_api_schemas_per_api": CROSS_API_SCHEMAS_PER_API,
                "cross_api_samples": CROSS_API_SAMPLES,
                "cross_api_questions_per_sample": CROSS_API_QUESTIONS_PER_SAMPLE,
            },
            "total_questions": len(unique_questions),
            "questions_by_category": {
                cat: len([q for q in unique_questions if q.category == cat])
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
        questions=unique_questions,
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
