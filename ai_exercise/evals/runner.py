"""Evaluation runner CLI with async parallel execution.

Usage:
    python -m ai_exercise.evals.runner run --config c0
    python -m ai_exercise.evals.runner run --config c0 --type retrieval
    python -m ai_exercise.evals.runner compare --configs c0,c1
    python -m ai_exercise.evals.runner report --output reports/
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import click
from tqdm.asyncio import tqdm_asyncio

from ai_exercise.configs.base import get_config
from ai_exercise.constants import SETTINGS, chroma_client
from ai_exercise.evals.datasets import EvalQuestion, load_eval_dataset
from ai_exercise.evals.judges import judge_answer_async
from ai_exercise.evals.metrics import (
    AnswerMetrics,
    EvalResult,
    RetrievalMetrics,
    compute_abstention_metrics,
    compute_answer_metrics,
    compute_retrieval_metrics,
    keyword_coverage,
)
from ai_exercise.llm.completions import create_prompt
from ai_exercise.llm.embeddings import openai_ef
from ai_exercise.loading.chunk_json import generate_chunk_id, normalize_chunk_id
from ai_exercise.retrieval.retrieval import get_relevant_chunks_with_ids
from ai_exercise.retrieval.vector_store import create_collection


def normalize_ground_truth_chunk(gt_chunk: str, relevant_apis: list[str]) -> list[str]:
    """Convert a ground truth chunk reference to possible chunk IDs.

    Ground truth chunks may be in formats like:
    - "/unified/hris/employees" (path)
    - "Schema EmployeeResult" (schema reference)
    - "ATS /unified/ats/applications" (with API prefix)

    Returns list of possible chunk IDs that could match.
    """
    possible_ids = []
    gt_lower = gt_chunk.lower().strip()

    # Handle "Schema X" format
    if gt_lower.startswith("schema "):
        schema_name = gt_chunk[7:].strip()  # Remove "Schema " prefix
        for api in relevant_apis:
            possible_ids.append(generate_chunk_id(api, "components", schema_name))
    # Handle "API /path" format (e.g., "ATS /unified/ats/applications")
    elif " /" in gt_chunk:
        parts = gt_chunk.split(" /", 1)
        api_hint = parts[0].lower().strip()
        path = "/" + parts[1].strip()
        # Try the hinted API first, then all relevant APIs
        if api_hint in relevant_apis:
            possible_ids.append(generate_chunk_id(api_hint, "paths", path))
        for api in relevant_apis:
            if api != api_hint:
                possible_ids.append(generate_chunk_id(api, "paths", path))
    # Handle plain path format
    elif gt_chunk.startswith("/"):
        for api in relevant_apis:
            possible_ids.append(generate_chunk_id(api, "paths", gt_chunk))
    # Handle webhook format
    elif "webhook" in gt_lower:
        webhook_name = gt_chunk.replace("Webhook ", "").replace("webhook ", "").strip()
        for api in relevant_apis:
            possible_ids.append(generate_chunk_id(api, "webhooks", webhook_name))
    # Generic - try all combinations
    else:
        normalized = normalize_chunk_id(gt_chunk)
        for api in relevant_apis:
            for source in ["paths", "components", "webhooks"]:
                possible_ids.append(f"{api}_{source}_{normalized}")

    return possible_ids


async def evaluate_single_question_async(
    question: EvalQuestion,
    collection: Any,
    async_client: Any,
    run_judges: bool = True,
) -> EvalResult:
    """Evaluate a single question against the RAG system.

    Args:
        question: The evaluation question to test.
        collection: ChromaDB collection with loaded documents.
        async_client: Async OpenAI client for parallel completions.
        run_judges: Whether to run LLM judges (slower but more detailed).

    Returns:
        EvalResult with all metrics for this question.
    """
    import asyncio

    # Retrieve relevant chunks with IDs (run in thread to not block)
    retrieved_chunk_objs = await asyncio.to_thread(
        get_relevant_chunks_with_ids,
        collection=collection,
        query=question.question,
        k=SETTINGS.k_neighbors,
    )

    # Extract content and IDs
    retrieved_chunks = [chunk.content for chunk in retrieved_chunk_objs]
    retrieved_chunk_ids = [chunk.chunk_id for chunk in retrieved_chunk_objs]

    # Generate answer using RAG with async client
    prompt = create_prompt(query=question.question, context=retrieved_chunks)
    response = await async_client.chat.completions.create(
        model=SETTINGS.openai_model,
        messages=[{"role": "user", "content": prompt}],
    )
    generated_answer = response.choices[0].message.content or ""

    # Convert ground truth chunks to possible chunk IDs
    gt_chunk_ids = set()
    for gt_chunk in question.ground_truth_chunks:
        possible_ids = normalize_ground_truth_chunk(gt_chunk, question.relevant_apis)
        gt_chunk_ids.update(possible_ids)

    # Check retrieval hit using chunk ID matching
    # Also support parent_chunk_id matching for split chunks
    retrieval_hit = False
    first_relevant_rank = None

    for rank, chunk_obj in enumerate(retrieved_chunk_objs, 1):
        chunk_id = chunk_obj.chunk_id
        parent_id = chunk_obj.metadata.get("parent_chunk_id")

        # Check if this chunk ID matches any ground truth ID
        # Use prefix matching to handle split chunks (e.g., "id_part0" matches "id")
        is_match = False
        for gt_id in gt_chunk_ids:
            if chunk_id == gt_id or chunk_id.startswith(gt_id + "_part"):
                is_match = True
                break
            if parent_id and parent_id == gt_id:
                is_match = True
                break

        if is_match:
            retrieval_hit = True
            if first_relevant_rank is None:
                first_relevant_rank = rank
            break  # Found first relevant, no need to continue

    # Calculate keyword coverage
    kw_coverage = keyword_coverage(generated_answer, question.required_keywords)

    # Run LLM judges if requested
    accuracy_score = 3  # Default middle score
    completeness_score = 3
    has_hallucination = False

    if run_judges:
        try:
            accuracy, completeness, hallucination = await judge_answer_async(
                question=question.question,
                answer=generated_answer,
                ground_truth=question.ground_truth_answer,
                context=retrieved_chunks,
            )
            accuracy_score = accuracy.score
            completeness_score = completeness.score
            has_hallucination = hallucination.has_hallucination
        except Exception as e:
            click.echo(f"Warning: Judge failed for {question.id}: {e}")

    return EvalResult(
        question_id=question.id,
        question=question.question,
        category=question.category,
        retrieved_chunks=retrieved_chunks,
        retrieved_chunk_ids=retrieved_chunk_ids,
        generated_answer=generated_answer,
        ground_truth_answer=question.ground_truth_answer,
        ground_truth_chunks=question.ground_truth_chunks,
        ground_truth_chunk_ids=list(gt_chunk_ids),
        relevant_apis=question.relevant_apis,
        retrieval_hit=retrieval_hit,
        first_relevant_rank=first_relevant_rank,
        keyword_coverage=kw_coverage,
        accuracy_score=accuracy_score,
        completeness_score=completeness_score,
        has_hallucination=has_hallucination,
    )


async def run_evaluation_async(
    config_name: str,
    eval_type: str = "all",
    run_judges: bool = True,
) -> tuple[list[EvalResult], RetrievalMetrics, AnswerMetrics, dict[str, float]]:
    """Run evaluation on the dataset using specified config.

    Args:
        config_name: Name of the system config to use (e.g., "c0").
        eval_type: Type of evaluation ("retrieval", "e2e", or "all").
        run_judges: Whether to run LLM judges.

    Returns:
        Tuple of (results, retrieval_metrics, answer_metrics, abstention_metrics).
    """
    from openai import AsyncOpenAI

    config = get_config(config_name)
    click.echo(f"\nRunning evaluation for config: {config.name}")
    click.echo(f"Description: {config.description}")

    # Load evaluation dataset
    questions = load_eval_dataset()
    click.echo(f"Loaded {len(questions)} evaluation questions")

    # Get or create collection
    collection = create_collection(chroma_client, openai_ef, SETTINGS.collection_name)

    # Check if collection has documents
    doc_count = collection.count()
    if doc_count == 0:
        raise click.ClickException(
            "No documents in collection. Run 'make load-data' first."
        )
    click.echo(f"Collection has {doc_count} documents")

    # Create async OpenAI client for parallel requests
    async_client = AsyncOpenAI(api_key=SETTINGS.openai_api_key.get_secret_value())

    # Run evaluations in parallel
    click.echo(f"\nEvaluating {len(questions)} questions...")

    # Create tasks for parallel execution
    tasks = [
        evaluate_single_question_async(
            q, collection, async_client, run_judges=run_judges
        )
        for q in questions
    ]

    # Run with progress bar
    results = await tqdm_asyncio.gather(
        *tasks,
        desc="Evaluating questions",
        unit="question",
    )

    # Compute metrics
    retrieval_metrics = compute_retrieval_metrics(results)
    answer_metrics = compute_answer_metrics(results)
    abstention_metrics = compute_abstention_metrics(results)

    return results, retrieval_metrics, answer_metrics, abstention_metrics


def save_results(
    config_name: str,
    results: list[EvalResult],
    retrieval_metrics: RetrievalMetrics,
    answer_metrics: AnswerMetrics,
    abstention_metrics: dict[str, float],
    output_dir: Path,
) -> Path:
    """Save evaluation results to JSON file.

    Args:
        config_name: Name of the config used.
        results: List of evaluation results.
        retrieval_metrics: Computed retrieval metrics.
        answer_metrics: Computed answer metrics.
        abstention_metrics: Computed abstention metrics.
        output_dir: Directory to save results.

    Returns:
        Path to the saved results file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "config": config_name,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_questions": len(results),
            "retrieval": {
                "hit_rate_at_k": retrieval_metrics.hit_rate_at_k,
                "mrr": retrieval_metrics.mrr,
                "precision_at_k": retrieval_metrics.precision_at_k,
                "recall_at_k": retrieval_metrics.recall_at_k,
                "k": retrieval_metrics.k,
            },
            "answer_quality": {
                "keyword_coverage": answer_metrics.keyword_coverage,
                "accuracy_score": answer_metrics.accuracy_score,
                "completeness_score": answer_metrics.completeness_score,
                "hallucination_rate": answer_metrics.hallucination_rate,
            },
            "abstention": abstention_metrics,
        },
        "results_by_category": {},
        "detailed_results": [],
    }

    # Group results by category
    categories = {r.category for r in results}
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        output["results_by_category"][cat] = {
            "count": len(cat_results),
            "avg_keyword_coverage": sum(r.keyword_coverage for r in cat_results)
            / len(cat_results),
            "avg_accuracy": sum(r.accuracy_score for r in cat_results)
            / len(cat_results),
            "retrieval_hit_rate": sum(1 for r in cat_results if r.retrieval_hit)
            / len(cat_results),
        }

    # Add detailed results
    for r in results:
        output["detailed_results"].append(
            {
                "question_id": r.question_id,
                "question": r.question,
                "category": r.category,
                "relevant_apis": r.relevant_apis,
                "generated_answer": r.generated_answer,
                "ground_truth_answer": r.ground_truth_answer,
                "ground_truth_chunks": r.ground_truth_chunks,
                "ground_truth_chunk_ids": r.ground_truth_chunk_ids,
                "retrieved_chunk_ids": r.retrieved_chunk_ids,
                "retrieval_hit": r.retrieval_hit,
                "first_relevant_rank": r.first_relevant_rank,
                "keyword_coverage": r.keyword_coverage,
                "accuracy_score": r.accuracy_score,
                "completeness_score": r.completeness_score,
                "has_hallucination": r.has_hallucination,
            }
        )

    output_path = output_dir / f"{config_name}_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return output_path


@click.group()
def cli() -> None:
    """Evaluation runner for StackOne RAG system."""
    pass


@cli.command()
@click.option("--config", required=True, help="Config name (e.g., c0, c1)")
@click.option(
    "--type",
    "eval_type",
    default="all",
    type=click.Choice(["retrieval", "e2e", "all"]),
    help="Type of evaluation to run",
)
@click.option(
    "--no-judges",
    is_flag=True,
    help="Skip LLM judges (faster but less detailed)",
)
@click.option(
    "--output",
    default="reports/results",
    help="Output directory for results",
)
def run(config: str, eval_type: str, no_judges: bool, output: str) -> None:
    """Run evaluation on the dataset."""
    output_dir = Path(output)

    results, retrieval_metrics, answer_metrics, abstention_metrics = asyncio.run(
        run_evaluation_async(config, eval_type, run_judges=not no_judges)
    )

    # Print summary
    click.echo("\n" + "=" * 60)
    click.echo("EVALUATION RESULTS")
    click.echo("=" * 60)

    click.echo(f"\nConfig: {config}")
    click.echo(f"Total questions: {len(results)}")

    click.echo("\n--- Retrieval Metrics ---")
    click.echo(f"Hit Rate@{retrieval_metrics.k}: {retrieval_metrics.hit_rate_at_k:.3f}")
    click.echo(f"MRR: {retrieval_metrics.mrr:.3f}")
    prec = retrieval_metrics.precision_at_k
    click.echo(f"Precision@{retrieval_metrics.k}: {prec:.3f}")
    click.echo(f"Recall@{retrieval_metrics.k}: {retrieval_metrics.recall_at_k:.3f}")

    click.echo("\n--- Answer Quality Metrics ---")
    click.echo(f"Keyword Coverage: {answer_metrics.keyword_coverage:.3f}")
    click.echo(f"Accuracy Score (1-5): {answer_metrics.accuracy_score:.2f}")
    click.echo(f"Completeness Score (1-5): {answer_metrics.completeness_score:.2f}")
    click.echo(f"Hallucination Rate: {answer_metrics.hallucination_rate:.3f}")

    click.echo("\n--- Abstention Metrics ---")
    click.echo(
        f"Correct Abstention Rate: {abstention_metrics['correct_abstention_rate']:.3f}"
    )
    click.echo(
        f"False Abstention Rate: {abstention_metrics['false_abstention_rate']:.3f}"
    )

    # Save results
    output_path = save_results(
        config, results, retrieval_metrics, answer_metrics,
        abstention_metrics, output_dir,
    )
    click.echo(f"\nResults saved to: {output_path}")


@cli.command()
@click.option("--configs", required=True, help="Comma-separated config names")
@click.option("--output", default="reports/results", help="Results directory")
def compare(configs: str, output: str) -> None:
    """Compare results from multiple configs."""
    output_dir = Path(output)
    config_names = [c.strip() for c in configs.split(",")]

    click.echo(f"\nComparing configs: {config_names}")

    results_data = []
    for name in config_names:
        results_file = output_dir / f"{name}_results.json"
        if not results_file.exists():
            click.echo(f"Warning: Results not found for {name}, skipping...")
            continue
        with open(results_file) as f:
            results_data.append((name, json.load(f)))

    if len(results_data) < 2:
        raise click.ClickException("Need at least 2 result files to compare")

    # Print comparison table
    click.echo("\n" + "=" * 80)
    click.echo("COMPARISON TABLE")
    click.echo("=" * 80)

    header_line = " ".join(f"{name:>12}" for name, _ in results_data)
    click.echo(f"\n{'Metric':<30} " + header_line)
    click.echo("-" * (30 + 13 * len(results_data)))

    def _get(section: str, key: str):
        def getter(d: dict) -> float:
            return d["summary"][section][key]
        return getter

    metrics = [
        ("Hit Rate@K", _get("retrieval", "hit_rate_at_k")),
        ("MRR", _get("retrieval", "mrr")),
        ("Precision@K", _get("retrieval", "precision_at_k")),
        ("Recall@K", _get("retrieval", "recall_at_k")),
        ("Keyword Coverage", _get("answer_quality", "keyword_coverage")),
        ("Accuracy (1-5)", _get("answer_quality", "accuracy_score")),
        ("Completeness (1-5)", _get("answer_quality", "completeness_score")),
        ("Hallucination Rate", _get("answer_quality", "hallucination_rate")),
        ("Correct Abstention", _get("abstention", "correct_abstention_rate")),
        ("False Abstention", _get("abstention", "false_abstention_rate")),
    ]

    for metric_name, getter in metrics:
        values = []
        for _, data in results_data:
            try:
                values.append(f"{getter(data):>12.3f}")
            except (KeyError, TypeError):
                values.append(f"{'N/A':>12}")
        click.echo(f"{metric_name:<30} " + " ".join(values))


@cli.command()
@click.option("--output", default="reports", help="Output directory for report")
def report(output: str) -> None:
    """Generate markdown report from all available results."""
    from ai_exercise.evals.reports import generate_markdown_report

    output_dir = Path(output)
    results_dir = output_dir / "results"

    if not results_dir.exists():
        raise click.ClickException(f"Results directory not found: {results_dir}")

    report_path = generate_markdown_report(results_dir, output_dir)
    click.echo(f"Report generated: {report_path}")


if __name__ == "__main__":
    cli()
