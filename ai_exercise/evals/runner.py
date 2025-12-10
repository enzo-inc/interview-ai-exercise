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

from ai_exercise.configs.base import SystemConfig, get_config
from ai_exercise.constants import SETTINGS, chroma_client
from ai_exercise.evals.datasets import EvalQuestion, load_eval_dataset
from ai_exercise.evals.judges import judge_abstention_async, judge_answer_async
from ai_exercise.evals.metrics import (
    AnswerMetrics,
    EvalResult,
    RetrievalMetrics,
    compute_answer_metrics,
    compute_retrieval_metrics,
)
from ai_exercise.llm.completions import create_prompt
from ai_exercise.llm.embeddings import openai_ef
from ai_exercise.retrieval.bm25_index import BM25Index
from ai_exercise.retrieval.hybrid_search import get_relevant_chunks_hybrid
from ai_exercise.retrieval.intent_detection import detect_query_intent
from ai_exercise.retrieval.reranker import rerank_chunks
from ai_exercise.retrieval.retrieval import RetrievedChunk, get_relevant_chunks_with_ids
from ai_exercise.retrieval.vector_store import create_collection

# BM25 index storage path
BM25_INDEX_DIR = Path(".bm25_index")


def get_bm25_index_path(collection_name: str) -> Path:
    """Get the path for a BM25 index file."""
    return BM25_INDEX_DIR / f"{collection_name}.pkl"


def load_bm25_index_if_exists(collection_name: str) -> BM25Index | None:
    """Load BM25 index from disk if it exists."""
    path = get_bm25_index_path(collection_name)
    if path.exists():
        try:
            return BM25Index.load(path)
        except Exception as e:
            click.echo(f"Warning: Failed to load BM25 index: {e}")
    return None


async def evaluate_single_question_async(
    question: EvalQuestion,
    collection: Any,
    async_client: Any,
    run_judges: bool = True,
    config: SystemConfig | None = None,
    bm25_index: BM25Index | None = None,
) -> EvalResult:
    """Evaluate a single question against the RAG system.

    Args:
        question: The evaluation question to test.
        collection: ChromaDB collection with loaded documents.
        async_client: Async OpenAI client for completions, intent detection, and reranking.
        run_judges: Whether to run LLM judges (slower but more detailed).
        config: System configuration to use for retrieval settings.
        bm25_index: BM25 index for hybrid search (required if config.use_hybrid_search).

    Returns:
        EvalResult with all metrics for this question.
    """
    import asyncio

    # Determine retrieval method based on config
    use_hybrid = (
        config is not None
        and config.use_hybrid_search
        and bm25_index is not None
    )
    use_metadata_filtering = config is not None and config.use_metadata_filtering
    use_reranking = config is not None and config.use_reranking
    use_unknown_detection = config is not None and config.use_unknown_detection

    # Detect query intent for metadata filtering if enabled (async)
    api_filter: list[str] | None = None
    if use_metadata_filtering:
        api_filter = await detect_query_intent(
            client=async_client,
            query=question.question,
        )

    # When reranking is enabled, retrieve more candidates
    retrieval_k = SETTINGS.k_neighbors * 3 if use_reranking else SETTINGS.k_neighbors

    # Retrieve relevant chunks with IDs (run in thread to not block)
    if use_hybrid:
        retrieved_chunk_objs: list[RetrievedChunk] = await asyncio.to_thread(
            get_relevant_chunks_hybrid,
            collection=collection,
            bm25_index=bm25_index,
            query=question.question,
            k=retrieval_k,
            api_filter=api_filter,
        )
    else:
        retrieved_chunk_objs = await asyncio.to_thread(
            get_relevant_chunks_with_ids,
            collection=collection,
            query=question.question,
            k=retrieval_k,
            api_filter=api_filter,
        )

    # Apply LLM-based reranking if enabled (async)
    if use_reranking:
        retrieved_chunk_objs = await rerank_chunks(
            client=async_client,
            query=question.question,
            chunks=retrieved_chunk_objs,
            top_k=SETTINGS.k_neighbors,
        )

    # Extract content and IDs
    retrieved_chunks = [chunk.content for chunk in retrieved_chunk_objs]
    retrieved_chunk_ids = [chunk.chunk_id for chunk in retrieved_chunk_objs]

    # Generate answer using RAG with async client
    prompt = create_prompt(
        query=question.question,
        context=retrieved_chunks,
        use_unknown_detection=use_unknown_detection,
    )
    response = await async_client.chat.completions.create(
        model=SETTINGS.openai_model,
        messages=[{"role": "user", "content": prompt}],
    )
    generated_answer = response.choices[0].message.content or ""

    # Check retrieval hit using structural IDs
    retrieval_hit = False
    first_relevant_rank = None
    relevant_ids = set(question.relevant_structural_ids)

    for rank, chunk_obj in enumerate(retrieved_chunk_objs, 1):
        # Get the structural IDs this chunk covers (stored as JSON string)
        covers_raw = chunk_obj.metadata.get("covers", "[]")
        if isinstance(covers_raw, str):
            try:
                covers = json.loads(covers_raw)
            except json.JSONDecodeError:
                covers = [covers_raw]  # Fallback to single-item list
        else:
            covers = covers_raw if isinstance(covers_raw, list) else [covers_raw]

        # Check if any covered structural ID matches a relevant ID
        if any(cov_id in relevant_ids for cov_id in covers):
            retrieval_hit = True
            if first_relevant_rank is None:
                first_relevant_rank = rank
            break

    # Run LLM judge if requested
    accuracy_score = 3  # Default middle score
    abstention_score = None  # Only set for out_of_scope questions

    if run_judges:
        try:
            accuracy = await judge_answer_async(
                question=question.question,
                answer=generated_answer,
                ground_truth=question.ground_truth_answer,
            )
            accuracy_score = accuracy.score
        except Exception as e:
            click.echo(f"Warning: Judge failed for {question.id}: {e}")

        # Run abstention judge for out_of_scope questions
        if question.category == "out_of_scope":
            try:
                abstention = await judge_abstention_async(
                    question=question.question,
                    answer=generated_answer,
                    retrieved_context=retrieved_chunks,
                )
                abstention_score = abstention.score
            except Exception as e:
                click.echo(f"Warning: Abstention judge failed for {question.id}: {e}")

    return EvalResult(
        question_id=question.id,
        question=question.question,
        category=question.category,
        retrieved_chunks=retrieved_chunks,
        retrieved_chunk_ids=retrieved_chunk_ids,
        generated_answer=generated_answer,
        ground_truth_answer=question.ground_truth_answer,
        relevant_structural_ids=question.relevant_structural_ids,
        relevant_apis=question.relevant_apis,
        retrieval_hit=retrieval_hit,
        first_relevant_rank=first_relevant_rank,
        accuracy_score=accuracy_score,
        abstention_score=abstention_score,
    )


async def run_evaluation_async(
    config_name: str,
    eval_type: str = "all",
    run_judges: bool = True,
) -> tuple[list[EvalResult], RetrievalMetrics, AnswerMetrics]:
    """Run evaluation on the dataset using specified config.

    Args:
        config_name: Name of the system config to use (e.g., "c0").
        eval_type: Type of evaluation ("retrieval", "e2e", or "all").
        run_judges: Whether to run LLM judges.

    Returns:
        Tuple of (results, retrieval_metrics, answer_metrics).
    """
    from openai import AsyncOpenAI

    config = get_config(config_name)
    click.echo(f"\nRunning evaluation for config: {config.name}")
    click.echo(f"Description: {config.description}")

    # Load evaluation dataset
    questions = load_eval_dataset()
    click.echo(f"Loaded {len(questions)} evaluation questions")

    # Get or create collection - use config-specific collection
    collection_name = f"{config_name}_vector_index"
    collection = create_collection(chroma_client, openai_ef, collection_name)

    # Check if collection has documents
    doc_count = collection.count()
    if doc_count == 0:
        raise click.ClickException(
            f"No documents in collection '{collection_name}'. "
            f"Run 'make load-data CONFIG={config_name}' first."
        )
    click.echo(f"Collection has {doc_count} documents")

    # Load BM25 index if hybrid search is enabled
    bm25_index: BM25Index | None = None
    if config.use_hybrid_search:
        bm25_index = load_bm25_index_if_exists(collection_name)
        if bm25_index is not None:
            click.echo(f"BM25 index loaded with {len(bm25_index)} documents")
            click.echo("Using HYBRID search (BM25 + Vector)")
        else:
            click.echo("Warning: Hybrid search enabled but BM25 index not found.")
            click.echo("Falling back to vector-only search.")
    else:
        click.echo("Using VECTOR-only search")

    # Show additional config features
    if config.use_metadata_filtering:
        click.echo("Using METADATA FILTERING (query intent detection)")
    if config.use_reranking:
        click.echo("Using LLM RE-RANKING")
    if config.use_unknown_detection:
        click.echo("Using UNKNOWN DETECTION prompting")

    # Create async OpenAI client
    async_client = AsyncOpenAI(api_key=SETTINGS.openai_api_key.get_secret_value())

    # Run evaluations in parallel
    click.echo(f"\nEvaluating {len(questions)} questions...")

    # Create tasks for parallel execution
    tasks = [
        evaluate_single_question_async(
            q, collection, async_client,
            run_judges=run_judges,
            config=config,
            bm25_index=bm25_index,
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

    return results, retrieval_metrics, answer_metrics


def save_results(
    config_name: str,
    results: list[EvalResult],
    retrieval_metrics: RetrievalMetrics,
    answer_metrics: AnswerMetrics,
    output_dir: Path,
) -> Path:
    """Save evaluation results to JSON file.

    Args:
        config_name: Name of the config used.
        results: List of evaluation results.
        retrieval_metrics: Computed retrieval metrics.
        answer_metrics: Computed answer metrics.
        output_dir: Directory to save results.

    Returns:
        Path to the saved results file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    answer_quality_summary = {
        "accuracy_score": answer_metrics.accuracy_score,
    }
    if answer_metrics.abstention_accuracy is not None:
        answer_quality_summary["abstention_accuracy"] = answer_metrics.abstention_accuracy

    output = {
        "config": config_name,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_questions": len(results),
            "retrieval": {
                "hit_rate_at_k": retrieval_metrics.hit_rate_at_k,
                "mrr": retrieval_metrics.mrr,
                "k": retrieval_metrics.k,
            },
            "answer_quality": answer_quality_summary,
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
            "avg_accuracy": sum(r.accuracy_score for r in cat_results)
            / len(cat_results),
            "retrieval_hit_rate": sum(1 for r in cat_results if r.retrieval_hit)
            / len(cat_results),
        }

    # Add detailed results
    for r in results:
        result_entry = {
            "question_id": r.question_id,
            "question": r.question,
            "category": r.category,
            "relevant_apis": r.relevant_apis,
            "generated_answer": r.generated_answer,
            "ground_truth_answer": r.ground_truth_answer,
            "relevant_structural_ids": r.relevant_structural_ids,
            "retrieved_chunk_ids": r.retrieved_chunk_ids,
            "retrieval_hit": r.retrieval_hit,
            "first_relevant_rank": r.first_relevant_rank,
            "accuracy_score": r.accuracy_score,
        }
        if r.abstention_score is not None:
            result_entry["abstention_score"] = r.abstention_score
        output["detailed_results"].append(result_entry)

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

    results, retrieval_metrics, answer_metrics = asyncio.run(
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

    click.echo("\n--- Answer Quality Metrics ---")
    click.echo(f"Accuracy Score (1-5): {answer_metrics.accuracy_score:.2f}")
    if answer_metrics.abstention_accuracy is not None:
        click.echo(f"Abstention Accuracy (0-1): {answer_metrics.abstention_accuracy:.2f}")

    # Save results
    output_path = save_results(
        config, results, retrieval_metrics, answer_metrics, output_dir,
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
        ("Accuracy (1-5)", _get("answer_quality", "accuracy_score")),
        ("Abstention Accuracy", _get("answer_quality", "abstention_accuracy")),
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
