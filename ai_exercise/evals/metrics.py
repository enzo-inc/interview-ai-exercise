"""Evaluation metrics for RAG system."""

from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation metrics."""

    hit_rate_at_k: float
    mrr: float
    k: int


@dataclass
class AnswerMetrics:
    """Container for answer quality metrics."""

    accuracy_score: float


@dataclass
class EvalResult:
    """Result for a single evaluation question."""

    question_id: str
    question: str
    category: str
    retrieved_chunks: list[str]  # Chunk content
    retrieved_chunk_ids: list[str]  # Chunk IDs for proper matching
    generated_answer: str
    ground_truth_answer: str
    relevant_structural_ids: list[str]  # Structural IDs for config-agnostic matching
    relevant_apis: list[str]  # APIs relevant to this question
    retrieval_hit: bool
    first_relevant_rank: int | None  # Rank of first relevant chunk, None if no hit
    accuracy_score: int


def chunk_id_matches(retrieved_id: str, gt_id: str) -> bool:
    """Check if a retrieved chunk ID matches a ground truth chunk ID.

    Handles split chunks (e.g., "hris_paths_x_part0" matches "hris_paths_x").
    """
    if retrieved_id == gt_id:
        return True
    # Handle split chunks - retrieved might be "base_id_partN"
    return retrieved_id.startswith(gt_id + "_part")


def hit_rate_at_k(
    retrieved_chunk_ids_list: list[list[str]],
    ground_truth_chunk_ids_list: list[list[str]],
    k: int = 5,
) -> float:
    """Calculate Hit Rate@K using chunk ID matching.

    Hit Rate@K measures the percentage of queries where at least one
    relevant chunk appears in the top-K retrieved results.

    Args:
        retrieved_chunk_ids_list: List of retrieved chunk ID lists per query.
        ground_truth_chunk_ids_list: List of ground truth chunk ID lists per query.
        k: Number of top results to consider.

    Returns:
        Hit rate as a float between 0 and 1.
    """
    if not retrieved_chunk_ids_list:
        return 0.0

    hits = 0
    for retrieved_ids, gt_ids in zip(
        retrieved_chunk_ids_list, ground_truth_chunk_ids_list, strict=False
    ):
        top_k = retrieved_ids[:k]
        # Check if any ground truth chunk appears in retrieved chunks using ID matching
        for gt_id in gt_ids:
            if any(chunk_id_matches(ret_id, gt_id) for ret_id in top_k):
                hits += 1
                break

    return hits / len(retrieved_chunk_ids_list)


def mrr(
    retrieved_chunk_ids_list: list[list[str]],
    ground_truth_chunk_ids_list: list[list[str]],
) -> float:
    """Calculate Mean Reciprocal Rank (MRR) using chunk ID matching.

    MRR measures the average of reciprocal ranks of the first relevant
    result across all queries.

    Args:
        retrieved_chunk_ids_list: List of retrieved chunk ID lists per query.
        ground_truth_chunk_ids_list: List of ground truth chunk ID lists per query.

    Returns:
        MRR as a float between 0 and 1.
    """
    if not retrieved_chunk_ids_list:
        return 0.0

    reciprocal_ranks = []
    for retrieved_ids, gt_ids in zip(
        retrieved_chunk_ids_list, ground_truth_chunk_ids_list, strict=False
    ):
        rank = 0
        for i, ret_id in enumerate(retrieved_ids, 1):
            for gt_id in gt_ids:
                if chunk_id_matches(ret_id, gt_id):
                    rank = i
                    break
            if rank > 0:
                break

        if rank > 0:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def compute_retrieval_metrics(
    eval_results: list[EvalResult], k: int = 5
) -> RetrievalMetrics:
    """Compute aggregate retrieval metrics from evaluation results.

    Args:
        eval_results: List of evaluation results.
        k: Number of top results to consider.

    Returns:
        RetrievalMetrics with aggregated scores.
    """
    # Filter out out-of-scope questions (they have no ground truth chunks)
    in_scope_results = [r for r in eval_results if r.category != "out_of_scope"]

    if not in_scope_results:
        return RetrievalMetrics(hit_rate_at_k=0.0, mrr=0.0, k=k)

    # Hit Rate@K: % of queries with at least one relevant chunk in top-K
    hits = sum(1 for r in in_scope_results if r.retrieval_hit)
    hit_rate = hits / len(in_scope_results)

    # MRR: Mean Reciprocal Rank - uses first_relevant_rank from EvalResult
    reciprocal_ranks = []
    for r in in_scope_results:
        if r.first_relevant_rank is not None and r.first_relevant_rank <= k:
            reciprocal_ranks.append(1.0 / r.first_relevant_rank)
        else:
            reciprocal_ranks.append(0.0)
    mrr_value = sum(reciprocal_ranks) / len(reciprocal_ranks)

    return RetrievalMetrics(
        hit_rate_at_k=hit_rate,
        mrr=mrr_value,
        k=k,
    )


def compute_answer_metrics(eval_results: list[EvalResult]) -> AnswerMetrics:
    """Compute aggregate answer quality metrics from evaluation results.

    Args:
        eval_results: List of evaluation results.

    Returns:
        AnswerMetrics with aggregated scores.
    """
    if not eval_results:
        return AnswerMetrics(accuracy_score=0.0)

    avg_accuracy = sum(r.accuracy_score for r in eval_results) / len(eval_results)

    return AnswerMetrics(accuracy_score=avg_accuracy)
