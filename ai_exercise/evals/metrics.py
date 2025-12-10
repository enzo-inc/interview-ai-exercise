"""Evaluation metrics for RAG system."""

from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation metrics."""

    hit_rate_at_k: float
    mrr: float
    precision_at_k: float
    recall_at_k: float
    k: int


@dataclass
class AnswerMetrics:
    """Container for answer quality metrics."""

    keyword_coverage: float
    accuracy_score: float
    completeness_score: float
    hallucination_rate: float


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
    ground_truth_chunks: list[str]  # Human-readable chunk references
    ground_truth_chunk_ids: list[str]  # Normalized chunk IDs for matching
    relevant_apis: list[str]  # APIs relevant to this question
    retrieval_hit: bool
    first_relevant_rank: int | None  # Rank of first relevant chunk (1-indexed), None if no hit
    keyword_coverage: float
    accuracy_score: int
    completeness_score: int
    has_hallucination: bool


def chunk_id_matches(retrieved_id: str, gt_id: str) -> bool:
    """Check if a retrieved chunk ID matches a ground truth chunk ID.

    Handles split chunks (e.g., "hris_paths_x_part0" matches "hris_paths_x").
    """
    if retrieved_id == gt_id:
        return True
    # Handle split chunks - retrieved might be "base_id_partN"
    if retrieved_id.startswith(gt_id + "_part"):
        return True
    return False


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


def precision_at_k(
    retrieved_chunk_ids_list: list[list[str]],
    ground_truth_chunk_ids_list: list[list[str]],
    k: int = 5,
) -> float:
    """Calculate Precision@K using chunk ID matching.

    Precision@K measures the proportion of relevant chunks among
    the top-K retrieved results, averaged across all queries.

    Args:
        retrieved_chunk_ids_list: List of retrieved chunk ID lists per query.
        ground_truth_chunk_ids_list: List of ground truth chunk ID lists per query.
        k: Number of top results to consider.

    Returns:
        Precision@K as a float between 0 and 1.
    """
    if not retrieved_chunk_ids_list:
        return 0.0

    precisions = []
    for retrieved_ids, gt_ids in zip(
        retrieved_chunk_ids_list, ground_truth_chunk_ids_list, strict=False
    ):
        top_k = retrieved_ids[:k]
        if not top_k:
            precisions.append(0.0)
            continue

        relevant_count = 0
        for retrieved_id in top_k:
            for gt_id in gt_ids:
                if chunk_id_matches(retrieved_id, gt_id):
                    relevant_count += 1
                    break

        precisions.append(relevant_count / len(top_k))

    return sum(precisions) / len(precisions)


def recall_at_k(
    retrieved_chunk_ids_list: list[list[str]],
    ground_truth_chunk_ids_list: list[list[str]],
    k: int = 5,
) -> float:
    """Calculate Recall@K using chunk ID matching.

    Recall@K measures the proportion of all relevant chunks that
    appear in the top-K retrieved results, averaged across all queries.

    Args:
        retrieved_chunk_ids_list: List of retrieved chunk ID lists per query.
        ground_truth_chunk_ids_list: List of ground truth chunk ID lists per query.
        k: Number of top results to consider.

    Returns:
        Recall@K as a float between 0 and 1.
    """
    if not retrieved_chunk_ids_list:
        return 0.0

    recalls = []
    for retrieved_ids, gt_ids in zip(
        retrieved_chunk_ids_list, ground_truth_chunk_ids_list, strict=False
    ):
        if not gt_ids:
            recalls.append(1.0)  # No ground truth = trivially all retrieved
            continue

        top_k = retrieved_ids[:k]
        found_count = 0
        for gt_id in gt_ids:
            if any(chunk_id_matches(ret_id, gt_id) for ret_id in top_k):
                found_count += 1

        recalls.append(found_count / len(gt_ids))

    return sum(recalls) / len(recalls)


def keyword_coverage(answer: str, required_keywords: list[str]) -> float:
    """Calculate keyword coverage in an answer.

    Args:
        answer: The generated answer text.
        required_keywords: List of keywords that should appear.

    Returns:
        Proportion of keywords found (0 to 1).
    """
    if not required_keywords:
        return 1.0

    answer_lower = answer.lower()
    found = sum(1 for kw in required_keywords if kw.lower() in answer_lower)
    return found / len(required_keywords)


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
        return RetrievalMetrics(
            hit_rate_at_k=0.0, mrr=0.0, precision_at_k=0.0, recall_at_k=0.0, k=k
        )

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

    # Precision@K and Recall@K: Use chunk ID matching
    retrieved_chunk_ids_list = [r.retrieved_chunk_ids for r in in_scope_results]
    ground_truth_chunk_ids_list = [r.ground_truth_chunk_ids for r in in_scope_results]

    precision_value = precision_at_k(retrieved_chunk_ids_list, ground_truth_chunk_ids_list, k)
    recall_value = recall_at_k(retrieved_chunk_ids_list, ground_truth_chunk_ids_list, k)

    return RetrievalMetrics(
        hit_rate_at_k=hit_rate,
        mrr=mrr_value,
        precision_at_k=precision_value,
        recall_at_k=recall_value,
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
        return AnswerMetrics(
            keyword_coverage=0.0,
            accuracy_score=0.0,
            completeness_score=0.0,
            hallucination_rate=0.0,
        )

    # Filter out out-of-scope for keyword coverage
    in_scope_results = [r for r in eval_results if r.category != "out_of_scope"]

    avg_keyword_coverage = (
        sum(r.keyword_coverage for r in in_scope_results) / len(in_scope_results)
        if in_scope_results
        else 0.0
    )

    avg_accuracy = sum(r.accuracy_score for r in eval_results) / len(eval_results)
    avg_completeness = sum(r.completeness_score for r in eval_results) / len(
        eval_results
    )
    hallucination_rate = sum(1 for r in eval_results if r.has_hallucination) / len(
        eval_results
    )

    return AnswerMetrics(
        keyword_coverage=avg_keyword_coverage,
        accuracy_score=avg_accuracy,
        completeness_score=avg_completeness,
        hallucination_rate=hallucination_rate,
    )


def compute_abstention_metrics(
    eval_results: list[EvalResult],
) -> dict[str, float]:
    """Compute abstention metrics for out-of-scope questions.

    Args:
        eval_results: List of evaluation results.

    Returns:
        Dictionary with abstention metrics.
    """
    out_of_scope = [r for r in eval_results if r.category == "out_of_scope"]
    in_scope = [r for r in eval_results if r.category != "out_of_scope"]

    # Check if answer contains abstention phrases
    abstention_phrases = [
        "don't know",
        "don't have",
        "not available",
        "cannot find",
        "no information",
        "not in the documentation",
        "unable to find",
        "i'm not sure",
    ]

    def is_abstention(answer: str) -> bool:
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in abstention_phrases)

    # Correct abstention: out-of-scope questions where system abstains
    correct_abstentions = sum(
        1 for r in out_of_scope if is_abstention(r.generated_answer)
    )
    correct_abstention_rate = (
        correct_abstentions / len(out_of_scope) if out_of_scope else 0.0
    )

    # False abstention: in-scope questions where system incorrectly abstains
    false_abstentions = sum(1 for r in in_scope if is_abstention(r.generated_answer))
    false_abstention_rate = false_abstentions / len(in_scope) if in_scope else 0.0

    return {
        "correct_abstention_rate": correct_abstention_rate,
        "false_abstention_rate": false_abstention_rate,
        "out_of_scope_count": len(out_of_scope),
        "in_scope_count": len(in_scope),
    }
