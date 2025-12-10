"""LLM-as-judge evaluation for answer quality.

Uses async execution with OpenAI for parallel judging.
"""

import asyncio

from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm_asyncio

from ai_exercise.constants import SETTINGS

# Initialize async OpenAI client
async_openai_client = AsyncOpenAI(api_key=SETTINGS.openai_api_key.get_secret_value())


class AccuracyJudgment(BaseModel):
    """Structured output for accuracy judgment."""

    score: int = Field(
        ge=1,
        le=5,
        description=(
            "Accuracy score from 1 (completely wrong) to 5 (perfectly accurate)"
        ),
    )
    reasoning: str = Field(description="Brief explanation for the score")


class CompletenessJudgment(BaseModel):
    """Structured output for completeness judgment."""

    score: int = Field(
        ge=1,
        le=5,
        description=(
            "Completeness score from 1 (missing most info) to 5 (fully complete)"
        ),
    )
    reasoning: str = Field(description="Brief explanation for the score")


class HallucinationJudgment(BaseModel):
    """Structured output for hallucination detection."""

    has_hallucination: bool = Field(
        description="True if answer contains unsupported claims"
    )
    hallucinated_claims: list[str] = Field(
        default_factory=list, description="List of specific hallucinated claims if any"
    )


async def judge_accuracy_async(
    question: str,
    answer: str,
    ground_truth: str,
) -> AccuracyJudgment:
    """Judge the factual accuracy of an answer using LLM.

    Args:
        question: The original question asked.
        answer: The generated answer to evaluate.
        ground_truth: The expected correct answer.

    Returns:
        AccuracyJudgment with score (1-5) and reasoning.
    """
    prompt = f"""You are evaluating the factual accuracy of a RAG system's answer.

QUESTION: {question}

EXPECTED ANSWER (Ground Truth):
{ground_truth}

GENERATED ANSWER:
{answer}

Rate the accuracy of the generated answer on a scale of 1-5:
1 = Completely wrong or contradicts ground truth
2 = Mostly wrong with some correct elements
3 = Partially correct but missing key facts or has errors
4 = Mostly correct with minor inaccuracies
5 = Perfectly accurate, matches ground truth

Focus on factual correctness, not style or verbosity."""

    response = await async_openai_client.beta.chat.completions.parse(
        model=SETTINGS.openai_model,
        messages=[{"role": "user", "content": prompt}],
        response_format=AccuracyJudgment,
    )

    result = response.choices[0].message.parsed
    if result is None:
        return AccuracyJudgment(score=1, reasoning="Failed to parse judgment")
    return result


async def judge_completeness_async(
    question: str,
    answer: str,
    ground_truth: str,
) -> CompletenessJudgment:
    """Judge the completeness of an answer using LLM.

    Args:
        question: The original question asked.
        answer: The generated answer to evaluate.
        ground_truth: The expected correct answer.

    Returns:
        CompletenessJudgment with score (1-5) and reasoning.
    """
    prompt = f"""You are evaluating the completeness of a RAG system's answer.

QUESTION: {question}

EXPECTED ANSWER (Ground Truth):
{ground_truth}

GENERATED ANSWER:
{answer}

Rate the completeness of the generated answer on a scale of 1-5:
1 = Missing almost all relevant information
2 = Missing most key information
3 = Contains some key information but incomplete
4 = Contains most key information with minor gaps
5 = Fully complete, covers all aspects of the question

Focus on whether all important parts of the answer are present."""

    response = await async_openai_client.beta.chat.completions.parse(
        model=SETTINGS.openai_model,
        messages=[{"role": "user", "content": prompt}],
        response_format=CompletenessJudgment,
    )

    result = response.choices[0].message.parsed
    if result is None:
        return CompletenessJudgment(score=1, reasoning="Failed to parse judgment")
    return result


async def detect_hallucination_async(
    answer: str,
    context: list[str],
) -> HallucinationJudgment:
    """Detect if an answer contains hallucinated (unsupported) claims.

    Args:
        answer: The generated answer to check.
        context: The retrieved context chunks used to generate the answer.

    Returns:
        HallucinationJudgment indicating if hallucinations were found.
    """
    context_text = "\n\n---\n\n".join(context) if context else "No context provided"

    prompt = f"""You are detecting hallucinations in a RAG system's answer.

A hallucination is a claim in the answer that is NOT supported by the provided context.

CONTEXT (Retrieved Chunks):
{context_text}

GENERATED ANSWER:
{answer}

Analyze the answer and identify any claims that are:
1. Not mentioned in the context
2. Contradict the context
3. Are made up or fabricated

Be strict: if specific details (numbers, names, technical specs) appear in the answer
but not in the context, they are likely hallucinations."""

    response = await async_openai_client.beta.chat.completions.parse(
        model=SETTINGS.openai_model,
        messages=[{"role": "user", "content": prompt}],
        response_format=HallucinationJudgment,
    )

    result = response.choices[0].message.parsed
    if result is None:
        return HallucinationJudgment(has_hallucination=False, hallucinated_claims=[])
    return result


async def judge_answer_async(
    question: str,
    answer: str,
    ground_truth: str,
    context: list[str],
) -> tuple[AccuracyJudgment, CompletenessJudgment, HallucinationJudgment]:
    """Run all judges in parallel for a single answer.

    Args:
        question: The original question.
        answer: The generated answer.
        ground_truth: The expected answer.
        context: Retrieved context chunks.

    Returns:
        Tuple of (accuracy, completeness, hallucination) judgments.
    """
    accuracy, completeness, hallucination = await asyncio.gather(
        judge_accuracy_async(question, answer, ground_truth),
        judge_completeness_async(question, answer, ground_truth),
        detect_hallucination_async(answer, context),
    )
    return accuracy, completeness, hallucination


async def judge_answers_batch_async(
    questions: list[str],
    answers: list[str],
    ground_truths: list[str],
    contexts: list[list[str]],
) -> list[tuple[AccuracyJudgment, CompletenessJudgment, HallucinationJudgment]]:
    """Judge multiple answers in parallel with progress bar.

    Args:
        questions: List of questions.
        answers: List of generated answers.
        ground_truths: List of expected answers.
        contexts: List of context chunk lists.

    Returns:
        List of judgment tuples for each answer.
    """
    tasks = [
        judge_answer_async(q, a, gt, ctx)
        for q, a, gt, ctx in zip(
            questions, answers, ground_truths, contexts, strict=False
        )
    ]

    results = await tqdm_asyncio.gather(
        *tasks,
        desc="Judging answers",
        unit="answer",
    )

    return results
