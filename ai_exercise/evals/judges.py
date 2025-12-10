"""LLM-as-judge evaluation for answer quality.

Uses async execution with OpenAI for parallel judging.
"""

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


async def judge_answer_async(
    question: str,
    answer: str,
    ground_truth: str,
) -> AccuracyJudgment:
    """Judge a single answer for accuracy.

    Args:
        question: The original question.
        answer: The generated answer.
        ground_truth: The expected answer.

    Returns:
        AccuracyJudgment with score (1-5) and reasoning.
    """
    return await judge_accuracy_async(question, answer, ground_truth)


async def judge_answers_batch_async(
    questions: list[str],
    answers: list[str],
    ground_truths: list[str],
) -> list[AccuracyJudgment]:
    """Judge multiple answers in parallel with progress bar.

    Args:
        questions: List of questions.
        answers: List of generated answers.
        ground_truths: List of expected answers.

    Returns:
        List of AccuracyJudgment for each answer.
    """
    tasks = [
        judge_answer_async(q, a, gt)
        for q, a, gt in zip(questions, answers, ground_truths, strict=False)
    ]

    results = await tqdm_asyncio.gather(
        *tasks,
        desc="Judging answers",
        unit="answer",
    )

    return results
