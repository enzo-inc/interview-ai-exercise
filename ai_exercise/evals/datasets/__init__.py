"""Evaluation datasets module."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class EvalQuestion(BaseModel):
    """Schema for an evaluation question."""

    id: str = Field(description="Unique identifier for this question")
    question: str = Field(description="The question to ask the RAG system")
    category: Literal[
        "factual", "endpoint", "schema", "cross_api", "auth", "out_of_scope"
    ] = Field(description="Category of the question")
    relevant_apis: list[str] = Field(
        description="List of API names relevant to this question"
    )
    ground_truth_answer: str = Field(
        description="The expected correct answer based on the API documentation"
    )
    relevant_structural_ids: list[str] = Field(
        description="Structural IDs that identify relevant chunks (config-agnostic). "
        "Format: 'api_name.section.key[.method]' e.g., "
        "'stackone.paths./connect_sessions.post' "
        "or 'stackone.components.ConnectSessionCreate'"
    )
    required_keywords: list[str] = Field(
        description="Keywords that should appear in a correct answer"
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        description="Difficulty level of the question"
    )


class GeneratedDataset(BaseModel):
    """Schema for the full evaluation dataset."""

    version: str = Field(default="1.0", description="Dataset version")
    description: str = Field(
        default="Evaluation dataset for StackOne RAG system",
        description="Dataset description",
    )
    generation_metadata: dict = Field(
        default_factory=dict, description="Metadata about how dataset was generated"
    )
    questions: list[EvalQuestion] = Field(description="List of evaluation questions")


def load_eval_dataset() -> list[EvalQuestion]:
    """Load the evaluation dataset from JSON file."""
    dataset_path = Path(__file__).parent / "eval_dataset.json"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Eval dataset not found at {dataset_path}. "
            "Run 'make generate-eval-questions' to generate it."
        )
    with open(dataset_path) as f:
        data = json.load(f)
    return [EvalQuestion(**q) for q in data["questions"]]


def get_questions_by_category(category: str) -> list[EvalQuestion]:
    """Get all questions of a specific category."""
    questions = load_eval_dataset()
    return [q for q in questions if q.category == category]


def save_eval_dataset(dataset: GeneratedDataset, path: Path | None = None) -> None:
    """Save the evaluation dataset to JSON file."""
    if path is None:
        path = Path(__file__).parent / "eval_dataset.json"
    with open(path, "w") as f:
        json.dump(dataset.model_dump(), f, indent=2)
