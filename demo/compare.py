"""Streamlit app for comparing evaluation results across configurations.

Start from project root with:
```bash
uv run streamlit run demo/compare.py
```
"""

import json
from pathlib import Path

import streamlit as st

RESULTS_DIR = Path("reports/results")

# Category display order
CATEGORY_ORDER = ["factual", "endpoint", "schema", "auth", "cross_api", "out_of_scope"]


def load_results() -> dict[str, dict]:
    """Load all result files from the reports/results directory."""
    results = {}
    if not RESULTS_DIR.exists():
        return results

    for file_path in sorted(RESULTS_DIR.glob("*_results.json")):
        config_name = file_path.stem.replace("_results", "")
        with open(file_path) as f:
            results[config_name] = json.load(f)

    return results


def get_questions_by_id(results: dict[str, dict]) -> dict[str, dict]:
    """Index all questions by their question_id across all configs."""
    questions = {}
    for config_name, data in results.items():
        for result in data.get("detailed_results", []):
            q_id = result["question_id"]
            if q_id not in questions:
                questions[q_id] = {
                    "question": result["question"],
                    "category": result["category"],
                    "relevant_apis": result.get("relevant_apis", []),
                    "ground_truth": result.get("ground_truth_answer", ""),
                    "configs": {},
                }
            questions[q_id]["configs"][config_name] = {
                "generated_answer": result.get("generated_answer", ""),
                "accuracy_score": result.get("accuracy_score"),
                "retrieval_hit": result.get("retrieval_hit"),
                "first_relevant_rank": result.get("first_relevant_rank"),
                "retrieved_chunk_ids": result.get("retrieved_chunk_ids", []),
            }
    return questions


def render_summary_metrics(results: dict[str, dict]) -> None:
    """Render summary metrics comparison table."""
    if not results:
        return

    st.subheader("Summary Metrics")

    cols = st.columns(len(results))
    for i, (config_name, data) in enumerate(sorted(results.items())):
        with cols[i]:
            st.markdown(f"**{config_name.upper()}**")
            summary = data.get("summary", {})

            retrieval = summary.get("retrieval", {})
            answer = summary.get("answer_quality", {})

            st.metric("Hit Rate@K", f"{retrieval.get('hit_rate_at_k', 0):.1%}")
            st.metric("MRR", f"{retrieval.get('mrr', 0):.2f}")
            st.metric("Avg Accuracy", f"{answer.get('accuracy_score', 0):.2f}/5")
            st.caption(f"Total: {summary.get('total_questions', 0)} questions")


def render_category_breakdown(results: dict[str, dict]) -> None:
    """Render category breakdown comparison."""
    if not results:
        return

    st.subheader("Results by Category")

    # Get all categories
    all_categories = set()
    for data in results.values():
        all_categories.update(data.get("results_by_category", {}).keys())

    # Sort categories
    sorted_categories = [c for c in CATEGORY_ORDER if c in all_categories]
    sorted_categories += [c for c in sorted(all_categories) if c not in CATEGORY_ORDER]

    # Create comparison table
    table_data = []
    for category in sorted_categories:
        row = {"Category": category}
        for config_name in sorted(results.keys()):
            cat_data = results[config_name].get("results_by_category", {}).get(category, {})
            accuracy = cat_data.get("avg_accuracy", 0)
            hit_rate = cat_data.get("retrieval_hit_rate", 0)
            count = cat_data.get("count", 0)
            row[f"{config_name.upper()} Accuracy"] = f"{accuracy:.2f}"
            row[f"{config_name.upper()} Hit Rate"] = f"{hit_rate:.1%}"
            row[f"{config_name.upper()} Count"] = count
        table_data.append(row)

    st.dataframe(table_data, use_container_width=True)


QUESTIONS_PER_PAGE = 10


def render_question_comparison(
    questions: dict[str, dict],
    config_names: list[str],
    selected_category: str | None,
    search_query: str,
) -> None:
    """Render question-by-question comparison."""
    st.subheader("Question Comparison")

    # Filter questions
    filtered_questions = {}
    for q_id, q_data in questions.items():
        # Category filter
        if selected_category and q_data["category"] != selected_category:
            continue
        # Search filter
        if search_query:
            search_lower = search_query.lower()
            if (
                search_lower not in q_data["question"].lower()
                and search_lower not in q_id.lower()
            ):
                continue
        filtered_questions[q_id] = q_data

    # Sort and paginate
    sorted_ids = sorted(filtered_questions.keys())
    total_questions = len(sorted_ids)
    total_pages = max(1, (total_questions + QUESTIONS_PER_PAGE - 1) // QUESTIONS_PER_PAGE)

    # Pagination controls
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        page = st.number_input(
            f"Page (1-{total_pages})",
            min_value=1,
            max_value=total_pages,
            value=1,
            key="question_page",
        )

    start_idx = (page - 1) * QUESTIONS_PER_PAGE
    end_idx = min(start_idx + QUESTIONS_PER_PAGE, total_questions)
    page_ids = sorted_ids[start_idx:end_idx]

    st.caption(f"Showing {start_idx + 1}-{end_idx} of {total_questions} questions")

    # Render only current page
    for q_id in page_ids:
        q_data = filtered_questions[q_id]

        # Build header with scores
        scores_str = " | ".join(
            f"{cfg.upper()}: {q_data['configs'].get(cfg, {}).get('accuracy_score', '-')}"
            for cfg in config_names
        )
        header = f"**{q_id}** ({q_data['category']}) - {scores_str}"

        with st.expander(header):
            # Question
            st.markdown("**Question:**")
            st.info(q_data["question"])

            # Ground truth
            st.markdown("**Ground Truth:**")
            st.success(q_data["ground_truth"])

            # Answers from each config
            st.markdown("**Generated Answers:**")
            tabs = st.tabs([cfg.upper() for cfg in config_names])

            for i, config_name in enumerate(config_names):
                with tabs[i]:
                    config_data = q_data["configs"].get(config_name, {})

                    if config_data:
                        # Metrics row
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            score = config_data.get("accuracy_score")
                            if score is not None:
                                color = "green" if score >= 4 else "orange" if score >= 3 else "red"
                                st.markdown(f"**Accuracy:** :{color}[{score}/5]")
                        with col2:
                            hit = config_data.get("retrieval_hit")
                            if hit is not None:
                                st.markdown(f"**Retrieval Hit:** {'✅' if hit else '❌'}")
                        with col3:
                            rank = config_data.get("first_relevant_rank")
                            if rank is not None:
                                st.markdown(f"**First Relevant Rank:** {rank}")

                        # Answer
                        st.markdown(config_data.get("generated_answer", "No answer"))

                        # Retrieved chunks
                        chunks = config_data.get("retrieved_chunk_ids", [])
                        if chunks:
                            st.markdown(f"**Retrieved Chunks ({len(chunks)}):**")
                            st.code("\n".join(chunks))
                    else:
                        st.warning("No results for this config")


def main():
    st.set_page_config(
        page_title="Eval Comparison",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Evaluation Results Comparison")

    # Load results
    results = load_results()

    if not results:
        st.warning(
            f"No result files found in `{RESULTS_DIR}`. "
            "Run evaluations first with `make eval CONFIG=<config>`"
        )
        return

    config_names = sorted(results.keys())
    st.sidebar.success(f"Loaded {len(config_names)} configs: {', '.join(c.upper() for c in config_names)}")

    # Index questions
    questions = get_questions_by_id(results)

    # Sidebar filters
    st.sidebar.header("Filters")

    # Category filter
    all_categories = sorted(set(q["category"] for q in questions.values()))
    selected_category = st.sidebar.selectbox(
        "Category",
        options=["All"] + all_categories,
        index=0,
    )
    if selected_category == "All":
        selected_category = None

    # Search filter
    search_query = st.sidebar.text_input("Search questions", placeholder="Search by question text or ID...")

    # Score difference filter
    show_differences = st.sidebar.checkbox("Show only score differences", value=False)

    if show_differences and len(config_names) > 1:
        # Filter to questions where scores differ
        diff_questions = {}
        for q_id, q_data in questions.items():
            scores = [
                q_data["configs"].get(cfg, {}).get("accuracy_score")
                for cfg in config_names
            ]
            scores = [s for s in scores if s is not None]
            if len(set(scores)) > 1:  # Scores differ
                diff_questions[q_id] = q_data
        questions = diff_questions

    # Summary metrics
    render_summary_metrics(results)

    st.divider()

    # Category breakdown
    render_category_breakdown(results)

    st.divider()

    # Question comparison
    render_question_comparison(questions, config_names, selected_category, search_query)


if __name__ == "__main__":
    main()
