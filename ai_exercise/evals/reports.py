"""Report generation for evaluation results."""

import json
from datetime import datetime
from pathlib import Path

from tabulate import tabulate


def generate_markdown_report(results_dir: Path, output_dir: Path) -> Path:
    """Generate a markdown report from all result files.

    Args:
        results_dir: Directory containing JSON result files.
        output_dir: Directory to write the report.

    Returns:
        Path to the generated report.
    """
    # Load all result files
    result_files = list(results_dir.glob("*_results.json"))
    if not result_files:
        raise ValueError(f"No result files found in {results_dir}")

    configs_data = []
    for f in sorted(result_files):
        with open(f) as fp:
            data = json.load(fp)
            configs_data.append((data["config"], data))

    # Generate report
    report_lines = [
        "# StackOne RAG Evaluation Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"Configurations evaluated: {len(configs_data)}",
        "",
    ]

    # Retrieval metrics table
    report_lines.extend([
        "## Retrieval Metrics",
        "",
    ])

    retrieval_headers = ["Config", "Hit Rate@K", "MRR", "Precision@K", "Recall@K"]
    retrieval_rows = []
    for name, data in configs_data:
        r = data["summary"]["retrieval"]
        retrieval_rows.append([
            name,
            f"{r['hit_rate_at_k']:.3f}",
            f"{r['mrr']:.3f}",
            f"{r['precision_at_k']:.3f}",
            f"{r['recall_at_k']:.3f}",
        ])

    report_lines.append(
        tabulate(retrieval_rows, headers=retrieval_headers, tablefmt="pipe")
    )
    report_lines.append("")

    # Answer quality metrics table
    report_lines.extend([
        "## Answer Quality Metrics",
        "",
    ])

    answer_headers = [
        "Config", "Keyword Cov.", "Accuracy (1-5)",
        "Completeness (1-5)", "Hallucination Rate",
    ]
    answer_rows = []
    for name, data in configs_data:
        a = data["summary"]["answer_quality"]
        answer_rows.append([
            name,
            f"{a['keyword_coverage']:.3f}",
            f"{a['accuracy_score']:.2f}",
            f"{a['completeness_score']:.2f}",
            f"{a['hallucination_rate']:.3f}",
        ])

    report_lines.append(
        tabulate(answer_rows, headers=answer_headers, tablefmt="pipe")
    )
    report_lines.append("")

    # Abstention metrics table
    report_lines.extend([
        "## Abstention Metrics",
        "",
    ])

    abstention_headers = [
        "Config", "Correct Abstention", "False Abstention",
        "Out-of-Scope", "In-Scope",
    ]
    abstention_rows = []
    for name, data in configs_data:
        ab = data["summary"]["abstention"]
        abstention_rows.append([
            name,
            f"{ab['correct_abstention_rate']:.3f}",
            f"{ab['false_abstention_rate']:.3f}",
            str(int(ab.get("out_of_scope_count", 0))),
            str(int(ab.get("in_scope_count", 0))),
        ])

    report_lines.append(
        tabulate(abstention_rows, headers=abstention_headers, tablefmt="pipe")
    )
    report_lines.append("")

    # Per-category breakdown for each config
    report_lines.extend([
        "## Results by Category",
        "",
    ])

    for name, data in configs_data:
        report_lines.extend([
            f"### {name}",
            "",
        ])

        if "results_by_category" in data:
            cat_headers = [
                "Category", "Count", "Keyword Cov.", "Accuracy", "Retrieval Hit",
            ]
            cat_rows = []
            for cat, cat_data in sorted(data["results_by_category"].items()):
                cat_rows.append([
                    cat,
                    str(cat_data["count"]),
                    f"{cat_data['avg_keyword_coverage']:.3f}",
                    f"{cat_data['avg_accuracy']:.2f}",
                    f"{cat_data['retrieval_hit_rate']:.3f}",
                ])
            report_lines.append(
                tabulate(cat_rows, headers=cat_headers, tablefmt="pipe")
            )
            report_lines.append("")

    # Write report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "evaluation_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    return report_path


def format_comparison_table(
    configs_data: list[tuple[str, dict]],
) -> str:
    """Format a comparison table as a string.

    Args:
        configs_data: List of (config_name, data) tuples.

    Returns:
        Formatted table string.
    """
    def _get_retrieval(key: str):
        def getter(d: dict) -> float:
            return d["summary"]["retrieval"][key]
        return getter

    def _get_answer(key: str):
        def getter(d: dict) -> float:
            return d["summary"]["answer_quality"][key]
        return getter

    metrics = [
        ("Hit Rate@K", _get_retrieval("hit_rate_at_k")),
        ("MRR", _get_retrieval("mrr")),
        ("Precision@K", _get_retrieval("precision_at_k")),
        ("Recall@K", _get_retrieval("recall_at_k")),
        ("Keyword Coverage", _get_answer("keyword_coverage")),
        ("Accuracy (1-5)", _get_answer("accuracy_score")),
        ("Completeness (1-5)", _get_answer("completeness_score")),
        ("Hallucination Rate", _get_answer("hallucination_rate")),
    ]

    headers = ["Metric"] + [name for name, _ in configs_data]
    rows = []

    for metric_name, getter in metrics:
        row = [metric_name]
        for _, data in configs_data:
            try:
                row.append(f"{getter(data):.3f}")
            except (KeyError, TypeError):
                row.append("N/A")
        rows.append(row)

    return tabulate(rows, headers=headers, tablefmt="pipe")
