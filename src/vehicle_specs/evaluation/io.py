"""Gold-case loading and atomic report persistence."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import ValidationError

from .models import EvaluationReport, GoldQuery


def load_gold_queries(path: str | Path) -> tuple[GoldQuery, ...]:
    """Load and validate UTF-8 JSON Lines gold cases."""
    input_path = Path(path)
    queries: list[GoldQuery] = []
    seen_queries: set[str] = set()

    with input_path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                query = GoldQuery.model_validate_json(line)
            except (ValidationError, ValueError) as error:
                raise ValueError(
                    f"invalid gold case at {input_path}:{line_number}"
                ) from error
            normalized = query.query.casefold()
            if normalized in seen_queries:
                raise ValueError(f"duplicate gold query at {input_path}:{line_number}")
            seen_queries.add(normalized)
            queries.append(query)

    if not queries:
        raise ValueError(f"gold query file is empty: {input_path}")
    return tuple(queries)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(content)
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def render_report_markdown(report: EvaluationReport) -> str:
    """Render a compact report suitable for source review."""
    retrieval = report.retrieval
    lines = [
        "# Evaluation Report",
        "",
        "## Retrieval",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Evaluated queries | {retrieval.evaluated_queries} |",
        f"| Recall@5 | {retrieval.recall_at_5:.3f} |",
        f"| Recall@10 | {retrieval.recall_at_10:.3f} |",
        f"| Mean reciprocal rank | {retrieval.mean_reciprocal_rank:.3f} |",
    ]
    if report.extraction is not None:
        extraction = report.extraction
        lines.extend(
            [
                "",
                "## Extraction",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Evaluated queries | {extraction.evaluated_queries} |",
                f"| Expected results | {extraction.expected_result_count} |",
                f"| Returned results | {extraction.returned_result_count} |",
                f"| Status accuracy | {extraction.status_accuracy:.3f} |",
                f"| Ambiguity accuracy | {extraction.ambiguity_accuracy:.3f} |",
                f"| Abstention accuracy | {extraction.abstention_accuracy:.3f} |",
                f"| Component accuracy | {extraction.component_accuracy:.3f} |",
                f"| Value accuracy | {extraction.value_accuracy:.3f} |",
                f"| Unit accuracy | {extraction.unit_accuracy:.3f} |",
                f"| Citation accuracy | {extraction.citation_accuracy:.3f} |",
                f"| Result precision | {extraction.result_precision:.3f} |",
                f"| Exact result accuracy | {extraction.exact_result_accuracy:.3f} |",
                (
                    "| Exact response accuracy | "
                    f"{extraction.exact_response_accuracy:.3f} |"
                ),
            ]
        )
    if report.failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(
            f"- `{failure.stage}` — {failure.query}: {failure.error}"
            for failure in report.failures
        )
    mismatches = [
        case for case in report.extraction_cases if not case.exact_response_match
    ]
    if mismatches:
        lines.extend(["", "## Extraction mismatches", ""])
        for case in mismatches:
            actual = case.actual_status or "error"
            detail = (
                f"; {case.error}"
                if case.error is not None
                else (
                    f"; exact results {case.exact_result_count}/"
                    f"{case.expected_result_count}; returned "
                    f"{case.returned_result_count}"
                )
            )
            lines.append(
                f"- {case.query}: expected `{case.expected_status}`, "
                f"got `{actual}`{detail}"
            )
    return "\n".join(lines) + "\n"


def write_evaluation_report(
    report: EvaluationReport,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    """Atomically persist machine- and human-readable reports."""
    _atomic_write(Path(json_path), report.model_dump_json(indent=2) + "\n")
    _atomic_write(Path(markdown_path), render_report_markdown(report))
