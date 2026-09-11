"""Gold-case loading, scoring, regression checks, and report output."""

from .evaluator import (
    EvaluationExecutionError,
    EvaluationRegressionError,
    enforce_no_evaluation_failures,
    enforce_retrieval_thresholds,
    evaluate,
    evaluate_extraction,
    evaluate_retrieval,
)
from .io import load_gold_queries, render_report_markdown, write_evaluation_report
from .models import (
    EvaluationFailure,
    EvaluationReport,
    ExtractionCaseEvaluation,
    ExtractionMetrics,
    GoldQuery,
    GoldSpecification,
    RetrievalMetrics,
)

__all__ = [
    "EvaluationExecutionError",
    "EvaluationFailure",
    "EvaluationRegressionError",
    "EvaluationReport",
    "ExtractionCaseEvaluation",
    "ExtractionMetrics",
    "GoldQuery",
    "GoldSpecification",
    "RetrievalMetrics",
    "enforce_no_evaluation_failures",
    "enforce_retrieval_thresholds",
    "evaluate",
    "evaluate_extraction",
    "evaluate_retrieval",
    "load_gold_queries",
    "render_report_markdown",
    "write_evaluation_report",
]
