"""Retrieval and end-to-end extraction evaluation."""

from collections.abc import Sequence
from typing import Protocol

from vehicle_specs.extraction import ExtractionResponse, SpecificationResult
from vehicle_specs.retrieval import RetrievalHit

from .models import (
    EvaluationFailure,
    EvaluationReport,
    ExtractionCaseEvaluation,
    ExtractionMetrics,
    GoldQuery,
    GoldSpecification,
    RetrievalMetrics,
)


class EvaluationRetriever(Protocol):
    """Retrieval method required by the evaluator."""

    def retrieve(self, query: str) -> list[RetrievalHit]: ...


class EvaluationPipeline(Protocol):
    """Extraction method required by the evaluator."""

    def query(self, query: str) -> ExtractionResponse: ...


class EvaluationRegressionError(AssertionError):
    """Raised when measured retrieval drops below required thresholds."""


class EvaluationExecutionError(RuntimeError):
    """Raised when one or more gold cases could not be executed."""


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_retrieval(
    gold_queries: Sequence[GoldQuery],
    retriever: EvaluationRetriever,
) -> tuple[RetrievalMetrics, tuple[EvaluationFailure, ...]]:
    """Calculate macro recall and reciprocal rank for answerable queries."""
    recalls_at_5: list[float] = []
    recalls_at_10: list[float] = []
    reciprocal_ranks: list[float] = []
    failures: list[EvaluationFailure] = []

    for gold in gold_queries:
        if not gold.relevant_chunk_ids:
            continue
        try:
            hits = retriever.retrieve(gold.query)
        except Exception as error:  # noqa: BLE001 - an evaluation must record, not abort
            failures.append(
                EvaluationFailure(
                    query=gold.query,
                    stage="retrieval",
                    error=f"{type(error).__name__}: {error}",
                )
            )
            hits = []

        ranked_ids = [hit.chunk_id for hit in hits]
        relevant = set(gold.relevant_chunk_ids)
        recalls_at_5.append(len(relevant.intersection(ranked_ids[:5])) / len(relevant))
        recalls_at_10.append(
            len(relevant.intersection(ranked_ids[:10])) / len(relevant)
        )
        first_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(ranked_ids, start=1)
                if chunk_id in relevant
            ),
            None,
        )
        reciprocal_ranks.append(1 / first_rank if first_rank is not None else 0.0)

    metrics = RetrievalMetrics(
        evaluated_queries=len(recalls_at_5),
        recall_at_5=_mean(recalls_at_5),
        recall_at_10=_mean(recalls_at_10),
        mean_reciprocal_rank=_mean(reciprocal_ranks),
    )
    return metrics, tuple(failures)


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _optional_normalized(value: str | None) -> str | None:
    return _normalized(value) if value is not None else None


def _field_matches(
    expected: GoldSpecification,
    result: SpecificationResult,
) -> tuple[bool, bool, bool, bool, bool]:
    component_match = _normalized(expected.component_contains) in _normalized(
        result.component
    )
    value_match = _normalized(expected.value) == _normalized(result.value)
    unit_match = _optional_normalized(expected.unit) == _optional_normalized(
        result.unit
    )
    citation_match = expected.pdf_page == result.source.pdf_page
    type_match = expected.spec_type == result.spec_type
    return component_match, value_match, unit_match, citation_match, type_match


def _best_result_match(
    expected: GoldSpecification,
    response: ExtractionResponse,
) -> tuple[bool, bool, bool, bool, bool]:
    matches = [_field_matches(expected, result) for result in response.results]
    if not matches:
        return False, False, False, False, False
    return (
        any(match[0] for match in matches),
        any(match[1] for match in matches),
        any(match[2] for match in matches),
        any(match[3] for match in matches),
        any(match[4] for match in matches),
    )


def _maximum_exact_matches(
    expected_results: Sequence[GoldSpecification],
    response: ExtractionResponse,
) -> int:
    """Return a one-to-one maximum matching between expected and actual results."""
    candidates = [
        [
            result_index
            for result_index, result in enumerate(response.results)
            if all(_field_matches(expected, result))
        ]
        for expected in expected_results
    ]
    expected_for_result: dict[int, int] = {}

    def assign(expected_index: int, seen_results: set[int]) -> bool:
        for result_index in candidates[expected_index]:
            if result_index in seen_results:
                continue
            seen_results.add(result_index)
            previous_expected = expected_for_result.get(result_index)
            if previous_expected is None or assign(previous_expected, seen_results):
                expected_for_result[result_index] = expected_index
                return True
        return False

    return sum(
        assign(expected_index, set()) for expected_index in range(len(expected_results))
    )


def _evaluate_extraction_detailed(
    gold_queries: Sequence[GoldQuery],
    pipeline: EvaluationPipeline,
) -> tuple[
    ExtractionMetrics,
    tuple[EvaluationFailure, ...],
    tuple[ExtractionCaseEvaluation, ...],
]:
    """Measure status and field accuracy for validated pipeline responses."""
    status_matches = 0
    component_matches = 0
    value_matches = 0
    unit_matches = 0
    citation_matches = 0
    exact_matches = 0
    exact_response_matches = 0
    returned_count = 0
    ambiguous_cases = 0
    ambiguous_matches = 0
    abstention_cases = 0
    abstention_matches = 0
    expected_count = sum(len(gold.expected_results) for gold in gold_queries)
    failures: list[EvaluationFailure] = []
    cases: list[ExtractionCaseEvaluation] = []

    for gold in gold_queries:
        try:
            response = pipeline.query(gold.query)
        except Exception as error:  # noqa: BLE001 - an evaluation must record, not abort
            error_message = f"{type(error).__name__}: {error}"
            failures.append(
                EvaluationFailure(
                    query=gold.query,
                    stage="extraction",
                    error=error_message,
                )
            )
            cases.append(
                ExtractionCaseEvaluation(
                    query=gold.query,
                    expected_status=gold.expected_status,
                    expected_result_count=len(gold.expected_results),
                    returned_result_count=0,
                    exact_result_count=0,
                    status_match=False,
                    exact_response_match=False,
                    error=error_message,
                )
            )
            continue

        status_match = response.status == gold.expected_status
        status_matches += status_match
        returned_count += len(response.results)
        query_exact_matches = _maximum_exact_matches(gold.expected_results, response)
        exact_matches += query_exact_matches
        exact_response_match = bool(
            status_match
            and query_exact_matches == len(gold.expected_results)
            and query_exact_matches == len(response.results)
        )
        exact_response_matches += exact_response_match
        if gold.expected_status == "ambiguous":
            ambiguous_cases += 1
            ambiguous_matches += response.status == "ambiguous"
        if gold.expected_status == "not_found":
            abstention_cases += 1
            abstention_matches += response.status == "not_found"
        for expected in gold.expected_results:
            component, value, unit, citation, _ = _best_result_match(
                expected,
                response,
            )
            component_matches += component
            value_matches += value
            unit_matches += unit
            citation_matches += citation
        cases.append(
            ExtractionCaseEvaluation(
                query=gold.query,
                expected_status=gold.expected_status,
                actual_status=response.status,
                expected_result_count=len(gold.expected_results),
                returned_result_count=len(response.results),
                exact_result_count=query_exact_matches,
                status_match=status_match,
                exact_response_match=exact_response_match,
                response=response,
            )
        )

    denominator = expected_count or 1
    precision_denominator = returned_count or 1
    query_denominator = len(gold_queries) or 1
    metrics = ExtractionMetrics(
        evaluated_queries=len(gold_queries),
        expected_result_count=expected_count,
        returned_result_count=returned_count,
        status_accuracy=status_matches / query_denominator,
        ambiguity_accuracy=ambiguous_matches / (ambiguous_cases or 1),
        abstention_accuracy=abstention_matches / (abstention_cases or 1),
        component_accuracy=component_matches / denominator,
        value_accuracy=value_matches / denominator,
        unit_accuracy=unit_matches / denominator,
        citation_accuracy=citation_matches / denominator,
        result_precision=(
            exact_matches / precision_denominator
            if returned_count
            else float(expected_count == 0)
        ),
        exact_result_accuracy=exact_matches / denominator,
        exact_response_accuracy=exact_response_matches / query_denominator,
    )
    return metrics, tuple(failures), tuple(cases)


def evaluate_extraction(
    gold_queries: Sequence[GoldQuery],
    pipeline: EvaluationPipeline,
) -> tuple[ExtractionMetrics, tuple[EvaluationFailure, ...]]:
    """Measure extraction while preserving the original aggregate-only API."""
    metrics, failures, _ = _evaluate_extraction_detailed(gold_queries, pipeline)
    return metrics, failures


def evaluate(
    gold_queries: Sequence[GoldQuery],
    retriever: EvaluationRetriever,
    *,
    pipeline: EvaluationPipeline | None = None,
) -> EvaluationReport:
    """Run retrieval evaluation and optionally the LLM extraction evaluation."""
    retrieval, retrieval_failures = evaluate_retrieval(gold_queries, retriever)
    extraction = None
    extraction_failures: tuple[EvaluationFailure, ...] = ()
    extraction_cases: tuple[ExtractionCaseEvaluation, ...] = ()
    if pipeline is not None:
        extraction, extraction_failures, extraction_cases = (
            _evaluate_extraction_detailed(gold_queries, pipeline)
        )
    return EvaluationReport(
        retrieval=retrieval,
        extraction=extraction,
        extraction_cases=extraction_cases,
        failures=(*retrieval_failures, *extraction_failures),
    )


def enforce_retrieval_thresholds(
    report: EvaluationReport,
    *,
    minimum_recall_at_5: float,
    minimum_recall_at_10: float,
) -> None:
    """Fail a regression run when retrieval recall drops below its contract."""
    failures: list[str] = []
    if report.retrieval.recall_at_5 < minimum_recall_at_5:
        failures.append(
            f"Recall@5 {report.retrieval.recall_at_5:.3f} < {minimum_recall_at_5:.3f}"
        )
    if report.retrieval.recall_at_10 < minimum_recall_at_10:
        failures.append(
            f"Recall@10 {report.retrieval.recall_at_10:.3f} < "
            f"{minimum_recall_at_10:.3f}"
        )
    if failures:
        raise EvaluationRegressionError("; ".join(failures))


def enforce_no_evaluation_failures(report: EvaluationReport) -> None:
    """Fail a release gate when any gold case raised an operational error."""
    if report.failures:
        stages = sorted({failure.stage for failure in report.failures})
        raise EvaluationExecutionError(
            f"{len(report.failures)} case(s) failed during {', '.join(stages)}"
        )
