import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from vehicle_specs.evaluation import (
    EvaluationExecutionError,
    EvaluationFailure,
    EvaluationRegressionError,
    EvaluationReport,
    GoldQuery,
    GoldSpecification,
    enforce_no_evaluation_failures,
    enforce_retrieval_thresholds,
    evaluate,
    evaluate_extraction,
    evaluate_retrieval,
    load_gold_queries,
    render_report_markdown,
    write_evaluation_report,
)
from vehicle_specs.extraction import (
    ExtractionResponse,
    SourceEvidence,
    SpecificationResult,
)
from vehicle_specs.indexing import PipelineVersions
from vehicle_specs.retrieval import RetrievalHit


def make_hit(chunk_id: str, rank: int) -> RetrievalHit:
    return RetrievalHit(
        point_id=rank,
        chunk_id=chunk_id,
        text="Brake caliper guide pin bolts | 37 | Nm",
        source="manual.pdf",
        kind="table",
        pdf_pages=(636,),
        pipeline_versions=PipelineVersions(),
        section_id="206-03",
        rank=rank,
        score=1 / rank,
    )


def make_gold(
    query: str,
    chunk_id: str,
    *,
    component: str = "guide pin bolts",
    value: str = "37",
) -> GoldQuery:
    return GoldQuery(
        query=query,
        relevant_chunk_ids=(chunk_id,),
        expected_status="found",
        expected_results=(
            GoldSpecification(
                component_contains=component,
                spec_type="torque",
                value=value,
                unit="Nm",
                pdf_page=636,
            ),
        ),
    )


class MappingRetriever:
    def __init__(self, hits: dict[str, list[RetrievalHit]]) -> None:
        self.hits = hits

    def retrieve(self, query: str) -> list[RetrievalHit]:
        return self.hits[query]


def test_retrieval_metrics_use_relevant_rank_and_answerable_cases_only() -> None:
    gold = [
        make_gold("first", "relevant-1"),
        make_gold("second", "relevant-2"),
        GoldQuery(query="unsupported", expected_status="not_found"),
    ]
    retriever = MappingRetriever(
        {
            "first": [make_hit("irrelevant", 1), make_hit("relevant-1", 2)],
            "second": [make_hit(f"miss-{rank}", rank) for rank in range(1, 11)],
        }
    )

    metrics, failures = evaluate_retrieval(gold, retriever)

    assert metrics.evaluated_queries == 2
    assert metrics.recall_at_5 == 0.5
    assert metrics.recall_at_10 == 0.5
    assert metrics.mean_reciprocal_rank == 0.25
    assert failures == ()


def make_result(component: str, value: str) -> SpecificationResult:
    return SpecificationResult(
        component=component,
        spec_type="torque",
        value=value,
        unit="Nm",
        source=SourceEvidence(
            chunk_id="relevant",
            pdf_page=636,
            section_id="206-03",
            evidence=f"{component} | {value} | Nm",
        ),
    )


class StaticPipeline:
    def __init__(self, responses: dict[str, ExtractionResponse]) -> None:
        self.responses = responses

    def query(self, query: str) -> ExtractionResponse:
        return self.responses[query]


def response(
    status: str,
    results: tuple[SpecificationResult, ...] = (),
    *,
    clarification: str | None = None,
) -> ExtractionResponse:
    return ExtractionResponse(
        status=status,
        results=results,
        clarification=clarification,
        model_name="test-model",
        prompt_version="test-prompt",
        pipeline_version="test-pipeline",
    )


def test_extraction_exact_match_requires_one_result_to_match_every_field() -> None:
    gold = [make_gold("split match", "relevant")]
    pipeline = StaticPipeline(
        {
            "split match": response(
                "found",
                (
                    make_result("guide pin bolts", "99"),
                    make_result("another component", "37"),
                ),
            )
        }
    )

    metrics, failures = evaluate_extraction(gold, pipeline)

    assert metrics.component_accuracy == 1
    assert metrics.value_accuracy == 1
    assert metrics.exact_result_accuracy == 0
    assert metrics.result_precision == 0
    assert metrics.exact_response_accuracy == 0
    assert failures == ()


def test_extraction_precision_and_exact_response_penalize_extra_results() -> None:
    gold = [make_gold("extra", "relevant")]
    pipeline = StaticPipeline(
        {
            "extra": response(
                "found",
                (
                    make_result("guide pin bolts", "37"),
                    make_result("unrequested bracket bolt", "48"),
                ),
            )
        }
    )

    metrics, _ = evaluate_extraction(gold, pipeline)

    assert metrics.exact_result_accuracy == 1
    assert metrics.result_precision == 0.5
    assert metrics.exact_response_accuracy == 0


def test_extraction_scores_ambiguity_and_abstention_separately() -> None:
    ambiguous = GoldQuery(
        query="which brake bolt",
        relevant_chunk_ids=("front", "rear"),
        expected_status="ambiguous",
        expected_results=(
            GoldSpecification(
                component_contains="front",
                spec_type="torque",
                value="37",
                unit="Nm",
                pdf_page=636,
            ),
            GoldSpecification(
                component_contains="rear",
                spec_type="torque",
                value="33",
                unit="Nm",
                pdf_page=652,
            ),
        ),
    )
    abstain = GoldQuery(query="spark gap", expected_status="not_found")
    pipeline = StaticPipeline(
        {
            "which brake bolt": response(
                "ambiguous",
                (make_result("front", "37"), make_result("rear", "33")),
                clarification="Which axle and fastener?",
            ),
            "spark gap": response("not_found"),
        }
    )

    metrics, _ = evaluate_extraction([ambiguous, abstain], pipeline)

    assert metrics.status_accuracy == 1
    assert metrics.ambiguity_accuracy == 1
    assert metrics.abstention_accuracy == 1


def test_full_evaluation_retains_per_case_extraction_diagnostics() -> None:
    gold = [make_gold("exact", "relevant")]
    retriever = MappingRetriever({"exact": [make_hit("relevant", 1)]})
    pipeline = StaticPipeline(
        {"exact": response("found", (make_result("guide pin bolts", "37"),))}
    )

    report = evaluate(gold, retriever, pipeline=pipeline)

    assert len(report.extraction_cases) == 1
    case = report.extraction_cases[0]
    assert case.exact_response_match is True
    assert case.response is not None
    assert case.response.results[0].value == "37"


def test_gold_contract_loading_reports_line_and_duplicate(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="answerable gold cases"):
        GoldQuery(query="missing evidence", expected_status="found")

    path = tmp_path / "gold.jsonl"
    valid = make_gold("same", "relevant").model_dump_json()
    path.write_text(f"{valid}\n{valid}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"duplicate gold query.*:2"):
        load_gold_queries(path)


def test_reports_are_atomic_and_thresholds_are_enforced(tmp_path: Path) -> None:
    metrics, _ = evaluate_retrieval(
        [make_gold("query", "relevant")],
        MappingRetriever({"query": [make_hit("relevant", 1)]}),
    )
    report = EvaluationReport(retrieval=metrics)
    json_path = tmp_path / "reports" / "evaluation.json"
    markdown_path = tmp_path / "reports" / "evaluation.md"

    write_evaluation_report(
        report,
        json_path=json_path,
        markdown_path=markdown_path,
    )

    assert '"recall_at_5": 1.0' in json_path.read_text(encoding="utf-8")
    assert "| Recall@10 | 1.000 |" in render_report_markdown(report)
    enforce_retrieval_thresholds(
        report,
        minimum_recall_at_5=1,
        minimum_recall_at_10=1,
    )
    with pytest.raises(EvaluationRegressionError, match="Recall@5"):
        enforce_retrieval_thresholds(
            report.model_copy(
                update={"retrieval": metrics.model_copy(update={"recall_at_5": 0.5})}
            ),
            minimum_recall_at_5=0.8,
            minimum_recall_at_10=0.8,
        )

    failed_report = report.model_copy(
        update={
            "failures": (
                EvaluationFailure(
                    query="query",
                    stage="retrieval",
                    error="model unavailable",
                ),
            )
        }
    )
    with pytest.raises(EvaluationExecutionError, match=r"1 case.*retrieval"):
        enforce_no_evaluation_failures(failed_report)


def test_checked_in_retrieval_results_meet_regression_floor() -> None:
    results_path = Path(__file__).resolve().parents[1] / "eval/retrieval_results.json"
    payload = json.loads(results_path.read_text(encoding="utf-8"))

    assert payload["configurations"]["hybrid_rrf"]["recall_at_5"] >= 0.95
    assert payload["configurations"]["hybrid_rrf_reranked"]["recall_at_10"] >= 0.99
