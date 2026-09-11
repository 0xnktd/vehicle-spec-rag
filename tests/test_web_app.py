from pathlib import Path

from streamlit.testing.v1 import AppTest

import vehicle_specs.web.app as web_app
from vehicle_specs.extraction import (
    ExtractionResponse,
    SourceEvidence,
    SpecificationResult,
)
from vehicle_specs.indexing import IndexNotFoundError, PipelineVersions
from vehicle_specs.pipeline import PipelineRun
from vehicle_specs.retrieval import RetrievalHit

APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def make_run() -> PipelineRun:
    evidence = "Battery monitoring sensor nut 8 Nm"
    return PipelineRun(
        query="battery sensor nut torque",
        response=ExtractionResponse(
            status="found",
            results=(
                SpecificationResult(
                    component="Battery monitoring sensor nut",
                    spec_type="torque",
                    value="8",
                    unit="Nm",
                    source=SourceEvidence(
                        chunk_id="battery-monitoring-sensor-nut",
                        pdf_page=42,
                        section_id="414-01",
                        evidence=evidence,
                    ),
                ),
            ),
            model_name="fake-model",
            prompt_version="fake-prompt",
            pipeline_version="fake-pipeline",
        ),
        retrieved=(
            RetrievalHit(
                point_id=1,
                chunk_id="battery-monitoring-sensor-nut",
                text=evidence,
                source="manual.pdf",
                kind="prose",
                pdf_pages=(42,),
                pipeline_versions=PipelineVersions(),
                section_id="414-01",
                section_title="Battery",
                rank=1,
                score=0.9,
                rerank_score=0.98,
            ),
        ),
    )


class FakeQueryService:
    def run(self, query: str) -> PipelineRun:
        assert query == "battery sensor nut torque"
        return make_run()


def test_initial_page_is_a_small_query_form() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=10).run()

    assert not app.exception
    assert app.title[0].value == "Vehicle specification search"
    assert app.text_input(key="query").label == "Question"
    assert app.button(key="submit-query").label == "Search manual"


def test_blank_submission_is_rejected_without_loading_models() -> None:
    app = AppTest.from_file(APP_PATH, default_timeout=10).run()

    app.text_input(key="query").input("   ")
    app.button(key="submit-query").click().run()

    assert not app.exception
    assert app.warning[0].value == "Enter a question before searching."


def test_successful_query_renders_answer_and_citation(monkeypatch) -> None:
    monkeypatch.setattr(web_app, "_query_service", lambda: FakeQueryService())
    app = AppTest.from_file(APP_PATH, default_timeout=10).run()

    app.text_input(key="query").input("battery sensor nut torque")
    app.button(key="submit-query").click().run()

    assert not app.exception
    assert app.success[0].value == "Found 1 supported specification."
    assert app.metric[0].value == "8 Nm"
    assert any("PDF page 42" in caption.value for caption in app.caption)
    assert any(code.value == "Battery monitoring sensor nut 8 Nm" for code in app.code)


def test_index_errors_have_actionable_messages() -> None:
    message = web_app._error_message(IndexNotFoundError("missing"))

    assert (
        message == "The local index is missing. Run the ingest command before querying."
    )
