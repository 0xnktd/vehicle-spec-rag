import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

import vehicle_specs.cli as cli
from vehicle_specs.extraction import (
    CandidateExtraction,
    SourceEvidence,
    SpecificationResult,
)
from vehicle_specs.indexing import PipelineVersions
from vehicle_specs.pipeline import PipelineRun, VehicleSpecificationPipeline
from vehicle_specs.retrieval import RetrievalFilters, RetrievalHit

runner = CliRunner()


def make_hit() -> RetrievalHit:
    return RetrievalHit(
        point_id=1,
        chunk_id="front-guide-pin",
        text="Brake caliper guide pin bolts | 37 | Nm",
        source="manual.pdf",
        kind="table",
        pdf_pages=(636,),
        pipeline_versions=PipelineVersions(),
        section_id="206-03",
        section_title="Front Disc Brake",
        rank=1,
        score=0.9,
    )


@contextmanager
def fake_open_index(*args: Any, **kwargs: Any) -> Iterator[object]:
    yield object()


class FakeRetriever:
    last_filters: RetrievalFilters | None = None
    last_config: Any = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if len(args) > 1:
            type(self).last_config = args[1]

    def retrieve(
        self,
        query: str,
        *,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievalHit]:
        assert query == "guide pin torque"
        type(self).last_filters = filters
        return [make_hit()]


class FakeExtractor:
    model_name = "fake-model"
    prompt_version = "fake-prompt"

    def extract(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
    ) -> CandidateExtraction:
        assert query == "guide pin torque"
        assert len(hits) == 1
        return CandidateExtraction(
            status="found",
            results=(
                SpecificationResult(
                    component="Brake caliper guide pin bolts",
                    spec_type="torque",
                    value="37",
                    unit="Nm",
                    source=SourceEvidence(
                        chunk_id="front-guide-pin",
                        pdf_page=636,
                        section_id="206-03",
                        evidence="Brake caliper guide pin bolts | 37 | Nm",
                    ),
                ),
            ),
        )


class FakeQueryService:
    def __init__(self, retrieval_config: Any) -> None:
        self.retrieval_config = retrieval_config

    def run(
        self,
        query: str,
        *,
        filters: RetrievalFilters | None = None,
    ) -> PipelineRun:
        retriever = FakeRetriever(object(), self.retrieval_config)
        return VehicleSpecificationPipeline(retriever, FakeExtractor()).run(
            query,
            filters=filters,
        )


def test_root_help_exposes_complete_local_workflow() -> None:
    result = runner.invoke(cli.app, ["--help"])

    assert result.exit_code == 0
    for command in ("ingest", "retrieve", "query", "evaluate", "index-status"):
        assert command in result.stdout


def test_retrieve_command_emits_machine_readable_ranked_hits(monkeypatch) -> None:
    monkeypatch.setattr(cli, "open_index", fake_open_index)
    monkeypatch.setattr(cli, "VehicleSpecRetriever", FakeRetriever)

    result = runner.invoke(
        cli.app,
        ["retrieve", "guide pin torque", "--section", "206-03"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload[0]["chunk_id"] == "front-guide-pin"
    assert FakeRetriever.last_filters == RetrievalFilters(section_ids=("206-03",))


def test_query_command_runs_validation_and_can_include_debug_context(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_create_query_service(*args: Any, **kwargs: Any) -> FakeQueryService:
        captured.update(kwargs)
        return FakeQueryService(kwargs["retrieval_config"])

    monkeypatch.setattr(cli, "create_query_service", fake_create_query_service)

    result = runner.invoke(
        cli.app,
        ["query", "guide pin torque", "--section", "206-03", "--debug"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["response"]["results"][0]["value"] == "37"
    assert payload["retrieved"][0]["pdf_pages"] == [636]
    assert FakeRetriever.last_config.context_k == 5
    assert FakeRetriever.last_filters == RetrievalFilters(section_ids=("206-03",))
    assert captured["rerank"] is False


def test_environment_settings_can_redirect_artifact_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    redirected = tmp_path / "local-index"
    monkeypatch.setenv("VEHICLE_SPECS_INDEX_PATH", str(redirected))

    assert cli.AppSettings().index_path == redirected
