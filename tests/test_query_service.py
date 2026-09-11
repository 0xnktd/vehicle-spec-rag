from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, cast

import pytest
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant.sparse_embeddings import SparseEmbeddings

import vehicle_specs.query_service as service_module
from vehicle_specs.extraction import (
    CandidateExtraction,
    SourceEvidence,
    SpecificationResult,
)
from vehicle_specs.indexing import IndexConfig, PipelineVersions
from vehicle_specs.query_service import QueryService
from vehicle_specs.retrieval import RetrievalConfig, RetrievalFilters, RetrievalHit


def make_hit() -> RetrievalHit:
    return RetrievalHit(
        point_id=1,
        chunk_id="battery-monitoring-sensor-nut",
        text="Battery monitoring sensor nut 8 Nm",
        source="manual.pdf",
        kind="prose",
        pdf_pages=(42,),
        pipeline_versions=PipelineVersions(),
        section_id="414-01",
        section_title="Battery",
        rank=1,
        score=0.9,
        rerank_score=0.98,
    )


class FakeExtractor:
    model_name = "fake-model"
    prompt_version = "fake-prompt"

    def extract(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
    ) -> CandidateExtraction:
        assert query == "battery sensor nut torque"
        assert hits == [make_hit()]
        return CandidateExtraction(
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
                        evidence="Battery monitoring sensor nut 8 Nm",
                    ),
                ),
            ),
        )


class FakeRetriever:
    last_filters: RetrievalFilters | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def retrieve(
        self,
        query: str,
        *,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievalHit]:
        type(self).last_filters = filters
        return [make_hit()]


def make_service() -> QueryService:
    return QueryService(
        index_config=IndexConfig(
            path="unused",
            dense_model="fake-dense",
            sparse_model="fake-sparse",
        ),
        retrieval_config=RetrievalConfig(candidate_k=5, context_k=1),
        extractor=FakeExtractor(),
        dense_embeddings=cast(Embeddings, object()),
        sparse_embeddings=cast(SparseEmbeddings, object()),
    )


def test_query_service_runs_validated_pipeline_and_releases_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_state = {"opened": 0, "closed": 0}

    @contextmanager
    def fake_open_index(*args: Any, **kwargs: Any) -> Iterator[QdrantVectorStore]:
        index_state["opened"] += 1
        try:
            yield cast(QdrantVectorStore, object())
        finally:
            index_state["closed"] += 1

    monkeypatch.setattr(service_module, "open_index", fake_open_index)
    monkeypatch.setattr(service_module, "VehicleSpecRetriever", FakeRetriever)
    filters = RetrievalFilters(section_ids=("414-01",))

    run = make_service().run("  battery sensor nut torque  ", filters=filters)

    assert run.response.status == "found"
    assert run.response.results[0].value == "8"
    assert run.query == "battery sensor nut torque"
    assert FakeRetriever.last_filters == filters
    assert index_state == {"opened": 1, "closed": 1}


def test_query_service_rejects_blank_query_before_opening_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_open(*args: Any, **kwargs: Any) -> None:
        pytest.fail("the index should not open for an empty query")

    monkeypatch.setattr(service_module, "open_index", unexpected_open)

    with pytest.raises(ValueError, match="query cannot be empty"):
        make_service().run("   ")
