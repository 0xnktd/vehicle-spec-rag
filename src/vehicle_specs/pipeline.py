"""Synchronous retrieval, extraction, and validation orchestration."""

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from vehicle_specs.extraction import (
    ExtractionResponse,
    StructuredExtractor,
    assemble_table_evidence,
    complete_capacity_table_results,
    complete_dimension_table_results,
    complete_torque_table_results,
    normalize_candidate,
    normalize_table_fields,
    validate_candidate,
)
from vehicle_specs.retrieval import RetrievalFilters, RetrievalHit

PIPELINE_VERSION = "1.5.0"


class Retriever(Protocol):
    """Minimal retrieval boundary required by the pipeline."""

    def retrieve(
        self,
        query: str,
        *,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievalHit]: ...


class PipelineRun(BaseModel):
    """Public answer plus optional retrieval diagnostics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1)
    response: ExtractionResponse
    retrieved: tuple[RetrievalHit, ...]


class VehicleSpecificationPipeline:
    """Run retrieval and fail-closed structured specification extraction."""

    def __init__(
        self,
        retriever: Retriever,
        extractor: StructuredExtractor,
    ) -> None:
        self.retriever = retriever
        self.extractor = extractor

    def run(
        self,
        query: str,
        *,
        filters: RetrievalFilters | None = None,
    ) -> PipelineRun:
        """Retrieve context, extract claims, then validate every citation."""
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")

        hits: Sequence[RetrievalHit] = self.retriever.retrieve(query, filters=filters)
        candidate = self.extractor.extract(query, hits)
        candidate = complete_torque_table_results(candidate, hits, query)
        candidate = complete_capacity_table_results(candidate, hits, query)
        candidate = complete_dimension_table_results(candidate, hits, query)
        candidate = assemble_table_evidence(candidate, hits, query)
        candidate = normalize_table_fields(candidate, hits, query)
        candidate = normalize_candidate(candidate, hits, query)
        validated = validate_candidate(candidate, hits)
        response = ExtractionResponse(
            **validated.model_dump(),
            model_name=self.extractor.model_name,
            prompt_version=self.extractor.prompt_version,
            pipeline_version=PIPELINE_VERSION,
        )
        return PipelineRun(query=query, response=response, retrieved=tuple(hits))

    def query(
        self,
        query: str,
        *,
        filters: RetrievalFilters | None = None,
    ) -> ExtractionResponse:
        """Return only the validated public answer."""
        return self.run(query, filters=filters).response
