"""Validated gold cases and aggregate evaluation results."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vehicle_specs.extraction import ExtractionResponse, ExtractionStatus, SpecType


class GoldSpecification(BaseModel):
    """Minimum facts needed to score one expected specification."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    component_contains: str = Field(min_length=1)
    spec_type: SpecType
    value: str = Field(min_length=1)
    unit: str | None = Field(default=None, min_length=1)
    pdf_page: int = Field(ge=1)


class GoldQuery(BaseModel):
    """One manually verified retrieval and extraction case."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    query: str = Field(min_length=1)
    relevant_chunk_ids: tuple[str, ...] = ()
    expected_status: ExtractionStatus
    expected_results: tuple[GoldSpecification, ...] = ()

    @model_validator(mode="after")
    def validate_expected_outcome(self) -> Self:
        if self.expected_status == "not_found":
            if self.relevant_chunk_ids or self.expected_results:
                raise ValueError(
                    "not_found gold cases cannot contain expected evidence"
                )
        else:
            if not self.relevant_chunk_ids:
                raise ValueError("answerable gold cases require relevant chunks")
            if not self.expected_results:
                raise ValueError("answerable gold cases require expected results")
        if self.expected_status == "ambiguous" and len(self.expected_results) < 2:
            raise ValueError("ambiguous gold cases require at least two results")
        if len(set(self.relevant_chunk_ids)) != len(self.relevant_chunk_ids):
            raise ValueError("relevant chunk IDs must be unique")
        return self


class RetrievalMetrics(BaseModel):
    """Retrieval measurements averaged over answerable gold queries."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True)

    evaluated_queries: int = Field(ge=0)
    recall_at_5: float = Field(ge=0, le=1)
    recall_at_10: float = Field(ge=0, le=1)
    mean_reciprocal_rank: float = Field(ge=0, le=1)


class ExtractionMetrics(BaseModel):
    """Structured-answer measurements over all applicable gold fields."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True)

    evaluated_queries: int = Field(ge=0)
    expected_result_count: int = Field(ge=0)
    returned_result_count: int = Field(ge=0)
    status_accuracy: float = Field(ge=0, le=1)
    ambiguity_accuracy: float = Field(ge=0, le=1)
    abstention_accuracy: float = Field(ge=0, le=1)
    component_accuracy: float = Field(ge=0, le=1)
    value_accuracy: float = Field(ge=0, le=1)
    unit_accuracy: float = Field(ge=0, le=1)
    citation_accuracy: float = Field(ge=0, le=1)
    result_precision: float = Field(ge=0, le=1)
    exact_result_accuracy: float = Field(ge=0, le=1)
    exact_response_accuracy: float = Field(ge=0, le=1)


class EvaluationFailure(BaseModel):
    """A query that could not be evaluated without stopping the suite."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    error: str = Field(min_length=1)


class ExtractionCaseEvaluation(BaseModel):
    """Per-query outcome retained for reproducible error analysis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1)
    expected_status: ExtractionStatus
    actual_status: ExtractionStatus | None = None
    expected_result_count: int = Field(ge=0)
    returned_result_count: int = Field(ge=0)
    exact_result_count: int = Field(ge=0)
    status_match: bool
    exact_response_match: bool
    response: ExtractionResponse | None = None
    error: str | None = Field(default=None, min_length=1)


class EvaluationReport(BaseModel):
    """Machine-readable retrieval and optional extraction evaluation report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    retrieval: RetrievalMetrics
    extraction: ExtractionMetrics | None = None
    extraction_cases: tuple[ExtractionCaseEvaluation, ...] = ()
    failures: tuple[EvaluationFailure, ...] = ()
