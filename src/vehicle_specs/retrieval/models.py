"""Validated configuration and results for vector retrieval."""

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vehicle_specs.chunking.models import ChunkKind
from vehicle_specs.indexing.models import PipelineVersions

RetrievalMode = Literal["dense", "sparse", "hybrid"]
Applicability = Literal["front", "rear"]


class RetrievalConfig(BaseModel):
    """Candidate-generation and context-selection settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: RetrievalMode = "hybrid"
    candidate_k: int = Field(default=20, ge=1)
    context_k: int = Field(default=8, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    dense_weight: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    sparse_weight: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    score_threshold: float | None = Field(default=None, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_limits_and_weights(self) -> Self:
        if self.context_k > self.candidate_k:
            raise ValueError("context_k cannot exceed candidate_k")
        if self.dense_weight == 0 and self.sparse_weight == 0:
            raise ValueError("at least one retrieval weight must be positive")
        return self


class RetrievalFilters(BaseModel):
    """Reliable metadata constraints that may be applied before ranking."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    source: str | None = Field(default=None, min_length=1)
    section_ids: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    applicability: Applicability | None = None


class RetrievalHit(BaseModel):
    """A retrieved chunk with branch-level ranking diagnostics."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    point_id: int | str
    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: str = Field(min_length=1)
    kind: ChunkKind
    pdf_pages: tuple[int, ...] = Field(min_length=1)
    pipeline_versions: PipelineVersions
    section_id: str | None = Field(default=None, min_length=1)
    section_title: str | None = Field(default=None, min_length=1)
    category: str | None = Field(default=None, min_length=1)
    article_title: str | None = Field(default=None, min_length=1)
    article_start_pdf_page: int | None = Field(default=None, ge=1)
    rank: int = Field(ge=1)
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    dense_rank: int | None = Field(default=None, ge=1)
    sparse_rank: int | None = Field(default=None, ge=1)
    rerank_score: float | None = None

    @field_validator("pdf_pages")
    @classmethod
    def validate_pdf_pages(cls, pages: tuple[int, ...]) -> tuple[int, ...]:
        if any(page < 1 for page in pages):
            raise ValueError("pdf_pages must contain positive page numbers")
        if pages != tuple(sorted(set(pages))):
            raise ValueError("pdf_pages must be sorted and unique")
        return pages

    @property
    def effective_score(self) -> float:
        """Return the cross-encoder score when present, otherwise retrieval score."""
        return self.rerank_score if self.rerank_score is not None else self.score

    @property
    def metadata(self) -> dict[str, Any]:
        """Return citation metadata in the same shape stored in Qdrant."""
        metadata: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "kind": self.kind,
            "pdf_pages": list(self.pdf_pages),
            "pipeline_versions": self.pipeline_versions.model_dump(mode="json"),
        }
        optional = {
            "section_id": self.section_id,
            "section_title": self.section_title,
            "category": self.category,
            "article_title": self.article_title,
            "article_start_pdf_page": self.article_start_pdf_page,
        }
        metadata.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        return metadata
