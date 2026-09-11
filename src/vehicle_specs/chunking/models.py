"""Validated models and configuration for structure-aware chunks."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vehicle_specs.pdf.models import SectionContext

ChunkKind = Literal["prose", "table"]


class ChunkingConfig(BaseModel):
    """Boundaries for deterministic lexical-token chunking."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_tokens: int = Field(default=400, ge=32)
    overlap_tokens: int = Field(default=40, ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> Self:
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        return self


class TextChunk(BaseModel):
    """A retrieval unit with exact source and article provenance."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    chunk_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    kind: ChunkKind
    text: str = Field(min_length=1)
    pdf_pages: tuple[int, ...] = Field(min_length=1)
    section: SectionContext | None = None

    @field_validator("pdf_pages")
    @classmethod
    def validate_pdf_pages(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(page < 1 for page in value):
            raise ValueError("pdf_pages must contain positive page numbers")
        if value != tuple(sorted(set(value))):
            raise ValueError("pdf_pages must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_section_origin(self) -> Self:
        if self.section and self.section.start_pdf_page > self.pdf_pages[0]:
            raise ValueError("section context cannot start after chunk content")
        return self
