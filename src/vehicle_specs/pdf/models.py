"""Validated data models produced by the PDF processing pipeline."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TextBlock(BaseModel):
    """A text block extracted from a single PDF page."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
    )

    bbox: tuple[float, float, float, float]
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text cannot be empty or whitespace only")
        return value

    @model_validator(mode="after")
    def validate_bbox(self) -> Self:
        x0, y0, x1, y1 = self.bbox

        if x1 < x0:
            raise ValueError("bbox x1 must be greater than or equal to x0")
        if y1 < y0:
            raise ValueError("bbox y1 must be greater than or equal to y0")

        return self


class PageRecord(BaseModel):
    """Text extraction result for one PDF page."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
    )

    pdf_page: int = Field(ge=1)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    blocks: tuple[TextBlock, ...] = Field(default_factory=tuple)

    @property
    def raw_text(self) -> str:
        """Return text blocks in their established reading order."""
        return "\n".join(block.text for block in self.blocks)


class SectionContext(BaseModel):
    """Article context inherited by pages until the next article header."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    section_id: str = Field(min_length=1)
    section_title: str = Field(min_length=1)
    category: str = Field(min_length=1)
    article_title: str | None = Field(default=None, min_length=1)
    start_pdf_page: int = Field(ge=1)


class ContextualPageRecord(BaseModel):
    """A page paired with the most recent article context."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    page: PageRecord
    section: SectionContext | None = None
