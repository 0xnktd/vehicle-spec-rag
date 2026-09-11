"""Page-level extraction coverage diagnostics."""

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from .models import PageRecord


class PageQuality(BaseModel):
    """Inspectable text coverage measurements for one extracted page."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True)

    pdf_page: int = Field(ge=1)
    block_count: int = Field(ge=0)
    text_character_count: int = Field(ge=0)
    block_area_ratio: float = Field(ge=0, le=1)
    suspiciously_sparse: bool


def assess_page_quality(
    page: PageRecord,
    *,
    minimum_characters: int = 40,
    minimum_area_ratio: float = 0.001,
) -> PageQuality:
    """Measure extracted text volume and flag pages needing inspection."""
    text_characters = sum(len(block.text.strip()) for block in page.blocks)
    block_area = sum(
        max(0.0, block.bbox[2] - block.bbox[0])
        * max(0.0, block.bbox[3] - block.bbox[1])
        for block in page.blocks
    )
    block_area_ratio = min(1.0, block_area / (page.width * page.height))
    suspicious = (
        text_characters < minimum_characters or block_area_ratio < minimum_area_ratio
    )
    return PageQuality(
        pdf_page=page.pdf_page,
        block_count=len(page.blocks),
        text_character_count=text_characters,
        block_area_ratio=block_area_ratio,
        suspiciously_sparse=suspicious,
    )


def assess_pages(pages: Iterable[PageRecord]) -> tuple[PageQuality, ...]:
    """Assess pages while preserving their input order."""
    return tuple(assess_page_quality(page) for page in pages)
