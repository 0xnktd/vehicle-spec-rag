from pathlib import Path

import pymupdf
import pytest
from pydantic import ValidationError

from vehicle_specs.chunking import (
    ChunkingConfig,
    TextChunk,
    count_tokens,
    iter_chunks,
    iter_pdf_chunks,
)
from vehicle_specs.pdf.cleaner import clean_page_record
from vehicle_specs.pdf.extractor import extract_clean_page, extract_page
from vehicle_specs.pdf.models import ContextualPageRecord, PageRecord, SectionContext, TextBlock
from vehicle_specs.pdf.sections import iter_contextual_pages


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_MANUAL = PROJECT_ROOT / "docs" / "sample-service-manual 1.pdf"


def make_context(
    start_pdf_page: int = 1,
    article_title: str = "Brake Caliper",
) -> SectionContext:
    return SectionContext(
        section_id="206-03",
        section_title="Front Disc Brake",
        category="REMOVAL AND INSTALLATION",
        article_title=article_title,
        start_pdf_page=start_pdf_page,
    )


def make_page(
    pdf_page: int,
    *texts: str,
    context: SectionContext | None = None,
) -> ContextualPageRecord:
    blocks = tuple(
        TextBlock(bbox=(50, 100 + index * 20, 500, 115 + index * 20), text=text)
        for index, text in enumerate(texts)
    )
    page = PageRecord(pdf_page=pdf_page, width=595, height=842, blocks=blocks)
    return ContextualPageRecord(page=page, section=context)


def test_chunking_config_rejects_invalid_overlap() -> None:
    with pytest.raises(ValidationError, match="overlap_tokens must be smaller"):
        ChunkingConfig(max_tokens=32, overlap_tokens=32)


def test_chunk_model_rejects_unsorted_or_duplicate_pages() -> None:
    with pytest.raises(ValidationError, match="sorted and unique"):
        TextChunk(
            chunk_id="invalid-pages",
            source="manual.pdf",
            kind="prose",
            text="Some content",
            pdf_pages=(2, 1, 2),
        )


def test_chunks_never_cross_article_boundaries_and_keep_context() -> None:
    first_context = make_context(start_pdf_page=1, article_title="Brake Caliper")
    second_context = make_context(start_pdf_page=3, article_title="Brake Pads")
    pages = (
        make_page(1, "First article content.", context=first_context),
        make_page(2, "First article continuation.", context=first_context),
        make_page(3, "Second article content.", context=second_context),
    )

    chunks = tuple(iter_chunks(pages, source="manual.pdf"))

    assert len(chunks) == 2
    assert chunks[0].section == first_context
    assert chunks[0].pdf_pages == (1, 2)
    assert chunks[1].section == second_context
    assert chunks[1].pdf_pages == (3,)
    assert "Article: Brake Caliper" in chunks[0].text
    assert "Article: Brake Pads" in chunks[1].text


def test_wrapped_section_header_is_metadata_not_duplicate_chunk_content() -> None:
    context = SectionContext(
        section_id="206-09",
        section_title="Anti-Lock Brake System (ABS) and Stability Control",
        category="DIAGNOSIS AND TESTING",
        article_title=None,
        start_pdf_page=1,
    )
    page = make_page(
        1,
        "SECTION 206-09: Anti-Lock Brake System (ABS)\n"
        "and Stability Control\n"
        "DIAGNOSIS AND TESTING",
        "Procedure content.",
        context=context,
    )

    (chunk,) = tuple(iter_chunks((page,), source="manual.pdf"))

    assert "Section 206-09: Anti-Lock Brake System" in chunk.text
    assert "SECTION 206-09" not in chunk.text


def test_prose_split_respects_budget_and_has_bounded_overlap() -> None:
    words = tuple(f"word{index}" for index in range(1, 81))
    page = make_page(1, " ".join(words))
    config = ChunkingConfig(max_tokens=32, overlap_tokens=3)

    chunks = tuple(iter_chunks((page,), source="manual.pdf", config=config))

    assert len(chunks) == 3
    assert all(chunk.kind == "prose" for chunk in chunks)
    assert all(count_tokens(chunk.text) <= config.max_tokens for chunk in chunks)
    assert chunks[0].text.split()[-3:] == chunks[1].text.split()[:3]
    assert chunks[1].text.split()[-3:] == chunks[2].text.split()[:3]


def test_chunk_ids_are_readable_and_deterministic_without_a_content_hash() -> None:
    context = make_context()
    pages = (make_page(1, "Brake caliper procedure text.", context=context),)

    first = tuple(iter_chunks(pages, source="Service Manual.pdf"))
    second = tuple(iter_chunks(pages, source="Service Manual.pdf"))

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert first[0].chunk_id == (
        "service-manual-206-03-a0001-p0001-prose-001"
    )


def test_empty_pages_do_not_create_empty_chunks() -> None:
    assert tuple(iter_chunks((make_page(1),), source="manual.pdf")) == ()


def test_front_brake_specification_tables_remain_atomic() -> None:
    with pymupdf.open(SAMPLE_MANUAL) as document:
        pages = (
            extract_clean_page(document.load_page(635)),
            clean_page_record(extract_page(document.load_page(636))),
        )
        contextual_pages = tuple(iter_contextual_pages(pages))

    chunks = tuple(
        iter_chunks(
            contextual_pages,
            source=SAMPLE_MANUAL.name,
            config=ChunkingConfig(max_tokens=64, overlap_tokens=8),
        )
    )

    assert len(chunks) == 3
    assert all(chunk.kind == "table" for chunk in chunks)
    assert all(chunk.pdf_pages == (636,) for chunk in chunks)

    torque_chunk = next(chunk for chunk in chunks if "Torque Specifications" in chunk.text)
    assert "| Brake caliper anchor plate bolts | 250 | 184 | — |" in torque_chunk.text
    assert "| Brake caliper guide pin bolts | 37 | 27 | — |" in torque_chunk.text
    assert torque_chunk.text.count("Brake caliper guide pin bolts") == 1
    assert count_tokens(torque_chunk.text) > 64


def test_complete_manual_chunking_invariants() -> None:
    config = ChunkingConfig()
    chunks = tuple(iter_pdf_chunks(SAMPLE_MANUAL, config=config))

    assert len(chunks) == 637
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert sum(chunk.kind == "table" for chunk in chunks) == 100
    assert all(chunk.section is not None for chunk in chunks)
    assert all(chunk.source == SAMPLE_MANUAL.name for chunk in chunks)
    assert all(
        count_tokens(chunk.text) <= config.max_tokens
        for chunk in chunks
        if chunk.kind == "prose"
    )

    represented_articles = {
        chunk.section.start_pdf_page
        for chunk in chunks
        if chunk.section is not None
    }
    assert len(represented_articles) == 177
