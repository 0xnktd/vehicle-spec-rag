from pathlib import Path

import pymupdf
import pytest

from vehicle_specs.pdf.cleaner import clean_page_record
from vehicle_specs.pdf.extractor import (
    detect_section_context,
    extract_page,
    iter_contextual_pages,
)
from vehicle_specs.pdf.models import PageRecord, SectionContext, TextBlock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_MANUAL = PROJECT_ROOT / "docs" / "sample-service-manual 1.pdf"


def make_page(pdf_page: int, *texts: str) -> PageRecord:
    blocks = tuple(
        TextBlock(
            bbox=(78, 120 + index * 30, 500, 140 + index * 30),
            text=text,
        )
        for index, text in enumerate(texts)
    )
    return PageRecord(pdf_page=pdf_page, width=595, height=842, blocks=blocks)


def test_detects_section_category_and_article_title() -> None:
    page = make_page(
        640,
        "SECTION 206-03: Front Disc Brake\nREMOVAL AND INSTALLATION",
        "Brake Caliper Anchor Plate",
    )

    assert detect_section_context(page) == SectionContext(
        section_id="206-03",
        section_title="Front Disc Brake",
        category="REMOVAL AND INSTALLATION",
        article_title="Brake Caliper Anchor Plate",
        start_pdf_page=640,
    )


def test_specification_context_does_not_treat_table_heading_as_article_title() -> None:
    page = make_page(
        636,
        "SECTION 206-03: Front Disc Brake\nSPECIFICATIONS",
        "Material",
    )

    context = detect_section_context(page)

    assert context is not None
    assert context.article_title is None


@pytest.mark.parametrize(
    "reference",
    (
        "Section 205-03.",
        "Section 204-01A for Rear Wheel Drive vehicles",
        "SECTION 205-03: Front Drive Axle/Differential",
    ),
)
def test_does_not_treat_section_references_as_article_headers(reference: str) -> None:
    assert detect_section_context(make_page(1, reference)) is None


def test_propagates_context_and_updates_it_only_at_a_new_article() -> None:
    pages = (
        make_page(1, "Preface"),
        make_page(
            2,
            "SECTION 206-03: Front Disc Brake\nREMOVAL AND INSTALLATION",
            "Brake Caliper",
        ),
        make_page(3, "Continued procedure text"),
        make_page(
            4,
            "SECTION 206-04: Rear Disc Brake\nSPECIFICATIONS",
            "Torque Specifications",
        ),
    )

    contextual_pages = tuple(iter_contextual_pages(pages))

    assert contextual_pages[0].section is None
    assert contextual_pages[1].section == contextual_pages[2].section
    assert contextual_pages[2].section is not None
    assert contextual_pages[2].section.start_pdf_page == 2
    assert contextual_pages[3].section is not None
    assert contextual_pages[3].section.section_id == "206-04"
    assert contextual_pages[3].section.start_pdf_page == 4


def test_rejects_out_of_order_pages() -> None:
    pages = (make_page(2, "Page two"), make_page(1, "Page one"))

    with pytest.raises(ValueError, match="increasing PDF page number"):
        tuple(iter_contextual_pages(pages))


def test_detects_every_article_boundary_in_the_manual() -> None:
    with pymupdf.open(SAMPLE_MANUAL) as document:
        cleaned_pages = (clean_page_record(extract_page(page)) for page in document)
        contextual_pages = tuple(iter_contextual_pages(cleaned_pages))

    assert len(contextual_pages) == 852
    assert all(record.section is not None for record in contextual_pages)

    article_start_pages = {
        record.section.start_pdf_page
        for record in contextual_pages
        if record.section is not None
    }
    section_ids = {
        record.section.section_id
        for record in contextual_pages
        if record.section is not None
    }

    assert len(article_start_pages) == 177
    assert len(section_ids) == 19

    front_specification = contextual_pages[635].section
    blank_continuation = contextual_pages[636].section
    front_description = contextual_pages[637].section

    assert front_specification is not None
    assert front_specification.category == "SPECIFICATIONS"
    assert blank_continuation == front_specification
    assert front_description is not None
    assert front_description.category == "DESCRIPTION AND OPERATION"
    assert front_description.article_title == "Front Disc Brake"
