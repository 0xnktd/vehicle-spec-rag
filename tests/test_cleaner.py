import pytest

from vehicle_specs.pdf.cleaner import clean_block_text, clean_page_record
from vehicle_specs.pdf.models import PageRecord, TextBlock


def make_block(text: str, y0: float = 100, y1: float = 120) -> TextBlock:
    return TextBlock(bbox=(50, y0, 500, y1), text=text)


def make_page(*blocks: TextBlock) -> PageRecord:
    return PageRecord(pdf_page=1, width=595, height=842, blocks=blocks)


@pytest.mark.parametrize(
    "boilerplate",
    (
        "Page 1 sur 2\n2014 F-150 Workshop Manual\n",
        ("2014-03-01\nfile:///C:/TSO/cache/manual.HTM\nrepair4less\n"),
    ),
)
def test_removes_only_recognized_header_and_footer_blocks(boilerplate: str) -> None:
    assert clean_block_text(boilerplate) == ""


def test_preserves_dates_and_top_of_page_content_that_are_not_boilerplate() -> None:
    text = "Service date: 2024-01-01\nInspect the front brake assembly.\n"

    assert clean_block_text(text) == (
        "Service date: 2024-01-01 Inspect the front brake assembly."
    )


def test_cleans_mixed_section_header_without_losing_section_context() -> None:
    text = (
        "SECTION 206-03: Front Disc Brake \n"
        "2014 F-150 Workshop Manual \n"
        "SPECIFICATIONS \n"
        "Procedure revision date: 10/25/2013 \n"
    )

    assert clean_block_text(text) == (
        "SECTION 206-03: Front Disc Brake\nSPECIFICATIONS"
    )


def test_normalizes_pdf_bullets_but_preserves_meaningful_unicode() -> None:
    text = "z Parent item \n\x84 Nested item ± 5° ™ ® — \n"

    assert clean_block_text(text) == "- Parent item\n- Nested item ± 5° ™ ® —"


def test_reflows_visual_wraps_and_preserves_hyphens_and_numbered_steps() -> None:
    text = (
        "22.\n"
        "Take readings using an Nm (lb-\n"
        "in) torque wrench. \n"
        "23. Install the heavy-\n"
        "duty component. \n"
    )

    assert clean_block_text(text) == (
        "22. Take readings using an Nm (lb-in) torque wrench.\n"
        "23. Install the heavy-duty component."
    )


def test_normalizes_whitespace_before_punctuation() -> None:
    assert clean_block_text("Refer to Section 100-04 . BCM , PID : value ?\n") == (
        "Refer to Section 100-04. BCM, PID: value?"
    )


def test_replaces_blocks_inside_a_table_and_retains_reading_order() -> None:
    heading = make_block("Torque Specifications\n", 150, 165)
    raw_table_cell = make_block("Bolt\n35\n", 180, 200)
    table = make_block("| Item | Nm |\n| --- | --- |\n| Bolt | 35 |", 170, 220)
    page = make_page(heading, raw_table_cell)

    cleaned = clean_page_record(page, (table,))

    assert cleaned.blocks == (
        make_block("Torque Specifications", 150, 165),
        table,
    )


def test_deduplicates_only_adjacent_heading_blocks() -> None:
    first = make_block("Symptom Chart — NVH\n", 100, 110)
    duplicate = make_block("Symptom Chart — NVH\n", 130, 140)
    later_body_value = make_block("Symptom Chart — NVH.\n", 300, 310)

    cleaned = clean_page_record(make_page(first, duplicate, later_body_value))

    assert [block.text for block in cleaned.blocks] == [
        "Symptom Chart — NVH",
        "Symptom Chart — NVH.",
    ]


def test_cleaning_is_idempotent() -> None:
    page = make_page(make_block("z Check the BCM . \n"))

    once = clean_page_record(page)
    twice = clean_page_record(once)

    assert twice == once
