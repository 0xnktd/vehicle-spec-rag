import re
from pathlib import Path

import pymupdf

from vehicle_specs.pdf.cleaner import clean_page_record
from vehicle_specs.pdf.extractor import extract_clean_page, extract_page
from vehicle_specs.pdf.tables import extract_specification_table_blocks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_MANUAL = PROJECT_ROOT / "docs" / "sample-service-manual 1.pdf"


def test_front_brake_page_is_clean_and_has_reconstructed_torque_rows() -> None:
    with pymupdf.open(SAMPLE_MANUAL) as document:
        page = document.load_page(635)
        table_blocks = extract_specification_table_blocks(page)
        record = extract_clean_page(page)

    assert len(table_blocks) == 3
    assert "Page 1 sur 2" not in record.raw_text
    assert "2014 F-150 Workshop Manual" not in record.raw_text
    assert "Procedure revision date" not in record.raw_text
    assert "file:///" not in record.raw_text
    assert "| Description | Nm | lb-ft | lb-in |" in record.raw_text
    assert "| Brake caliper guide pin bolts | 37 | 27 | — |" in record.raw_text
    assert record.raw_text.count("Brake caliper guide pin bolts") == 1


def test_rear_brake_page_preserves_distinct_rear_torque_values() -> None:
    with pymupdf.open(SAMPLE_MANUAL) as document:
        record = extract_clean_page(document.load_page(651))

    assert "SECTION 206-04: Rear Disc Brake\nSPECIFICATIONS" in record.raw_text
    assert "| Brake caliper guide pin bolts | 33 | 24 | — |" in record.raw_text
    assert "| Brake caliper support bracket bolts | 150 | 111 | — |" in record.raw_text


def test_uncertain_diagnostic_table_uses_raw_text_fallback() -> None:
    with pymupdf.open(SAMPLE_MANUAL) as document:
        page = document.load_page(162)
        table_blocks = extract_specification_table_blocks(page)
        record = extract_clean_page(page)

    assert table_blocks == ()
    assert "TPM_PRES_LF" in record.raw_text
    assert "\x84" not in record.raw_text
    assert not any(line.startswith("z ") for line in record.raw_text.splitlines())


def test_prose_that_mentions_specification_is_not_mistaken_for_a_table_header() -> None:
    with pymupdf.open(SAMPLE_MANUAL) as document:
        table_blocks = extract_specification_table_blocks(document.load_page(4))

    assert table_blocks == ()


def test_hyphenated_unit_is_joined_without_dropping_the_hyphen() -> None:
    with pymupdf.open(SAMPLE_MANUAL) as document:
        record = extract_clean_page(document.load_page(476))

    assert "Nm (lb-in) torque wrench" in record.raw_text
    assert "lb-\nin" not in record.raw_text


def test_duplicate_heading_is_removed_conservatively() -> None:
    with pymupdf.open(SAMPLE_MANUAL) as document:
        record = extract_clean_page(document.load_page(3))

    assert record.raw_text.count("Symptom Chart — NVH") == 1


def test_boilerplate_only_page_becomes_an_empty_page_record() -> None:
    with pymupdf.open(SAMPLE_MANUAL) as document:
        record = extract_clean_page(document.load_page(636))

    assert record.pdf_page == 637
    assert record.blocks == ()
    assert record.raw_text == ""


def test_cleaner_invariants_hold_across_the_complete_manual() -> None:
    empty_pages: list[int] = []

    with pymupdf.open(SAMPLE_MANUAL) as document:
        for page in document:
            record = clean_page_record(extract_page(page))
            if not record.blocks:
                empty_pages.append(record.pdf_page)

            for block in record.blocks:
                assert "2014 F-150 Workshop Manual" not in block.text
                assert "file:///" not in block.text
                assert "\x84" not in block.text
                assert not re.search(r"^Page\s+\d+\s+sur\s+\d+$", block.text, re.MULTILINE)
                assert not re.search(r"^z\s+", block.text, re.MULTILINE)
                assert not re.search(r"\w-\n\w", block.text)
                assert not re.search(r"[ \t]+$", block.text, re.MULTILINE)

    assert len(empty_pages) == 69
    assert 637 in empty_pages
