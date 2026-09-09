from pathlib import Path

import pymupdf
import pytest

from vehicle_specs.pdf.extractor import (
    extract_page,
    iter_page_records,
    read_document_metadata,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_MANUAL = PROJECT_ROOT / "docs" / "sample-service-manual 1.pdf"


def test_reads_document_metadata() -> None:
    metadata = read_document_metadata(SAMPLE_MANUAL)

    assert metadata.filename == SAMPLE_MANUAL.name
    assert metadata.page_count == 852


def test_extracts_expected_front_brake_specification_page() -> None:
    with pymupdf.open(SAMPLE_MANUAL) as document:
        record = extract_page(document.load_page(635))

    assert record.pdf_page == 636
    assert record.width == pytest.approx(595.0)
    assert record.height == pytest.approx(842.0)
    assert record.blocks
    assert "SECTION 206-03" in record.raw_text
    assert "Brake caliper guide pin bolts" in record.raw_text
    assert "37" in record.raw_text


def test_page_iterator_starts_with_pdf_page_one() -> None:
    records = iter_page_records(SAMPLE_MANUAL)

    try:
        first_page = next(records)
    finally:
        records.close()

    assert first_page.pdf_page == 1


def test_missing_pdf_raises_file_not_found(tmp_path: Path) -> None:
    missing_pdf = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError, match="PDF not found"):
        read_document_metadata(missing_pdf)
