from pathlib import Path

import pymupdf
import pytest

from vehicle_specs.pdf.extractor import (
    extract_page,
    iter_clean_page_records,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_MANUAL = PROJECT_ROOT / "docs" / "sample-service-manual 1.pdf"


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


def test_missing_pdf_raises_file_not_found(tmp_path: Path) -> None:
    missing_pdf = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError, match="PDF not found"):
        next(iter_clean_page_records(missing_pdf))
