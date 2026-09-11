import json
from pathlib import Path

import pymupdf

from vehicle_specs.pdf import (
    PageRecord,
    TextBlock,
    assess_page_quality,
    extract_clean_page,
    write_page_records_jsonl,
)


def test_page_quality_flags_empty_and_accepts_substantial_text() -> None:
    empty = PageRecord(pdf_page=1, width=600, height=800)
    populated = PageRecord(
        pdf_page=2,
        width=600,
        height=800,
        blocks=(
            TextBlock(
                bbox=(50, 100, 550, 300),
                text=(
                    "A sufficiently substantial extracted block of service manual text."
                ),
            ),
        ),
    )

    assert assess_page_quality(empty).suspiciously_sparse is True
    quality = assess_page_quality(populated)
    assert quality.suspiciously_sparse is False
    assert quality.text_character_count == len(populated.blocks[0].text)
    assert quality.block_area_ratio > 0


def test_page_records_jsonl_is_atomic_and_round_trips(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "pages.jsonl"
    pages = [
        PageRecord(
            pdf_page=1,
            width=600,
            height=800,
            blocks=(TextBlock(bbox=(1, 2, 3, 4), text="Page text"),),
        )
    ]

    count = write_page_records_jsonl(pages, output)

    assert count == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert PageRecord.model_validate(payload) == pages[0]
    assert not tuple(output.parent.glob("*.tmp"))


def test_table_detection_does_not_pollute_machine_readable_stdout(
    capsys,
) -> None:
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text(
            (50, 80),
            "Torque Specifications\nDescription Nm\nGuide pin bolt 37",
        )

        extract_clean_page(page)

    assert capsys.readouterr().out == ""
