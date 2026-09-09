import json

from vehicle_specs.chunking import TextChunk, write_chunks_jsonl


def test_writes_chunks_as_utf8_json_lines(tmp_path) -> None:
    chunks = (
        TextChunk(
            chunk_id="manual-unsectioned-a0001-p0001-prose-001",
            source="manual.pdf",
            kind="prose",
            text="Torque is 35 Nm — verified.",
            pdf_pages=(1,),
        ),
        TextChunk(
            chunk_id="manual-unsectioned-a0001-p0002-table-002",
            source="manual.pdf",
            kind="table",
            text="| Item | Nm |\n| --- | --- |\n| Bolt | 35 |",
            pdf_pages=(2,),
        ),
    )
    output_path = tmp_path / "nested" / "chunks.jsonl"

    count = write_chunks_jsonl(chunks, output_path)
    records = tuple(json.loads(line) for line in output_path.read_text().splitlines())

    assert count == 2
    assert [record["chunk_id"] for record in records] == [
        chunk.chunk_id for chunk in chunks
    ]
    assert records[0]["text"] == "Torque is 35 Nm — verified."
    assert not tuple(output_path.parent.glob("*.tmp"))
