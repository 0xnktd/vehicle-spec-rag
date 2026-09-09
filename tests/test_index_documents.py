from vehicle_specs.chunking import TextChunk
from vehicle_specs.indexing import PipelineVersions, chunk_to_document
from vehicle_specs.pdf import SectionContext


def test_chunk_becomes_langchain_document_with_citation_metadata() -> None:
    section = SectionContext(
        section_id="206-03",
        section_title="Front Disc Brake",
        category="SPECIFICATIONS",
        article_title="Brake Caliper",
        start_pdf_page=636,
    )
    chunk = TextChunk(
        chunk_id="manual-206-03-a0636-p0636-table-003",
        source="manual.pdf",
        kind="table",
        text="| Item | Nm |\n| --- | --- |\n| Guide pin bolts | 37 |",
        pdf_pages=(636,),
        section=section,
    )
    versions = PipelineVersions(
        pdf_parser="parser-test",
        pdf_cleaner="cleaner-test",
        chunker="chunker-test",
    )

    document = chunk_to_document(chunk, versions)

    assert document.id == chunk.chunk_id
    assert document.page_content == chunk.text
    assert document.metadata == {
        "chunk_id": chunk.chunk_id,
        "source": "manual.pdf",
        "kind": "table",
        "pdf_pages": [636],
        "pipeline_versions": {
            "pdf_parser": "parser-test",
            "pdf_cleaner": "cleaner-test",
            "chunker": "chunker-test",
        },
        "section_id": "206-03",
        "section_title": "Front Disc Brake",
        "category": "SPECIFICATIONS",
        "article_title": "Brake Caliper",
        "article_start_pdf_page": 636,
    }


def test_unsectioned_chunk_does_not_invent_section_metadata() -> None:
    chunk = TextChunk(
        chunk_id="manual-unsectioned-a0001-p0001-prose-001",
        source="manual.pdf",
        kind="prose",
        text="Introduction",
        pdf_pages=(1,),
    )

    document = chunk_to_document(chunk)

    assert "section_id" not in document.metadata
    assert "article_title" not in document.metadata
