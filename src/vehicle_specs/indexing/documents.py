"""Conversion between pipeline chunks and LangChain documents."""

from collections.abc import Iterable
from typing import Any

from langchain_core.documents import Document

from vehicle_specs.chunking.models import TextChunk

from .models import PipelineVersions


def chunk_metadata(
    chunk: TextChunk,
    pipeline_versions: PipelineVersions,
) -> dict[str, Any]:
    """Build JSON-compatible citation and filtering metadata for a chunk."""
    metadata: dict[str, Any] = {
        "chunk_id": chunk.chunk_id,
        "source": chunk.source,
        "kind": chunk.kind,
        "pdf_pages": list(chunk.pdf_pages),
        "pipeline_versions": pipeline_versions.model_dump(mode="json"),
    }

    if chunk.section is not None:
        metadata.update(
            {
                "section_id": chunk.section.section_id,
                "section_title": chunk.section.section_title,
                "category": chunk.section.category,
                "article_title": chunk.section.article_title,
                "article_start_pdf_page": chunk.section.start_pdf_page,
            }
        )

    return metadata


def chunk_to_document(
    chunk: TextChunk,
    pipeline_versions: PipelineVersions | None = None,
) -> Document:
    """Represent one validated chunk using LangChain's interchange type."""
    versions = pipeline_versions or PipelineVersions()
    return Document(
        id=chunk.chunk_id,
        page_content=chunk.text,
        metadata=chunk_metadata(chunk, versions),
    )


def chunks_to_documents(
    chunks: Iterable[TextChunk],
    pipeline_versions: PipelineVersions | None = None,
) -> list[Document]:
    """Convert chunks while preserving their input order."""
    versions = pipeline_versions or PipelineVersions()
    return [chunk_to_document(chunk, versions) for chunk in chunks]
