"""Synchronous PDF-to-index ingestion orchestration."""

from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_qdrant.sparse_embeddings import SparseEmbeddings
from pydantic import BaseModel, ConfigDict, Field

from vehicle_specs.chunking import ChunkingConfig, iter_chunks, write_chunks_jsonl
from vehicle_specs.indexing import IndexBuildResult, IndexConfig, build_index
from vehicle_specs.pdf import (
    assess_pages,
    iter_clean_page_records,
    iter_contextual_pages,
    write_page_quality_jsonl,
    write_page_records_jsonl,
)


class IngestionResult(BaseModel):
    """Inspectable summary of a completed PDF-to-index run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pdf_path: Path
    pages_path: Path
    chunks_path: Path
    page_quality_path: Path
    page_count: int = Field(ge=1)
    chunk_count: int = Field(ge=0)
    table_chunk_count: int = Field(ge=0)
    suspicious_pdf_pages: tuple[int, ...] = ()
    index: IndexBuildResult


def ingest_pdf(
    pdf_path: str | Path,
    *,
    index_config: IndexConfig,
    chunking_config: ChunkingConfig | None = None,
    pages_path: str | Path = "artifacts/pages.jsonl",
    chunks_path: str | Path = "artifacts/chunks.jsonl",
    page_quality_path: str | Path = "artifacts/page_quality.jsonl",
    rebuild: bool = False,
    dense_embeddings: Embeddings | None = None,
    sparse_embeddings: SparseEmbeddings | None = None,
) -> IngestionResult:
    """Extract, inspect, chunk, persist, and index one PDF synchronously."""
    source_path = Path(pdf_path)
    page_output = Path(pages_path)
    chunk_output = Path(chunks_path)
    quality_output = Path(page_quality_path)

    pages = tuple(iter_clean_page_records(source_path))
    write_page_records_jsonl(pages, page_output)

    qualities = assess_pages(pages)
    write_page_quality_jsonl(qualities, quality_output)

    chunks = tuple(
        iter_chunks(
            iter_contextual_pages(pages),
            source=source_path.name,
            config=chunking_config,
        )
    )
    write_chunks_jsonl(chunks, chunk_output)

    index_result = build_index(
        chunks,
        index_config,
        rebuild=rebuild,
        dense_embeddings=dense_embeddings,
        sparse_embeddings=sparse_embeddings,
    )
    return IngestionResult(
        pdf_path=source_path,
        pages_path=page_output,
        chunks_path=chunk_output,
        page_quality_path=quality_output,
        page_count=len(pages),
        chunk_count=len(chunks),
        table_chunk_count=sum(chunk.kind == "table" for chunk in chunks),
        suspicious_pdf_pages=tuple(
            quality.pdf_page for quality in qualities if quality.suspiciously_sparse
        ),
        index=index_result,
    )
