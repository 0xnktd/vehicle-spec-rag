from pathlib import Path
from typing import ClassVar

import pymupdf
from langchain_core.embeddings import Embeddings
from langchain_qdrant.sparse_embeddings import SparseEmbeddings, SparseVector

from vehicle_specs.indexing import IndexConfig, open_index
from vehicle_specs.ingestion import ingest_pdf


class ConstantDenseEmbeddings(Embeddings):
    model_name = "constant-dense"
    dimension = 2

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class ConstantSparseEmbeddings(SparseEmbeddings):
    model_name = "constant-sparse"
    index: ClassVar[int] = 1

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [SparseVector(indices=[self.index], values=[1.0]) for _ in texts]

    def embed_query(self, text: str) -> SparseVector:
        return SparseVector(indices=[self.index], values=[1.0])


def make_pdf(path: Path) -> None:
    with pymupdf.open() as document:
        page = document.new_page(width=600, height=800)
        page.insert_text(
            (50, 80),
            "SECTION 206-03: Front Disc Brake\nSPECIFICATIONS\n"
            "Torque Specifications\nBrake caliper guide pin bolts 37 Nm",
        )
        document.save(path)


def test_ingestion_persists_inspection_artifacts_and_reopenable_index(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "manual.pdf"
    make_pdf(pdf_path)
    dense = ConstantDenseEmbeddings()
    sparse = ConstantSparseEmbeddings()
    config = IndexConfig(
        path=tmp_path / "qdrant",
        collection_name="ingestion_test",
        dense_model=dense.model_name,
        sparse_model=sparse.model_name,
    )

    result = ingest_pdf(
        pdf_path,
        index_config=config,
        pages_path=tmp_path / "pages.jsonl",
        chunks_path=tmp_path / "chunks.jsonl",
        page_quality_path=tmp_path / "quality.jsonl",
        dense_embeddings=dense,
        sparse_embeddings=sparse,
    )

    assert result.page_count == 1
    assert result.chunk_count >= 1
    assert result.index.indexed_chunk_count == result.chunk_count
    assert result.pages_path.read_text(encoding="utf-8").count("\n") == 1
    assert result.chunks_path.read_text(encoding="utf-8").count("\n") >= 1
    assert result.page_quality_path.read_text(encoding="utf-8").count("\n") == 1

    with open_index(
        config,
        dense_embeddings=dense,
        sparse_embeddings=sparse,
    ) as store:
        assert store.client.count(config.collection_name, exact=True).count >= 1
