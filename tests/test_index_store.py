from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant.sparse_embeddings import SparseEmbeddings, SparseVector
from qdrant_client import QdrantClient, models

from vehicle_specs.chunking import TextChunk
from vehicle_specs.indexing import (
    DuplicateChunkIdError,
    IndexAlreadyExistsError,
    IndexCompatibilityError,
    IndexConfig,
    PipelineVersions,
    build_index,
    open_index,
    read_index_manifest,
)
from vehicle_specs.pdf import SectionContext


class FakeDenseEmbeddings(Embeddings):
    model_name = "fake-dense"
    dimension = 3

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    @staticmethod
    def _vector(text: str) -> list[float]:
        return [float(len(text)), float(text.count("bolt") + 1), 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeSparseEmbeddings(SparseEmbeddings):
    model_name = "fake-sparse"

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        self.batch_sizes.append(len(texts))
        return [SparseVector(indices=[1, 7], values=[1.0, 1.0]) for _ in texts]

    def embed_query(self, text: str) -> SparseVector:
        return SparseVector(indices=[1], values=[1.0])


def make_config(path: Path, *, pipeline_suffix: str = "test") -> IndexConfig:
    return IndexConfig(
        path=path,
        collection_name="manual_specs",
        dense_model="fake-dense",
        sparse_model="fake-sparse",
        batch_size=2,
        pipeline_versions=PipelineVersions(
            pdf_parser=f"parser-{pipeline_suffix}",
            pdf_cleaner=f"cleaner-{pipeline_suffix}",
            chunker=f"chunker-{pipeline_suffix}",
        ),
    )


def make_chunk(ordinal: int, *, text: str | None = None) -> TextChunk:
    page = 635 + ordinal
    section = SectionContext(
        section_id="206-03",
        section_title="Front Disc Brake",
        category="SPECIFICATIONS",
        article_title="Specifications",
        start_pdf_page=636,
    )
    return TextChunk(
        chunk_id=f"manual-206-03-a0636-p{page:04d}-table-{ordinal:03d}",
        source="manual.pdf",
        kind="table",
        text=text or f"Brake bolt {ordinal}: {ordinal * 10} Nm",
        pdf_pages=(page,),
        section=section,
    )


def test_builds_batched_hybrid_index_and_reopens_from_disk(tmp_path: Path) -> None:
    config = make_config(tmp_path / "qdrant")
    dense = FakeDenseEmbeddings()
    sparse = FakeSparseEmbeddings()
    chunks = [make_chunk(1), make_chunk(2), make_chunk(3)]

    result = build_index(
        chunks,
        config,
        dense_embeddings=dense,
        sparse_embeddings=sparse,
    )

    assert result.input_chunk_count == 3
    assert result.indexed_chunk_count == 3
    assert result.duplicate_chunk_count == 0
    assert result.manifest.dense_dimension == 3
    assert dense.batch_sizes == [2, 1]
    assert sparse.batch_sizes == [2, 1]

    manifest = read_index_manifest(config)
    assert manifest == result.manifest

    client = QdrantClient(path=str(config.path))
    try:
        collection = client.get_collection(config.collection_name)
        assert set(collection.config.params.vectors) == {"dense"}
        assert set(collection.config.params.sparse_vectors or {}) == {"sparse"}
        sparse_config = collection.config.params.sparse_vectors["sparse"]
        assert sparse_config.modifier == models.Modifier.IDF

        points, next_offset = client.scroll(
            config.collection_name,
            limit=10,
            with_payload=True,
            with_vectors=True,
        )
        assert next_offset is None
        assert [point.id for point in points] == [1, 2, 3]
        assert all(set(point.vector) == {"dense", "sparse"} for point in points)
        first_payload = points[0].payload
        assert first_payload["page_content"] == chunks[0].text
        assert first_payload["metadata"]["chunk_id"] == chunks[0].chunk_id
        assert first_payload["metadata"]["pdf_pages"] == [636]
        assert first_payload["metadata"]["section_id"] == "206-03"
        assert first_payload["metadata"]["pipeline_versions"] == {
            "pdf_parser": "parser-test",
            "pdf_cleaner": "cleaner-test",
            "chunker": "chunker-test",
        }
    finally:
        client.close()

    with open_index(
        config,
        dense_embeddings=FakeDenseEmbeddings(),
        sparse_embeddings=FakeSparseEmbeddings(),
    ) as vector_store:
        assert isinstance(vector_store, QdrantVectorStore)

    assert read_index_manifest(config).chunk_count == 3


def test_identical_duplicate_chunk_ids_are_indexed_once(tmp_path: Path) -> None:
    config = make_config(tmp_path / "qdrant")
    first = make_chunk(1)
    second = make_chunk(2)

    result = build_index(
        [first, first, second],
        config,
        dense_embeddings=FakeDenseEmbeddings(),
        sparse_embeddings=FakeSparseEmbeddings(),
    )

    assert result.input_chunk_count == 3
    assert result.indexed_chunk_count == 2
    assert result.duplicate_chunk_count == 1
    assert read_index_manifest(config).chunk_count == 2


def test_conflicting_duplicate_chunk_ids_fail_before_creating_index(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path / "qdrant")
    original = make_chunk(1)
    conflicting = original.model_copy(update={"text": "Different content"})

    with pytest.raises(DuplicateChunkIdError, match=original.chunk_id):
        build_index(
            [original, conflicting],
            config,
            dense_embeddings=FakeDenseEmbeddings(),
            sparse_embeddings=FakeSparseEmbeddings(),
        )

    assert not config.path.exists()


def test_existing_collection_requires_explicit_rebuild(tmp_path: Path) -> None:
    config = make_config(tmp_path / "qdrant")
    embeddings = {
        "dense_embeddings": FakeDenseEmbeddings(),
        "sparse_embeddings": FakeSparseEmbeddings(),
    }
    build_index([make_chunk(1), make_chunk(2)], config, **embeddings)

    with pytest.raises(IndexAlreadyExistsError, match="rebuild=True"):
        build_index([make_chunk(3)], config, **embeddings)

    assert read_index_manifest(config).chunk_count == 2

    rebuilt = build_index([make_chunk(3)], config, rebuild=True, **embeddings)

    assert rebuilt.indexed_chunk_count == 1
    assert read_index_manifest(config).chunk_count == 1


def test_changed_pipeline_version_rejects_stale_index(tmp_path: Path) -> None:
    config = make_config(tmp_path / "qdrant", pipeline_suffix="one")
    build_index(
        [make_chunk(1)],
        config,
        dense_embeddings=FakeDenseEmbeddings(),
        sparse_embeddings=FakeSparseEmbeddings(),
    )
    changed_config = make_config(config.path, pipeline_suffix="two")

    with pytest.raises(IndexCompatibilityError, match="pipeline_versions"):
        read_index_manifest(changed_config)
