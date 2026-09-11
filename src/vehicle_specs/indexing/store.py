"""LangChain document conversion and persistent local Qdrant storage."""

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_qdrant.sparse_embeddings import SparseEmbeddings
from pydantic import ValidationError
from qdrant_client import QdrantClient, models

from vehicle_specs.chunking.models import TextChunk

from .embeddings import FastEmbedDenseEmbeddings, FastEmbedSparseEmbeddings
from .models import (
    INDEX_FORMAT_VERSION,
    IndexBuildResult,
    IndexConfig,
    IndexManifest,
    PipelineVersions,
)


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


class IndexAlreadyExistsError(RuntimeError):
    """Raised when a build would overwrite a collection without permission."""


class IndexNotFoundError(FileNotFoundError):
    """Raised when the configured collection does not exist."""


class IndexCompatibilityError(RuntimeError):
    """Raised when persisted index settings do not match runtime settings."""


class IndexIntegrityError(RuntimeError):
    """Raised when persisted index contents fail an integrity check."""


class DuplicateChunkIdError(ValueError):
    """Raised when one chunk ID refers to conflicting chunk contents."""


def _deduplicate_chunks(
    chunks: Iterable[TextChunk],
) -> tuple[list[TextChunk], int, int]:
    unique_by_id: dict[str, TextChunk] = {}
    input_count = 0
    duplicate_count = 0

    for chunk in chunks:
        input_count += 1
        existing = unique_by_id.get(chunk.chunk_id)
        if existing is None:
            unique_by_id[chunk.chunk_id] = chunk
        elif existing == chunk:
            duplicate_count += 1
        else:
            raise DuplicateChunkIdError(
                f"chunk ID refers to different content: {chunk.chunk_id}"
            )

    return list(unique_by_id.values()), input_count, duplicate_count


def _model_cache_dir(config: IndexConfig) -> str | None:
    if config.model_cache_path is None:
        return None
    return str(config.model_cache_path)


def _create_dense_embeddings(config: IndexConfig) -> FastEmbedDenseEmbeddings:
    return FastEmbedDenseEmbeddings(
        model_name=config.dense_model,
        batch_size=config.batch_size,
        cache_dir=_model_cache_dir(config),
    )


def _create_sparse_embeddings(config: IndexConfig) -> FastEmbedSparseEmbeddings:
    return FastEmbedSparseEmbeddings(
        model_name=config.sparse_model,
        batch_size=config.batch_size,
        cache_dir=_model_cache_dir(config),
    )


def _validate_embedding_model(
    embeddings: Any,
    configured_model: str,
    *,
    kind: str,
) -> None:
    actual_model = getattr(embeddings, "model_name", None)
    if actual_model is not None and actual_model != configured_model:
        raise ValueError(
            f"configured {kind} model {configured_model!r} does not match "
            f"embedding provider {actual_model!r}"
        )


def _dense_dimension(embeddings: Embeddings) -> int:
    dimension = getattr(embeddings, "dimension", None)
    if dimension is None:
        dimension = len(embeddings.embed_query("embedding dimension probe"))
    return int(dimension)


def _manifest_for(
    config: IndexConfig,
    *,
    dense_dimension: int,
    chunk_count: int,
) -> IndexManifest:
    return IndexManifest(
        collection_name=config.collection_name,
        dense_model=config.dense_model,
        sparse_model=config.sparse_model,
        dense_vector_name=config.dense_vector_name,
        sparse_vector_name=config.sparse_vector_name,
        dense_dimension=dense_dimension,
        chunk_count=chunk_count,
        pipeline_versions=config.pipeline_versions,
    )


def _read_manifest(client: QdrantClient, config: IndexConfig) -> IndexManifest:
    if not client.collection_exists(config.collection_name):
        raise IndexNotFoundError(
            f"Qdrant collection does not exist: {config.collection_name}"
        )

    metadata = client.get_collection(config.collection_name).config.metadata
    if metadata is None:
        raise IndexCompatibilityError(
            "collection has no index manifest; rebuild it with this pipeline"
        )

    try:
        manifest = IndexManifest.model_validate(metadata)
    except ValidationError as error:
        raise IndexCompatibilityError(
            "collection index manifest is invalid; rebuild it with this pipeline"
        ) from error

    stored_count = client.count(config.collection_name, exact=True).count
    if stored_count != manifest.chunk_count:
        raise IndexIntegrityError(
            f"collection contains {stored_count} points but manifest expects "
            f"{manifest.chunk_count}"
        )
    return manifest


def _validate_manifest(config: IndexConfig, manifest: IndexManifest) -> None:
    expected: dict[str, Any] = {
        "index_format_version": INDEX_FORMAT_VERSION,
        "collection_name": config.collection_name,
        "dense_model": config.dense_model,
        "sparse_model": config.sparse_model,
        "dense_vector_name": config.dense_vector_name,
        "sparse_vector_name": config.sparse_vector_name,
        "pipeline_versions": config.pipeline_versions,
    }
    mismatches = [
        name
        for name, expected_value in expected.items()
        if getattr(manifest, name) != expected_value
    ]
    if mismatches:
        raise IndexCompatibilityError(
            "index is incompatible in "
            f"{', '.join(mismatches)}; rebuild the collection explicitly"
        )


def read_index_manifest(config: IndexConfig) -> IndexManifest:
    """Read and verify the manifest from a closed-and-reopenable local index."""
    client = QdrantClient(path=str(config.path))
    try:
        manifest = _read_manifest(client, config)
        _validate_manifest(config, manifest)
        return manifest
    finally:
        client.close()


def build_index(
    chunks: Iterable[TextChunk],
    config: IndexConfig,
    *,
    rebuild: bool = False,
    dense_embeddings: Embeddings | None = None,
    sparse_embeddings: SparseEmbeddings | None = None,
) -> IndexBuildResult:
    """Build a fresh dense-plus-BM25 local index and close it before returning."""
    unique_chunks, input_count, duplicate_count = _deduplicate_chunks(chunks)
    dense = dense_embeddings or _create_dense_embeddings(config)
    sparse = sparse_embeddings or _create_sparse_embeddings(config)
    _validate_embedding_model(dense, config.dense_model, kind="dense")
    _validate_embedding_model(sparse, config.sparse_model, kind="sparse")

    dense_dimension = _dense_dimension(dense)
    manifest = _manifest_for(
        config,
        dense_dimension=dense_dimension,
        chunk_count=len(unique_chunks),
    )
    documents = chunks_to_documents(unique_chunks, config.pipeline_versions)

    index_path = Path(config.path)
    index_path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(index_path))
    created_collection = False

    try:
        if client.collection_exists(config.collection_name):
            if not rebuild:
                raise IndexAlreadyExistsError(
                    f"Qdrant collection already exists: {config.collection_name}; "
                    "pass rebuild=True to replace it"
                )
            client.delete_collection(config.collection_name)

        client.create_collection(
            collection_name=config.collection_name,
            vectors_config={
                config.dense_vector_name: models.VectorParams(
                    size=dense_dimension,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                config.sparse_vector_name: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
            metadata=manifest.model_dump(mode="json"),
        )
        created_collection = True

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=config.collection_name,
            embedding=dense,
            sparse_embedding=sparse,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name=config.dense_vector_name,
            sparse_vector_name=config.sparse_vector_name,
            validate_collection_config=False,
        )
        point_ids = list(range(1, len(documents) + 1))
        vector_store.add_documents(
            documents,
            ids=point_ids,
            batch_size=config.batch_size,
            wait=True,
        )

        stored_count = client.count(config.collection_name, exact=True).count
        if stored_count != len(documents):
            raise IndexIntegrityError(
                f"Qdrant stored {stored_count} of {len(documents)} chunks"
            )
    except Exception:
        if created_collection and client.collection_exists(config.collection_name):
            client.delete_collection(config.collection_name)
        raise
    finally:
        client.close()

    return IndexBuildResult(
        path=index_path,
        input_chunk_count=input_count,
        duplicate_chunk_count=duplicate_count,
        manifest=manifest,
    )


@contextmanager
def open_index(
    config: IndexConfig,
    *,
    dense_embeddings: Embeddings | None = None,
    sparse_embeddings: SparseEmbeddings | None = None,
) -> Iterator[QdrantVectorStore]:
    """Open a compatible index as a LangChain store and always release its lock."""
    client = QdrantClient(path=str(config.path))
    try:
        manifest = _read_manifest(client, config)
        _validate_manifest(config, manifest)

        dense = dense_embeddings or _create_dense_embeddings(config)
        sparse = sparse_embeddings or _create_sparse_embeddings(config)
        _validate_embedding_model(dense, config.dense_model, kind="dense")
        _validate_embedding_model(sparse, config.sparse_model, kind="sparse")

        yield QdrantVectorStore(
            client=client,
            collection_name=config.collection_name,
            embedding=dense,
            sparse_embedding=sparse,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name=config.dense_vector_name,
            sparse_vector_name=config.sparse_vector_name,
        )
    finally:
        client.close()
