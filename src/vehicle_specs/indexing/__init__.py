"""Local dense/sparse embeddings and persistent Qdrant indexing."""

from .documents import chunk_metadata, chunk_to_document, chunks_to_documents
from .embeddings import FastEmbedDenseEmbeddings, FastEmbedSparseEmbeddings
from .models import (
    DEFAULT_DENSE_MODEL,
    DEFAULT_SPARSE_MODEL,
    INDEX_FORMAT_VERSION,
    IndexBuildResult,
    IndexConfig,
    IndexManifest,
    PipelineVersions,
)
from .store import (
    DuplicateChunkIdError,
    IndexAlreadyExistsError,
    IndexCompatibilityError,
    IndexIntegrityError,
    IndexNotFoundError,
    build_index,
    open_index,
    read_index_manifest,
)

__all__ = [
    "DEFAULT_DENSE_MODEL",
    "DEFAULT_SPARSE_MODEL",
    "INDEX_FORMAT_VERSION",
    "DuplicateChunkIdError",
    "FastEmbedDenseEmbeddings",
    "FastEmbedSparseEmbeddings",
    "IndexAlreadyExistsError",
    "IndexBuildResult",
    "IndexCompatibilityError",
    "IndexConfig",
    "IndexIntegrityError",
    "IndexManifest",
    "IndexNotFoundError",
    "PipelineVersions",
    "build_index",
    "chunk_metadata",
    "chunk_to_document",
    "chunks_to_documents",
    "open_index",
    "read_index_manifest",
]
