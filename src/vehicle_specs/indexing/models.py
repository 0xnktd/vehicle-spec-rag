"""Validated configuration and results for the local vector index."""

from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vehicle_specs.chunking.chunker import CHUNKER_VERSION
from vehicle_specs.pdf.cleaner import PDF_CLEANER_VERSION
from vehicle_specs.pdf.extractor import PDF_PARSER_VERSION

INDEX_FORMAT_VERSION = "1.0.0"
DEFAULT_DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_SPARSE_MODEL = "Qdrant/bm25"


class PipelineVersions(BaseModel):
    """Versions of the transformations represented by an indexed chunk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pdf_parser: str = Field(default=PDF_PARSER_VERSION, min_length=1)
    pdf_cleaner: str = Field(default=PDF_CLEANER_VERSION, min_length=1)
    chunker: str = Field(default=CHUNKER_VERSION, min_length=1)


class IndexConfig(BaseModel):
    """Configuration for one local, persistent hybrid index."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    path: Path
    collection_name: str = Field(default="vehicle_specs", min_length=1)
    dense_model: str = Field(default=DEFAULT_DENSE_MODEL, min_length=1)
    sparse_model: str = Field(default=DEFAULT_SPARSE_MODEL, min_length=1)
    dense_vector_name: str = Field(default="dense", min_length=1)
    sparse_vector_name: str = Field(default="sparse", min_length=1)
    batch_size: int = Field(default=64, ge=1)
    model_cache_path: Path | None = None
    pipeline_versions: PipelineVersions = Field(default_factory=PipelineVersions)

    @model_validator(mode="after")
    def validate_vector_names(self) -> Self:
        if self.dense_vector_name == self.sparse_vector_name:
            raise ValueError("dense and sparse vector names must be different")
        return self


class IndexManifest(BaseModel):
    """Reproducibility metadata stored with a Qdrant collection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    index_format_version: str = Field(default=INDEX_FORMAT_VERSION, min_length=1)
    collection_name: str = Field(min_length=1)
    dense_model: str = Field(min_length=1)
    sparse_model: str = Field(min_length=1)
    dense_vector_name: str = Field(min_length=1)
    sparse_vector_name: str = Field(min_length=1)
    dense_dimension: int = Field(gt=0)
    chunk_count: int = Field(ge=0)
    pipeline_versions: PipelineVersions


class IndexBuildResult(BaseModel):
    """Counts and manifest returned after a successful index build."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    input_chunk_count: int = Field(ge=0)
    duplicate_chunk_count: int = Field(ge=0)
    manifest: IndexManifest

    @property
    def indexed_chunk_count(self) -> int:
        return self.manifest.chunk_count
