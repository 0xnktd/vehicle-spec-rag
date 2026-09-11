"""Typed environment-backed settings for the synchronous application."""

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from vehicle_specs.chunking import ChunkingConfig
from vehicle_specs.indexing import (
    DEFAULT_DENSE_MODEL,
    DEFAULT_SPARSE_MODEL,
    IndexConfig,
)
from vehicle_specs.retrieval import (
    DEFAULT_RERANKER_MODEL,
    RetrievalConfig,
    RetrievalMode,
)


class AppSettings(BaseSettings):
    """Application defaults overridable with `VEHICLE_SPECS_` variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="VEHICLE_SPECS_",
        extra="ignore",
        frozen=True,
    )

    pdf_path: Path = Path("docs/sample-service-manual 1.pdf")
    index_path: Path = Path("artifacts/qdrant")
    pages_path: Path = Path("artifacts/pages.jsonl")
    chunks_path: Path = Path("artifacts/chunks.jsonl")
    page_quality_path: Path = Path("artifacts/page_quality.jsonl")
    model_cache_path: Path = Path("artifacts/models")
    collection_name: str = Field(default="vehicle_specs", min_length=1)

    dense_model: str = Field(default=DEFAULT_DENSE_MODEL, min_length=1)
    sparse_model: str = Field(default=DEFAULT_SPARSE_MODEL, min_length=1)
    embedding_batch_size: int = Field(default=64, ge=1)

    chunk_max_tokens: int = Field(default=400, ge=32)
    chunk_overlap_tokens: int = Field(default=40, ge=0)

    retrieval_mode: RetrievalMode = "hybrid"
    retrieval_candidate_k: int = Field(default=20, ge=1)
    retrieval_context_k: int = Field(default=8, ge=1)
    retrieval_rrf_k: int = Field(default=60, ge=1)
    extraction_context_k: int = Field(default=5, ge=1)

    reranker_model: str = Field(default=DEFAULT_RERANKER_MODEL, min_length=1)
    reranker_batch_size: int = Field(default=32, ge=1)

    llm_model: str = Field(
        default="RedHatAI/Qwen3.5-4B-quantized.w4a16",
        min_length=1,
    )
    llm_base_url: str = Field(default="http://localhost:8000/v1", min_length=1)
    llm_api_key: SecretStr = SecretStr("local-vllm")
    llm_max_output_tokens: int = Field(default=2048, ge=1)
    llm_request_timeout_seconds: float = Field(default=180, gt=0)

    def chunking_config(self) -> ChunkingConfig:
        """Build the validated chunker configuration."""
        return ChunkingConfig(
            max_tokens=self.chunk_max_tokens,
            overlap_tokens=self.chunk_overlap_tokens,
        )

    def index_config(self, *, path: Path | None = None) -> IndexConfig:
        """Build the validated persistent-index configuration."""
        return IndexConfig(
            path=path or self.index_path,
            collection_name=self.collection_name,
            dense_model=self.dense_model,
            sparse_model=self.sparse_model,
            batch_size=self.embedding_batch_size,
            model_cache_path=self.model_cache_path,
        )

    def retrieval_config(
        self,
        *,
        mode: RetrievalMode | None = None,
        candidate_k: int | None = None,
        context_k: int | None = None,
    ) -> RetrievalConfig:
        """Build validated retrieval limits from settings and command overrides."""
        return RetrievalConfig(
            mode=mode or self.retrieval_mode,
            candidate_k=candidate_k or self.retrieval_candidate_k,
            context_k=context_k or self.retrieval_context_k,
            rrf_k=self.retrieval_rrf_k,
        )
