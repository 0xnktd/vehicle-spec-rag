"""Reusable assembly and execution of the synchronous query pipeline."""

from pathlib import Path
from threading import Lock

from langchain_core.embeddings import Embeddings
from langchain_qdrant.sparse_embeddings import SparseEmbeddings

from vehicle_specs.config import AppSettings
from vehicle_specs.extraction import (
    StructuredExtractor,
    create_openai_compatible_extractor,
)
from vehicle_specs.indexing import (
    FastEmbedDenseEmbeddings,
    FastEmbedSparseEmbeddings,
    IndexConfig,
    open_index,
)
from vehicle_specs.pipeline import PipelineRun, VehicleSpecificationPipeline
from vehicle_specs.retrieval import (
    FastEmbedReranker,
    Reranker,
    RetrievalConfig,
    RetrievalFilters,
    VehicleSpecRetriever,
)


class QueryService:
    """Own expensive query models and serialize access to the local pipeline."""

    def __init__(
        self,
        *,
        index_config: IndexConfig,
        retrieval_config: RetrievalConfig,
        extractor: StructuredExtractor,
        dense_embeddings: Embeddings,
        sparse_embeddings: SparseEmbeddings,
        reranker: Reranker | None = None,
    ) -> None:
        self._index_config = index_config
        self._retrieval_config = retrieval_config
        self._extractor = extractor
        self._dense_embeddings = dense_embeddings
        self._sparse_embeddings = sparse_embeddings
        self._reranker = reranker
        self._query_lock = Lock()

    def run(
        self,
        query: str,
        *,
        filters: RetrievalFilters | None = None,
    ) -> PipelineRun:
        """Run one query while keeping the embedded index lock request-scoped."""
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")

        # A UI may share this service between sessions. Serializing the synchronous
        # run protects local model clients and matches vLLM's one-sequence GPU setup.
        with self._query_lock:
            with open_index(
                self._index_config,
                dense_embeddings=self._dense_embeddings,
                sparse_embeddings=self._sparse_embeddings,
            ) as store:
                retriever = VehicleSpecRetriever(
                    store,
                    self._retrieval_config,
                    reranker=self._reranker,
                )
                pipeline = VehicleSpecificationPipeline(retriever, self._extractor)
                return pipeline.run(query, filters=filters)


def create_query_service(
    settings: AppSettings | None = None,
    *,
    index_path: Path | None = None,
    retrieval_config: RetrievalConfig | None = None,
    rerank: bool = True,
) -> QueryService:
    """Build query components from environment settings and explicit overrides."""
    settings = settings or AppSettings()
    cache_dir = str(settings.model_cache_path)
    configured_reranker: Reranker | None = None
    if rerank:
        configured_reranker = FastEmbedReranker(
            model_name=settings.reranker_model,
            batch_size=settings.reranker_batch_size,
            cache_dir=cache_dir,
        )

    if retrieval_config is None:
        retrieval_config = settings.retrieval_config(
            context_k=settings.extraction_context_k,
        )

    return QueryService(
        index_config=settings.index_config(path=index_path),
        retrieval_config=retrieval_config,
        extractor=create_openai_compatible_extractor(
            settings.llm_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            max_output_tokens=settings.llm_max_output_tokens,
            request_timeout_seconds=settings.llm_request_timeout_seconds,
        ),
        dense_embeddings=FastEmbedDenseEmbeddings(
            model_name=settings.dense_model,
            batch_size=settings.embedding_batch_size,
            cache_dir=cache_dir,
        ),
        sparse_embeddings=FastEmbedSparseEmbeddings(
            model_name=settings.sparse_model,
            batch_size=settings.embedding_batch_size,
            cache_dir=cache_dir,
        ),
        reranker=configured_reranker,
    )
