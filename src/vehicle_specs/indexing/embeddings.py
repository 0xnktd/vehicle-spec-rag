"""Local FastEmbed implementations of LangChain embedding interfaces."""

from collections.abc import Iterable
from typing import Any, Protocol

from langchain_core.embeddings import Embeddings
from langchain_qdrant.sparse_embeddings import SparseEmbeddings, SparseVector

from .models import DEFAULT_DENSE_MODEL, DEFAULT_SPARSE_MODEL


class _DenseBackend(Protocol):
    def embed(
        self,
        documents: Iterable[str],
        *,
        batch_size: int,
    ) -> Iterable[Any]: ...

    def query_embed(self, query: str, **kwargs: Any) -> Iterable[Any]: ...


class _SparseBackend(Protocol):
    def embed(
        self,
        documents: Iterable[str],
        *,
        batch_size: int,
    ) -> Iterable[Any]: ...

    def query_embed(self, query: str, **kwargs: Any) -> Iterable[Any]: ...


def _as_list(values: Any) -> list[Any]:
    converted = values.tolist() if hasattr(values, "tolist") else values
    return list(converted)


def _dense_model_dimension(model_name: str) -> int:
    from fastembed import TextEmbedding

    for description in TextEmbedding.list_supported_models():
        if description["model"] == model_name:
            return int(description["dim"])
    raise ValueError(f"FastEmbed dense model is not supported: {model_name}")


def _dense_vectors(raw_vectors: Iterable[Any]) -> list[list[float]]:
    return [[float(value) for value in _as_list(vector)] for vector in raw_vectors]


class FastEmbedDenseEmbeddings(Embeddings):
    """Run a FastEmbed dense model behind LangChain's standard interface."""

    def __init__(
        self,
        model_name: str = DEFAULT_DENSE_MODEL,
        *,
        batch_size: int = 64,
        cache_dir: str | None = None,
        threads: int | None = None,
        _backend: _DenseBackend | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.dimension = _dense_model_dimension(model_name)

        if _backend is None:
            from fastembed import TextEmbedding

            _backend = TextEmbedding(
                model_name=model_name,
                cache_dir=cache_dir,
                threads=threads,
                lazy_load=True,
            )
        self._backend = _backend

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages locally in configured batches."""
        if not texts:
            return []
        vectors = self._backend.embed(texts, batch_size=self.batch_size)
        return _dense_vectors(vectors)

    def embed_query(self, text: str) -> list[float]:
        """Embed one retrieval query locally."""
        vectors = self._backend.query_embed(text, batch_size=self.batch_size)
        return _dense_vectors(vectors)[0]


def _sparse_vectors(raw_vectors: Iterable[Any]) -> list[SparseVector]:
    vectors: list[SparseVector] = []

    for raw_vector in raw_vectors:
        indices = [int(index) for index in _as_list(raw_vector.indices)]
        values = [float(value) for value in _as_list(raw_vector.values)]
        vectors.append(SparseVector(indices=indices, values=values))

    return vectors


class FastEmbedSparseEmbeddings(SparseEmbeddings):
    """Run FastEmbed BM25 behind LangChain Qdrant's sparse interface."""

    def __init__(
        self,
        model_name: str = DEFAULT_SPARSE_MODEL,
        *,
        batch_size: int = 64,
        cache_dir: str | None = None,
        threads: int | None = None,
        _backend: _SparseBackend | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

        if _backend is None:
            from fastembed import SparseTextEmbedding

            supported = {
                description["model"]
                for description in SparseTextEmbedding.list_supported_models()
            }
            if model_name not in supported:
                raise ValueError(
                    f"FastEmbed sparse model is not supported: {model_name}"
                )
            _backend = SparseTextEmbedding(
                model_name=model_name,
                cache_dir=cache_dir,
                threads=threads,
                lazy_load=True,
            )
        self._backend = _backend

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        """Embed passages as local sparse vectors."""
        if not texts:
            return []
        vectors = self._backend.embed(texts, batch_size=self.batch_size)
        return _sparse_vectors(vectors)

    def embed_query(self, text: str) -> SparseVector:
        """Embed one retrieval query as a local sparse vector."""
        vectors = self._backend.query_embed(text, batch_size=self.batch_size)
        return _sparse_vectors(vectors)[0]
