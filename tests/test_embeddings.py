from dataclasses import dataclass
from typing import Any

from vehicle_specs.indexing import (
    DEFAULT_DENSE_MODEL,
    DEFAULT_SPARSE_MODEL,
    FastEmbedDenseEmbeddings,
    FastEmbedSparseEmbeddings,
)


class FakeDenseBackend:
    def __init__(self, *, dimension: int = 384) -> None:
        self.dimension = dimension
        self.document_calls: list[tuple[list[str], int]] = []
        self.query_calls: list[tuple[str, int]] = []

    def _vector(self, marker: float) -> list[float]:
        return [marker, *([0.0] * (self.dimension - 1))]

    def embed(
        self,
        documents: list[str],
        *,
        batch_size: int,
    ) -> list[list[float]]:
        texts = list(documents)
        self.document_calls.append((texts, batch_size))
        return [self._vector(float(len(text))) for text in texts]

    def query_embed(self, query: str, **kwargs: Any) -> list[list[float]]:
        self.query_calls.append((query, kwargs["batch_size"]))
        return [self._vector(99.0)]


@dataclass
class RawSparseVector:
    indices: list[int]
    values: list[float]


class FakeSparseBackend:
    def __init__(self) -> None:
        self.document_calls: list[tuple[list[str], int]] = []
        self.query_calls: list[tuple[str, int]] = []

    def embed(
        self,
        documents: list[str],
        *,
        batch_size: int,
    ) -> list[RawSparseVector]:
        texts = list(documents)
        self.document_calls.append((texts, batch_size))
        return [RawSparseVector(indices=[1, 4], values=[1.0, 2.0]) for _ in texts]

    def query_embed(self, query: str, **kwargs: Any) -> list[RawSparseVector]:
        self.query_calls.append((query, kwargs["batch_size"]))
        return [RawSparseVector(indices=[4], values=[3.0])]


def test_dense_adapter_implements_document_and_query_embeddings() -> None:
    backend = FakeDenseBackend()
    embeddings = FastEmbedDenseEmbeddings(
        model_name=DEFAULT_DENSE_MODEL,
        batch_size=2,
        _backend=backend,
    )

    document_vectors = embeddings.embed_documents(["brake", "caliper"])
    query_vector = embeddings.embed_query("guide pin")

    assert embeddings.dimension == 384
    assert len(document_vectors) == 2
    assert all(len(vector) == 384 for vector in document_vectors)
    assert query_vector[0] == 99.0
    assert backend.document_calls == [(["brake", "caliper"], 2)]
    assert backend.query_calls == [("guide pin", 2)]


def test_sparse_adapter_implements_document_and_query_embeddings() -> None:
    backend = FakeSparseBackend()
    embeddings = FastEmbedSparseEmbeddings(
        model_name=DEFAULT_SPARSE_MODEL,
        batch_size=3,
        _backend=backend,
    )

    document_vectors = embeddings.embed_documents(["37 Nm", "guide pin"])
    query_vector = embeddings.embed_query("bolt torque")

    assert [vector.indices for vector in document_vectors] == [[1, 4], [1, 4]]
    assert [vector.values for vector in document_vectors] == [[1.0, 2.0], [1.0, 2.0]]
    assert query_vector.indices == [4]
    assert query_vector.values == [3.0]
    assert backend.document_calls == [(["37 Nm", "guide pin"], 3)]
    assert backend.query_calls == [("bolt torque", 3)]
