"""Dense, sparse, and explicit reciprocal-rank-fusion retrieval."""

from collections.abc import Sequence
from typing import Any

from langchain_qdrant import QdrantVectorStore
from qdrant_client import models
from qdrant_client.http.models import ScoredPoint

from .models import RetrievalConfig, RetrievalFilters, RetrievalHit
from .reranker import Reranker


class RetrievalPayloadError(RuntimeError):
    """Raised when an indexed point does not satisfy the payload contract."""


def _qdrant_filter(filters: RetrievalFilters | None) -> models.Filter | None:
    if filters is None:
        return None

    conditions: list[models.Condition] = []
    if filters.source is not None:
        conditions.append(
            models.FieldCondition(
                key="metadata.source",
                match=models.MatchValue(value=filters.source),
            )
        )
    if filters.section_ids:
        conditions.append(
            models.FieldCondition(
                key="metadata.section_id",
                match=models.MatchAny(any=list(filters.section_ids)),
            )
        )
    if filters.categories:
        conditions.append(
            models.FieldCondition(
                key="metadata.category",
                match=models.MatchAny(any=list(filters.categories)),
            )
        )
    return models.Filter(must=conditions) if conditions else None


def _payload_values(point: ScoredPoint) -> dict[str, Any]:
    payload = point.payload or {}
    metadata = payload.get("metadata")
    text = payload.get("page_content")
    if not isinstance(metadata, dict) or not isinstance(text, str):
        raise RetrievalPayloadError(
            f"Qdrant point {point.id!r} is missing LangChain text or metadata"
        )
    point_id = point.id if isinstance(point.id, (int, str)) else str(point.id)
    return {"point_id": point_id, "text": text, **metadata}


def _matches_applicability(point: ScoredPoint, applicability: str | None) -> bool:
    if applicability is None:
        return True
    metadata = (point.payload or {}).get("metadata")
    if not isinstance(metadata, dict):
        return False
    section_title = metadata.get("section_title")
    return isinstance(section_title, str) and applicability in section_title.casefold()


def _single_mode_hits(
    points: Sequence[ScoredPoint],
    *,
    mode: str,
) -> list[RetrievalHit]:
    hits: list[RetrievalHit] = []
    for rank, point in enumerate(points, start=1):
        branch_values: dict[str, Any] = {
            "dense_score": point.score if mode == "dense" else None,
            "sparse_score": point.score if mode == "sparse" else None,
            "dense_rank": rank if mode == "dense" else None,
            "sparse_rank": rank if mode == "sparse" else None,
        }
        hits.append(
            RetrievalHit(
                **_payload_values(point),
                rank=rank,
                score=point.score,
                **branch_values,
            )
        )
    return hits


def _hybrid_hits(
    dense_points: Sequence[ScoredPoint],
    sparse_points: Sequence[ScoredPoint],
    config: RetrievalConfig,
) -> list[RetrievalHit]:
    merged: dict[int | str, dict[str, Any]] = {}

    for rank, point in enumerate(dense_points, start=1):
        point_id = point.id if isinstance(point.id, (int, str)) else str(point.id)
        merged[point_id] = {
            "point": point,
            "dense_score": point.score,
            "dense_rank": rank,
            "sparse_score": None,
            "sparse_rank": None,
            "rrf_score": config.dense_weight / (config.rrf_k + rank),
        }

    for rank, point in enumerate(sparse_points, start=1):
        point_id = point.id if isinstance(point.id, (int, str)) else str(point.id)
        entry = merged.setdefault(
            point_id,
            {
                "point": point,
                "dense_score": None,
                "dense_rank": None,
                "sparse_score": None,
                "sparse_rank": None,
                "rrf_score": 0.0,
            },
        )
        entry["sparse_score"] = point.score
        entry["sparse_rank"] = rank
        entry["rrf_score"] += config.sparse_weight / (config.rrf_k + rank)

    ordered = sorted(
        merged.values(),
        key=lambda entry: (-entry["rrf_score"], str(entry["point"].id)),
    )
    hits: list[RetrievalHit] = []
    for rank, entry in enumerate(ordered, start=1):
        point = entry["point"]
        hits.append(
            RetrievalHit(
                **_payload_values(point),
                rank=rank,
                score=entry["rrf_score"],
                dense_score=entry["dense_score"],
                sparse_score=entry["sparse_score"],
                dense_rank=entry["dense_rank"],
                sparse_rank=entry["sparse_rank"],
            )
        )
    return hits


class VehicleSpecRetriever:
    """Retrieve citation-ready chunks from an open hybrid Qdrant store."""

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        config: RetrievalConfig | None = None,
        *,
        reranker: Reranker | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.config = config or RetrievalConfig()
        self.reranker = reranker

    def _dense_points(
        self,
        query: str,
        query_filter: models.Filter | None,
    ) -> list[ScoredPoint]:
        embeddings = self.vector_store.embeddings
        if embeddings is None:
            raise RuntimeError("dense embeddings are not configured")
        vector = embeddings.embed_query(query)
        return self.vector_store.client.query_points(
            collection_name=self.vector_store.collection_name,
            query=vector,
            using=self.vector_store.vector_name,
            query_filter=query_filter,
            limit=self.config.candidate_k,
            score_threshold=self.config.score_threshold,
            with_payload=True,
            with_vectors=False,
        ).points

    def _sparse_points(
        self,
        query: str,
        query_filter: models.Filter | None,
    ) -> list[ScoredPoint]:
        vector = self.vector_store.sparse_embeddings.embed_query(query)
        return self.vector_store.client.query_points(
            collection_name=self.vector_store.collection_name,
            query=models.SparseVector(indices=vector.indices, values=vector.values),
            using=self.vector_store.sparse_vector_name,
            query_filter=query_filter,
            limit=self.config.candidate_k,
            score_threshold=self.config.score_threshold,
            with_payload=True,
            with_vectors=False,
        ).points

    def retrieve(
        self,
        query: str,
        *,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievalHit]:
        """Retrieve broad candidates, optionally rerank, then narrow context."""
        query = query.strip()

        query_filter = _qdrant_filter(filters)
        dense_points: list[ScoredPoint] = []
        sparse_points: list[ScoredPoint] = []
        if self.config.mode in {"dense", "hybrid"}:
            dense_points = self._dense_points(query, query_filter)
        if self.config.mode in {"sparse", "hybrid"}:
            sparse_points = self._sparse_points(query, query_filter)

        applicability = filters.applicability if filters else None
        dense_points = [
            point
            for point in dense_points
            if _matches_applicability(point, applicability)
        ]
        sparse_points = [
            point
            for point in sparse_points
            if _matches_applicability(point, applicability)
        ]

        if self.config.mode == "dense":
            hits = _single_mode_hits(dense_points, mode="dense")
        elif self.config.mode == "sparse":
            hits = _single_mode_hits(sparse_points, mode="sparse")
        else:
            hits = _hybrid_hits(dense_points, sparse_points, self.config)

        if self.reranker is not None:
            return self.reranker.rerank(
                query,
                hits,
                limit=self.config.context_k,
            )
        return hits[: self.config.context_k]
