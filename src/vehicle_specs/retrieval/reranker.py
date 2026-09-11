"""Optional local cross-encoder reranking behind a small interface."""

from collections.abc import Iterable, Sequence
from typing import Protocol

from .models import RetrievalHit

DEFAULT_RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class Reranker(Protocol):
    """Interface used by retrieval orchestration."""

    model_name: str

    def rerank(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        *,
        limit: int,
    ) -> list[RetrievalHit]: ...


class _CrossEncoderBackend(Protocol):
    def rerank(
        self,
        query: str,
        documents: Iterable[str],
        *,
        batch_size: int,
    ) -> Iterable[float]: ...


class FastEmbedReranker:
    """Rerank retrieved chunks using a local FastEmbed cross-encoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        batch_size: int = 32,
        cache_dir: str | None = None,
        threads: int | None = None,
        _backend: _CrossEncoderBackend | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

        if _backend is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            supported = {
                description["model"]
                for description in TextCrossEncoder.list_supported_models()
            }
            if model_name not in supported:
                raise ValueError(
                    f"FastEmbed reranker model is not supported: {model_name}"
                )
            _backend = TextCrossEncoder(
                model_name=model_name,
                cache_dir=cache_dir,
                threads=threads,
                lazy_load=True,
            )
        self._backend = _backend

    def rerank(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        *,
        limit: int,
    ) -> list[RetrievalHit]:
        """Score every candidate, then return the strongest limited set."""
        if not hits:
            return []

        raw_scores = self._backend.rerank(
            query,
            [hit.text for hit in hits],
            batch_size=self.batch_size,
        )
        scores = [float(score) for score in raw_scores]

        scored = [
            hit.model_copy(update={"rerank_score": score})
            for hit, score in zip(hits, scores, strict=True)
        ]
        scored.sort(key=lambda hit: (-hit.effective_score, hit.rank, str(hit.point_id)))
        return [
            hit.model_copy(update={"rank": rank})
            for rank, hit in enumerate(scored[:limit], start=1)
        ]
