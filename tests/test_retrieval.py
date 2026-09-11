import re
from pathlib import Path
from typing import ClassVar

from langchain_core.embeddings import Embeddings
from langchain_qdrant.sparse_embeddings import SparseEmbeddings, SparseVector

from vehicle_specs.chunking import TextChunk
from vehicle_specs.indexing import (
    IndexConfig,
    PipelineVersions,
    build_index,
    open_index,
)
from vehicle_specs.pdf import SectionContext
from vehicle_specs.retrieval import (
    FastEmbedReranker,
    RetrievalConfig,
    RetrievalFilters,
    VehicleSpecRetriever,
)


class KeywordDenseEmbeddings(Embeddings):
    model_name = "keyword-dense"
    dimension = 3

    @staticmethod
    def _embed(text: str) -> list[float]:
        lowered = text.casefold()
        if "front" in lowered:
            return [1.0, 0.0, 0.0]
        if "rear" in lowered:
            return [0.0, 1.0, 0.0]
        if "fl-500-s" in lowered:
            return [0.0, 0.0, 1.0]
        return [0.7, 0.7, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class KeywordSparseEmbeddings(SparseEmbeddings):
    model_name = "keyword-sparse"
    vocabulary: ClassVar[dict[str, int]] = {
        "front": 1,
        "rear": 2,
        "brake": 3,
        "caliper": 4,
        "guide": 5,
        "pin": 6,
        "bolts": 7,
        "fl-500-s": 8,
    }

    @classmethod
    def _embed(cls, text: str) -> SparseVector:
        tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.casefold())
        counts: dict[int, float] = {}
        for token in tokens:
            if token in cls.vocabulary:
                index = cls.vocabulary[token]
                counts[index] = counts.get(index, 0.0) + 1.0
        return SparseVector(indices=list(counts), values=list(counts.values()))

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> SparseVector:
        return self._embed(text)


def make_chunk(
    chunk_id: str,
    text: str,
    page: int,
    section_id: str,
    section_title: str,
) -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id,
        source="manual.pdf",
        kind="table",
        text=text,
        pdf_pages=(page,),
        section=SectionContext(
            section_id=section_id,
            section_title=section_title,
            category="SPECIFICATIONS",
            article_title="Specifications",
            start_pdf_page=page,
        ),
    )


def test_dense_sparse_hybrid_and_filtered_retrieval(tmp_path: Path) -> None:
    dense = KeywordDenseEmbeddings()
    sparse = KeywordSparseEmbeddings()
    config = IndexConfig(
        path=tmp_path / "qdrant",
        collection_name="retrieval_test",
        dense_model=dense.model_name,
        sparse_model=sparse.model_name,
        pipeline_versions=PipelineVersions(),
    )
    chunks = [
        make_chunk(
            "front-guide-pin",
            "Front brake caliper guide pin bolts: 37 Nm",
            636,
            "206-03",
            "Front Disc Brake",
        ),
        make_chunk(
            "rear-guide-pin",
            "Rear brake caliper guide pin bolts: 33 Nm",
            652,
            "206-04",
            "Rear Disc Brake",
        ),
        make_chunk(
            "oil-filter-part",
            "Engine oil filter part number: FL-500-S",
            110,
            "303-01",
            "Engine",
        ),
    ]
    build_index(
        chunks,
        config,
        dense_embeddings=dense,
        sparse_embeddings=sparse,
    )

    with open_index(
        config,
        dense_embeddings=dense,
        sparse_embeddings=sparse,
    ) as store:
        dense_hits = VehicleSpecRetriever(
            store,
            RetrievalConfig(mode="dense", candidate_k=3, context_k=2),
        ).retrieve("front stopping hardware")
        assert dense_hits[0].chunk_id == "front-guide-pin"
        assert dense_hits[0].dense_score is not None
        assert dense_hits[0].sparse_score is None

        sparse_hits = VehicleSpecRetriever(
            store,
            RetrievalConfig(mode="sparse", candidate_k=3, context_k=2),
        ).retrieve("FL-500-S")
        assert sparse_hits[0].chunk_id == "oil-filter-part"
        assert sparse_hits[0].sparse_score is not None

        hybrid_hits = VehicleSpecRetriever(
            store,
            RetrievalConfig(mode="hybrid", candidate_k=3, context_k=3),
        ).retrieve("brake caliper guide pin bolts")
        assert {hit.chunk_id for hit in hybrid_hits[:2]} == {
            "front-guide-pin",
            "rear-guide-pin",
        }
        assert all(hit.dense_rank is not None for hit in hybrid_hits[:2])
        assert all(hit.sparse_rank is not None for hit in hybrid_hits[:2])

        rear_only = VehicleSpecRetriever(
            store,
            RetrievalConfig(candidate_k=3, context_k=3),
        ).retrieve(
            "brake caliper guide pin bolts",
            filters=RetrievalFilters(section_ids=("206-04",)),
        )
        assert [hit.chunk_id for hit in rear_only] == ["rear-guide-pin"]

        front_only = VehicleSpecRetriever(
            store,
            RetrievalConfig(candidate_k=3, context_k=3),
        ).retrieve(
            "brake caliper guide pin bolts",
            filters=RetrievalFilters(applicability="front"),
        )
        assert [hit.chunk_id for hit in front_only] == ["front-guide-pin"]


class FakeCrossEncoder:
    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        batch_size: int,
    ) -> list[float]:
        assert query == "guide pin"
        assert batch_size == 4
        return [0.1, 0.9]


def test_cross_encoder_reranker_reorders_and_limits_hits() -> None:
    versions = PipelineVersions()
    hits = [
        {
            "point_id": ordinal,
            "chunk_id": f"chunk-{ordinal}",
            "text": f"text {ordinal}",
            "source": "manual.pdf",
            "kind": "prose",
            "pdf_pages": (ordinal,),
            "pipeline_versions": versions,
            "rank": ordinal,
            "score": 1 / ordinal,
        }
        for ordinal in (1, 2)
    ]
    from vehicle_specs.retrieval import RetrievalHit

    reranker = FastEmbedReranker(
        batch_size=4,
        _backend=FakeCrossEncoder(),
    )
    reranked = reranker.rerank(
        "guide pin",
        [RetrievalHit(**hit) for hit in hits],
        limit=1,
    )

    assert len(reranked) == 1
    assert reranked[0].chunk_id == "chunk-2"
    assert reranked[0].rank == 1
    assert reranked[0].rerank_score == 0.9
