import math
from typing import Protocol

from domain.schema import RetrievedEvidence
from ingestion.manual_indexer import Embedder, VersionedStore
from persistence.chroma_store import ChromaStore


class Reranker(Protocol):
    def score(self, question: str, passages: list[str]) -> list[float]: ...


class HybridRetriever:
    """Merge current-version vector and BM25 candidates using reciprocal ranks."""

    def __init__(
        self,
        sqlite_store: VersionedStore,
        chroma_store: ChromaStore,
        ollama_client: Embedder,
        vector_candidate_limit: int,
        keyword_candidate_limit: int,
        result_limit: int = 10,
        reranker: Reranker | None = None,
        rerank_candidate_limit: int = 30,
    ) -> None:
        if min(vector_candidate_limit, keyword_candidate_limit, result_limit) < 1:
            raise ValueError("Retrieval limits must be at least 1.")
        if reranker is not None and rerank_candidate_limit < result_limit:
            raise ValueError("Reranking candidate limit must cover the result limit.")
        self.reranker = reranker
        self.rerank_candidate_limit = rerank_candidate_limit
        self.sqlite_store = sqlite_store
        self.chroma_store = chroma_store
        self.ollama_client = ollama_client
        self.vector_candidate_limit = vector_candidate_limit
        self.keyword_candidate_limit = keyword_candidate_limit
        self.result_limit = result_limit

    def retrieve(self, question: str) -> list[RetrievedEvidence]:
        question = question.strip()
        if not question:
            return []

        # Capture one eligibility list so both channels search the same versions.
        passage_ids = self.sqlite_store.latest_passage_ids()
        if not passage_ids:
            return []
        query_embedding = self.ollama_client.embed([question])[0]
        vector_results = self.chroma_store.search(
            query_embedding, self.vector_candidate_limit, passage_ids=passage_ids
        )
        keyword_results = self.sqlite_store.keyword_search(
            question, self.keyword_candidate_limit, passage_ids=passage_ids
        )
        evidence_by_id: dict[str, RetrievedEvidence] = {}
        channels = (
            ("vector", [passage_id for passage_id, _ in vector_results]),
            ("keyword", [passage.id for passage in keyword_results]),
        )
        for source, ranked_ids in channels:
            for rank, passage_id in enumerate(ranked_ids, start=1):
                # RRF combines ranks, not incompatible cosine and BM25 scores.
                contribution = 1.0 / (60 + rank)
                existing = evidence_by_id.get(passage_id)
                if existing is not None:
                    existing.score += contribution
                    existing.retrieval_source += f"+{source}"
                    continue
                passage = self.sqlite_store.get_passage(passage_id)
                if passage is not None:
                    evidence_by_id[passage_id] = RetrievedEvidence(
                        passage=passage, score=contribution, retrieval_source=source
                    )

        ranked = sorted(
            evidence_by_id.values(),
            key=lambda item: (-item.score, item.passage.id),
        )
        if self.reranker is not None and ranked:
            ranked = ranked[: self.rerank_candidate_limit]
            texts = [
                "\n".join(
                    filter(
                        None, [" > ".join(item.passage.section_path), item.passage.text]
                    )
                )
                for item in ranked
            ]
            scores = self.reranker.score(question, texts)
            if len(scores) != len(ranked) or not all(map(math.isfinite, scores)):
                raise ValueError("Reranker must return one finite score per candidate.")
            for item, score in zip(ranked, scores, strict=True):
                item.rerank_score = score
            # Stable ties retain the original RRF ordering.
            ranked = [
                item
                for _, item in sorted(
                    zip(scores, ranked, strict=True), key=lambda pair: -pair[0]
                )
            ]
        return ranked[: self.result_limit]
