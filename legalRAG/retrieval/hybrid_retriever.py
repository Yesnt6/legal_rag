from domain.schema import RetrievedEvidence
from llm.ollama_client import OllamaClient
from persistence.chroma_store import ChromaStore
from persistence.sqlite_store import SQLiteStore


class HybridRetriever:
    def __init__(
        self,
        sqlite_store: SQLiteStore,
        chroma_store: ChromaStore,
        ollama_client: OllamaClient,
        vector_candidate_limit: int,
        keyword_candidate_limit: int,
    ) -> None:
        self.sqlite_store = sqlite_store
        self.chroma_store = chroma_store
        self.ollama_client = ollama_client
        self.vector_candidate_limit = vector_candidate_limit
        self.keyword_candidate_limit = keyword_candidate_limit

    def retrieve(self, question: str) -> list[RetrievedEvidence]:
        if not question.strip():
            return []

        query_embedding = self.ollama_client.embed([question])[0]
        evidence_by_passage_id: dict[str, RetrievedEvidence] = {}

        for passage_id, score in self.chroma_store.search(
            query_embedding, self.vector_candidate_limit
        ):
            passage = self.sqlite_store.get_passage(passage_id)
            if passage is not None:
                evidence_by_passage_id[passage.id] = RetrievedEvidence(
                    passage=passage,
                    score=score,
                    retrieval_source="vector",
                )

        for passage in self.sqlite_store.keyword_search(
            question, self.keyword_candidate_limit
        ):
            existing_evidence = evidence_by_passage_id.get(passage.id)
            if existing_evidence is None:
                evidence_by_passage_id[passage.id] = RetrievedEvidence(
                    passage=passage,
                    score=0.0,
                    retrieval_source="keyword",
                )
            else:
                existing_evidence.retrieval_source += "+keyword"

        return list(evidence_by_passage_id.values())
