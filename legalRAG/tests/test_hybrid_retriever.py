from pathlib import Path

from domain.schema import Document, DocumentKind, Passage
from persistence.chroma_store import ChromaStore
from persistence.sqlite_store import SQLiteStore
from retrieval.hybrid_retriever import HybridRetriever


class FakeOllamaClient:
    def embed(self, texts: list[str]) -> list[list[float]]:
        assert texts == ["approval"]
        return [[1.0, 0.0]]


def test_retrieve_combines_and_deduplicates_candidates(tmp_path: Path) -> None:
    sqlite_store = SQLiteStore(tmp_path / "legal_rag.sqlite3")
    sqlite_store.create_schema()
    chroma_store = ChromaStore(tmp_path / "chroma")
    document = Document(
        id="document-1",
        title="Society Self-Redevelopment Guide",
        kind=DocumentKind.CIRCULAR,
        source_path=Path("documents/guide.pdf"),
    )
    passages = [
        Passage(
            id="passage-1",
            document_id=document.id,
            text="The committee needs approval before appointing a consultant.",
            page_start=4,
            page_end=4,
        ),
        Passage(
            id="passage-2",
            document_id=document.id,
            text="Member approval is recorded at the special general body meeting.",
            page_start=5,
            page_end=5,
        ),
    ]
    sqlite_store.upsert_document(document)
    for passage in passages:
        sqlite_store.upsert_passage(passage)
    chroma_store.upsert_passages(passages, [[1.0, 0.0], [0.0, 1.0]])
    retriever = HybridRetriever(
        sqlite_store=sqlite_store,
        chroma_store=chroma_store,
        ollama_client=FakeOllamaClient(),
        vector_candidate_limit=1,
        keyword_candidate_limit=10,
    )

    evidence = retriever.retrieve("approval")

    assert [item.passage.id for item in evidence] == ["passage-1", "passage-2"]
    assert evidence[0].retrieval_source == "vector+keyword"
    assert evidence[1].retrieval_source == "keyword"
