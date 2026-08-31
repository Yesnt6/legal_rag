from pathlib import Path

from domain.models import Document, DocumentKind, Passage
from persistence.sqlite_store import SQLiteStore


def test_store_round_trip_and_keyword_search(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "legal_rag.sqlite3")
    store.create_schema()

    document = Document(
        id="document-1",
        title="Society Self-Redevelopment Guide",
        kind=DocumentKind.CIRCULAR,
        source_path=Path("documents/guide.pdf"),
    )
    passage = Passage(
        id="passage-1",
        document_id=document.id,
        text="The society must appoint a project management consultant.",
        page_start=4,
        page_end=4,
        section_path=["Chapter 2", "Consultant appointment"],
    )

    store.upsert_document(document)
    store.upsert_passage(passage)

    assert store.get_passage(passage.id) == passage
    assert store.keyword_search("CONSULTANT", limit=10) == [passage]
