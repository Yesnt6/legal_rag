from datetime import date
from pathlib import Path

from domain.schema import DocumentKind, Passage
from ingestion.manual_indexer import DocumentMetadata, IndexedPassage, VersionedStore


def save_version(
    store: VersionedStore,
    version_id: str,
    passages: list[Passage],
    *,
    document_id: str = "document-1",
    retrievable: bool = True,
) -> None:
    metadata = DocumentMetadata(
        document_id=document_id,
        title="Test document",
        kind=DocumentKind.CIRCULAR,
        issuing_authority="Test authority",
        publication_date=date(2009, 1, 3),
        jurisdiction="Maharashtra, India",
    )
    store.save(
        metadata,
        version_id.encode(),
        version_id,
        version_id,
        [
            IndexedPassage(
                p.model_copy(update={"document_id": document_id}),
                "RETRIEVAL",
                0,
                len(p.text),
                retrievable,
            )
            for p in passages
        ],
    )


def passage(passage_id: str, text: str, **kwargs) -> Passage:
    return Passage(
        id=passage_id,
        document_id="document-1",
        text=text,
        page_start=1,
        page_end=1,
        **kwargs,
    )


def test_keyword_search_matches_terms_and_headings_in_bm25_order(tmp_path: Path):
    store = VersionedStore(tmp_path / "sources.sqlite3")
    save_version(
        store,
        "v1",
        [
            passage("irrelevant", "Members meet annually."),
            passage("partial", "The bank opens an account."),
            passage("best", "A bank guarantee must be supplied."),
            passage("heading", "Appoint an expert.", section_path=["Consultant"]),
        ],
    )

    results = store.keyword_search("What bank guarantee is required?", limit=10)
    assert [p.id for p in results] == ["best", "partial"]
    assert [p.id for p in store.keyword_search("consultant", 10)] == ["heading"]


def test_only_latest_retrievable_passages_are_searchable(tmp_path: Path):
    store = VersionedStore(tmp_path / "sources.sqlite3")
    save_version(store, "v1", [passage("old", "bank guarantee")])
    save_version(store, "v2", [passage("new", "bank guarantee")])
    save_version(
        store,
        "parent-v1",
        [passage("parent", "bank guarantee")],
        document_id="parent-document",
        retrievable=False,
    )
    assert store.latest_passage_ids() == ["new"]
    assert [p.id for p in store.keyword_search("bank guarantee", 1)] == ["new"]
    assert store.get_passage("old").text == "bank guarantee"


def test_backfills_existing_passages_and_reopening_does_not_duplicate(tmp_path: Path):
    store = VersionedStore(tmp_path / "sources.sqlite3")
    save_version(store, "v1", [passage("p1", "bank guarantee")])
    # Simulate the pre-FTS schema, retaining the existing authoritative passages.
    with store.connect() as connection:
        connection.executescript(
            "DROP TRIGGER version_passages_fts_insert; DROP TABLE passage_fts;"
        )
    for _ in range(2):
        store = VersionedStore(store.path)
        assert [p.id for p in store.keyword_search("bank", 10)] == ["p1"]


def test_keyword_search_handles_punctuation_and_fts_operators(tmp_path: Path):
    store = VersionedStore(tmp_path / "sources.sqlite3")
    save_version(store, "v1", [passage("p1", "bank guarantee")])
    assert store.keyword_search('!?* " ()', 10) == []
    assert [p.id for p in store.keyword_search('bank: "guarantee" OR *', 10)] == ["p1"]


def test_latest_selection_is_per_document(tmp_path: Path):
    store = VersionedStore(tmp_path / "sources.sqlite3")
    save_version(store, "a1", [passage("a-old", "bank guarantee")])
    save_version(store, "a2", [passage("a-new", "bank guarantee")])
    save_version(
        store, "b1", [passage("b-current", "bank guarantee")], document_id="document-2"
    )
    assert store.latest_passage_ids() == ["a-new", "b-current"]
    assert {p.id for p in store.keyword_search("bank", 10)} == {"a-new", "b-current"}
