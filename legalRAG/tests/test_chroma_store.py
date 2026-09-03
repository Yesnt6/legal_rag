from domain.schema import Passage
from persistence.chroma_store import ChromaStore


def test_upsert_and_search_passages(tmp_path) -> None:
    store = ChromaStore(tmp_path / "chroma")
    passages = [
        Passage(
            id="passage-1",
            document_id="document-1",
            text="Society must appoint a consultant.",
            page_start=4,
            page_end=4,
        ),
        Passage(
            id="passage-2",
            document_id="document-1",
            text="Members vote at a special general body meeting.",
            page_start=5,
            page_end=5,
        ),
    ]

    store.upsert_passages(passages, [[1.0, 0.0], [0.0, 1.0]])

    assert store.search([1.0, 0.0], limit=1) == [("passage-1", 1.0)]
