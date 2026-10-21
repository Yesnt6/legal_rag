from domain.schema import RetrievedEvidence
from ingestion.manual_indexer import parse_pages, segment
from retrieval.hybrid_retriever import HybridRetriever


def list_items(limit=1800):
    text, pages = parse_pages(
        "[PAGE 1]\nFollowing duties:\n1. Inspect the site.\n2. Submit the report."
    )
    return segment("doc", "v", text, pages, limit)


def test_complete_list_and_individual_items_are_searchable():
    items = list_items()
    group = next(i for i in items if i.kind == "LIST")
    assert group.retrievable
    assert "1. Inspect" in group.passage.text and "2. Submit" in group.passage.text
    assert len([i for i in items if i.kind == "RETRIEVAL"]) == 2


def test_oversized_list_is_not_embedded_as_a_complete_passage():
    group = next(i for i in list_items(25) if i.kind == "LIST")
    assert not group.retrievable


def test_highest_ranked_granularity_wins_without_duplicate_children():
    items = list_items()
    passages = {i.passage.id: i.passage for i in items}

    class Store:
        def get_passage(self, pid):
            return passages.get(pid)

    retriever = HybridRetriever(Store(), None, None, 20, 20, result_limit=5)
    group = next(i.passage for i in items if i.kind == "LIST")
    children = [i.passage for i in items if i.kind == "RETRIEVAL"]

    def evidence(ps):
        return [
            RetrievedEvidence(passage=p, score=1, retrieval_source="vector") for p in ps
        ]

    assert [
        e.passage.id for e in retriever.select_evidence(evidence([group, *children]))
    ] == [group.id]
    # A focused child ranked first must not be replaced by its lower-ranked parent.
    assert [
        e.passage.id
        for e in retriever.select_evidence(evidence([children[0], group, children[1]]))
    ] == [p.id for p in children]
