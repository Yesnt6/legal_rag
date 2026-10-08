from pathlib import Path

import pytest
from test_versioned_search import passage, save_version

from ingestion.manual_indexer import VersionedStore
from persistence.chroma_store import ChromaStore
from retrieval.hybrid_retriever import HybridRetriever


class FakeOllamaClient:
    def embed(self, texts: list[str]) -> list[list[float]]:
        assert texts == ["bank guarantee"]
        return [[1.0, 0.0]]


def test_retrieve_filters_versions_before_limits_and_fuses_ranks(tmp_path: Path):
    store = VersionedStore(tmp_path / "sources.sqlite3")
    chroma = ChromaStore(tmp_path / "chroma")
    old = passage("old", "bank guarantee")
    save_version(store, "v1", [old])
    current = [
        passage("semantic", "Financial security must be provided."),
        passage("both", "bank guarantee"),
        passage("keyword", "The bank opens an account."),
    ]
    save_version(store, "v2", current)
    chroma.upsert_passages(
        [old, passage("orphan", "bank guarantee"), *current],
        [[1.0, 0.0], [1.0, 0.0], [0.99, 0.1], [0.8, 0.2], [0.0, 1.0]],
    )
    retriever = HybridRetriever(store, chroma, FakeOllamaClient(), 2, 2)

    evidence = retriever.retrieve("bank guarantee")

    assert [item.passage.id for item in evidence] == ["both", "semantic", "keyword"]
    assert [item.retrieval_source for item in evidence] == [
        "vector+keyword",
        "vector",
        "keyword",
    ]
    assert [item.score for item in evidence] == pytest.approx(
        [1 / 62 + 1 / 61, 1 / 61, 1 / 62]
    )
    assert retriever.retrieve("  ") == []
    limited = HybridRetriever(store, chroma, FakeOllamaClient(), 2, 2, result_limit=1)
    assert [e.passage.id for e in limited.retrieve("bank guarantee")] == ["both"]


def test_empty_versioned_store_returns_no_evidence(tmp_path: Path):
    store = VersionedStore(tmp_path / "sources.sqlite3")
    chroma = ChromaStore(tmp_path / "chroma")
    retriever = HybridRetriever(store, chroma, FakeOllamaClient(), 2, 2)
    # The fake rejects this query if embedding is unnecessarily attempted.
    assert retriever.retrieve("empty corpus") == []
