import hashlib
from datetime import date
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest
from test_segmentation import SOURCE

from domain.schema import DocumentKind, Passage
from ingestion.manual_indexer import (
    DocumentMetadata,
    IndexedPassage,
    VersionedStore,
    index_document,
    parse_pages,
)
from persistence.chroma_store import ChromaStore


class Embeddings:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FailedEmbeddings:
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding unavailable")


def test_reindex_keeps_source_and_old_passages_then_activates_new_profile(
    tmp_path: Path,
):
    source = tmp_path / "source.txt"
    source.write_text(SOURCE)
    store = VersionedStore(tmp_path / "source.sqlite3")
    chroma = ChromaStore(tmp_path / "chroma")
    metadata = DocumentMetadata(
        "doc",
        "Title",
        DocumentKind.CIRCULAR,
        "Authority",
        date(2009, 1, 3),
        "Maharashtra",
    )
    text, _ = parse_pages(SOURCE)
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    version_id = str(uuid5(NAMESPACE_URL, f"doc:{checksum}"))
    old = Passage(id="legacy", document_id="doc", text=text, page_start=1, page_end=2)
    store.save(
        metadata,
        source.read_bytes(),
        text,
        version_id,
        [IndexedPassage(old, "RETRIEVAL", 0, len(text), True)],
    )
    chroma.upsert_passages([old], [[1.0, 0.0]])

    with pytest.raises(RuntimeError, match="embedding unavailable"):
        index_document(source, metadata, store, chroma, FailedEmbeddings())
    assert store.latest_passage_ids() == ["legacy"]

    result = index_document(source, metadata, store, chroma, Embeddings())
    assert result.reused_version
    assert result.version_id == version_id
    assert "legacy" not in result.passage_ids
    assert set(store.latest_passage_ids()) == set(result.passage_ids)
    assert store.get_passage("legacy") == old
    assert store.list_versions("doc") == [version_id]
    bank = [
        p for p in store.retrieval_passages(version_id) if "Bank Guarantee" in p.text
    ]
    assert len(bank) == 1 and "(3)" not in bank[0].text
    assert set(chroma.collection.get(ids=list(result.passage_ids))["ids"]) == set(
        result.passage_ids
    )
    assert index_document(source, metadata, store, chroma, Embeddings()) == result

    shorter = index_document(
        source, metadata, store, chroma, Embeddings(), max_chars=60
    )
    assert set(shorter.passage_ids).isdisjoint(result.passage_ids)
    assert set(store.latest_passage_ids()) == set(shorter.passage_ids)
    assert all(store.get_passage(pid) is not None for pid in result.passage_ids)


def test_legacy_database_migration_preserves_existing_passages(tmp_path: Path):
    import sqlite3

    path = tmp_path / "legacy.sqlite3"
    source_text = "Legacy bank guarantee."
    old = Passage(
        id="old", document_id="doc", text=source_text, page_start=1, page_end=1
    )
    # Exact pre-profile table shape, populated before VersionedStore opens it.
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE source_versions (
                id TEXT PRIMARY KEY, document_id TEXT NOT NULL,
                version_number INTEGER NOT NULL, sha256 TEXT UNIQUE NOT NULL,
                source_bytes BLOB NOT NULL, normalized_text TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(document_id, version_number)
            );
            CREATE TABLE version_passages (
                id TEXT PRIMARY KEY, version_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL, kind TEXT NOT NULL,
                char_start INTEGER NOT NULL, char_end INTEGER NOT NULL,
                retrievable INTEGER NOT NULL, passage_json TEXT NOT NULL,
                UNIQUE(version_id, ordinal)
            );
        """)
        connection.execute(
            "INSERT INTO source_versions(id,document_id,version_number,sha256,"
            "source_bytes,normalized_text,metadata_json) VALUES(?,?,?,?,?,?,?)",
            ("v", "doc", 1, "checksum", b"source", source_text, "{}"),
        )
        connection.execute(
            "INSERT INTO version_passages VALUES(?,?,?,?,?,?,?,?)",
            ("old", "v", 0, "RETRIEVAL", 0, len(source_text), 1, old.model_dump_json()),
        )
    for _ in range(2):
        store = VersionedStore(path)
        assert store.latest_passage_ids() == ["old"]
        assert store.retrieval_passages("v") == [old]
        assert store.keyword_search("guarantee", 10) == [old]


def test_failed_new_source_version_keeps_previous_version_searchable(tmp_path: Path):
    source = tmp_path / "source.txt"
    source.write_text(SOURCE)
    store = VersionedStore(tmp_path / "source.sqlite3")
    chroma = ChromaStore(tmp_path / "chroma")
    metadata = DocumentMetadata(
        "doc",
        "Title",
        DocumentKind.CIRCULAR,
        "Authority",
        date(2009, 1, 3),
        "Maharashtra",
    )
    original = index_document(source, metadata, store, chroma, Embeddings())
    source.write_text(SOURCE.replace("20%", "30%"))
    with pytest.raises(RuntimeError):
        index_document(source, metadata, store, chroma, FailedEmbeddings())
    store = VersionedStore(store.path)
    assert set(store.latest_passage_ids()) == set(original.passage_ids)
    updated = index_document(source, metadata, store, chroma, Embeddings())
    assert updated.version_id != original.version_id
    assert set(store.latest_passage_ids()) == set(updated.passage_ids)
