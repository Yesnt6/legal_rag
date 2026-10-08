from datetime import date
from pathlib import Path

from domain.schema import DocumentKind
from ingestion.manual_indexer import (
    DocumentMetadata,
    VersionedStore,
    index_document,
)
from llm.ollama_client import OllamaClient
from persistence.chroma_store import ChromaStore

BASE = Path(__file__).resolve().parent


def main() -> None:
    store = VersionedStore(BASE / "data/indexer_test.sqlite3")
    chroma = ChromaStore(
        BASE / "data/indexer_test_chroma",
        collection_name="indexer_test",
    )

    result = index_document(
        source_path=BASE / "../outputs/report_1.txt",
        metadata=DocumentMetadata(
            document_id="test-document",
            title="Directive for Redevelopment of Building of Co-operative Housing Society",
            kind=DocumentKind.CIRCULAR,
            issuing_authority="Government of Maharashtra",
            publication_date=date(2009, 1, 3),  # Replace with actual date.
            jurisdiction="Maharashtra, India",
        ),
        store=store,
        chroma=chroma,
        embedder=OllamaClient(
            base_url="http://localhost:11434",
            embedding_model="embeddinggemma",
            timeout_seconds=45,
        ),
    )

    passages = store.retrieval_passages(result.version_id)
    assert passages, "No searchable passages were stored."
    assert tuple(p.id for p in passages) == result.passage_ids

    indexed = chroma.collection.get(ids=list(result.passage_ids))
    assert set(indexed["ids"]) == set(result.passage_ids)

    print(f"Version: {result.version_id}")
    print(f"Reused existing version: {result.reused_version}")
    print(f"Passages in SQLite and Chroma: {len(passages)}")

    for passage in passages[:3]:
        print(f"\nPages {passage.page_start}–{passage.page_end}")
        print(f"Heading path: {passage.section_path}")
        print(passage.text[:300])


if __name__ == "__main__":
    main()