"""Search the versioned SQLite/Chroma corpus created by ingestion.run_indexer."""

from pathlib import Path

from ingestion.manual_indexer import VersionedStore
from llm.ollama_client import OllamaClient
from persistence.chroma_store import ChromaStore
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "ingestion/data"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_DEVICE = None  # Automatic device selection; set "cpu" if needed.
QUESTION = "What bank guarantee must the developer provide?"


def main() -> None:
    sqlite_path = DATA_DIR / "indexer_test.sqlite3"
    chroma_path = DATA_DIR / "indexer_test_chroma"
    if not sqlite_path.is_file() or not chroma_path.is_dir():
        raise FileNotFoundError(
            f"Indexed data not found in {DATA_DIR}. "
            "Run python -m ingestion.run_indexer first, or edit DATA_DIR."
        )
    store = VersionedStore(sqlite_path)
    retriever = HybridRetriever(
        sqlite_store=store,
        chroma_store=ChromaStore(chroma_path, collection_name="indexer_test"),
        ollama_client=OllamaClient(
            base_url="http://localhost:11434",
            embedding_model="embeddinggemma",
            timeout_seconds=45,
        ),
        vector_candidate_limit=20,
        keyword_candidate_limit=20,
        result_limit=5,
        reranker=CrossEncoderReranker(RERANK_MODEL, device=RERANK_DEVICE),
        rerank_candidate_limit=30,
    )
    print(f"Question: {QUESTION}")
    print(f"Reranker: {RERANK_MODEL} (first run may download model weights)")
    evidence = retriever.retrieve(QUESTION)
    if not evidence:
        print("No passages found.")
    for rank, item in enumerate(evidence, start=1):
        passage = item.passage
        print(f"\n{rank}. {passage.id}")
        print(f"Origin: {item.retrieval_source} | RRF score: {item.score:.6f}")
        if item.rerank_score is not None:
            print(f"Reranker score: {item.rerank_score:.6f}")
        print(f"Document: {passage.document_id}")
        print(f"Pages {passage.page_start}-{passage.page_end}")
        print(f"Heading: {' > '.join(passage.section_path) or '(none)'}")
        print(passage.text)


if __name__ == "__main__":
    main()
