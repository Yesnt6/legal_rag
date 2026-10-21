"""Generate a cited answer about indexed sources, without asserting current law."""

from contextlib import closing

from config import load_settings
from ingestion.manual_indexer import VersionedStore
from llm.answer_generator import AnswerGenerator
from llm.ollama_client import OllamaClient
from persistence.chroma_store import ChromaStore
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker
from retrieval.run_retriever import DATA_DIR

QUESTION = (
    "What is the nature of complaints relating to development?"
)


def main():
    settings = load_settings()
    path = DATA_DIR / "indexer_test.sqlite3"
    if not path.is_file():
        raise FileNotFoundError("Run python -m ingestion.run_indexer first.")
    store = VersionedStore(path)
    client = OllamaClient(
        settings.ollama_base_url,
        settings.embedding_model,
        settings.answer_timeout_seconds,
    )
    retriever = HybridRetriever(
        store,
        ChromaStore(DATA_DIR / "indexer_test_chroma", collection_name="indexer_test"),
        client,
        20,
        20,
        result_limit=15,
        reranker=CrossEncoderReranker(settings.reranker_model),
    )
    print(f"Question: {QUESTION}\nModel: {settings.main_model}", flush=True)
    print("Retrieving evidence...", flush=True)
    evidence = retriever.retrieve(QUESTION)
    print(f"Retrieved {len(evidence)} passages.", flush=True)
    result = AnswerGenerator(
        client,
        settings.main_model,
        progress=lambda message: print(message, flush=True),
    ).answer(QUESTION, evidence)
    print(result.answer)
    if result.abstained:
        print(f"Reason: {result.abstention_reason}")
    for number, citation in enumerate(result.citations, 1):
        with closing(store.connect()) as connection:
            row = connection.execute(
                "SELECT p.version_id, s.metadata_json FROM version_passages p "
                "JOIN source_versions s ON s.id=p.version_id WHERE p.id=?",
                (citation.passage_id,),
            ).fetchone()
        passage = store.get_passage(citation.passage_id)
        print(
            f"[{number}] Document: {citation.document_id}; "
            f"version: {row['version_id']}; "
            f"pages {citation.page_start}-{citation.page_end}"
        )
        print(f"Section: {' > '.join(passage.section_path)}")
        print(f"Exact source quote: {citation.quoted_text}")
    print(
        "Scope: supplied source documents only. Current "
        "legal validity is not established. "
        "Automated verification can be wrong; "
        "completeness and source conflicts need evaluation."
    )


if __name__ == "__main__":
    main()
