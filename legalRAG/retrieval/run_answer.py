"""Generate a cited answer about indexed sources, without asserting current law."""

from contextlib import closing

from config import load_settings
from ingestion.manual_indexer import VersionedStore
from llm.answer_generator import AnswerGenerator
from llm.ollama_client import OllamaClient
from persistence.chroma_store import ChromaStore
from retrieval.evaluate_retrieval import CASES
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker
from retrieval.run_retriever import DATA_DIR

# Choose "single" to run one question, or "all" to run all ten.
MODE = "all"
QUESTION_INDEX = 4  # One-based index into CASES.


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
        result_limit=20,
        reranker=CrossEncoderReranker(settings.reranker_model),
    )
    if MODE not in {"single", "all"}:
        raise ValueError("MODE must be 'single' or 'all'.")
    if not 1 <= QUESTION_INDEX <= len(CASES):
        raise ValueError(f"QUESTION_INDEX must be between 1 and {len(CASES)}.")
    questions = CASES if MODE == "all" else [CASES[QUESTION_INDEX - 1]]

    answer_generator = AnswerGenerator(
        client,
        settings.main_model,
        progress=lambda message: print(message, flush=True),
    )
    print(f"Mode: {MODE}; model: {settings.main_model}", flush=True)
    for question, _ in questions:
        print(f"\n{'=' * 80}\nQuestion: {question}", flush=True)
        print("Retrieving evidence...", flush=True)
        evidence = retriever.retrieve(question)
        print(f"Retrieved {len(evidence)} passages.", flush=True)
        for rank, item in enumerate(evidence, 1):
            passage = item.passage
            section = " > ".join(passage.section_path) or "(no section)"
            if len(section) > 90:
                section = f"{section[:87]}..."
            rerank = (
                f"{item.rerank_score:.3f}"
                if item.rerank_score is not None
                else "n/a"
            )
            print(
                f"  {rank:02d} id={passage.id[:12]} "
                f"pages={passage.page_start}-{passage.page_end} "
                f"source={item.retrieval_source} "
                f"rrf={item.score:.4f} rerank={rerank} "
                f"section={section}",
                flush=True,
            )
        result = answer_generator.answer(question, evidence)
        print(result.answer)
        if result.abstained:
            print(f"Reason: {result.abstention_reason}")
        for number, citation in enumerate(result.citations, 1):
            with closing(store.connect()) as connection:
                row = connection.execute(
                    "SELECT p.version_id FROM version_passages p "
                    "WHERE p.id=?",
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
