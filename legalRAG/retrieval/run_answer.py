"""Generate a cited answer about indexed sources, without asserting current law."""

import gc
import json
from contextlib import closing

from config import load_settings
from domain.schema import Answer
from ingestion.manual_indexer import VersionedStore
from llm.answer_generator import AnswerGenerator
from llm.ollama_client import OllamaClient
from persistence.chroma_store import ChromaStore
from retrieval.evaluate_retrieval import CASES
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker
from retrieval.run_retriever import DATA_DIR

# Choose "single" to run one question, or "all" to run all ten.
MODE = "single"
QUESTION_INDEX = 3  # One-based index into CASES.
ANSWER_PASSAGE_LIMIT = 8
ANSWER_CONTEXT_CHARS = 7000
SHOW_DEBUG = True


def print_answer(result: Answer, store: VersionedStore) -> None:
    print("\nLLM Response:")
    print(result.answer)
    if result.abstained:
        print(f"Verification: {result.abstention_reason}")

    print("\nSources:")
    if not result.citations:
        print("None verified.")

    by_passage = {}
    for number, citation in enumerate(result.citations, start=1):
        by_passage.setdefault(citation.passage_id, []).append((number, citation))

    for passage_id, references in by_passage.items():
        with closing(store.connect()) as connection:
            row = connection.execute(
                "SELECT p.version_id, v.metadata_json "
                "FROM version_passages p "
                "JOIN source_versions v ON v.id = p.version_id "
                "WHERE p.id = ?",
                (passage_id,),
            ).fetchone()
        passage = store.get_passage(passage_id)
        if row is None or passage is None:
            raise ValueError(f"Cited passage {passage_id} is missing from SQLite.")
        metadata = json.loads(row["metadata_json"])
        numbers = ", ".join(f"[{number}]" for number, _ in references)
        print(f"\nSource {numbers}:")
        print(f"Document: {metadata['title']} (ID: {passage.document_id})")
        print(f"Authority: {metadata['issuing_authority']}")
        print(f"Publication date: {metadata['publication_date']}")
        print(f"Version: {row['version_id']}")
        print(f"Pages: {passage.page_start}–{passage.page_end}")
        print(f"Section: {' > '.join(passage.section_path) or '(no section)'}")
        print("Source passage (exact indexed text):")
        print(passage.text)
        print("Cited excerpts:")
        for number, citation in references:
            print(f"[{number}] {citation.quoted_text}")

    print(
        "\nScope: Indexed source documents only; "
        "current legal validity is not verified."
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
        embedding_keep_alive="0",
        generation_context_tokens=4096,
    )
    if MODE not in {"single", "all"}:
        raise ValueError("MODE must be 'single' or 'all'.")
    if not 1 <= QUESTION_INDEX <= len(CASES):
        raise ValueError(f"QUESTION_INDEX must be between 1 and {len(CASES)}.")
    questions = CASES if MODE == "all" else [CASES[QUESTION_INDEX - 1]]

    answer_generator = AnswerGenerator(
        client,
        settings.main_model,
        max_context_chars=ANSWER_CONTEXT_CHARS,
        progress=(lambda message: print(message, flush=True)) if SHOW_DEBUG else None,
    )
    if SHOW_DEBUG:
        print(f"Mode: {MODE}; model: {settings.main_model}", flush=True)
    for question, _ in questions:
        retriever = HybridRetriever(
            store,
            ChromaStore(
                DATA_DIR / "indexer_test_chroma", collection_name="indexer_test"
            ),
            client,
            20,
            20,
            result_limit=20,
            reranker=CrossEncoderReranker(settings.reranker_model),
        )
        print(f"\n{'=' * 80}\nQuestion: {question}", flush=True)
        print(f"Model: {settings.main_model}", flush=True)
        print("Retrieving evidence...", flush=True)
        evidence = retriever.retrieve(question)
        del retriever
        gc.collect()
        if SHOW_DEBUG:
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
        print("Generating answer...", flush=True)
        result = answer_generator.answer(question, evidence[:ANSWER_PASSAGE_LIMIT])
        print_answer(result, store)


if __name__ == "__main__":
    main()
