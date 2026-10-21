"""Evaluate the user's ten questions against the indexed 2009 directive.

Gold excerpts are manually selected from report_1.txt, independently of rankings.
This is a development set, not a held-out test of general legal accuracy.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from ingestion.manual_indexer import VersionedStore
from llm.ollama_client import OllamaClient
from persistence.chroma_store import ChromaStore
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker
from retrieval.run_retriever import DATA_DIR, RERANK_DEVICE, RERANK_MODEL

CASES = [
    (
        "Nature of complaints relating to development",
        [
            ["Not taking the members in confidence"],
            ["There is no transparency in tender"],
            ["Appointing contractors arbitrarily"],
            ["To work by violating provisions"],
            ["No orderliness in the work of Architect"],
            ["Not planning Redevelopment Project Report"],
            ["Not adopting proper procedure"],
            ["There is no similarity in agreements"],
        ],
    ),
    (
        "What is the requisition for convening Speical General "
        "Body Meeting for Redevelopment",
        [
            ["Not less that ¼ members"],
        ],
    ),
    (
        "What are the list of things the society must do before convening the meeting",
        [
            ["On receipt of an application as per Directive No. 1"],
            ["Before convening the said meeting"],
            ["Members of the Society will be entitled to submit in writing"],
        ],
    ),
    (
        "What is the quorum for the special general body "
        "meeting? What happens in the case where quorum is not formed",
        [
            [
                "Quorum for the Special General Body Meeting convened "
                "for redevelopment",
                "If the quorum of ¾ members out of total members",
            ],
        ],
    ),
    (
        "What is the work to be done by the Architect in the initial stages?",
        [
            ["To survey Society’s building and land"],
            ["To obtain information about conveyance"],
            ["To take into consideration prevailing policy"],
            ["To take into consideration suggestions and recommendations"],
            ["should prepare the project report within two months"],
        ],
    ),
    (
        "What are the actions after receiving the project report",
        [
            ["On receipt of Redevelopment Project Report as above"],
            ["Seven days prior to joint meeting"],
            ["There will be a detailed discussion in the Joint meeting"],
        ],
    ),
    (
        "What business will be transacted in the special "
        "general body meeting for redevelopment",
        [
            ["To take preliminary decision", "On formation of quorum for the meeting"],
            [
                "To select expert and experienced Architect",
                "To selected expert and experienced Architect",
            ],
            [
                "To submit outline of the programme",
                "To submit an outline of the programme",
            ],
        ],
    ),
    (
        "What will the builder do for the residents during the period of redevelopment",
        [
            ["During the period of redevelopment, the Developer will make available"],
        ],
    ),
    (
        "Who all should be present during the proceedings of "
        "the special general body meeting",
        [
            ["After appointment of authorized officer, with his prior permission"],
            [
                "In the Special General Body Meeting to be convened "
                "for selection of Developer"
            ],
        ],
    ),
    (
        "What are the actions to be taken after the receipt of "
        "the Occupation Certificate?",
        [
            ["After receipt of Occupation Certificate"],
        ],
    ),
]


def normalize(text):
    return " ".join(text.split())


def coverage(ids, groups):
    return sum(bool(set(ids) & set(group)) for group in groups) / len(groups)


def metrics(items, groups):
    ids = [item.passage.id for item in items]
    relevant = set().union(*map(set, groups))
    rank = next((i for i, pid in enumerate(ids, 1) if pid in relevant), None)
    return {
        "hit_at_1": bool(ids and ids[0] in relevant),
        "hit_at_5": bool(set(ids[:5]) & relevant),
        "coverage_at_5": coverage(ids[:5], groups),
        "complete_at_5": coverage(ids[:5], groups) == 1,
        "first_relevant_rank": rank,
        "reciprocal_rank": 1 / rank if rank else 0,
    }


class TimedReranker(CrossEncoderReranker):
    def score(self, question, passages):
        start = perf_counter()
        scores = super().score(question, passages)
        self.seconds = perf_counter() - start
        return scores


def main():
    store = VersionedStore(DATA_DIR / "indexer_test.sqlite3")
    passages = [store.get_passage(pid) for pid in store.latest_passage_ids()]
    passages = [
        p for p in passages if p is not None and p.document_id == "test-document"
    ]
    gold = []
    for question, groups in CASES:
        resolved = [
            [
                p.id
                for p in passages
                if any(
                    normalize(excerpt) in normalize(p.text) for excerpt in alternatives
                )
            ]
            for alternatives in groups
        ]
        if not all(resolved):
            raise ValueError(
                f"Gold evidence missing from active index: {question}: {resolved}"
            )
        gold.append(resolved)
    reranker = TimedReranker(RERANK_MODEL, device=RERANK_DEVICE)
    # Measure loading/initial inference separately from the ten query timings.
    reranker.score("warmup", ["A redevelopment document."])
    warmup = reranker.seconds
    search = HybridRetriever(
        store,
        ChromaStore(DATA_DIR / "indexer_test_chroma", collection_name="indexer_test"),
        OllamaClient(
            base_url="http://localhost:11434",
            embedding_model="embeddinggemma",
            timeout_seconds=45,
        ),
        20,
        20,
        result_limit=30,
        reranker=reranker,
        rerank_candidate_limit=30,
    )
    results = []
    for number, ((question, excerpts), groups) in enumerate(
        zip(CASES, gold, strict=True), 1
    ):
        start = perf_counter()
        ranked = search.retrieve_candidates(question)
        total = perf_counter() - start
        baseline = sorted(ranked, key=lambda e: (-e.score, e.passage.id))
        # Both orders use precisely the same merged top-30 candidate set.
        result = {
            "number": number,
            "question": question,
            "gold_excerpt_groups": excerpts,
            "gold_passage_groups": groups,
            "baseline": metrics(search.select_evidence(baseline), groups),
            "reranked": metrics(search.select_evidence(ranked), groups),
            "candidate_coverage": coverage([e.passage.id for e in ranked], groups),
            "retrieval_seconds": total - reranker.seconds,
            "rerank_seconds": reranker.seconds,
            "total_seconds": total,
            "baseline_ids": [e.passage.id for e in baseline],
            "reranked_evidence": [e.model_dump(mode="json") for e in ranked],
        }
        results.append(result)
        print(
            f"Q{number}: coverage@5 {result['baseline']['coverage_at_5']:.0%} -> "
            f"{result['reranked']['coverage_at_5']:.0%}; {total:.2f}s",
            flush=True,
        )
    output = Path(__file__).resolve().parents[1] / "outputs/retrieval_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "2009 supplied directive only; provisional "
        "gold labels, not current-law validation",
        "assumptions": "Q3/Q7 initial meeting; Q9 developer "
        "selection; Q4 accepts either quorum clause",
        "model": RERANK_MODEL,
        "warmup_seconds": warmup,
        "candidate_limit": 30,
        "channel_limits": 20,
        "result_limit": 5,
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
