from unittest.mock import Mock

import pytest
from test_versioned_search import passage

from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker


def retriever(scores, result_limit=1, candidate_limit=3):
    items = [passage(str(i), f"Clause {i}") for i in range(3)]
    items[2].section_path = ["Guarantee", "(2)"]
    store = Mock()
    store.latest_passage_ids.return_value = [p.id for p in items]
    store.keyword_search.return_value = items
    store.get_passage.side_effect = lambda identifier: items[int(identifier)]
    chroma = Mock()
    chroma.search.return_value = [(p.id, 0.0) for p in items]
    embedder = Mock()
    embedder.embed.return_value = [[1.0]]
    reranker = Mock()
    reranker.score.return_value = scores
    return HybridRetriever(
        store,
        chroma,
        embedder,
        3,
        3,
        result_limit,
        reranker=reranker,
        rerank_candidate_limit=candidate_limit,
    ), reranker


def test_reranks_before_final_limit_and_preserves_evidence():
    search, scorer = retriever([0.1, 0.2, 0.9])
    result = search.retrieve("guarantee")
    assert result[0].passage.id == "2"
    assert result[0].rerank_score == 0.9
    assert result[0].score == pytest.approx(2 / 63)
    assert result[0].retrieval_source == "vector+keyword"
    assert result[0].passage.section_path == ["Guarantee", "(2)"]
    assert scorer.score.call_args.args == (
        "guarantee",
        ["Clause 0", "Clause 1", "Guarantee > (2)\nClause 2"],
    )


def test_candidate_cap_and_ties_preserve_rrf_order():
    search, scorer = retriever([0.5, 0.5], result_limit=2, candidate_limit=2)
    assert [e.passage.id for e in search.retrieve("q")] == ["0", "1"]
    assert len(scorer.score.call_args.args[1]) == 2


@pytest.mark.parametrize("scores", [[0.2], [0.2, float("nan"), 0.3]])
def test_invalid_model_output_fails_explicitly(scores):
    search, _ = retriever(scores)
    with pytest.raises(ValueError, match="finite score"):
        search.retrieve("q")


def test_empty_query_skips_reranker():
    search, scorer = retriever([])
    assert search.retrieve("  ") == []
    scorer.score.assert_not_called()


def test_cross_encoder_pairs_and_model_reuse():
    reranker = CrossEncoderReranker()
    model = Mock()
    model.predict.return_value = [0.2, 0.8]
    reranker._model = model
    assert reranker.score("q", ["first", "second"]) == [0.2, 0.8]
    assert model.predict.call_args.args[0] == [("q", "first"), ("q", "second")]
    assert reranker.score("q", []) == []
    assert model.predict.call_count == 1
