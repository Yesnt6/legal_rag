from types import SimpleNamespace

import pytest

from retrieval.evaluate_retrieval import coverage, metrics


def items(*ids):
    return [SimpleNamespace(passage=SimpleNamespace(id=pid)) for pid in ids]


def test_partial_answers_are_not_counted_as_complete():
    groups = [["a", "alternative"], ["b"], ["c"]]
    result = metrics(items("noise", "alternative", "b"), groups)
    assert not result["hit_at_1"]
    assert result["hit_at_5"]
    assert result["coverage_at_5"] == pytest.approx(2 / 3)
    assert not result["complete_at_5"]
    assert result["reciprocal_rank"] == 0.5


def test_relevant_result_beyond_cutoff_does_not_count_as_top_five():
    result = metrics(items("1", "2", "3", "4", "5", "answer"), [["answer"]])
    assert not result["hit_at_5"]
    assert result["first_relevant_rank"] == 6
    assert result["coverage_at_5"] == 0


def test_empty_results_and_alternative_support():
    assert metrics([], [["a"]])["first_relevant_rank"] is None
    assert coverage(["alternative"], [["a", "alternative"]]) == 1
