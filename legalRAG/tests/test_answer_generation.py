import json
from unittest.mock import Mock

import httpx
import pytest

from domain.schema import Passage, RetrievedEvidence
from llm.answer_generator import AnswerGenerator, Draft, Verdict
from llm.ollama_client import OllamaClient


def evidence():
    return [
        RetrievedEvidence(
            passage=Passage(
                id="p1",
                document_id="doc",
                text="Developer must give a 20% bank guarantee.",
                page_start=5,
                page_end=5,
            ),
            score=0.5,
            retrieval_source="vector",
        )
    ]


def draft():
    return Draft(
        schema_version="1",
        insufficient_evidence=False,
        claims=[
            {
                "text": "The developer must give a 20% bank guarantee.",
                "support": [{"passage_id": "p1", "quote": "20% bank guarantee"}],
            }
        ],
    )


def test_verified_claim_has_application_owned_citation_and_fresh_verifier_context():
    client = Mock()
    client.generate_json.side_effect = [
        draft(),
        Verdict(
            schema_version="1",
            status="SUPPORTED",
            support=[{"passage_id": "p1", "quote": "20% bank guarantee"}],
        ),
    ]
    answer = AnswerGenerator(client, "test").answer("How much?", evidence())
    assert not answer.abstained
    assert answer.answer.endswith("[1]")
    assert answer.citations[0].page_start == 5
    verification = json.loads(
        client.generate_json.call_args_list[1].args[1][1]["content"]
    )
    assert set(verification) == {"claim", "passages"}
    assert "question" not in verification


@pytest.mark.parametrize("status", ["NOT_ESTABLISHED", "CONTRADICTED"])
def test_unverified_answer_is_withheld(status):
    client = Mock()
    client.generate_json.side_effect = [
        draft(),
        Verdict(schema_version="1", status=status, support=[]),
    ]
    result = AnswerGenerator(client, "test").answer("q", evidence())
    assert result.abstained and result.citations == []


@pytest.mark.parametrize("pid,quote", [("invented", "20%"), ("p1", "50%"), ("p1", " ")])
def test_unknown_ids_and_nonverbatim_quotes_are_rejected(pid, quote):
    value = draft()
    value.claims[0].support[0].passage_id = pid
    value.claims[0].support[0].quote = quote
    client = Mock()
    client.generate_json.return_value = value
    assert AnswerGenerator(client, "test").answer("q", evidence()).abstained
    assert client.generate_json.call_count == 1


def test_empty_or_over_budget_evidence_skips_model():
    client = Mock()
    generator = AnswerGenerator(client, "test", max_context_chars=1)
    assert generator.answer("q", []).abstained
    assert generator.answer("q", evidence()).abstained
    client.generate_json.assert_not_called()


def test_model_can_abstain():
    client = Mock()
    client.generate_json.return_value = Draft(
        schema_version="1", claims=[], insufficient_evidence=True
    )
    assert AnswerGenerator(client, "test").answer("q", evidence()).abstained


def test_supported_verdict_still_requires_exact_quotes():
    client = Mock()
    client.generate_json.side_effect = [
        draft(),
        Verdict(schema_version="1", status="SUPPORTED", support=[]),
    ]
    assert AnswerGenerator(client, "test").answer("q", evidence()).abstained


def test_structured_http_retry_and_schema(monkeypatch):
    post = Mock(
        side_effect=[
            httpx.Response(
                200,
                json={"message": {"content": "invalid"}},
                request=httpx.Request("POST", "http://test"),
            ),
            httpx.Response(
                200,
                json={"message": {"content": draft().model_dump_json()}},
                request=httpx.Request("POST", "http://test"),
            ),
        ]
    )
    monkeypatch.setattr(httpx, "post", post)
    client = OllamaClient("http://test", "embed", 10)
    assert client.generate_json("main", [], Draft) == draft()
    assert post.call_count == 2
    payload = post.call_args.kwargs["json"]
    assert payload["format"] == Draft.model_json_schema()
    assert payload["stream"] is False
    post.side_effect = None
    post.return_value = httpx.Response(
        200,
        json={"message": {"content": "{}"}},
        request=httpx.Request("POST", "http://test"),
    )
    with pytest.raises(ValueError, match="twice"):
        client.generate_json("main", [], Draft)
