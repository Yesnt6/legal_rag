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
    source = evidence()
    source[0].passage.section_path = ["Developer agreement", "Bank guarantee"]
    answer = AnswerGenerator(client, "test").answer("How much?", source)
    assert not answer.abstained
    assert answer.answer.endswith("[1]")
    assert answer.citations[0].page_start == 5
    verification = json.loads(
        client.generate_json.call_args_list[1].args[1][1]["content"]
    )
    assert set(verification) == {"claim", "passages"}
    assert "question" not in verification
    assert verification["passages"][0]["heading"] == [
        "Developer agreement", "Bank guarantee"
    ]


def test_short_model_alias_maps_back_to_full_passage_id():
    source = evidence()
    source[0].passage.id = "2c0711f7bbf6a-long-stored-passage-id"
    value = draft()
    value.claims[0].support[0].passage_id = "P1"
    client = Mock()
    client.generate_json.side_effect = [
        value,
        Verdict(
            schema_version="1",
            status="SUPPORTED",
            support=[{"passage_id": "P1", "quote": "20% bank guarantee"}],
        ),
    ]

    answer = AnswerGenerator(client, "test").answer("How much?", source)

    assert not answer.abstained
    assert answer.citations[0].passage_id == source[0].passage.id
    draft_input = json.loads(
        client.generate_json.call_args_list[0].args[1][1]["content"]
    )
    assert draft_input["passages"][0]["passage_id"] == "P1"


def test_paraphrased_draft_support_is_accepted_when_verifier_quotes_source():
    value = draft()
    value.claims[0].support[0].quote = "The guarantee has to equal one fifth."
    client = Mock()
    client.generate_json.side_effect = [
        value,
        Verdict(
            schema_version="1",
            status="SUPPORTED",
            support=[{"passage_id": "p1", "quote": "20% bank guarantee"}],
        ),
    ]

    answer = AnswerGenerator(client, "test").answer("How much?", evidence())

    assert not answer.abstained
    assert answer.citations[0].quoted_text == "20% bank guarantee"


@pytest.mark.parametrize("status", ["NOT_ESTABLISHED", "CONTRADICTED"])
def test_unverified_answer_is_withheld(status):
    client = Mock()
    client.generate_json.side_effect = [
        draft(),
        Verdict(schema_version="1", status=status, support=[]),
    ]
    result = AnswerGenerator(client, "test").answer("q", evidence())
    assert result.abstained and result.citations == []
    assert status in result.abstention_reason


def test_unknown_passage_ids_are_rejected():
    value = draft()
    value.claims[0].support[0].passage_id = "invented"
    client = Mock()
    client.generate_json.return_value = value
    assert AnswerGenerator(client, "test").answer("q", evidence()).abstained
    assert client.generate_json.call_count == 1


def test_draft_quote_can_paraphrase_while_exact_verifier_quote_is_preserved():
    value = draft()
    value.claims[0].support[0].quote = "A paraphrase of the source."
    client = Mock()
    client.generate_json.side_effect = [
        value,
        Verdict(
            schema_version="1",
            status="SUPPORTED",
            support=[{"passage_id": "p1", "quote": "a 20% bank guarantee"}],
        ),
    ]

    result = AnswerGenerator(client, "test").answer("q", evidence())

    assert not result.abstained
    assert result.citations[0].quoted_text == "a 20% bank guarantee"
    assert "A paraphrase of the source." not in result.citations[0].quoted_text


def test_verifier_quote_can_cross_extracted_page_label():
    source = evidence()
    source[0].passage.text = (
        "Society should obtain quotations from minimum 5 experienced persons "
        "for \nPage 2 of 5 \npreparing project report."
    )
    value = draft()
    value.claims[0].text = "Society should obtain quotations from at least five people."
    client = Mock()
    client.generate_json.side_effect = [
        value,
        Verdict(
            schema_version="1",
            status="SUPPORTED",
            support=[
                {
                    "passage_id": "p1",
                    "quote": "for preparing project report",
                }
            ],
        ),
    ]

    result = AnswerGenerator(client, "test").answer("q", source)

    assert not result.abstained
    assert result.citations[0].quoted_text == (
        "for \nPage 2 of 5 \npreparing project report"
    )
    draft_prompt = client.generate_json.call_args_list[0].args[1][0]["content"]
    verifier_prompt = client.generate_json.call_args_list[1].args[1][0]["content"]
    assert "should" in draft_prompt and "must" in draft_prompt
    assert "should" in verifier_prompt and "must" in verifier_prompt


def test_verifier_quote_recovers_pdf_spacing_and_curly_apostrophe():
    source = evidence()
    source[0].passage.text = (
        "1. To take preliminary decision by taking into consideration demand "
        "of the members for redevelopment of society’s building.\n"
        "2. To select expert and experienced Architect."
    )
    quote = (
        "1. To take preliminary decision bytaking into consideration demand "
        "of the members for redevelopment of society's building.\n"
        "2. To selectexpert and experienced Architect."
    )
    client = Mock()
    client.generate_json.side_effect = [
        draft(),
        Verdict(
            schema_version="1",
            status="SUPPORTED",
            support=[{"passage_id": "p1", "quote": quote}],
        ),
    ]

    result = AnswerGenerator(client, "test").answer("q", source)

    assert not result.abstained
    assert result.citations[0].quoted_text == source[0].passage.text


def test_supported_paraphrase_cites_exact_source_passage():
    source = evidence()
    source[0].passage.text = (
        "Before convening the said meeting, Society should obtain list of "
        "Architects / Project Management Consultants on the panel of "
        "Government / Local Authority and obtain quotations from minimum "
        "5 experienced persons for\nPage 2 of 5\npreparing project report."
    )
    value = draft()
    value.claims[0].text = (
        "The society should obtain quotations from at least five experienced persons."
    )
    client = Mock()
    client.generate_json.side_effect = [
        value,
        Verdict(
            schema_version="1",
            status="SUPPORTED",
            support=[{
                "passage_id": "p1",
                "quote": "Society should obtain quotations from minimum 5 experienced "
                "persons for preparing project report",
            }],
        ),
    ]

    answer = AnswerGenerator(client, "test").answer("q", source)

    assert not answer.abstained
    assert answer.citations[0].quoted_text == source[0].passage.text


def test_supported_verdict_with_unknown_source_still_abstains():
    client = Mock()
    client.generate_json.side_effect = [
        draft(),
        Verdict(
            schema_version="1",
            status="SUPPORTED",
            support=[{"passage_id": "invented", "quote": "20% bank guarantee"}],
        ),
    ]

    assert AnswerGenerator(client, "test").answer("q", evidence()).abstained


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


def test_inconsistent_draft_is_reported_before_verification():
    value = draft()
    value.insufficient_evidence = True
    client = Mock()
    client.generate_json.return_value = value

    result = AnswerGenerator(client, "test").answer("q", evidence())

    assert result.abstained
    assert "inconsistent" in result.abstention_reason
    assert client.generate_json.call_count == 1


def test_supported_verdict_still_requires_exact_source_quotes():
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
