import json
from unittest.mock import Mock

from domain.schema import Answer, Citation, Passage
from retrieval.run_answer import print_answer


def test_output_groups_citations_by_exact_source_passage(capsys):
    passage = Passage(
        id="passage-1",
        document_id="document-1",
        text="First source sentence.\nSecond source sentence.",
        page_start=1,
        page_end=2,
        section_path=["Meeting", "Business"],
    )
    result = Answer(
        question="What happens?",
        answer="First claim. [1]\nSecond claim. [2]",
        citations=[
            Citation(
                document_id="document-1",
                passage_id="passage-1",
                page_start=1,
                page_end=2,
                quoted_text="First source sentence.",
            ),
            Citation(
                document_id="document-1",
                passage_id="passage-1",
                page_start=1,
                page_end=2,
                quoted_text="Second source sentence.",
            ),
        ],
    )
    connection = Mock()
    connection.execute.return_value.fetchone.return_value = {
        "version_id": "version-1",
        "metadata_json": json.dumps(
            {
                "title": "Redevelopment Directive",
                "issuing_authority": "Government of Maharashtra",
                "publication_date": "2009-01-03",
            }
        ),
    }
    store = Mock()
    store.connect.return_value = connection
    store.get_passage.return_value = passage

    print_answer(result, store)
    output = capsys.readouterr().out

    assert "LLM Response:\nFirst claim. [1]\nSecond claim. [2]" in output
    assert "Source [1], [2]:" in output
    assert "Document: Redevelopment Directive (ID: document-1)" in output
    assert "Pages: 1–2" in output
    assert "Section: Meeting > Business" in output
    assert output.count("Source passage (exact indexed text):") == 1
    assert passage.text in output
    assert "[1] First source sentence." in output
    assert "[2] Second source sentence." in output
    connection.execute.assert_called_once()


def test_abstention_has_reason_and_no_sources(capsys):
    result = Answer(
        question="What happens?",
        answer="Insufficient verified evidence to answer.",
        citations=[],
        abstained=True,
        abstention_reason="Claim was not established.",
    )
    store = Mock()

    print_answer(result, store)
    output = capsys.readouterr().out

    assert "LLM Response:\nInsufficient verified evidence to answer." in output
    assert "Verification: Claim was not established." in output
    assert "Sources:\nNone verified." in output
    store.connect.assert_not_called()
