from pathlib import Path

from ingestion.manual_indexer import parse_pages, segment


def parsed(source):
    text, pages = parse_pages(source)
    items = segment("doc", "v", text, pages, 1800)
    assert all(p.passage.text == text[p.char_start : p.char_end] for p in items)
    return items


def test_list_and_sections_share_numbering_without_sharing_parent():
    items = parsed("""[PAGE 1]
The following complaints were received:-
1. Missing notice.
2. Unfair selection.

This explanatory paragraph is outside the list.

1. Requisition:-
Submit a request.
2. Convening:-
Send notice.
""")
    group = next(p for p in items if p.kind == "LIST")
    children = [p for p in items if p.passage.parent_id == group.passage.id]
    assert [p.passage.section_path[-1] for p in children] == ["1.", "2."]
    assert "explanatory" not in group.passage.text
    assert all(p.kind == "CLAUSE" for p in children)
    sections = [p for p in items if p.kind == "SECTION"]
    assert all(p.passage.parent_id != group.passage.id for p in sections)
    prose = next(p for p in items if p.retrievable and "explanatory" in p.passage.text)
    assert prose.passage.section_path == []


def test_list_inside_section_preserves_outer_heading_and_cross_page_items():
    items = parsed("""[PAGE 1]
2. Meeting:-
Some introductory prose.

Following business will be transacted:-
1. Discuss the
[PAGE 2]
proposal.

2. Select a consultant.

3. Approve the programme.

3. Suggestions:-
Submit suggestions.
""")
    group = next(p for p in items if p.kind == "LIST")
    assert group.passage.section_path == [
        "2. Meeting:-",
        "Following business will be transacted:-",
    ]
    first = next(p for p in items if p.kind == "RETRIEVAL" and "Discuss" in p.passage.text)
    assert first.passage.page_start == 1 and first.passage.page_end == 2
    assert first.passage.section_path[-2:] == [
        "Following business will be transacted:-",
        "1.",
    ]
    assert not any(
        p.retrievable and p.passage.text == "Following business will be transacted:-"
        for p in items
    )


def test_colon_or_numbering_alone_does_not_create_a_list():
    for source in [
        "Topic:\n1. First block.\n2. Second block.",
        "1. First block.\n2. Second block.",
        "Following topics:\n1. First section:-\nBody.\n2. Second section:-\nBody.",
        "Following items:\n1. First.\n3. Third.",
    ]:
        assert not any(p.kind == "LIST" for p in parsed("[PAGE 1]\n" + source))


def test_actual_document_complaints_and_agenda_have_list_context():
    source = Path(__file__).resolve().parents[1] / "outputs/report_1.txt"
    if not source.exists():
        import pytest

        pytest.skip("Local extracted document is not available")
    items = parsed(source.read_text())
    lists = [p for p in items if p.kind == "LIST"]
    complaints = next(
        p for p in lists if "nature of complaints" in " ".join(p.passage.text.split())
    )
    agenda = next(
        p for p in lists if "Following business will be transacted" in p.passage.text
    )
    assert len([p for p in items if p.passage.parent_id == complaints.passage.id]) == 8
    assert len([p for p in items if p.passage.parent_id == agenda.passage.id]) == 3
    assert "Whereas there is no concrete policy" not in complaints.passage.text
    assert (
        agenda.passage.section_path[0] == "2. Convening Special General Body Meeting :-"
    )
