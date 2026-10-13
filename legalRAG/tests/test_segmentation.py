from ingestion.manual_indexer import parse_pages, segment

SOURCE = """[PAGE 1]
11. Agreement to be entered into with Developer:-

An agreement must be made.
(1)
Complete the project within two years.
[PAGE 2]
(2)
Developer will give a Bank Guarantee for amount equal to 20% of the
project cost.
(3) Provide alternative accommodation.
"""


def test_numbered_clauses_keep_section_context_and_exact_spans():
    text, pages = parse_pages(SOURCE)
    items = segment("doc", "version", text, pages, 1800)
    bank = [p for p in items if p.retrievable and "Bank Guarantee" in p.passage.text]
    assert len(bank) == 1
    assert bank[0].passage.text == (
        "(2)\nDeveloper will give a Bank Guarantee for amount equal to 20% of the\n"
        "project cost."
    )
    assert bank[0].passage.section_path == [
        "11. Agreement to be entered into with Developer:-",
        "(2)",
    ]
    assert bank[0].passage.page_start == bank[0].passage.page_end == 2
    by_id = {p.passage.id: p for p in items}
    assert by_id[bank[0].passage.parent_id].kind == "CLAUSE"
    assert not any(
        p.retrievable and p.passage.text.endswith("Developer:-") for p in items
    )
    for item in items:
        assert item.passage.text == text[item.char_start : item.char_end]
    assert [p.passage.id for p in items] == [
        p.passage.id for p in segment("doc", "version", text, pages, 1800)
    ]


def test_split_line_headings_and_nested_alpha_roman_clauses():
    source = """[PAGE 1]
# Redevelopment
11.
Agreement with the
Developer:-

a) Selection procedure:-

i) Invite bids.
ii) Evaluate bids.
b) Sign the contract.
12. Completion:-

Issue certificates.
"""
    text, pages = parse_pages(source)
    items = segment("doc", "v", text, pages, 1800)
    found = next(
        p for p in items if p.retrievable and p.passage.text == "ii) Evaluate bids."
    )
    assert found.passage.section_path == [
        "Redevelopment",
        "11. Agreement with the Developer:-",
        "a) Selection procedure:-",
        "ii)",
    ]
    sibling = next(
        p for p in items if p.retrievable and "Sign the contract" in p.passage.text
    )
    assert sibling.passage.section_path[-1] == "b)"
    assert "a) Selection procedure:-" not in sibling.passage.section_path
    completion = next(
        p for p in items if p.retrievable and "Issue certificates" in p.passage.text
    )
    assert completion.passage.section_path == ["Redevelopment", "12. Completion:-"]


def test_cross_page_clause_is_one_passage_with_both_pages():
    text, pages = parse_pages(
        "[PAGE 1]\n(1) Members must\n[PAGE 2]\napprove the plan.\n(2) Keep records."
    )
    items = [p for p in segment("doc", "v", text, pages, 1800) if p.retrievable]
    assert items[0].passage.text == "(1) Members must\napprove the plan."
    assert (items[0].passage.page_start, items[0].passage.page_end) == (1, 2)


def test_long_clause_prefers_sentence_boundary_and_retains_parent():
    text, pages = parse_pages(
        "[PAGE 1]\n(1) First sentence ends here. Second sentence must remain intact."
    )
    items = segment("doc", "v", text, pages, 45)
    chunks = [p for p in items if p.retrievable]
    assert chunks[0].passage.text == "(1) First sentence ends here. "
    assert chunks[1].passage.text == "Second sentence must remain intact."
    assert chunks[0].passage.parent_id == chunks[1].passage.parent_id
    assert all(len(p.passage.text) <= 45 for p in chunks)
    assert "".join(p.passage.text for p in chunks) == text


def test_decimal_amounts_and_plain_text_are_not_numbered_boundaries():
    text, pages = parse_pages("[PAGE 1]\n20.5 percent is payable.\n\nKeep records.")
    items = segment("doc", "v", text, pages, 1800)
    assert all(p.kind != "CLAUSE" for p in items)
    assert [p.passage.text for p in items if p.retrievable] == [
        "20.5 percent is payable.",
        "Keep records.",
    ]


def test_numbered_heading_does_not_require_blank_line_before_body():
    text, pages = parse_pages(
        "[PAGE 1]\n11. Agreement with Developer:-\n"
        "An agreement must be made.\n(1) Complete the project."
    )
    items = segment("doc", "v", text, pages, 1800)
    chunks = [p.passage for p in items if p.retrievable]
    assert chunks[0].text == "An agreement must be made."
    assert chunks[1].section_path == ["11. Agreement with Developer:-", "(1)"]


def test_long_numbered_sentence_never_emits_only_the_number():
    text, pages = parse_pages(
        "[PAGE 1]\n11. The developer must provide a sufficient bank guarantee "
        "before work begins on the building."
    )
    items = segment("doc", "v", text, pages, 40)
    chunks = [p.passage.text for p in items if p.retrievable]
    assert chunks[0] != "11. "
    assert "developer" in chunks[0]
    assert "".join(chunks) == text
