from pathlib import Path

import pymupdf
import pytest

from ingestion.pdf_extractor import extract_marked_text


def make_pdf(path: Path, texts: list[str | None]) -> None:
    if not texts:
        # Minimal zero-page PDF; PyMuPDF cannot save a new empty document.
        path.write_bytes(
            b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
            b"trailer\n<< /Root 1 0 R /Size 3 >>\n%%EOF\n"
        )
        return
    with pymupdf.open() as document:
        for text in texts:
            page = document.new_page(width=612, height=792)
            if text is not None:
                page.insert_text((72, 72), text)
        document.save(path)


def test_extracts_pages_in_order_including_blank_pages(tmp_path: Path) -> None:
    path = tmp_path / "source.pdf"
    make_pdf(path, ["Cover", None, "Provision 1"])

    assert extract_marked_text(path) == (
        "[PAGE 1]\nCover\n[PAGE 2]\n\n[PAGE 3]\nProvision 1\n"
    )


@pytest.mark.parametrize("pages", [[], [None], [None, None]])
def test_rejects_documents_without_text(tmp_path: Path, pages) -> None:
    path = tmp_path / "empty.pdf"
    make_pdf(path, pages)

    with pytest.raises(ValueError, match="no extractable text"):
        extract_marked_text(path)


def test_rejects_source_text_that_looks_like_a_page_marker(tmp_path: Path) -> None:
    path = tmp_path / "marker.pdf"
    make_pdf(path, ["[PAGE 99]"])

    with pytest.raises(ValueError, match="reserved page marker"):
        extract_marked_text(path)


def test_flags_image_only_page_and_preserves_its_marker(tmp_path: Path) -> None:
    path = tmp_path / "mixed.pdf"
    make_pdf(path, ["Native text"])
    mixed = tmp_path / "with_image.pdf"
    with pymupdf.open(path) as document:
        page = document.new_page(width=612, height=792)
        image = pymupdf.Pixmap(pymupdf.csRGB, (0, 0, 1, 1))
        image.clear_with(0)
        page.insert_image(pymupdf.Rect(0, 0, 100, 100), pixmap=image)
        document.save(mixed)

    with pytest.warns(UserWarning, match="1 image-only pages: 2"):
        text = extract_marked_text(mixed)
    assert text == "[PAGE 1]\nNative text\n[PAGE 2]\n\n"


def test_groups_repeated_images_without_blocking_native_text(tmp_path: Path) -> None:
    path = tmp_path / "source.pdf"
    make_pdf(path, ["Cover", "Clause with image", "Plain text", "Another image"])
    mixed = tmp_path / "mixed.pdf"
    with pymupdf.open(path) as document:
        image = pymupdf.Pixmap(pymupdf.csRGB, (0, 0, 1, 1))
        image.clear_with(0)
        for index in (1, 3):
            document[index].insert_image(pymupdf.Rect(0, 0, 100, 100), pixmap=image)
        document.save(mixed)

    with pytest.warns(UserWarning) as flags:
        text = extract_marked_text(mixed)
    message = str(flags[0].message)
    assert "0 image-only pages: none" in message
    assert "1 unique images" in message
    assert "image 1: pages 2, 4" in message
    assert text == (
        "[PAGE 1]\nCover\n[PAGE 2]\nClause with image\n"
        "[PAGE 3]\nPlain text\n[PAGE 4]\nAnother image\n"
    )


def test_image_only_document_flags_distinct_images_and_excludes_blank_page(
    tmp_path: Path,
) -> None:
    path = tmp_path / "images.pdf"
    with pymupdf.open() as document:
        for color in (0, 255):
            page = document.new_page()
            image = pymupdf.Pixmap(pymupdf.csRGB, (0, 0, 1, 1))
            image.clear_with(color)
            page.insert_image(pymupdf.Rect(0, 0, 100, 100), pixmap=image)
            page.insert_image(pymupdf.Rect(100, 0, 200, 100), pixmap=image)
        document.new_page()
        document.save(path)

    with pytest.warns(UserWarning) as flags:
        text = extract_marked_text(path)
    message = str(flags[0].message)
    assert "2 image-only pages: 1, 2" in message
    assert "2 unique images (image 1: pages 1; image 2: pages 2)" in message
    assert text == "[PAGE 1]\n\n[PAGE 2]\n\n[PAGE 3]\n\n"
