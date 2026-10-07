from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
)

from ingestion.pdf_extractor import extract_marked_text


def make_pdf(path: Path, texts: list[str | None]) -> None:
    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=612, height=792)
        if text is not None:
            font = DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                }
            )
            page[NameObject("/Resources")] = DictionaryObject(
                {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
            )
            stream = DecodedStreamObject()
            stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode())
            page[NameObject("/Contents")] = stream
    writer.write(path)


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


def test_rejects_image_only_page_in_otherwise_text_pdf(tmp_path: Path) -> None:
    path = tmp_path / "mixed.pdf"
    make_pdf(path, ["Native text"])
    writer = PdfWriter(clone_from=path)
    page = writer.add_blank_page(width=612, height=792)
    image = DecodedStreamObject()
    image.update(
        {
            NameObject("/Type"): NameObject("/XObject"),
            NameObject("/Subtype"): NameObject("/Image"),
            NameObject("/Width"): NumberObject(1),
            NameObject("/Height"): NumberObject(1),
            NameObject("/ColorSpace"): NameObject("/DeviceRGB"),
            NameObject("/BitsPerComponent"): NumberObject(8),
        }
    )
    image.set_data(bytes([0, 0, 0]))
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/XObject"): DictionaryObject({NameObject("/Im1"): image})}
    )
    stream = DecodedStreamObject()
    stream.set_data(b"q 100 0 0 100 0 0 cm /Im1 Do Q")
    page[NameObject("/Contents")] = stream
    mixed = tmp_path / "with_image.pdf"
    writer.write(mixed)

    with pytest.raises(ValueError, match="Page 2.*OCR"):
        extract_marked_text(mixed)
