"""Extract native PDF text with physical-page markers for manual indexing."""

import argparse
import re
from pathlib import Path

from pypdf import PdfReader


def extract_marked_text(source_path: Path) -> str:
    """Keep physical page positions, including blank pages; do not perform OCR."""
    parts: list[str] = []
    has_text = False
    with source_path.open("rb") as source:
        reader = PdfReader(source)
        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported.")
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").replace("\r\n", "\n")
            text = text.replace("\r", "\n").rstrip("\n")
            if not text.strip() and len(page.images):
                raise ValueError(
                    f"Page {number} has images but no text; OCR is required."
                )
            if re.search(r"^\[PAGE [1-9]\d*\][ \t]*$", text, re.MULTILINE):
                raise ValueError(f"Page {number} contains a reserved page marker.")
            has_text = has_text or bool(text.strip())
            parts.append(f"[PAGE {number}]\n{text}\n")
    if not has_text:
        raise ValueError("PDF contains no extractable text; OCR is not supported.")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Input native-text PDF")
    parser.add_argument("output", type=Path, help="New UTF-8 text file")
    args = parser.parse_args()
    text = extract_marked_text(args.source)
    # Exclusive creation prevents overwriting either an existing output or source.
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        output.write(text)


if __name__ == "__main__":
    main()
