"""Extract native PDF text with physical-page markers for manual indexing."""

import re
from pathlib import Path

import pymupdf

PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_DIR / "documents/source.pdf"
OUTPUT_PATH = PROJECT_DIR / "documents/source.txt"


def extract_marked_text(source_path: Path) -> str:
    """Keep physical page positions, including blank pages; do not perform OCR."""
    parts: list[str] = []
    has_text = False
    with pymupdf.open(source_path) as reader:
        if not reader.is_pdf:
            raise ValueError("Input must be a PDF.")
        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported.")
        for number, page in enumerate(reader, start=1):
            text = (page.get_text("text") or "").replace("\r\n", "\n")
            text = text.replace("\r", "\n").rstrip("\n")
            if not text.strip() and page.get_image_info():
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
    text = extract_marked_text(SOURCE_PATH)
    # Exclusive creation prevents overwriting either an existing output or source.
    with OUTPUT_PATH.open("x", encoding="utf-8", newline="\n") as output:
        output.write(text)


if __name__ == "__main__":
    main()
