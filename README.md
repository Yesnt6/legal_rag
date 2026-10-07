# Closed-Corpus Legal Research Assistant

> [!IMPORTANT]
> **Work in Progress**
>
> This project is under active development and is not production-ready. The flow below represents the system I intend to build; some components may not yet be implemented.

## Goal

Build a legal research assistant that answers questions using only an approved document collection. The system is intended to provide source-backed responses, verify citations, and avoid answering when the available evidence is insufficient.

## Intended Flow

```mermaid
flowchart LR
    A[User question] --> B[Search approved legal documents]
    B --> C[Retrieve relevant passages]
    C --> D[Generate a grounded answer]
    D --> E[Verify claims and citations]
    E --> F{Evidence sufficient?}
    F -- Yes --> G[Return answer with sources]
    F -- No --> H[Abstain or request clarification]
```

## Design Principles

- Use only documents from the closed corpus.
- Link important claims to supporting passages.
- Clearly separate retrieved evidence from generated text.
- Abstain when the evidence does not support an answer.

## Status

The project is currently in early development. Implementation details, setup instructions, tests, and evaluation results will be added as the system progresses.

## Disclaimer

This project is intended for research and learning purposes. It does not provide legal advice.

## Extract a PDF before indexing

From the `legalRAG` directory, install the project dependencies and run:

```bash
python -m pip install -e '.[dev]'
python -m ingestion.pdf_extractor documents/source.pdf documents/source.txt
```

The output is UTF-8 text with `[PAGE 1]`, `[PAGE 2]`, etc. based on physical
PDF page order, including blank pages. The output parent directory must exist;
existing files are not overwritten. Extraction finishes before output is written.

This step does not run indexing or infer Markdown/legal section headings.
Encrypted PDFs, documents without text, and image-only pages are rejected.
OCR is not performed. A text layer alone does not prove native-text origin:
OCR-derived text, mixed image/text content, tables, and reading order still
need source review before indexing. Printed footer numbers remain source text.
