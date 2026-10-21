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
python -m ingestion.pdf_extractor
```

Edit `SOURCE_PATH` and `OUTPUT_PATH` in `legalRAG/ingestion/pdf_extractor.py`
to choose the input and output. Defaults are `legalRAG/documents/source.pdf`
and `legalRAG/documents/source.txt`, independent of the working directory.
Extraction uses PyMuPDF.

The output is UTF-8 text with `[PAGE 1]`, `[PAGE 2]`, etc. based on physical
PDF page order, including blank pages. The output parent directory must exist;
existing files are not overwritten. Extraction finishes before output is written.

This step does not run indexing or infer Markdown/legal section headings.
Encrypted PDFs and documents with neither extractable text nor raster images
are rejected. Images no longer block extraction or require approval. A warning
reports the count and physical page numbers of image-only pages (raster images
present, no non-whitespace extracted text). Identical images are grouped by
PyMuPDF's image digest, with their page locations listed once per group.

Every page keeps its `[PAGE N]` marker. Mixed text/image pages retain their native
text; image-only pages contain just the marker and a blank line. Even an entirely
image-only PDF produces marked output, but has no text for the indexer to index.
Blank pages are not counted as image-only. Flags are console/Python warnings,
not inserted into source text. OCR is not performed; vector drawings are not
covered by this raster-image check. A readable footer does not mean the image
content was extracted. Printed footer numbers remain source text.

## Search the indexed document

With Ollama running and the indexed data available, run from `legalRAG`:

```bash
python -m retrieval.run_retriever
```

Edit `QUESTION` and `DATA_DIR` in `retrieval/run_retriever.py`. The default data
location matches the current indexer: `legalRAG/ingestion/data/`. The runner
uses Chroma collection `indexer_test` and the same `embeddinggemma` model used
for indexing. It refuses to silently create an empty corpus if files are missing.

Opening `VersionedStore` creates/backfills an FTS5 index over searchable passage
text and heading paths. New passages are indexed in the same SQLite transaction.
No re-embedding is required. Queries use quoted literal terms joined with OR,
ranked by SQLite BM25. Both keyword and semantic search are restricted to the
same latest activated source version and active segmentation per document before applying candidate limits.
"Latest" means highest ingestion version number, not legally applicable as of
a date; effective-date filtering is not implemented yet.

Results merge by passage ID and use reciprocal rank fusion with k=60:
`sum(1 / (60 + channel_rank))`. Higher scores rank first; these scores are not
cosine similarities or confidence values. Ties break by passage ID. The runner
prints source pages and origins (`vector`, `keyword`, or `vector+keyword`).
The former unversioned SQLiteStore remains separate; HybridRetriever now expects
VersionedStore. Graph retrieval, model reranking, and answer generation remain
future steps. For this small local corpus, eligible passage IDs are materialized
in memory and passed to Chroma as an ID filter.

## Clause-aware segmentation and re-indexing

The indexer recognizes Markdown headings, numbered `11.` sections, `(2)` / `2)`
clauses, and letter/Roman items such as `a)` and `ii)`, including markers on a
separate line. Numbered blocks ending in `:` or `:-` are treated as headings;
other numbered text remains clause content. This is deterministic formatting
recognition, not a universal legal-layout parser. Ambiguous numbering, unusual
headings, tables, and PDF reading order still need inspection.

Heading-only nodes are context, not independent search candidates. Clause
passages retain parent IDs, heading paths, and exact offsets into normalized
source text. Clauses spanning PDF pages retain both page numbers. Oversized
clauses prefer sentence boundaries, then whitespace, then a hard character
limit when necessary; their complete parent clause remains available.

Re-run `python -m ingestion.run_indexer` to apply the new segmentation, then
`python -m retrieval.run_retriever` to test retrieval. The source version ID
stays unchanged for an unchanged file. A profile combines the segmentation
algorithm version and `max_chars`, producing distinct passage IDs. Existing
SQLite databases are migrated automatically and legacy passages remain readable
by ID. Old Chroma embeddings are retained but excluded from active retrieval.
The new profile becomes active only after all embedding batches succeed;
failed imports can be retried without losing the previous active results.
`Reused existing version: True` refers to the source, not necessarily its
segmentation. The runner prints the active segmentation profile.

## Local reranking

`python -m retrieval.run_retriever` now merges BM25 and vector results with RRF,
reranks up to 30 candidates using `BAAI/bge-reranker-v2-m3`, and prints the best
five. Each question is paired with the passage's heading path and source text.
The output retains the original RRF score and adds a separate reranker score;
neither is an answer-confidence probability. Citations and source text are unchanged.

The first nonempty search downloads the model from Hugging Face. Subsequent runs
reuse its local cache. Configure `RERANK_MODEL` and `RERANK_DEVICE` in
`legalRAG/retrieval/run_retriever.py`; the adapter also exposes batch size and
maximum pair length (default 1024 tokens, with longer inputs truncated for scoring).
Model errors propagate instead of silently returning unreranked results.
For a baseline comparison, construct `HybridRetriever` without a reranker.
No reindexing is required. The ten-question quality evaluation is pending the
user's questions; automated tests verify pipeline behavior, not model quality.

## Ten-question retrieval evaluation

From `legalRAG`, run `python -m retrieval.evaluate_retrieval` with Ollama running.
The runner preserves the user's ten questions (including original spelling),
resolves manually selected evidence excerpts against the active source passages,
and compares RRF order against reranked order for the same top-30 candidates.
Q3 and Q7 concern the initial meeting; Q9 concerns developer selection. Q4 accepts
either of the document's quorum passages. These are provisional, source-grounded
gold labels for the supplied 2009 directive, not a current-law assessment.

Results are written to `outputs/retrieval_evaluation.json`, including full candidate
texts, page references, gold passage IDs, hit@1, hit@5, evidence-group coverage@5,
complete coverage@5, candidate coverage, and query/reranking timings. Alternative
passages supporting the same required point count once. Full coverage measures
the listed evidence groups, not whether a generated answer is correct.
Model warmup is recorded separately. The per-question timings are single-run
observations, not a latency benchmark. With eight complaint items, Q1 cannot
achieve complete coverage using five individual passages.

### Preserving explicit numbered lists

Segmentation profile `legal-clauses-v3` adds non-searchable `LIST` parents for
explicit prose introductions (enumeration cues such as “following” or “as under”,
ending in a colon) followed by at least two consecutive numeric items starting
at 1, with consistent numbering style and indentation. Numbered colon headings
are excluded. Each item keeps a clause parent beneath the list; its heading path
includes the source introduction and enclosing section. Exact text offsets and
physical page spans remain unchanged.

This conservative plain-text pass handles simple `1.` and `1)` lists. It does
not infer font information, general nested-list structure, or arbitrary unmarked
lists. A new paragraph within an item prevents further grouping; a blank line
after the final item ends the group so following prose is not absorbed. Thus
multi-paragraph items need richer extraction or additional rules. Existing
alpha/Roman clause parsing remains in place. Unrecognized groups retain the
existing segmentation behavior. Tests cover list/section numbering collisions,
ambiguous inputs, cross-page continuation, and the local directive when present.

Run `python -m ingestion.run_indexer` from `legalRAG` to activate the new profile.
This changes indexing context only: sibling expansion and the top-five retrieval
limit are unchanged. Keep the existing evaluation outputs as the pre-change
baseline before rerunning evaluation.

### Searching complete lists and individual clauses

Profile `legal-clauses-v4` makes `LIST` records searchable when their complete
source span fits `max_chars` (1800 by default). Those records enter both SQLite
FTS and Chroma alongside individual clauses; the usual hybrid ranking and
reranker choose the level of detail without a manual expansion mode. Larger
lists remain non-searchable parents with searchable individual clauses; complete
retrieval of oversized lists is not implemented yet.

After ranking, `HybridRetriever.select_evidence()` removes ancestor/descendant
overlap in rank order. A higher-ranked list excludes its children; a higher-ranked
child excludes its lower-ranked list, preserving focused results. Other siblings
remain eligible. Results retain their original scores and citations, and later
candidates fill available result slots. Parent IDs, not text similarity, determine
overlap. `retrieve_candidates()` exposes the ranking before this selection so
evaluation can compare RRF and reranking independently on the same candidates.

Reindex to activate v4. Existing evaluation reports are historical baselines and
are not automatically updated by reindexing. Run evaluation separately to measure
quality changes; whole-list results can satisfy multiple expected evidence groups.

## Source-bound answer generation

From `legalRAG`, install the configured generation model with `ollama pull qwen3.5:4b`,
then run `python -m retrieval.run_answer`. Edit `QUESTION` in that runner; the model
uses `LEGAL_RAG_MAIN_MODEL` from settings (default `qwen3.5:4b`). EmbeddingGemma is
only an embedding model and cannot generate answers. No reindexing is required.

The runner retrieves five non-overlapping passages, generates atomic claims using
Ollama JSON-schema output, and verifies each claim in a fresh model call containing
only that claim and its cited passages. Invalid passage IDs, missing/non-verbatim
quotes, or any verdict other than SUPPORTED with valid supporting quotes withhold
the answer. A malformed structured response gets one retry; HTTP/model/timeout
errors remain errors, not evidence-based abstentions. Citation numbers, document
IDs and page ranges come from stored records. The runner additionally displays
source version IDs, clause paths and exact supporting quotes.

This is a source-reading prototype, not the complete legal answer contract.
It does not establish temporal validity, exhaustively detect cross-document
conflicts, or prove answer completeness. Generation and verification use the same
model and can share mistakes. Evidence is bounded to 12000 source characters;
whole passages exceeding the remaining budget are omitted, never silently cut
mid-clause. The 180-second setting is a timeout per model request, not a guaranteed
end-to-end budget; verification adds one request per claim. Automated tests use
simulated model outputs; live quality evaluation remains a separate step.
