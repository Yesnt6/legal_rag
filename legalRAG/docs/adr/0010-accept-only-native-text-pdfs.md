# Accept only native text PDFs

The initial ingestion pipeline will accept only digitally generated PDFs with a
native text layer and stable page boundaries. Scanned documents, OCR-derived
PDFs, images, Word documents, and PDFs that fail text-extraction quality checks
will be rejected so that graph provenance and page citations remain dependable.

