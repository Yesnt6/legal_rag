# Require legal document metadata at indexing time

Indexing requires a document title, document type, issuing authority,
publication date, jurisdiction, and any known effective-from, effective-until,
official identifier, and source URL values. The system calculates checksum,
immutable version, page count, ingestion date, and extractor model and prompt
versions; missing temporal or authority metadata prevents deterministic
precedence decisions and causes conflicts to be disclosed rather than guessed.

