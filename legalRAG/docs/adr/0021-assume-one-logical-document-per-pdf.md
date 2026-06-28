# Assume one logical document per PDF

The initial ingestion pipeline will treat each uploaded PDF as one logical
document and will not spend computation detecting or rejecting bundled
documents. Uploaders provide the document title and type, annexures remain part
of the same document, and the interface will disclose that bundled PDFs may
weaken document metadata and citations.

