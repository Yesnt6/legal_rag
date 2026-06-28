# Use SQLite as the authoritative passage and metadata store

SQLite will hold authoritative document metadata, immutable versions, passage
text, hierarchy, page and character offsets, and ingestion records. Chroma will
store vector embeddings and Neo4j will store derived graph entities,
relationships, and provenance links, both keyed by stable passage identifiers;
the derived stores can therefore be rebuilt from SQLite and the immutable
original PDFs.

