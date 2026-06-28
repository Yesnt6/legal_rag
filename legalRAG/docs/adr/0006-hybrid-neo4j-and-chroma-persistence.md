# Use Neo4j and Chroma as a hybrid retrieval store

The local GraphRAG system will use Neo4j Community Edition for extracted graph
entities, relationships, document structure, and provenance links, while Chroma
stores vector embeddings. Neo4j will run as a single-instance service through
Docker Compose, and every derived graph fact must link back to stable supporting
passage identifiers. ADR-0030 later established SQLite as the authoritative
passage and metadata store.
