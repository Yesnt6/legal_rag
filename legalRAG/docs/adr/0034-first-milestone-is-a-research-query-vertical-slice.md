# Prove the research query path as the first vertical slice

The first implementation milestone will load one manually extracted legal
document with page markers, create hierarchical passages in SQLite, embed them
in Chroma, extract a provenance-grounded constrained graph into Neo4j, and
answer one research question through FastAPI using hybrid retrieval, reranking,
claim generation, verification, citations, abstention or conflict handling, and
retrieval traces. PDF parsing, Streamlit, compliance mode, and durable ingestion
remain deferred until this path works end to end.

