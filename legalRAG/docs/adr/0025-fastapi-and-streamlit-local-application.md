# Build the initial application with FastAPI and Streamlit

The initial local application will expose ingestion, document status, querying,
and retrieval traces through a FastAPI backend, with a Streamlit interface for
PDF upload, ingestion progress, document browsing, cited chat answers, and
evidence inspection. Neo4j Community Edition runs through Docker Compose,
Ollama serves local models, and Chroma persists passage embeddings on disk.

