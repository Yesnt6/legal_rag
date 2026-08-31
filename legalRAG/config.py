import os 
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field, SecretStr

load_dotenv()

class Settings(BaseModel):
    main_model: str = "qwen3:8b"
    embedding_model: str = "embeddinggemma"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    graph_max_hops: int = Field(default=3, ge=1, le=6)
    vector_candidate_limit: int = Field(default=20, ge=1)
    keyword_candidate_limit: int = Field(default=20, ge=1)
    graph_candidate_limit: int = Field(default=30, ge=1)
    rerank_limit: int = Field(default=8, ge=1)
    answer_timeout_seconds: float = Field(default=45, gt=0)

    ollama_base_url: str = "http://localhost:11434"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("")

    sqlite_path: Path = Path("data/legal_rag.sqlite3")
    chroma_path: Path = Path("data/chroma")

def load_settings() -> Settings:
    return Settings(
        main_model=os.getenv("LEGAL_RAG_MAIN_MODEL", "qwen3:8b"),
        embedding_model=os.getenv("LEGAL_RAG_EMBEDDING_MODEL", "embeddinggemma"),
        reranker_model=os.getenv(
            "LEGAL_RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
        ),
        graph_max_hops=os.getenv("LEGAL_RAG_GRAPH_MAX_HOPS", "3"),
        vector_candidate_limit=os.getenv("LEGAL_RAG_VECTOR_CANDIDATE_LIMIT", "20"),
        keyword_candidate_limit=os.getenv("LEGAL_RAG_KEYWORD_CANDIDATE_LIMIT", "20"),
        graph_candidate_limit=os.getenv("LEGAL_RAG_GRAPH_CANDIDATE_LIMIT", "30"),
        rerank_limit=os.getenv("LEGAL_RAG_RERANK_LIMIT", "8"),
        answer_timeout_seconds=os.getenv("LEGAL_RAG_ANSWER_TIMEOUT_SECONDS", "45"),
        ollama_base_url=os.getenv("LEGAL_RAG_OLLAMA_BASE_URL", "http://localhost:11434"),
        neo4j_uri=os.getenv("LEGAL_RAG_NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("LEGAL_RAG_NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("LEGAL_RAG_NEO4J_PASSWORD", ""),
        sqlite_path=os.getenv("LEGAL_RAG_SQLITE_PATH", "data/legal_rag.sqlite3"),
        chroma_path=os.getenv("LEGAL_RAG_CHROMA_PATH", "data/chroma"),
    )
