from pydantic import BaseModel, Field, SecretStr

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
