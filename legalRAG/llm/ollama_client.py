import httpx
from pydantic import BaseModel


class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]


class OllamaClient:
    def __init__(
        self, base_url: str, embedding_model: str, timeout_seconds: float
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.embedding_model = embedding_model
        self.timeout_seconds = timeout_seconds

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = httpx.post(
            f"{self.base_url}/api/embed",
            json={"model": self.embedding_model, "input": texts},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

        embeddings = EmbeddingResponse.model_validate(response.json()).embeddings
        if len(embeddings) != len(texts):
            raise ValueError(
                "Ollama returned a different number of embeddings than inputs."
            )

        return embeddings
