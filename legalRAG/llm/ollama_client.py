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

    def generate_json(
        self, model: str, messages: list[dict[str, str]], schema: type[BaseModel]
    ) -> BaseModel:
        """Constrain output to a schema; retry malformed output once."""
        conversation = list(messages)
        for attempt in range(2):
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": conversation,
                    "stream": False,
                    "think": False,
                    "format": schema.model_json_schema(),
                    "options": {"temperature": 0, "num_ctx": 8192},
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            try:
                content = response.json()["message"]["content"]
                return schema.model_validate_json(content)
            except (ValueError, KeyError, TypeError) as exc:
                if attempt:
                    raise ValueError(
                        "Model output failed structured validation twice."
                    ) from exc
                conversation = list(messages) + [
                    {
                        "role": "user",
                        "content": "Previous output did not match the "
                        "required JSON schema. Return only valid "
                        "schema-conforming JSON.",
                    }
                ]
        raise AssertionError("Unreachable")
