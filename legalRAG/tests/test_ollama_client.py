import httpx
import pytest

from llm.ollama_client import OllamaClient


def test_embed_posts_texts_and_returns_embeddings(monkeypatch) -> None:
    request: dict[str, object] = {}

    def fake_post(
        url: str, *, json: dict[str, object], timeout: float
    ) -> httpx.Response:
        request.update({"url": url, "json": json, "timeout": timeout})
        return httpx.Response(
            200,
            json={"embeddings": [[1.0, 0.0], [0.0, 1.0]]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("llm.ollama_client.httpx.post", fake_post)
    client = OllamaClient(
        base_url="http://localhost:11434/",
        embedding_model="embeddinggemma",
        timeout_seconds=45.0,
    )

    assert client.embed(["first text", "second text"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert request == {
        "url": "http://localhost:11434/api/embed",
        "json": {"model": "embeddinggemma", "input": ["first text", "second text"]},
        "timeout": 45.0,
    }


def test_embed_rejects_a_partial_response(monkeypatch) -> None:
    def fake_post(
        url: str, *, json: dict[str, object], timeout: float
    ) -> httpx.Response:
        return httpx.Response(
            200,
            json={"embeddings": [[1.0, 0.0]]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("llm.ollama_client.httpx.post", fake_post)
    client = OllamaClient("http://localhost:11434", "embeddinggemma", 45.0)

    with pytest.raises(ValueError, match="different number"):
        client.embed(["first text", "second text"])
