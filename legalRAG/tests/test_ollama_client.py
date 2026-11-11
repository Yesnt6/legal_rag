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


def test_answer_client_releases_embedding_model_and_uses_smaller_context(
    monkeypatch,
) -> None:
    requests = []

    def fake_post(url, *, json, timeout):
        requests.append((url, json, timeout))
        body = (
            {"embeddings": [[1.0, 0.0]]}
            if url.endswith("/api/embed")
            else {"message": {"content": '{"value": "ok"}'}}
        )
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    from pydantic import BaseModel

    class Response(BaseModel):
        value: str

    monkeypatch.setattr("llm.ollama_client.httpx.post", fake_post)
    client = OllamaClient(
        "http://localhost:11434", "embeddinggemma", 600,
        embedding_keep_alive="0", generation_context_tokens=4096,
    )

    assert client.embed(["question"]) == [[1.0, 0.0]]
    assert client.generate_json("qwen3.5:4b", [], Response).value == "ok"
    assert requests[0][1]["keep_alive"] == "0"
    assert requests[1][1]["options"]["num_ctx"] == 4096
