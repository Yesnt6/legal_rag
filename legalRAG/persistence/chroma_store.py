from collections.abc import Sequence
from pathlib import Path

import chromadb

from domain.schema import Passage


class ChromaStore:
    def __init__(self, db_path: Path, collection_name: str = "legal_passages"):
        client = chromadb.PersistentClient(path=str(db_path))
        self.collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    def upsert_passages(
        self, passages: list[Passage], embeddings: list[list[float]]
    ) -> None:
        if not passages:
            return

        if len(passages) != len(embeddings):
            raise ValueError("Each passage must have exactly one embedding.")

        vectors: list[Sequence[float] | Sequence[int]] = [
            vector for vector in embeddings
        ]
        self.collection.upsert(
            ids=[passage.id for passage in passages],
            embeddings=vectors,
            metadatas=[
                {
                    "document_id": passage.document_id,
                    "page_start": passage.page_start,
                    "page_end": passage.page_end,
                }
                for passage in passages
            ],
        )

    def search(
        self,
        query_embedding: list[float],
        limit: int,
        *,
        passage_ids: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        if limit < 1:
            raise ValueError("Search limit must be at least 1.")

        if passage_ids == []:
            return []

        results = self.collection.query(
            query_embeddings=query_embedding,
            ids=passage_ids,
            n_results=limit,
            include=["distances"],
        )
        ids = results["ids"][0]
        distance_rows = results["distances"]
        if distance_rows is None:
            raise ValueError("Chroma returned no distances for a vector search.")
        distances = distance_rows[0]

        return [
            (passage_id, 1.0 - distance)
            for passage_id, distance in zip(ids, distances, strict=True)
        ]
