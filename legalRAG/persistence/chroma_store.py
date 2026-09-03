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

        self.collection.upsert(
            ids=[passage.id for passage in passages],
            embeddings=embeddings,
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
        self, query_embedding: list[float], limit: int
    ) -> list[tuple[str, float]]:
        if limit < 1:
            raise ValueError("Search limit must be at least 1.")

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["distances"],
        )
        ids = results["ids"][0]
        distances = results["distances"][0]

        return [
            (passage_id, 1.0 - distance)
            for passage_id, distance in zip(ids, distances, strict=True)
        ]
