import json
import sqlite3
from pathlib import Path

from domain.models import Document, Passage


class SQLiteStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON;")
        return connection

    def create_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    jurisdiction TEXT NOT NULL,
                    effective_date TEXT,
                    version_label TEXT
                );
                
                CREATE TABLE IF NOT EXISTS passages (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    page_start INTEGER NOT NULL,
                    page_end INTEGER NOT NULL,
                    section_path TEXT NOT NULL,
                    parent_id TEXT,
                    FOREIGN KEY (document_id) REFERENCES documents(id)
                );
                """
            )

    def upsert_document(self, document: Document) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    id,
                    title,
                    kind,
                    source_path,
                    jurisdiction,
                    effective_date,
                    version_label
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    kind = excluded.kind,
                    source_path = excluded.source_path,
                    jurisdiction = excluded.jurisdiction,
                    effective_date = excluded.effective_date,
                    version_label = excluded.version_label;
                """,
                (
                    document.id,
                    document.title,
                    document.kind.value,
                    str(document.source_path),
                    document.jurisdiction,
                    document.effective_date,
                    document.version_label,
                ),
            )

    def upsert_passage(self, passage: Passage) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO passages (
                    id,
                    document_id,
                    text,
                    page_start,
                    page_end,
                    section_path,
                    parent_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    document_id = excluded.document_id,
                    text = excluded.text,
                    page_start = excluded.page_start,
                    page_end = excluded.page_end,
                    section_path = excluded.section_path,
                    parent_id = excluded.parent_id;
                """,
                (
                    passage.id,
                    passage.document_id,
                    passage.text,
                    passage.page_start,
                    passage.page_end,
                    json.dumps(passage.section_path),
                    passage.parent_id,
                ),
            )

    def get_passage(self, passage_id: str) -> Passage | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM passages WHERE id = ?;",
                (passage_id,),
            ).fetchone()

        if row is None:
            return None

        return self._passage_from_row(row)

    def keyword_search(self, query: str, limit: int) -> list[Passage]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM passages
                WHERE instr(lower(text), lower(?)) > 0
                ORDER BY page_start, id
                LIMIT ?;
                """,
                (query, limit),
            ).fetchall()

        return [self._passage_from_row(row) for row in rows]

    @staticmethod
    def _passage_from_row(row: sqlite3.Row) -> Passage:
        return Passage(
            id=row["id"],
            document_id=row["document_id"],
            text=row["text"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            section_path=json.loads(row["section_path"]),
            parent_id=row["parent_id"],
        )
