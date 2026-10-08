from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from domain.schema import DocumentKind, Passage
from persistence.chroma_store import ChromaStore

PAGE_MARKER = re.compile(r"^\[PAGE ([1-9]\d*)\][ \t]*$", re.MULTILINE)
HEADING = re.compile(r"^(#{1,6})[ \t]+(.+)$", re.MULTILINE)

class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    
@dataclass(frozen=True)
class DocumentMetadata:
    document_id: str
    title: str
    kind: DocumentKind
    issuing_authority: str
    publication_date: date
    jurisdiction: str
    effective_from: date | None = None
    effective_until: date | None = None
    
    def validate(self) -> None:
        for value in (
            self.document_id,
            self.title,
            self.issuing_authority,
            self.jurisdiction,
        ):
            if not value.strip():
                raise ValueError(f"Required document metadata cannot be blank.")
            
        if (
            self.effective_from
            and self.effective_until
            and self.effective_from > self.effective_until
        ):
            raise ValueError(f"Effective from date must be before effective until date.")
        
    def to_json(self) -> str:
        return json.dumps(
            {
                "document_id": self.document_id,
                "title": self.title,
                "kind": self.kind.value,
                "issuing_authority": self.issuing_authority,
                "publication_date": self.publication_date.isoformat(),
                "jurisdiction": self.jurisdiction,
                "effective_from": self.effective_from.isoformat() if self.effective_from else None,
                "effective_until": self.effective_until.isoformat() if self.effective_until else None,
            },
            sort_keys=True,
        )
    
@dataclass(frozen=True)
class SourcePage:
    number: int
    start: int
    end: int
    
@dataclass(frozen=True)
class IndexedPassage:
    passage: Passage
    kind: str 
    char_start: int
    char_end: int
    retrievable: bool

@dataclass(frozen=True)
class IndexResult:
    version_id: str
    passage_ids: tuple[str, ...]
    reused_version: bool
    
    
def parse_pages(source: str) -> tuple[str, list[SourcePage]]:
    source = source.replace("\r\n", "\n").replace("\r", "\n")
    markers = list(PAGE_MARKER.finditer(source))
    if not markers or source[:markers[0].start()].strip():
        raise ValueError("Input must begin with a [PAGE N] marker.")
    numbers = [int(marker.group(1)) for marker in markers]
    if numbers != list(range(1, len(numbers) + 1)):
        raise ValueError("Page numbers must be consecutive, starting at 1.")
    
    parts: list[str] = []
    pages: list[SourcePage] = []
    offset = 0
    
    for index, marker in enumerate(markers):
        start = marker.end()
        if source[start:start + 1] == "\n":
            start += 1
            
        end = (
            markers[index + 1].start()
            if index + 1 < len(markers)
            else len(source)
        )
        content = source[start:end]
        
        if index + 1 < len(markers) and not content.endswith("\n"):
            content += "\n"
            
        parts.append(content)
        pages.append(SourcePage(number=numbers[index], start=offset, end=offset + len(content)))
        offset += len(content)
        
        text = "".join(parts)
    if not text.strip():
        raise ValueError("The document contains no source text.")
    
    return text, pages

def segment(
    document_id: str,
    version_id: str,
    text: str,
    pages: list[SourcePage],
    max_chars: int,
) -> list[IndexedPassage]:
    
    if max_chars <= 0:
        raise ValueError("Maximum character length must be non-negative.")
    
    result: list[IndexedPassage] = []
    
    def add(
        kind: str,
        start: int,
        end: int,
        parent_id: str | None,
        path: list[str],
        retrievable: bool = False,
    ) -> str:
        identity = f"{version_id}:{kind}:{start}:{end}"
        passage_id = hashlib.sha256(identity.encode()).hexdigest()
        touched = [
            page for page in pages
            if page.start < end and page.end > start
        ]
        
        result.append(
            IndexedPassage(
                passage=Passage(
                    id=passage_id,
                    document_id=document_id,
                    text=text[start:end],
                    page_start=touched[0].number,
                    page_end=touched[-1].number,
                    section_path=list(path),
                    parent_id=parent_id,
                ),
                kind=kind,
                char_start=start,
                char_end=end,
                retrievable=retrievable,
            )
        )
        return passage_id
    
    root_id = add("DOCUMENT", 0, len(text), None, [])
    headings = list(HEADING.finditer(text))
    stack: list[tuple[int, str, str]] = []
    
    def add_body(start: int, end: int, parent_id: str, path: list[str]) -> None:
        cursor = start
        boundaries = [
            match.span()
            for match in re.finditer(r"\n[ \t]*\n+", text[start:end])
        ]
        
        for separator_start, separator_end in boundaries + [(end - start, end - start)]:
            stop = start + separator_start
            left, right = cursor, stop

            while left < right and text[left].isspace():
                left += 1
            while right > left and text[right - 1].isspace():
                right -= 1

            if left < right:
                paragraph_id = add("PARAGRAPH", left, right, parent_id, path)
                chunk_start = left

                while chunk_start < right:
                    chunk_end = min(chunk_start + max_chars, right)
                    if chunk_end < right:
                        spaces = list(
                            re.finditer(r"\s+", text[chunk_start:chunk_end])
                        )
                        if spaces:
                            chunk_end = chunk_start + spaces[-1].end()

                    add(
                        "RETRIEVAL",
                        chunk_start,
                        chunk_end,
                        paragraph_id,
                        path,
                        retrievable=True,
                    )
                    chunk_start = chunk_end

            cursor = start + separator_end

    add_body(0, headings[0].start() if headings else len(text), root_id, [])

    for index, heading in enumerate(headings):
        level = len(heading.group(1))
        while stack and stack[-1][0] >= level:
            stack.pop()

        parent_id = stack[-1][1] if stack else root_id
        path = [item[2] for item in stack] + [heading.group(2).strip()]

        section_end = len(text)
        for following in headings[index + 1:]:
            if len(following.group(1)) <= level:
                section_end = following.start()
                break

        section_id = add(
            "SECTION", heading.start(), section_end, parent_id, path
        )
        stack.append((level, section_id, heading.group(2).strip()))

        body_end = (
            headings[index + 1].start()
            if index + 1 < len(headings)
            else len(text)
        )
        add_body(heading.end(), body_end, section_id, path)

    return result

        

class VersionedStore:
    """SQLite owns immutable source snapshots and passage boundaries."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

        with closing(self.connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS source_versions (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    version_number INTEGER NOT NULL,
                    sha256 TEXT UNIQUE NOT NULL,
                    source_bytes BLOB NOT NULL,
                    normalized_text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(document_id, version_number)
                );

                CREATE TABLE IF NOT EXISTS version_passages (
                    id TEXT PRIMARY KEY,
                    version_id TEXT NOT NULL REFERENCES source_versions(id),
                    ordinal INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    char_start INTEGER NOT NULL,
                    char_end INTEGER NOT NULL,
                    retrievable INTEGER NOT NULL,
                    passage_json TEXT NOT NULL,
                    UNIQUE(version_id, ordinal)
                );
                """
            )

            connection.executescript(
                """
                BEGIN IMMEDIATE;
                CREATE VIRTUAL TABLE IF NOT EXISTS passage_fts USING fts5(
                    text, heading_path, tokenize='unicode61'
                );
                CREATE TRIGGER IF NOT EXISTS version_passages_fts_insert
                AFTER INSERT ON version_passages WHEN new.retrievable = 1
                BEGIN
                    INSERT INTO passage_fts(rowid, text, heading_path)
                    VALUES (
                        new.rowid,
                        json_extract(new.passage_json, '$.text'),
                        json_extract(new.passage_json, '$.section_path')
                    );
                END;
                INSERT INTO passage_fts(rowid, text, heading_path)
                SELECT p.rowid, json_extract(p.passage_json, '$.text'),
                       json_extract(p.passage_json, '$.section_path')
                FROM version_passages p
                WHERE p.retrievable = 1
                  AND p.rowid NOT IN (SELECT rowid FROM passage_fts);
                CREATE VIEW IF NOT EXISTS latest_retrieval_passages AS
                SELECT p.* FROM version_passages p
                JOIN source_versions v ON v.id = p.version_id
                WHERE p.retrievable = 1 AND v.version_number = (
                    SELECT MAX(newer.version_number) FROM source_versions newer
                    WHERE newer.document_id = v.document_id
                );
                COMMIT;
                """
            )

    def latest_passage_ids(self) -> list[str]:
        """Return retrievable IDs from the newest stored version per document."""
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT id FROM latest_retrieval_passages ORDER BY id"
            ).fetchall()
        return [row[0] for row in rows]

    def keyword_search(
        self, query: str, limit: int, *, passage_ids: list[str] | None = None
    ) -> list[Passage]:
        """Rank literal query terms by BM25, restricted to eligible passage IDs."""
        if limit < 1:
            raise ValueError("Search limit must be at least 1.")
        # Quote each term so punctuation and FTS operators cannot become syntax.
        terms = list(dict.fromkeys(re.findall(r"[^\W_]+", query.casefold())))
        if not terms:
            return []
        match_query = " OR ".join(f'"{term}"' for term in terms)
        if passage_ids is None:
            passage_ids = self.latest_passage_ids()
        if not passage_ids:
            return []
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """
                SELECT p.passage_json FROM passage_fts
                JOIN version_passages p ON p.rowid = passage_fts.rowid
                WHERE passage_fts MATCH ?
                  AND p.id IN (SELECT value FROM json_each(?))
                ORDER BY bm25(passage_fts), p.id
                LIMIT ?
                """,
                (match_query, json.dumps(passage_ids), limit),
            ).fetchall()
        return [Passage.model_validate_json(row[0]) for row in rows]

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def save(
        self,
        metadata: DocumentMetadata,
        source_bytes: bytes,
        normalized_text: str,
        version_id: str,
        passages: list[IndexedPassage],
    ) -> bool:
        """Returns True for an identical existing import."""
        checksum = hashlib.sha256(source_bytes).hexdigest()
        metadata_json = metadata.to_json()

        with closing(self.connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM source_versions WHERE sha256 = ?",
                (checksum,),
            ).fetchone()

            if existing is not None:
                if existing["document_id"] != metadata.document_id:
                    raise ValueError("This source belongs to another document.")
                if existing["metadata_json"] != metadata_json:
                    raise ValueError(
                        "Existing immutable version has different metadata."
                    )
                return True

            next_number = connection.execute(
                """
                SELECT COALESCE(MAX(version_number), 0) + 1
                FROM source_versions WHERE document_id = ?
                """,
                (metadata.document_id,),
            ).fetchone()[0]

            connection.execute(
                """
                INSERT INTO source_versions (
                    id, document_id, version_number, sha256,
                    source_bytes, normalized_text, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version_id,
                    metadata.document_id,
                    next_number,
                    checksum,
                    source_bytes,
                    normalized_text,
                    metadata_json,
                ),
            )
            connection.executemany(
                """
                INSERT INTO version_passages (
                    id, version_id, ordinal, kind, char_start,
                    char_end, retrievable, passage_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.passage.id,
                        version_id,
                        ordinal,
                        item.kind,
                        item.char_start,
                        item.char_end,
                        int(item.retrievable),
                        item.passage.model_dump_json(),
                    )
                    for ordinal, item in enumerate(passages)
                ],
            )

        return False

    def get_passage(self, passage_id: str) -> Passage | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT passage_json FROM version_passages WHERE id = ?",
                (passage_id,),
            ).fetchone()
        return Passage.model_validate_json(row[0]) if row else None

    def retrieval_passages(self, version_id: str) -> list[Passage]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """
                SELECT passage_json FROM version_passages
                WHERE version_id = ? AND retrievable = 1
                ORDER BY ordinal
                """,
                (version_id,),
            ).fetchall()
        return [Passage.model_validate_json(row[0]) for row in rows]

    def list_versions(self, document_id: str) -> list[str]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """
                SELECT id FROM source_versions
                WHERE document_id = ? ORDER BY version_number
                """,
                (document_id,),
            ).fetchall()
        return [row[0] for row in rows]


def index_document(
    source_path: Path,
    metadata: DocumentMetadata,
    store: VersionedStore,
    chroma: ChromaStore,
    embedder: Embedder,
    *,
    max_chars: int = 1800,
    batch_size: int = 16,
) -> IndexResult:
    metadata.validate()
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    source_bytes = source_path.read_bytes()
    text, pages = parse_pages(source_bytes.decode("utf-8"))
    checksum = hashlib.sha256(source_bytes).hexdigest()
    version_id = str(
        uuid5(NAMESPACE_URL, f"{metadata.document_id}:{checksum}")
    )

    passages = segment(
        metadata.document_id, version_id, text, pages, max_chars
    )
    if not any(item.retrievable for item in passages):
        raise ValueError("The document contains no body paragraphs.")

    reused = store.save(
        metadata, source_bytes, text, version_id, passages
    )

    # Always use persisted passages on retry, including their original IDs
    # and segmentation. Chroma is a rebuildable index.
    retrieval_passages = store.retrieval_passages(version_id)

    for start in range(0, len(retrieval_passages), batch_size):
        batch = retrieval_passages[start:start + batch_size]
        inputs = [
            "\n".join([*passage.section_path, passage.text])
            for passage in batch
        ]
        embeddings = embedder.embed(inputs)
        chroma.upsert_passages(batch, embeddings)

    return IndexResult(
        version_id=version_id,
        passage_ids=tuple(passage.id for passage in retrieval_passages),
        reused_version=reused,
    )
    
        
    