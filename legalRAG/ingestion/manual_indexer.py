from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass, replace
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
    segmentation_profile: str = "legacy"
    
    
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

SEGMENTATION_VERSION = "legal-clauses-v4"
LEGAL_MARKER = re.compile(
    r"^[ \t]*(?:(?P<markdown>#{1,6})[ \t]+(?P<title>[^\n]+)|"
    r"(?P<label>\d{1,3}[.)]|\(\d{1,3}\)|"
    r"(?:[a-z]|[ivxlcdm]+)\)|\((?:[a-z]|[ivxlcdm]+)\))(?=[ \t\n]|$))",
    re.MULTILINE | re.IGNORECASE,
)


@dataclass(frozen=True)
class _Boundary:
    start: int
    body_start: int
    level: int
    kind: str
    label: str
    heading: bool


def _legal_boundaries(text: str) -> list[_Boundary]:
    matches = list(LEGAL_MARKER.finditer(text))
    boundaries: list[_Boundary] = []
    last_alpha = ""
    for index, match in enumerate(matches):
        if match.group("markdown"):
            boundaries.append(
                _Boundary(
                    match.start(),
                    match.end(),
                    len(match.group("markdown")),
                    "SECTION",
                    match.group("title").strip(),
                    True,
                )
            )
            last_alpha = ""
            continue
        label = match.group("label")
        token = label.strip("().").lower()
        if token.isdigit():
            level = 10 if label.endswith(".") else 20
            last_alpha = ""
        elif len(token) > 1 or (
            token in {"i", "v", "x"} and last_alpha != chr(ord(token) - 1)
        ):
            level = 40
        else:
            level = 30
            last_alpha = token
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        gap = re.search(r"\n[ \t]*\n", text[match.start() : end])
        heading_end = match.start() + gap.start() if gap else end
        terminator = re.search(r":-?[ \t]*(?=\n|$)", text[match.start() : heading_end])
        if terminator:
            heading_end = match.start() + terminator.end()
        candidate = text[match.start() : heading_end].strip()
        # A colon-terminated numbered block is an explicit heading. Prose lists
        # remain clauses; we do not guess headings from capitalization alone.
        heading = (
            len(candidate) <= 400
            and candidate.count("\n") <= 5
            and candidate.endswith((":", ":-"))
        )
        boundaries.append(
            _Boundary(
                match.start(),
                heading_end if heading else match.start(),
                level,
                "SECTION" if heading and level == 10 else "CLAUSE",
                " ".join(candidate.split()) if heading else label,
                heading,
            )
        )
    return boundaries


# These cues express enumeration, not any particular legal topic.
_LIST_CUE = re.compile(
    r"\b(?:following|as\s+follows|as\s+under|include[sd]?|consists?\s+of)\b",
    re.IGNORECASE,
)


def _preserve_numbered_lists(text: str, boundaries: list[_Boundary]) -> list[_Boundary]:
    """Group only explicit introductions followed by simple numeric sequences.

    Plain text cannot reliably classify arbitrary multi-paragraph or nested lists.
    Ambiguous groups retain the original boundaries. RESUME is a parser event,
    not a persisted passage: it restores the surrounding section after a list.
    """
    result: list[_Boundary] = []
    index = 0
    while index < len(boundaries):
        first = boundaries[index]
        number = re.fullmatch(r"(\d+)([.)])", first.label)
        lower = boundaries[index - 1].body_start if index else 0
        prefix = text[lower : first.start]
        stripped = prefix.rstrip()
        gaps = list(re.finditer(r"\n[ \t]*\n", stripped))
        intro_start = lower + (gaps[-1].end() if gaps else 0)
        while intro_start < first.start and text[intro_start].isspace():
            intro_start += 1
        intro = text[intro_start : first.start].strip()
        eligible = (
            number is not None
            and number.group(1) == "1"
            and not first.heading
            and intro.endswith((":", ":-"))
            and len(intro) <= 1200
            and _LIST_CUE.search(intro) is not None
            and not LEGAL_MARKER.search(intro)
        )
        if not eligible:
            result.append(first)
            index += 1
            continue
        assert number is not None
        end_index = index
        while end_index < len(boundaries):
            item = boundaries[end_index]
            expected = f"{end_index - index + 1}{number.group(2)}"
            if item.heading or item.label != expected:
                break
            # Matching indentation is another structural signal.
            line = text[item.start :].split("\n", 1)[0]
            first_line = text[first.start :].split("\n", 1)[0]
            if len(line) - len(line.lstrip()) != len(first_line) - len(
                first_line.lstrip()
            ):
                break
            if end_index > index:
                previous = boundaries[end_index - 1]
                between = text[previous.start : item.start].strip()
                if re.search(r"\n[ \t]*\n\s*\S", between):
                    break
            end_index += 1
        if end_index - index < 2:
            result.append(first)
            index += 1
            continue
        limit = (
            boundaries[end_index].start if end_index < len(boundaries) else len(text)
        )
        last = boundaries[end_index - 1]
        gap = re.search(r"\n[ \t]*\n", text[last.start : limit])
        list_end = last.start + gap.start() if gap else limit
        # Only retain enclosing headings still above the list's numbering level.
        enclosing = [b for b in result if b.heading and b.level <= first.level]
        outer_level = enclosing[-1].level if enclosing else 0
        level = outer_level + 1
        result.append(
            _Boundary(
                intro_start, first.start, level, "LIST", " ".join(intro.split()), True
            )
        )
        result.extend(replace(b, level=level + 1) for b in boundaries[index:end_index])
        if list_end < len(text):
            result.append(_Boundary(list_end, list_end, level, "RESUME", "", False))
        index = end_index
    return result


def segmentation_profile(max_chars: int) -> str:
    return f"{SEGMENTATION_VERSION}:max_chars={max_chars}"


def segment(
    document_id: str,
    version_id: str,
    text: str,
    pages: list[SourcePage],
    max_chars: int,
) -> list[IndexedPassage]:
    """Segment explicit legal numbering while retaining exact source slices."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive.")
    if not text.strip():
        return []
    result: list[IndexedPassage] = []
    profile = segmentation_profile(max_chars)

    def add(
        kind: str,
        start: int,
        end: int,
        parent_id: str | None,
        path: list[str],
        retrievable: bool = False,
    ) -> str:
        identity = f"{version_id}:{profile}:{kind}:{start}:{end}"
        passage_id = hashlib.sha256(identity.encode()).hexdigest()
        touched = [page for page in pages if page.start < end and page.end > start]
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

    def emit(start: int, end: int, parent_id: str, path: list[str]) -> None:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        while start < end:
            stop = min(start + max_chars, end)
            if stop < end:
                window = text[start:stop]
                sentences = [
                    match
                    for match in re.finditer(r"(?<=[.!?])\s+", window)
                    if not re.fullmatch(r"\s*\d+\.", window[: match.start()])
                ]
                spaces = list(re.finditer(r"\s+", window))
                candidates = sentences or spaces
                if candidates:
                    stop = start + candidates[-1].end()
            add("RETRIEVAL", start, stop, parent_id, path, True)
            start = stop

    def paragraphs(start: int, end: int, parent_id: str, path: list[str]) -> None:
        cursor = start
        separators = list(re.finditer(r"\n[ \t]*\n+", text[start:end]))
        for stop in [start + match.start() for match in separators] + [end]:
            left, right = cursor, stop
            while left < right and text[left].isspace():
                left += 1
            while right > left and text[right - 1].isspace():
                right -= 1
            if left < right:
                paragraph_id = add("PARAGRAPH", left, right, parent_id, path)
                emit(left, right, paragraph_id, path)
            cursor = stop

    root_id = add("DOCUMENT", 0, len(text), None, [])
    boundaries = _preserve_numbered_lists(text, _legal_boundaries(text))
    paragraphs(0, boundaries[0].start if boundaries else len(text), root_id, [])
    stack: list[tuple[int, str, str]] = []
    for index, boundary in enumerate(boundaries):
        while stack and stack[-1][0] >= boundary.level:
            stack.pop()
        parent_id = stack[-1][1] if stack else root_id
        if boundary.kind == "RESUME":
            body_end = (
                boundaries[index + 1].start
                if index + 1 < len(boundaries)
                else len(text)
            )
            paragraphs(boundary.start, body_end, parent_id, [item[2] for item in stack])
            continue
        path = [item[2] for item in stack] + [boundary.label]
        node_end = next(
            (
                item.start
                for item in boundaries[index + 1 :]
                if item.level <= boundary.level
            ),
            len(text),
        )
        node_id = add(
            boundary.kind, boundary.start, node_end, parent_id, path,
            retrievable=boundary.kind == "LIST" and node_end - boundary.start <= max_chars,
        )
        stack.append((boundary.level, node_id, boundary.label))
        body_end = (
            boundaries[index + 1].start if index + 1 < len(boundaries) else len(text)
        )
        if boundary.heading:
            paragraphs(boundary.body_start, body_end, node_id, path)
        else:
            emit(boundary.start, body_end, node_id, path)
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

            connection.execute("BEGIN IMMEDIATE")
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(version_passages)")
            }
            if "segmentation_key" not in columns:
                connection.execute(
                    "ALTER TABLE version_passages ADD COLUMN "
                    "segmentation_key TEXT NOT NULL DEFAULT 'legacy'"
                )
            connection.commit()
            connection.executescript(
                """
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS active_segmentations (
                    version_id TEXT PRIMARY KEY REFERENCES source_versions(id),
                    profile TEXT NOT NULL
                );
                INSERT OR IGNORE INTO active_segmentations(version_id, profile)
                SELECT DISTINCT version_id, 'legacy' FROM version_passages
                WHERE segmentation_key = 'legacy';
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
                DROP VIEW IF EXISTS latest_retrieval_passages;
                CREATE VIEW latest_retrieval_passages AS
                SELECT p.* FROM version_passages p
                JOIN source_versions v ON v.id = p.version_id
                JOIN active_segmentations a ON a.version_id = p.version_id
                    AND a.profile = p.segmentation_key
                WHERE p.retrievable = 1 AND v.version_number = (
                    SELECT MAX(newer.version_number) FROM source_versions newer
                    JOIN active_segmentations ready ON ready.version_id = newer.id
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
        *,
        segmentation_key: str = "legacy",
        activate: bool = True,
    ) -> bool:
        """Keep source versions immutable; append a new passage profile if needed."""
        checksum = hashlib.sha256(source_bytes).hexdigest()
        metadata_json = metadata.to_json()
        with closing(self.connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM source_versions WHERE sha256 = ?", (checksum,)
            ).fetchone()
            reused = existing is not None
            if existing is not None:
                if existing["document_id"] != metadata.document_id:
                    raise ValueError("This source belongs to another document.")
                if existing["metadata_json"] != metadata_json:
                    raise ValueError(
                        "Existing immutable version has different metadata."
                    )
                if existing["id"] != version_id:
                    raise ValueError(
                        "Source version ID does not match existing source."
                    )
            else:
                next_number = connection.execute(
                    "SELECT COALESCE(MAX(version_number), 0) + 1 "
                    "FROM source_versions WHERE document_id = ?",
                    (metadata.document_id,),
                ).fetchone()[0]
                connection.execute(
                    """INSERT INTO source_versions (
                        id, document_id, version_number, sha256,
                        source_bytes, normalized_text, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
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
            present = connection.execute(
                "SELECT 1 FROM version_passages "
                "WHERE version_id = ? AND segmentation_key = ? LIMIT 1",
                (version_id, segmentation_key),
            ).fetchone()
            if present is None:
                # Preserve the existing UNIQUE(version_id, ordinal) constraint.
                first_ordinal = connection.execute(
                    "SELECT COALESCE(MAX(ordinal), -1) + 1 "
                    "FROM version_passages WHERE version_id = ?",
                    (version_id,),
                ).fetchone()[0]
                connection.executemany(
                    """INSERT INTO version_passages (
                        id, version_id, ordinal, kind, char_start, char_end,
                        retrievable, passage_json, segmentation_key
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        (
                            item.passage.id,
                            version_id,
                            first_ordinal + ordinal,
                            item.kind,
                            item.char_start,
                            item.char_end,
                            int(item.retrievable),
                            item.passage.model_dump_json(),
                            segmentation_key,
                        )
                        for ordinal, item in enumerate(passages)
                    ],
                )
            if activate:
                connection.execute(
                    "INSERT INTO active_segmentations VALUES (?, ?) "
                    "ON CONFLICT(version_id) DO UPDATE SET profile = excluded.profile",
                    (version_id, segmentation_key),
                )
        return reused

    def activate_segmentation(self, version_id: str, profile: str) -> None:
        """Publish a profile only after its derived embeddings have been stored."""
        with closing(self.connect()) as connection, connection:
            present = connection.execute(
                "SELECT 1 FROM version_passages WHERE version_id = ? "
                "AND segmentation_key = ? AND retrievable = 1 LIMIT 1",
                (version_id, profile),
            ).fetchone()
            if present is None:
                raise ValueError("No retrieval passages exist for this profile.")
            connection.execute(
                "INSERT INTO active_segmentations VALUES (?, ?) "
                "ON CONFLICT(version_id) DO UPDATE SET profile = excluded.profile",
                (version_id, profile),
            )

    def get_passage(self, passage_id: str) -> Passage | None:
        with closing(self.connect()) as connection:
            row = connection.execute(
                "SELECT passage_json FROM version_passages WHERE id = ?",
                (passage_id,),
            ).fetchone()
        return Passage.model_validate_json(row[0]) if row else None

    def retrieval_passages(
        self, version_id: str, *, profile: str | None = None
    ) -> list[Passage]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """SELECT passage_json FROM version_passages
                WHERE version_id = ? AND retrievable = 1
                  AND segmentation_key = COALESCE(?, (
                      SELECT profile FROM active_segmentations WHERE version_id = ?
                  ))
                ORDER BY ordinal""",
                (version_id, profile, version_id),
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
    version_id = str(uuid5(NAMESPACE_URL, f"{metadata.document_id}:{checksum}"))

    passages = segment(metadata.document_id, version_id, text, pages, max_chars)
    if not any(item.retrievable for item in passages):
        raise ValueError("The document contains no body paragraphs.")

    profile = segmentation_profile(max_chars)
    reused = store.save(
        metadata,
        source_bytes,
        text,
        version_id,
        passages,
        segmentation_key=profile,
        activate=False,
    )

    # Always use persisted passages on retry, including their original IDs
    # and segmentation. Chroma is a rebuildable index.
    retrieval_passages = store.retrieval_passages(version_id, profile=profile)

    for start in range(0, len(retrieval_passages), batch_size):
        batch = retrieval_passages[start : start + batch_size]
        inputs = ["\n".join([*passage.section_path, passage.text]) for passage in batch]
        embeddings = embedder.embed(inputs)
        chroma.upsert_passages(batch, embeddings)

    store.activate_segmentation(version_id, profile)
    return IndexResult(
        version_id=version_id,
        passage_ids=tuple(passage.id for passage in retrieval_passages),
        reused_version=reused,
        segmentation_profile=profile,
    )
