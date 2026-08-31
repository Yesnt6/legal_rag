from enum import StrEnum
from pathlib import Path
from pydantic import BaseModel, Field

class DocumentKind(StrEnum):
    ACT = "act"
    RULES = "rules"
    BYE_LAWS = "bye_laws"
    CIRCULAR = "circular"
    JUDGEMENT = "judgement"
    CONTRACT = "contract"
    OTHER = "other"

class Document(BaseModel):
    id: str
    title: str
    kind: DocumentKind
    source_path: Path
    jurisdiction: str = "India"
    effective_date: str | None = None
    version_label: str | None= None

class Passage(BaseModel):
    id: str
    document_id: str
    text: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    section_path: list[str] = Field(default_factory=list)
    parent_id: str | None = None

class Citation(BaseModel):
    document_id: str
    passage_id: str
    page_start: int
    page_end: int
    quoted_text: str


class GraphNodeType(StrEnum):
    LAW = "law"
    SECTION = "section"
    REQUIREMENT = "requirement"
    AUTHORITY = "authority"
    SOCIETY = "society"
    MANAGEMENT_COMMITTEE = "management_committee"
    DEVELOPER = "developer"
    MEMBER = "member"
    DOCUMENT = "document"
    PROCESS_STEP = "process_step"


class GraphNode(BaseModel):
    id: str
    type: GraphNodeType
    name: str
    source_passage_id: str


class GraphRelationshipType(StrEnum):
    REQUIRES = "requires"
    PERMITS = "permits"
    PROHIBITS = "prohibits"
    DEFINES = "defines"
    APPLIES_TO = "applies_to"
    ISSUED_BY = "issued_by"
    PART_OF = "part_of"
    PRECEDES = "precedes"


class GraphRelationship(BaseModel):
    id: str
    source_node_id: str
    target_node_id: str
    type: GraphRelationshipType
    source_passage_id: str
    supporting_text: str


class RetrievedEvidence(BaseModel):
    passage: Passage
    score: float
    retrieval_source: str


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    abstained: bool = False
    abstention_reason: str | None = None