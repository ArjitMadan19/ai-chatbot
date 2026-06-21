from typing import List, Literal, Optional

from pydantic import BaseModel


class SourceDocument(BaseModel):
    title: Optional[str] = None
    doc_type: Optional[str] = None
    file_name: Optional[str] = None
    chunk_id: Optional[int] = None
    source: Optional[str] = None


class AskRequest(BaseModel):
    question: str
    doc_type_filter: Optional[Literal["contract", "research_paper", "notes"]] = None
    session_id: Optional[int] = None


class AskResponse(BaseModel):
    session_id: int
    answer: str
    sources: List[SourceDocument]


class ChatMessageResponse(BaseModel):
    id: int
    question: str
    answer: str
    doc_type_filter: Optional[str] = None
    sources: List[SourceDocument]


class ChatSessionResponse(BaseModel):
    id: int
    messages: List[ChatMessageResponse]
