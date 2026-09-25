from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    history: list[dict] = Field(default_factory=list)
    retrieval_only: bool = False
    knowledge_group: str | None = None


class Source(BaseModel):
    file_name: str | None = None
    source: str | None = None
    page_number: int | None = None
    title: str | None = None
    section: str | None = None
    knowledge_group: str | None = None
    text: str


class ChatResponse(BaseModel):
    success: bool
    answer: str = ""
    query_used: str = ""
    sources: list[Source] = Field(default_factory=list)
    error: str | None = None
