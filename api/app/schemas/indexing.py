from __future__ import annotations

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    source_path: str
    group_path: str
    files: list[str]
    group: str = "general"


class IndexingRequest(BaseModel):
    source_path: str = Field(..., description="PDF file or folder containing PDFs.")


class IndexingResponse(BaseModel):
    success: bool
    source_path: str
    n_docs: int = 0
    n_chunks: int = 0
    n_vectors: int = 0
    groups: list[str] = Field(default_factory=list)
    loaded_from_disk: bool = False
    error: str | None = None
