from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.app.schemas.indexing import IndexingRequest, IndexingResponse, UploadResponse
from api.app.services.indexing_service import indexing_service

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_pdfs(
    files: list[UploadFile] = File(...),
    group: str = Form("general"),
) -> UploadResponse:
    try:
        return await indexing_service.save_uploads(files, group=group)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/run", response_model=IndexingResponse)
def run_indexing(request: IndexingRequest) -> IndexingResponse:
    try:
        return indexing_service.run(source_path=request.source_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/status")
def indexing_status() -> dict:
    return indexing_service.status()


@router.post("/load", response_model=IndexingResponse)
def load_existing_index() -> IndexingResponse:
    try:
        return indexing_service.load_existing(raise_if_missing=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
