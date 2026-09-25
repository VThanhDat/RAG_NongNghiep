from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check() -> dict:
    return {"status": "ok", "service": "rag-nong-nghiep-api"}
