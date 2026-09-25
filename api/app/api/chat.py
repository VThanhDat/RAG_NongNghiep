from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.app.schemas.chat import ChatRequest, ChatResponse
from api.app.services.chat_service import chat_service

router = APIRouter()


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        return chat_service.answer(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
