from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.app.api.chat import router as chat_router
from api.app.api.health import router as health_router
from api.app.api.indexing import router as indexing_router
from api.app.services.indexing_service import indexing_service


app = FastAPI(title="RAG Nong Nghiep API", version="1.0.0")

configured_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
)
cors_origins = [origin.strip() for origin in configured_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, tags=["health"])
app.include_router(indexing_router, prefix="/indexing", tags=["indexing"])
app.include_router(chat_router, prefix="/chat", tags=["chat"])


@app.on_event("startup")
def load_existing_index() -> None:
    try:
        indexing_service.load_existing(raise_if_missing=False)
    except Exception:
        pass
