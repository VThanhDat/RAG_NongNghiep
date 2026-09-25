from __future__ import annotations

from pipeline.generation_pipeline import GenerationPipeline

from api.app.schemas.chat import ChatRequest, ChatResponse, Source
from api.app.services.state import rag_state


class ChatService:
    def answer(self, request: ChatRequest) -> ChatResponse:
        with rag_state.lock:
            cfg = rag_state.cfg
            vdb_result = rag_state.vdb_result

        if not cfg or not vdb_result:
            raise ValueError("Index is not ready. Run indexing first.")

        result = GenerationPipeline().run(
            query=request.question.strip(),
            vdb_result=vdb_result,
            cfg=cfg,
            history=request.history,
            skip_generation=request.retrieval_only,
            metadata_filter=self._metadata_filter(request),
        )

        return ChatResponse(
            success=result.success,
            answer=result.answer,
            query_used=result.query_used,
            sources=[self._source_from_doc(doc) for doc in result.final_docs],
            error=result.error,
        )

    def _source_from_doc(self, doc) -> Source:
        metadata = getattr(doc, "metadata", {}) or {}
        return Source(
            file_name=metadata.get("file_name"),
            source=metadata.get("source"),
            page_number=metadata.get("page_number") or metadata.get("page"),
            title=metadata.get("title"),
            section=metadata.get("section"),
            knowledge_group=metadata.get("knowledge_group") or metadata.get("group"),
            text=(getattr(doc, "page_content", "") or "")[:800],
        )

    @staticmethod
    def _metadata_filter(request: ChatRequest) -> dict | None:
        group = (request.knowledge_group or "").strip()
        if not group or group == "all":
            return None
        return {"knowledge_group": group}


chat_service = ChatService()
