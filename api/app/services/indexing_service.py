from __future__ import annotations

from pathlib import Path
import unicodedata

from fastapi import UploadFile

from pipeline.indexing_pipeline import IndexingPipeline, load_config

from api.app.schemas.indexing import IndexingResponse, UploadResponse
from api.app.services.state import rag_state


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class IndexingService:
    def __init__(self, upload_dir: str = "data/uploads") -> None:
        self.upload_dir = self._resolve_path(upload_dir)
        # Ensure the upload directory exists on startup so users can copy
        # PDF files manually into subfolders without running upload first.
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    async def save_uploads(self, files: list[UploadFile], group: str = "general") -> UploadResponse:
        safe_group = self._sanitize_group(group)
        group_dir = self.upload_dir / safe_group
        group_dir.mkdir(parents=True, exist_ok=True)
        saved: list[str] = []

        for file in files:
            filename = Path(file.filename or "").name
            if not filename.lower().endswith(".pdf"):
                raise ValueError("Only PDF files are supported.")
            target = group_dir / filename
            target.write_bytes(await file.read())
            saved.append(str(target))

        if not saved:
            raise ValueError("No PDF files uploaded.")
        return UploadResponse(
            source_path=str(self.upload_dir),
            group_path=str(group_dir),
            files=saved,
            group=safe_group,
        )

    def run(self, source_path: str) -> IndexingResponse:
        path = self._resolve_path(source_path)
        if not path.exists():
            raise ValueError(
                f"Thư mục '{source_path}' chưa tồn tại. "
                "Hãy upload PDF trước hoặc tạo thư mục và copy file PDF vào "
                "các subfolder (ví dụ: data/uploads/lua/, data/uploads/ca-phe/)."
            )

        # Check that there is at least one PDF to index.
        has_pdfs = any(path.rglob("*.pdf")) if path.is_dir() else path.suffix.lower() == ".pdf"
        if not has_pdfs:
            raise ValueError(
                f"Không tìm thấy file PDF nào trong '{source_path}'. "
                "Hãy upload PDF trước hoặc copy file PDF vào subfolder tương ứng "
                "(ví dụ: data/uploads/lua/tai-lieu.pdf)."
            )

        cfg = load_config("config.yaml")
        result = IndexingPipeline().run(str(path), cfg)

        with rag_state.lock:
            rag_state.cfg = cfg
            rag_state.source_path = str(path)
            rag_state.indexing_result = result
            rag_state.vdb_result = result.vdb_result if result.success else None

        return IndexingResponse(
            success=result.success,
            source_path=str(path),
            n_docs=result.n_docs,
            n_chunks=result.n_chunks,
            n_vectors=result.n_vectors,
            groups=self._groups_from_documents(result.chunks),
            error=result.error,
        )

    def load_existing(self, raise_if_missing: bool = True) -> IndexingResponse:
        cfg = load_config("config.yaml")

        from embedding.factory import get_embedder_from_config
        from pipeline.artifacts import documents_path_from_config, index_artifacts_exist, load_documents
        from vector_db.factory import load_vector_store_from_config

        if not index_artifacts_exist(cfg):
            if raise_if_missing:
                raise ValueError("No saved vector index/chunks found. Run indexing first.")
            return IndexingResponse(success=False, source_path="", error="No saved index found.")

        docs_path = documents_path_from_config(cfg)
        chunks = load_documents(docs_path)
        embedder = get_embedder_from_config(cfg)
        vector_store = load_vector_store_from_config(chunks, embedder, cfg)

        db_cfg = cfg.get("indexing", {}).get("vector_db", {})
        vdb_result = {
            "vector_store": vector_store,
            "documents": chunks,
            "provider": db_cfg.get("provider", "pinecone"),
            "collection_name": db_cfg.get("collection_name", "rag_collection_cohere"),
            "persist_dir": db_cfg.get("persist_dir", "./storage/pinecone"),
            "n_vectors": len(chunks),
            "loaded_from_disk": True,
        }

        with rag_state.lock:
            rag_state.cfg = cfg
            rag_state.source_path = str(docs_path)
            rag_state.indexing_result = None
            rag_state.vdb_result = vdb_result

        return IndexingResponse(
            success=True,
            source_path=str(docs_path),
            n_docs=0,
            n_chunks=len(chunks),
            n_vectors=len(chunks),
            groups=self._groups_from_documents(chunks),
            loaded_from_disk=True,
        )

    def status(self) -> dict:
        with rag_state.lock:
            result = rag_state.indexing_result
            vdb_result = rag_state.vdb_result or {}
            docs = vdb_result.get("documents", [])
            return {
                "ready": bool(rag_state.vdb_result),
                "source_path": rag_state.source_path,
                "n_docs": getattr(result, "n_docs", 0) if result else 0,
                "n_chunks": getattr(result, "n_chunks", 0) if result else len(docs),
                "n_vectors": getattr(result, "n_vectors", 0) if result else int(vdb_result.get("n_vectors", 0)),
                "loaded_from_disk": bool(vdb_result.get("loaded_from_disk")),
                "groups": self._groups_from_documents(docs),
            }

    @staticmethod
    def _sanitize_group(group: str) -> str:
        """Return a filesystem-safe ASCII slug for the given group name.

        Steps (order matters):
        1. Strip whitespace and casefold.
        2. Replace ``đ``/``Đ`` *before* NFD decomposition because U+0111 (đ)
           has no NFD decomposition and would survive the diacritic-removal step.
        3. NFD-decompose and strip combining marks (diacritics).
        4. Replace non-alphanumeric characters with ``-``.
        5. Collapse consecutive ``-`` separators.
        """
        raw = (group or "general").strip().casefold()
        # Must replace Vietnamese Đ/đ before NFD because it has no decomposition.
        raw = raw.replace("đ", "d").replace("Đ", "D")
        raw = unicodedata.normalize("NFD", raw)
        raw = "".join(ch for ch in raw if unicodedata.category(ch) != "Mn")
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in raw)
        safe = "-".join(part for part in safe.split("-") if part)
        return safe or "general"

    @staticmethod
    def _resolve_path(path: str | Path) -> Path:
        p = Path(path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @staticmethod
    def _groups_from_documents(docs: list) -> list[str]:
        groups = {
            str(doc.metadata.get("knowledge_group") or doc.metadata.get("group") or "general")
            for doc in docs
        }
        return sorted(groups)


indexing_service = IndexingService()
