"""
Stage 1 indexing pipeline for RAG Nong Nghiep.

Flow:
    PDF loader -> recursive chunking -> Cohere embedding wrapper
    -> Pinecone vector DB
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml
from langchain_core.documents import Document

try:
    from dotenv import load_dotenv

    load_dotenv(override=True)
except Exception:
    pass


@dataclass
class StepResult:
    step: str
    success: bool
    n_items: int = 0
    meta: dict = field(default_factory=dict)
    error: str | None = None


@dataclass
class IndexingResult:
    success: bool
    steps: list[StepResult] = field(default_factory=list)
    docs: list[Document] = field(default_factory=list)
    chunks: list[Document] = field(default_factory=list)
    embedder: object | None = None
    vector_store: object | None = None
    vdb_result: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def n_docs(self) -> int:
        return len(self.docs)

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    @property
    def n_vectors(self) -> int:
        return int(self.vdb_result.get("n_vectors", len(self.chunks)))


class IndexingPipeline:
    def __init__(self, on_progress: Callable[[str, float, str], None] | None = None):
        self.on_progress = on_progress or (lambda *_: None)

    @classmethod
    def from_config_file(cls, config_path: str = "config.yaml", **kwargs) -> tuple["IndexingPipeline", dict]:
        with open(config_path, encoding="utf-8") as file:
            return cls(**kwargs), yaml.safe_load(file)

    def run(self, source_path: str, cfg: dict) -> IndexingResult:
        result = IndexingResult(success=False)

        try:
            docs = self._run_loader(source_path, cfg)
            self._apply_knowledge_groups(docs, source_path, cfg)
            result.docs = docs
            result.steps.append(StepResult("loader", True, len(docs)))

            chunks = self._run_chunking(docs, cfg)
            result.chunks = chunks
            result.steps.append(StepResult("chunking", True, len(chunks)))

            embedder = self._build_embedder(cfg)
            result.embedder = embedder
            result.steps.append(
                StepResult(
                    "embedding",
                    True,
                    len(chunks),
                    {"provider": embedder.dense_provider, "model": embedder.dense_model},
                )
            )

            vector_store, vdb_result = self._run_vector_db(chunks, embedder, cfg)
            result.vector_store = vector_store
            result.vdb_result = vdb_result
            result.steps.append(StepResult("vector_db", True, vdb_result.get("n_vectors", len(chunks))))

            self._save_artifacts(chunks, cfg)
            result.steps.append(StepResult("artifacts", True, len(chunks)))
        except Exception as exc:
            result.error = str(exc)
            result.steps.append(StepResult("failed", False, error=str(exc)))
            return result

        result.success = True
        return result

    def _run_loader(self, source_path: str, cfg: dict) -> list[Document]:
        self.on_progress("loader", 0.1, "Loading PDF documents")
        from loader import DocumentLoader

        loader_cfg = cfg.get("indexing", {}).get("loader", {})
        data_cfg = cfg.get("data", {})
        loader = DocumentLoader(
            language=data_cfg.get("language", "both"),
            pdf_strategy=loader_cfg.get("pdf_strategy", "pypdf"),
            extract_tables=loader_cfg.get("extract_tables", True),
            ocr=loader_cfg.get("ocr", "never"),
        )
        docs = loader.load(source_path)
        self.on_progress("loader", 1.0, f"Loaded {len(docs)} documents")
        return docs

    def _run_chunking(self, docs: list[Document], cfg: dict) -> list[Document]:
        self.on_progress("chunking", 0.1, "Chunking documents")
        from chunking.factory import chunk_documents_from_config

        chunks = chunk_documents_from_config(docs, cfg)
        self.on_progress("chunking", 1.0, f"Created {len(chunks)} chunks")
        return chunks

    def _build_embedder(self, cfg: dict):
        self.on_progress("embedding", 0.1, "Preparing Cohere embedding API")
        from embedding.factory import get_embedder_from_config

        embedder = get_embedder_from_config(cfg)
        self.on_progress("embedding", 1.0, "Embedding model ready")
        return embedder

    def _run_vector_db(self, chunks: list[Document], embedder, cfg: dict):
        self.on_progress("vector_db", 0.1, "Building configured vector index")
        from vector_db.factory import get_vector_store_from_config

        vector_store = get_vector_store_from_config(chunks, embedder, cfg)
        db_cfg = cfg.get("indexing", {}).get("vector_db", {})
        vdb_result = {
            "vector_store": vector_store,
            "documents": chunks,
            "provider": db_cfg.get("provider", "pinecone"),
            "collection_name": db_cfg.get("collection_name", "rag_collection_cohere"),
            "persist_dir": db_cfg.get("persist_dir", "./storage/pinecone"),
            "n_vectors": int(getattr(vector_store, "n_vectors", len(chunks))),
        }
        self.on_progress("vector_db", 1.0, f"Indexed {len(chunks)} vectors")
        return vector_store, vdb_result

    def _apply_knowledge_groups(self, docs: list[Document], source_path: str, cfg: dict) -> None:
        metadata_cfg = cfg.get("indexing", {}).get("metadata", {})
        default_group = str(metadata_cfg.get("default_group", "general") or "general")
        root = Path(source_path)
        root_dir = root if root.is_dir() or not root.suffix else root.parent

        for doc in docs:
            metadata = doc.metadata or {}
            source = metadata.get("source") or metadata.get("file_path") or ""
            group = metadata.get("knowledge_group") or self._infer_group(source, root_dir, default_group)
            metadata["knowledge_group"] = group
            metadata["group"] = group
            doc.metadata = metadata

    @staticmethod
    def _infer_group(source: str, root_dir: Path, default_group: str) -> str:
        """Infer knowledge group from subfolder name relative to the index root.

        After ``path.relative_to(root_dir)`` the leading root segments are
        already stripped, so ``relative.parts[0]`` is always the first
        subfolder directly inside ``root_dir`` — which is the group name.

        Example:
            root_dir = /project/data/uploads
            source   = /project/data/uploads/lua/tai-lieu-lua.pdf
            relative = lua/tai-lieu-lua.pdf  → parts[0] = "lua"
        """
        try:
            path = Path(str(source))
            relative = path.relative_to(root_dir)
            # File is inside a subfolder of root_dir → that subfolder is the group.
            if len(relative.parts) > 1:
                return relative.parts[0]
        except Exception:
            pass
        return default_group

    def _save_artifacts(self, chunks: list[Document], cfg: dict) -> None:
        self.on_progress("artifacts", 0.1, "Saving indexed chunks")
        from pipeline.artifacts import documents_path_from_config, save_documents

        save_documents(chunks, documents_path_from_config(cfg))
        self.on_progress("artifacts", 1.0, "Saved indexed chunks")


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the RAG Nong Nghiep indexing pipeline.")
    parser.add_argument("--source", required=True, help="PDF file or directory containing PDF files.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pipeline = IndexingPipeline()
    result = pipeline.run(args.source, cfg)
    if not result.success:
        raise SystemExit(f"Indexing failed: {result.error}")

    print(f"Documents: {result.n_docs}")
    print(f"Chunks: {result.n_chunks}")
    print(f"Vectors: {result.n_vectors}")


if __name__ == "__main__":
    _cli()
