from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document


def documents_path_from_config(cfg: dict) -> Path:
    db_cfg = cfg.get("indexing", {}).get("vector_db", {})
    default_dir = "./storage/pinecone"
    return Path(db_cfg.get("documents_path") or Path(db_cfg.get("persist_dir", default_dir)) / "documents.jsonl")


def save_documents(docs: list[Document], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for doc in docs:
            file.write(
                json.dumps(
                    {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata or {},
                    },
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )
    return output_path


def load_documents(path: str | Path) -> list[Document]:
    input_path = Path(path)
    docs: list[Document] = []
    with input_path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            docs.append(
                Document(
                    page_content=item.get("page_content") or item.get("text") or "",
                    metadata=item.get("metadata") or {},
                )
            )
    return docs


def index_artifacts_exist(cfg: dict) -> bool:
    return documents_path_from_config(cfg).exists()
