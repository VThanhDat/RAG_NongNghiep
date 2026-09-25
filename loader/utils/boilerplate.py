"""
loader/utils/boilerplate.py
===========================
Remove repeated header/footer boilerplate from page-level documents.
"""

from __future__ import annotations

from langchain_core.documents import Document

from loader.utils.text import clean_text


def strip_repeated_boilerplate(
    docs: list[Document],
    *,
    min_repeats: int = 3,
    edge_lines: int = 3,
    max_line_chars: int = 120,
) -> list[Document]:
    """
    Remove repeated short header/footer lines from page-level documents.

    The heuristic only considers the first/last `edge_lines` lines of each
    page, so repeated content inside the body is preserved.
    """
    if len(docs) < min_repeats:
        return docs

    line_counts: dict[str, int] = {}
    for doc in docs:
        lines = [line.strip() for line in doc.page_content.splitlines() if line.strip()]
        edge = lines[:edge_lines] + lines[-edge_lines:]
        for line in set(edge):
            if len(line) <= max_line_chars:
                line_counts[line] = line_counts.get(line, 0) + 1

    threshold = min(min_repeats, max(2, len(docs) // 2))
    repeated = {line for line, count in line_counts.items() if count >= threshold}
    if not repeated:
        return docs

    for doc in docs:
        lines = doc.page_content.splitlines()
        nonblank_indexes = [i for i, line in enumerate(lines) if line.strip()]
        edge_indexes = set(nonblank_indexes[:edge_lines] + nonblank_indexes[-edge_lines:])
        cleaned = [
            line
            for i, line in enumerate(lines)
            if not (i in edge_indexes and line.strip() in repeated)
        ]
        doc.page_content = clean_text("\n".join(cleaned))
        doc.metadata["boilerplate_removed"] = True
    return docs
