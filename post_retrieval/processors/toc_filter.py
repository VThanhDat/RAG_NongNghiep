"""
TOC (Table of Contents) filter for post-retrieval stage.

Mục lục PDF thường không mang thông tin thực sự và gây nhiễu khi được
đưa vào context generation. Filter này phát hiện và loại các chunk
nghi là mục lục dựa trên các heuristic sau:

1. dot_ratio     : tỉ lệ ký tự '.' quá cao → dấu hiệu của "Chương 1 ...... 5"
2. short_lines   : tỉ lệ dòng ngắn cao → mục lục thường là danh sách dòng ngắn
3. min_chars     : chunk quá ngắn không đủ ngữ cảnh → loại trực tiếp
4. early_page    : trang đầu tài liệu (trang 1–3) kết hợp với dòng ngắn hoặc
                   từ khóa mục lục → rất có khả năng là TOC/lời mở đầu không
                   có thông tin cụ thể.

Chunk bị loại khi thỏa ≥ 2 heuristic để tránh false positive.
"""
from __future__ import annotations

import logging
import re

from langchain_core.documents import Document

from post_retrieval.core.base import BasePostProcessor

logger = logging.getLogger(__name__)

# Chuỗi dấu chấm liên tiếp dài (≥ 4) — đặc trưng của dòng mục lục
_DOT_LEADER_RE = re.compile(r"\.{4,}")
# Số trang ở cuối dòng: khoảng trắng rồi 1-3 chữ số
_PAGE_NUM_RE = re.compile(r"\s+\d{1,3}\s*$", re.MULTILINE)
# Từ khóa mục lục phổ biến trong tiếng Việt và tiếng Anh
_TOC_KEYWORD_RE = re.compile(
    r"\b(mục\s*lục|table\s+of\s+contents|contents|danh\s+mục|chương\s+\d|phần\s+\d"
    r"|chapter\s+\d|section\s+\d)\b",
    re.IGNORECASE,
)


class TocFilter(BasePostProcessor):
    """Phát hiện và loại chunk nghi là mục lục khỏi context generation.

    Parameters
    ----------
    dot_ratio        : Ngưỡng tỉ lệ ký tự '.' / tổng chars để kích hoạt heuristic 1.
                       Mặc định 0.15 (15%).
    short_line_ratio : Ngưỡng tỉ lệ dòng ngắn (< 60 ký tự) / tổng số dòng để kích
                       hoạt heuristic 2. Mặc định 0.70 (70%).
    min_chars        : Chunk có ít hơn số ký tự này bị coi là quá ngắn (heuristic 3).
                       Mặc định 200.
    early_page_limit : Trang ≤ giá trị này được coi là "trang đầu" (heuristic 4).
                       Mặc định 3. Đặt 0 để tắt heuristic này.
    keep_min         : Luôn giữ lại ít nhất n doc dù tất cả đều nghi là TOC.
                       Mặc định 1.
    """

    def __init__(
        self,
        dot_ratio: float = 0.15,
        short_line_ratio: float = 0.70,
        min_chars: int = 200,
        early_page_limit: int = 3,
        keep_min: int = 1,
    ):
        self.dot_ratio = dot_ratio
        self.short_line_ratio = short_line_ratio
        self.min_chars = min_chars
        self.early_page_limit = early_page_limit
        self.keep_min = keep_min

    def process(self, query: str, docs: list[Document]) -> list[Document]:
        kept: list[Document] = []
        removed: list[int] = []

        for idx, doc in enumerate(docs, start=1):
            page = self._page_number(doc)
            if self._is_toc(doc.page_content, page_number=page):
                removed.append(idx)
                logger.info(
                    "TocFilter: loại NGUON %d (trang %s, nghi mục lục)", idx, page
                )
            else:
                kept.append(doc)

        # Đảm bảo luôn còn ít nhất keep_min doc
        if len(kept) < self.keep_min and docs:
            needed = self.keep_min - len(kept)
            filtered_back = [docs[i - 1] for i in removed[:needed]]
            logger.warning(
                "TocFilter: khôi phục %d doc do keep_min=%d", needed, self.keep_min
            )
            kept = filtered_back + kept

        return kept

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _page_number(doc: Document) -> int | None:
        """Trích số trang từ metadata (thử nhiều key khác nhau)."""
        for key in ("page_number", "page", "page_num"):
            val = doc.metadata.get(key)
            if val is not None:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    pass
        return None

    def _is_toc(self, text: str, page_number: int | None = None) -> bool:
        """Trả về True nếu chunk nghi là mục lục (thỏa ≥ 2 heuristic)."""
        hits = 0
        total_chars = len(text)

        # Heuristic 1 — dot_ratio: tỉ lệ '.' cao hoặc có chuỗi dấu chấm dài
        if total_chars > 0:
            has_dot_leaders = bool(_DOT_LEADER_RE.search(text))
            if has_dot_leaders or (text.count(".") / total_chars) >= self.dot_ratio:
                hits += 1

        # Heuristic 2 — short_line_ratio: phần lớn dòng rất ngắn
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if lines:
            short = sum(1 for ln in lines if len(ln.strip()) < 60)
            if (short / len(lines)) >= self.short_line_ratio:
                hits += 1

        # Heuristic 3 — min_chars: chunk quá ngắn
        if total_chars < self.min_chars:
            hits += 1

        # Bonus A: có nhiều số trang ở cuối dòng (≥ 3)
        if len(_PAGE_NUM_RE.findall(text)) >= 3:
            hits += 1

        # Heuristic 4 — early_page: trang đầu + dòng ngắn + từ khóa mục lục
        if (
            self.early_page_limit > 0
            and page_number is not None
            and page_number <= self.early_page_limit
        ):
            has_toc_keyword = bool(_TOC_KEYWORD_RE.search(text))
            # Trang đầu mà dòng ngắn chiếm nhiều hoặc có từ khóa mục lục = TOC
            short_line_flag = lines and (
                sum(1 for ln in lines if len(ln.strip()) < 60) / len(lines)
            ) >= 0.55  # threshold thấp hơn cho trang đầu
            if has_toc_keyword or short_line_flag:
                hits += 1

        return hits >= 2
