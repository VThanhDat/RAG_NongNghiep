"""
Citation prompt for agriculture RAG answers.
"""
from __future__ import annotations

import re

from langchain_core.documents import Document

from prompt.core.base import BasePromptBuilder, PromptResult


class CitationPromptBuilder(BasePromptBuilder):
    SYSTEM_PROMPT = (
        "Bạn là trợ lý RAG cho lĩnh vực nông nghiệp. "
        "Chỉ trả lời dựa trên các nguồn được cung cấp trong phần NGUỒN. "
        "Không bịa thông tin, không dùng kiến thức bên ngoài nếu nguồn không nói. "
        "Mỗi câu mang thông tin từ nguồn BẮT BUỘC phải có trích dẫn dạng [NGUON 1] ngay cuối câu. "
        "Nếu tài liệu không đủ thông tin, hãy nói rõ là chưa tìm thấy trong tài liệu."
    )

    USER_TEMPLATE = (
        "NGUỒN:\n{context}\n\n"
        "CÂU HỎI:\n{query}\n\n"
        "YÊU CẦU TRẢ LỜI:\n"
        "- Trả lời bằng tiếng Việt, rõ ràng và thực tế.\n"
        "- Tóm tắt và gộp các ý trùng nhau từ các nguồn; không lặp lại nguyên văn từng nguồn.\n"
        "- Ưu tiên câu trả lời ngắn gọn, khoảng 2-4 đoạn hoặc gạch đầu dòng khi phù hợp.\n"
        "- Không chỉ đưa ra tiêu đề; mỗi mục phải có phần giải thích hoặc thông tin cụ thể từ NGUỒN.\n"
        "- Với câu hỏi tổng quát, hãy nêu ít nhất 3 ý chính nếu NGUỒN có đủ dữ liệu.\n"
        "- Hãy bắt đầu ngay bằng phần tổng hợp nội dung, không mở đầu bằng việc mô tả cách bạn đã tìm nguồn.\n"
        "- Chỉ dùng thông tin trong NGUỒN.\n"
        "\n"
        "QUY TẮC TRÍCH DẪN BẮT BUỘC (KHÔNG ĐƯỢC BỎ QUA):\n"
        "- BẮT BUỘC: Mỗi câu hoặc ý mang thông tin từ nguồn PHẢI kết thúc bằng [NGUON X] ngay cuối câu,\n"
        "  trước dấu chấm hoặc xuống dòng. Không được để một câu thông tin nào thiếu citation.\n"
        '  Ví dụ ĐÚNG: "Đạm hỗ trợ sinh trưởng thân lá, lân thúc đẩy phát triển rễ [NGUON 1]."\n'
        '  Ví dụ ĐÚNG: "Bón phân cần phù hợp với loại cây và thời điểm [NGUON 1][NGUON 3]."\n'
        '  Ví dụ SAI:  "Đạm hỗ trợ sinh trưởng thân lá." (thiếu citation — KHÔNG chấp nhận)\n'
        "- BẮT BUỘC: Nếu một nguồn chỉ chứa mục lục (danh sách chương, số trang, dấu chấm liên tiếp\n"
        "  như '..........'), TUYỆT ĐỐI không dùng nguồn đó để làm dẫn chứng. Bỏ qua hoàn toàn.\n"
        "- BẮT BUỘC: Nếu một nguồn chỉ áp dụng cho một loại cây hoặc điều kiện cụ thể\n"
        "  (ví dụ: chỉ nói về cây lúa), hãy ghi rõ phạm vi đó trong câu trả lời\n"
        '  (ví dụ: "Đối với cây lúa, ... [NGUON 4]").\n'
        "- Nếu các nguồn mâu thuẫn, nêu rõ sự khác nhau giữa các nguồn.\n"
        "- Nếu không đủ căn cứ, trả lời: \"Tôi chưa tìm thấy thông tin này trong tài liệu được cung cấp.\"\n"
    )

    def __init__(self, validate_citations: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.validate_citations = validate_citations

    def build(
        self,
        query: str,
        docs: list[Document],
        history: list[dict] | None = None,
    ) -> PromptResult:
        context = self._format_context(docs)
        system_text = self.system_instruction.strip() or self.SYSTEM_PROMPT
        user_text = self.USER_TEMPLATE.format(context=context, query=query)

        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]

        return PromptResult(
            messages=messages,
            full_prompt=self._messages_to_string(messages),
            context_docs=docs,
            n_sources=len(docs),
            template_name="citation",
        )

    @staticmethod
    def extract_cited_indices(answer: str) -> list[int]:
        pattern = re.compile(r"\[(?:NGUON|NGUỒN|SOURCE)\s*(\d+(?:\s*,\s*\d+)*)\]", re.IGNORECASE)
        indices: list[int] = []
        for match in pattern.finditer(answer or ""):
            for value in re.split(r"\s*,\s*", match.group(1)):
                if value.isdigit():
                    indices.append(int(value))
        return sorted(set(indices))
