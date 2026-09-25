"""
loader/utils/text.py
====================
Text cleaning utilities.
"""

from __future__ import annotations

import re
import unicodedata


def clean_text(text: str) -> str:
    """
    Chuẩn hoá văn bản sau khi trích xuất:
    xoá null byte, trim trailing whitespace từng dòng,
    thu gọn ≥3 dòng trắng xuống còn 2, strip toàn bộ.
    """
    text = unicodedata.normalize("NFC", str(text or ""))
    text = text.replace("\x00", "")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in text.splitlines()]
    
    cleaned: list[str] = []
    blank_run = 0
    for line in lines:
        if line.strip() == "":
            blank_run += 1
            if blank_run <= 2:
                cleaned.append(line)
        else:
            blank_run = 0
            cleaned.append(line)
            
    return "\n".join(cleaned).strip()
