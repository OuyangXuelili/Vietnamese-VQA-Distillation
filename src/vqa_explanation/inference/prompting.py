from __future__ import annotations

import re
from dataclasses import dataclass

SYSTEM_PROMPT = (
    "Bạn là hệ thống hỏi đáp hình ảnh tiếng Việt. "
    "Chỉ sử dụng thông tin quan sát được trong ảnh."
)


def build_user_prompt(question: str) -> str:
    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")
    return (
        f"Câu hỏi: {question}\n"
        "Trả lời đúng hai phần theo định dạng:\n"
        "Đáp án: <câu trả lời ngắn>\n"
        "Giải thích: <một câu nêu bằng chứng nhìn thấy trong ảnh>"
    )


@dataclass(frozen=True)
class ParsedPrediction:
    answer: str
    explanation: str
    format_valid: bool


def parse_prediction(text: str) -> ParsedPrediction:
    cleaned = text.strip()
    answer_match = re.search(
        r"(?:^|\n)\s*Đáp\s*án\s*:\s*(.+?)(?=\n\s*Giải\s*thích\s*:|$)",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    explanation_match = re.search(
        r"(?:^|\n)\s*Giải\s*thích\s*:\s*(.+)$",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    answer = answer_match.group(1).strip() if answer_match else ""
    explanation = explanation_match.group(1).strip() if explanation_match else ""
    return ParsedPrediction(
        answer=answer,
        explanation=explanation,
        format_valid=bool(answer and explanation),
    )
