from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

ALLOWED_SPLITS = {"train", "validation", "test"}


def _non_empty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class VQASample:
    question_id: str
    image_id: str
    image_path: str
    question: str
    answer: str
    explanations: tuple[str, ...]
    split: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "VQASample":
        explanations = raw.get("explanations", raw.get("explanation"))
        if isinstance(explanations, str):
            explanations = [explanations]
        if not isinstance(explanations, Iterable) or isinstance(explanations, (str, bytes)):
            raise ValueError("explanations must be a string or a list of strings")
        normalized = tuple(_non_empty_text(item, "explanation") for item in explanations)
        if not normalized:
            raise ValueError("explanations must contain at least one item")
        split = _non_empty_text(raw.get("split"), "split").lower()
        if split == "val":
            split = "validation"
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"unsupported split: {split}")
        return cls(
            question_id=str(raw.get("question_id", "")).strip(),
            image_id=str(raw.get("image_id", "")).strip(),
            image_path=_non_empty_text(raw.get("image_path"), "image_path"),
            question=_non_empty_text(raw.get("question"), "question"),
            answer=_non_empty_text(raw.get("answer"), "answer"),
            explanations=normalized,
            split=split,
        )

    def validate_ids(self) -> None:
        if not self.question_id:
            raise ValueError("question_id must not be empty")
        if not self.image_id:
            raise ValueError("image_id must not be empty")
