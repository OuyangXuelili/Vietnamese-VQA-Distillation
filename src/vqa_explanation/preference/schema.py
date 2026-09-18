from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ERROR_TYPES = {"object", "attribute", "relation_action", "natural_model_error"}
REVIEW_STATES = {"pending", "accepted", "rejected", "uncertain"}


@dataclass(frozen=True)
class PreferencePair:
    pair_id: str
    source_question_id: str
    image_path: str
    prompt: str
    chosen: str
    rejected: str
    error_type: str
    generation_method: str
    review_status: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PreferencePair":
        required = [
            "pair_id",
            "source_question_id",
            "image_path",
            "prompt",
            "chosen",
            "rejected",
            "error_type",
            "generation_method",
            "review_status",
        ]
        missing = [key for key in required if not str(raw.get(key, "")).strip()]
        if missing:
            raise ValueError(f"missing fields: {missing}")
        if raw["chosen"].strip() == raw["rejected"].strip():
            raise ValueError("chosen and rejected must be different")
        if raw["error_type"] not in ERROR_TYPES:
            raise ValueError(f"unsupported error_type: {raw['error_type']}")
        if raw["review_status"] not in REVIEW_STATES:
            raise ValueError(f"unsupported review_status: {raw['review_status']}")
        return cls(**{key: str(raw[key]).strip() for key in required})
