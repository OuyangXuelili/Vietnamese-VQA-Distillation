import pytest

from vqa_explanation.preference.schema import PreferencePair


def test_preference_pair_rejects_identical_outputs() -> None:
    raw = {
        "pair_id": "p1",
        "source_question_id": "q1",
        "image_path": "image.jpg",
        "prompt": "Câu hỏi",
        "chosen": "Đáp án: mèo\nGiải thích: có một con mèo.",
        "rejected": "Đáp án: mèo\nGiải thích: có một con mèo.",
        "error_type": "object",
        "generation_method": "manual",
        "review_status": "accepted",
    }
    with pytest.raises(ValueError, match="must be different"):
        PreferencePair.from_dict(raw)

