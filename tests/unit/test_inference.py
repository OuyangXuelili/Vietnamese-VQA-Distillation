from vqa_explanation.inference.prompting import build_user_prompt, parse_prediction


def test_build_user_prompt_requires_a_question() -> None:
    try:
        build_user_prompt("   ")
    except ValueError as exc:
        assert "question" in str(exc)
    else:
        raise AssertionError("empty question should fail")


def test_parse_prediction_accepts_expected_format() -> None:
    result = parse_prediction(
        "Đáp án: hai người\nGiải thích: Trong ảnh có hai người đang đứng cạnh xe."
    )
    assert result.answer == "hai người"
    assert result.explanation == "Trong ảnh có hai người đang đứng cạnh xe."
    assert result.format_valid


def test_parse_prediction_marks_unstructured_text_invalid() -> None:
    result = parse_prediction("Tôi nhìn thấy hai người đứng cạnh xe.")
    assert result.answer == ""
    assert result.explanation == ""
    assert not result.format_valid
