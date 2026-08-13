"""Fast state and UI-flow tests that do not load a vision-language model."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import app


class DemoStateTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.image_path = Path(self.tempdir.name) / "sample.png"
        Image.new("RGB", (24, 16), "white").save(self.image_path)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_prompt_is_short_for_student_and_echo_is_detected(self):
        prompt = app.build_model_prompt(
            "Mô tả chi tiết hình ảnh",
            app.STUDENT_LABEL,
            "Mô tả chi tiết",
        )
        self.assertEqual(prompt, "Mô tả chi tiết hình ảnh\nMô tả ảnh bằng tiếng Việt.")
        self.assertTrue(app.looks_like_prompt_echo("Mô tả ảnh bằng tiếng Việt."))
        self.assertFalse(app.looks_like_prompt_echo("Một con mèo trắng nằm cạnh hoa."))

    def test_context_is_only_added_for_a_follow_up(self):
        context = "Người dùng: Có con mèo không?\nTrợ lý: Có."
        standalone = app.build_model_prompt(
            "Con mèo màu gì?", app.STUDENT_LABEL, "Cân bằng", context
        )
        follow_up = app.build_model_prompt(
            "Nó màu gì?", app.STUDENT_LABEL, "Cân bằng", context
        )
        self.assertNotIn("Ngữ cảnh gần nhất", standalone)
        self.assertIn("Ngữ cảnh gần nhất", follow_up)

    def test_stale_uploaded_image_is_removed_from_saved_browser_state(self):
        conversation = app.new_conversation()
        conversation["active_image_path"] = "Z:/not-present/image.png"
        conversation["messages"] = [
            {
                "role": "user",
                "content": [
                    {"path": "Z:/not-present/image.png", "alt_text": "Ảnh"},
                    "Ảnh có gì?",
                ],
            }
        ]
        normalized = app.normalize_conversations([conversation])[0]
        self.assertIsNone(normalized["active_image_path"])
        self.assertEqual(normalized["messages"][0]["content"], ["Ảnh có gì?"])

    def test_new_and_delete_conversations(self):
        first = app.new_conversation()
        created = app.create_product_chat([first])
        conversations = created[9]
        active_id = created[10]
        self.assertEqual(len(conversations), 2)

        deleted = app.delete_product_chat(conversations, active_id)
        self.assertEqual(len(deleted[9]), 1)
        self.assertNotEqual(deleted[10], active_id)

    def test_active_image_uses_clean_html_thumbnail(self):
        preview = app.active_image_html(str(self.image_path))
        self.assertIn('class="pinned-image"', preview)
        self.assertIn("data:image/jpeg;base64,", preview)
        self.assertNotIn(str(self.image_path), preview)

    def test_low_ram_error_is_friendly_and_hides_backend_details(self):
        error = RuntimeError(
            "Not enough free RAM for 2B CPU inference: 0.63 GB available, about 5 GB required"
        )
        message = app.friendly_inference_error(error)
        self.assertIn("Colab", message)
        self.assertNotIn("0.63", message)

    def test_submit_yields_pending_then_completed_state(self):
        conversation = app.new_conversation()
        fake_record = {
            "time": "01/01/2026 00:00:00",
            "model": app.STUDENT_LABEL,
            "question": "Ảnh có gì?",
            "answer": "Một hình vuông màu trắng.",
            "inference_seconds": 0.2,
            "load_seconds": 0.0,
        }

        def fake_answer(*_args, **_kwargs):
            return (
                fake_record["answer"],
                app.time_badge("Suy luận 0.2 giây"),
                [fake_record],
                app.history_rows([fake_record]),
                "status",
            )

        with (
            patch.object(app, "get_inference", return_value=(object(), 0.0)),
            patch.object(app, "answer", side_effect=fake_answer),
        ):
            states = list(
                app.answer_chat_gpt(
                    {"text": "Ảnh có gì?", "files": [str(self.image_path)]},
                    app.STUDENT_LABEL,
                    "Cân bằng",
                    64,
                    None,
                    [],
                    [],
                    [conversation],
                    conversation["id"],
                )
            )

        self.assertEqual(len(states), 3)
        self.assertIn("Đang xử lý ảnh", states[0][0][-1]["content"])
        self.assertIn("Đang", states[0][2])
        self.assertIn("Đang sinh câu trả lời", states[1][2])
        self.assertIn("Một hình vuông", states[2][0][-1]["content"])
        self.assertEqual(states[2][3], str(self.image_path))

    def test_missing_image_keeps_question_for_retry(self):
        conversation = app.new_conversation()
        states = list(
            app.answer_chat_gpt(
                {"text": "Ảnh có gì?", "files": []},
                app.STUDENT_LABEL,
                "Cân bằng",
                64,
                None,
                [],
                [],
                [conversation],
                conversation["id"],
            )
        )
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0][1]["text"], "Ảnh có gì?")
        self.assertIn("Đính kèm ảnh trước", states[0][2])


if __name__ == "__main__":
    unittest.main()
