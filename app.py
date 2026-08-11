"""Branded Gradio demo for Vietnamese VQA Distillation."""

from __future__ import annotations

import argparse
import gc
import html
import json
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

from src.models.inference import VQAInference
from src.models.model_loader import VQAModelLoader
from src.models.model_wrapper import StudentModel, TeacherModel


ROOT = Path(__file__).resolve().parent
APP_TITLE = "Vietnamese Generative VQA Distillation"
OWNER = "OuyangXuelili"

STUDENT_LABEL = "Student 2B · FP16"
TEACHER_LABEL = "Teacher 7B · 4-bit"

ADAPTERS = {
    STUDENT_LABEL: {
        "path": ROOT / "outputs-2B-distilled" / "final_lora",
        "repo": "Ouyangxuelili/Qwen-2B-ViVQA",
    },
    TEACHER_LABEL: {
        "path": ROOT / "outputs-7B" / "final_lora",
        "repo": "Ouyangxuelili/Qwen-7B-Vi-VQA",
    },
}

REQUIRED_ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "preprocessor_config.json",
    "tokenizer_config.json",
)

_inference_engines: dict[str, VQAInference] = {}
_load_lock = threading.Lock()
_active_model_choice: str | None = None

HISTORY_HEADERS = [
    "Thời gian",
    "Model",
    "Câu hỏi",
    "Câu trả lời",
    "Suy luận (giây)",
]


CUSTOM_CSS = """
:root {
  --ui-canvas: #f7f7f6;
  --ui-surface: #ffffff;
  --ui-surface-subtle: #fafaf9;
  --ui-ink: #18181b;
  --ui-muted: #6b6b72;
  --ui-line: #e4e4e7;
  --ui-line-strong: #d4d4d8;
  --ui-danger: #b91c1c;
}

html, body {
  background: var(--ui-canvas) !important;
}

.gradio-container {
  width: 100% !important;
  max-width: 1120px !important;
  margin: 0 auto !important;
  padding: 0 20px !important;
  color: var(--ui-ink) !important;
  background: transparent !important;
  font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container p,
.gradio-container button,
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  font-family: inherit !important;
}

.app-shell {
  width: 100% !important;
  gap: 14px !important;
  padding: 24px 0 32px !important;
}

.product-header {
  padding: 2px 2px 18px;
  border-bottom: 1px solid var(--ui-line);
}

.product-header h1 {
  margin: 0;
  color: var(--ui-ink);
  font-size: clamp(1.55rem, 3vw, 2rem);
  font-weight: 720;
  letter-spacing: -.035em;
  line-height: 1.18;
}

.product-header p {
  max-width: 700px;
  margin: 7px 0 0;
  color: var(--ui-muted);
  font-size: .9rem;
  line-height: 1.55;
}

.workspace {
  gap: 14px !important;
  margin-top: 0 !important;
  align-items: flex-start !important;
}

.panel,
.input-panel,
.query-panel {
  min-width: 0 !important;
  padding: 18px !important;
  border: 1px solid var(--ui-line) !important;
  border-radius: 12px !important;
  background: var(--ui-surface) !important;
  box-shadow: none !important;
}

.workspace .input-panel,
.workspace .query-panel {
  align-self: flex-start !important;
  height: auto !important;
}

.panel-heading {
  margin-bottom: 12px;
}

.panel-heading h2 {
  margin: 0;
  color: var(--ui-ink);
  font-size: 1rem;
  font-weight: 700;
  letter-spacing: -.015em;
}

.input-panel > .block,
.query-panel > .block {
  flex: 0 0 auto !important;
}

.input-panel .image-container,
.input-panel [data-testid="image"] {
  border-color: var(--ui-line) !important;
  border-radius: 10px !important;
}

.image-uploader {
  height: 320px !important;
  min-height: 0 !important;
}

.input-panel [data-testid="image"] .upload-container > button > .wrap {
  font-size: 0 !important;
}

.input-panel [data-testid="image"] .upload-container > button > .wrap::after {
  display: block;
  margin-top: 10px;
  color: var(--ui-muted);
  content: "Kéo thả hoặc chọn ảnh";
  font-size: .9rem;
  line-height: 1.4;
}

.input-panel [data-testid="image"] .upload-container .or {
  display: none !important;
}

.image-meta-output,
.image-meta-output > div {
  padding: 0 !important;
}

.image-meta {
  display: flex;
  align-items: center;
  min-height: 20px;
  margin-top: 1px;
  color: var(--ui-muted);
  font-size: .76rem;
  line-height: 1.4;
}

.image-meta::before {
  width: 6px;
  height: 6px;
  margin-right: 8px;
  border-radius: 999px;
  background: #a1a1aa;
  content: "";
}

.image-meta.ready::before {
  background: #16a34a;
}

.options-drawer {
  margin-top: 2px !important;
  border-color: var(--ui-line) !important;
  border-radius: 10px !important;
  background: var(--ui-surface-subtle) !important;
}

.primary-action {
  min-height: 46px !important;
  border-color: var(--ui-ink) !important;
  border-radius: 9px !important;
  color: #ffffff !important;
  background: var(--ui-ink) !important;
  box-shadow: none !important;
  font-weight: 680 !important;
}

.primary-action:hover {
  border-color: #27272a !important;
  background: #27272a !important;
}

.answer-card {
  margin-top: 5px;
  padding: 15px 0 0 !important;
  border: 0 !important;
  border-top: 1px solid var(--ui-line) !important;
  border-radius: 0 !important;
  background: transparent !important;
}

.answer-header {
  align-items: center !important;
  margin-bottom: 6px !important;
}

.answer-title {
  flex: 1 1 auto !important;
  min-width: 120px !important;
}

.answer-title h3 {
  margin: 0 !important;
  color: var(--ui-ink) !important;
  font-size: .9rem !important;
  font-weight: 700 !important;
}

.time-chip-output {
  flex: 0 0 auto !important;
  width: auto !important;
  min-width: 0 !important;
  margin-left: auto !important;
}

.time-chip-output > div { padding: 0 !important; }

.time-chip {
  color: var(--ui-muted);
  font-size: .73rem;
  font-weight: 500;
  white-space: nowrap;
}

.time-chip.error { color: var(--ui-danger); }

.result-box textarea {
  color: var(--ui-ink) !important;
  background: #ffffff !important;
  font-size: .96rem !important;
  line-height: 1.6 !important;
}

.result-box,
.result-box > div {
  border-color: var(--ui-line) !important;
  border-radius: 9px !important;
}

footer { display: none !important; }

@media (max-width: 860px) {
  .gradio-container { padding: 0 12px !important; }
  .app-shell { padding-top: 16px !important; }
  .product-header { padding-bottom: 14px; }
  .product-header h1 { font-size: 1.45rem; }
  .workspace { flex-direction: column !important; }
  .workspace .input-panel,
  .workspace .query-panel {
    width: 100% !important;
    max-width: none !important;
    flex: 1 1 100% !important;
  }
  .panel,
  .input-panel,
  .query-panel { padding: 15px !important; }
  .image-uploader { height: 240px !important; }
}
"""


def adapter_status() -> list[str]:
    """Describe whether each adapter will load locally or from the Hub."""
    messages: list[str] = []
    for label, adapter in ADAPTERS.items():
        directory = adapter["path"]
        absent = [name for name in REQUIRED_ADAPTER_FILES if not (directory / name).is_file()]
        if not absent:
            size_mb = sum(
                path.stat().st_size for path in directory.rglob("*") if path.is_file()
            ) / 1024**2
            messages.append(f"{label}: local ({size_mb:.1f} MB)")
        else:
            messages.append(f"{label}: Hugging Face ({adapter['repo']})")
    return messages


def _release_loaded_models() -> None:
    """Release all loaded engines without changing model precision or outputs."""
    global _active_model_choice

    _inference_engines.clear()
    _active_model_choice = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_inference(model_choice: str) -> tuple[VQAInference, float]:
    """Load one model at a time and return its one-time loading duration."""
    global _active_model_choice

    if model_choice in _inference_engines:
        return _inference_engines[model_choice], 0.0

    with _load_lock:
        if model_choice in _inference_engines:
            return _inference_engines[model_choice], 0.0

        # A free T4 should keep only one VLM in VRAM. Releasing a model does not
        # change answer quality; switching back simply requires loading it again.
        if _inference_engines:
            _release_loaded_models()

        started = time.perf_counter()
        if model_choice == TEACHER_LABEL:
            loader = VQAModelLoader(device="auto", load_in_4bit=True)
            model, processor = loader.load_teacher_model()
            wrapper = TeacherModel(model, processor)
        else:
            # FP16 is faster than 4-bit dequantization for the 2B model on a 15 GB T4.
            loader = VQAModelLoader(device="auto", load_in_4bit=False)
            model, processor = loader.load_student_model()
            wrapper = StudentModel(model, processor)

        engine = VQAInference(wrapper)
        _inference_engines[model_choice] = engine
        _active_model_choice = model_choice
        return engine, time.perf_counter() - started


def time_badge(message: str, state: str = "success") -> str:
    """Render a compact persistent timing badge for the answer card."""
    safe_message = html.escape(message)
    safe_state = state if state in {"success", "idle", "error"} else "idle"
    return f'<span class="time-chip {safe_state}">{safe_message}</span>'


def image_summary(image) -> str:
    """Return compact image metadata without retaining another image copy."""
    if image is None:
        return '<div class="image-meta">Chưa có ảnh được chọn</div>'
    width, height = image.size
    megapixels = width * height / 1_000_000
    return (
        '<div class="image-meta ready">'
        f'Kích thước: {width} × {height} px · {megapixels:.2f} MP'
        '</div>'
    )


def history_rows(history: list[dict] | None) -> list[list]:
    """Convert session records into rows accepted by gr.Dataframe."""
    return [
        [
            item["time"],
            item["model"],
            item["question"],
            item["answer"],
            item["inference_seconds"],
        ]
        for item in (history or [])
    ]


def system_status_html() -> str:
    """Render device, VRAM and active-model information."""
    active_model = _active_model_choice or "Chưa nạp model"
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        allocated_bytes = torch.cuda.memory_allocated(device_index)
        used_bytes = total_bytes - free_bytes
        device_value = html.escape(device_name)
        memory_value = f"{used_bytes / 1024**3:.1f} / {total_bytes / 1024**3:.1f} GB"
        allocated_value = f"{allocated_bytes / 1024**3:.1f} GB"
    else:
        device_value = "CPU"
        memory_value = "Không có CUDA"
        allocated_value = "0 GB"

    return f"""
    <div class="system-grid">
      <div class="system-card">
        <div class="system-label">Thiết bị</div>
        <div class="system-value">{device_value}</div>
      </div>
      <div class="system-card">
        <div class="system-label">VRAM đang dùng</div>
        <div class="system-value">{memory_value}</div>
      </div>
      <div class="system-card">
        <div class="system-label">Tensor được cấp phát</div>
        <div class="system-value">{allocated_value}</div>
      </div>
      <div class="system-card">
        <div class="system-label">Model đang hoạt động</div>
        <div class="system-value">{html.escape(active_model)}</div>
      </div>
    </div>
    """


def adapter_sources_html() -> str:
    """Render local/Hub adapter sources for the system tab."""
    items = "".join(f"<li>{html.escape(message)}</li>" for message in adapter_status())
    return f'<ul class="source-list">{items}</ul>'


def unload_models_ui() -> tuple[str, str]:
    """Release the active model and refresh the system panel."""
    with _load_lock:
        _release_loaded_models()
    return (
        system_status_html(),
        '<div class="action-note">Đã giải phóng model và CUDA cache. Lần chạy tiếp theo sẽ nạp lại model.</div>',
    )


def export_history(history: list[dict] | None):
    """Write session history to a temporary JSON file for Gradio download."""
    if not history:
        return None
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(tempfile.gettempdir()) / f"vqa-history-{timestamp}.json"
    path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)


def clear_history() -> tuple[list, list, None]:
    """Clear records from the current browser session."""
    return [], [], None


def answer(
    image,
    question: str,
    model_choice: str,
    response_style: str,
    max_new_tokens: int,
    history: list[dict] | None,
    conversation_context: str | None = None,
) -> tuple[str, str, list[dict], list[list], str]:
    """Answer one Vietnamese visual question from the web interface."""
    records = list(history or [])
    if image is None:
        return (
            "Vui lòng tải lên một hình ảnh trước.",
            time_badge("Chưa thực hiện", "idle"),
            records,
            history_rows(records),
            system_status_html(),
        )
    if not question or not question.strip():
        return (
            "Vui lòng nhập câu hỏi về hình ảnh.",
            time_badge("Chưa thực hiện", "idle"),
            records,
            history_rows(records),
            system_status_html(),
        )

    try:
        engine, load_seconds = get_inference(model_choice)
        prompt = question.strip()
        if conversation_context:
            prompt = (
                "Ngữ cảnh các lượt hỏi trước (chỉ dùng khi có liên quan đến ảnh hiện tại):\n"
                f"{conversation_context}\n\n"
                f"Câu hỏi hiện tại: {prompt}"
            )
        min_new_tokens = 0

        if response_style == "Mô tả chi tiết":
            prompt = (
                f"{prompt}\n\n"
                "Mô tả đầy đủ ảnh bằng tiếng Việt trong 2 đến 4 câu, "
                "chỉ nêu những điều có thể quan sát trực tiếp."
            )
        elif response_style == "Trả lời ngắn":
            prompt = f"{prompt}\nHãy trả lời ngắn gọn, trực tiếp bằng tiếng Việt."

        inference_started = time.perf_counter()
        prediction = engine.predict_single(
            image.convert("RGB"),
            prompt,
            int(max_new_tokens),
            min_new_tokens,
        )
        inference_seconds = time.perf_counter() - inference_started

        timing_parts = [f"Suy luận {inference_seconds:.1f} giây"]
        if load_seconds:
            timing_parts.insert(0, f"Nạp model {load_seconds:.1f} giây")

        answer_text = prediction.strip()
        width, height = image.size
        records.append(
            {
                "time": datetime.now().astimezone().strftime("%d/%m/%Y %H:%M:%S"),
                "model": model_choice,
                "question": question.strip(),
                "answer": answer_text,
                "response_style": response_style,
                "max_new_tokens": int(max_new_tokens),
                "image_size": f"{width}x{height}",
                "load_seconds": round(load_seconds, 3),
                "inference_seconds": round(inference_seconds, 3),
            }
        )
        records = records[-100:]
        return (
            answer_text,
            time_badge(" · ".join(timing_parts)),
            records,
            history_rows(records),
            system_status_html(),
        )
    except Exception as exc:
        safe_error = html.escape(str(exc))
        return (
            f"Không thể chạy model: {safe_error}",
            time_badge("Đã xảy ra lỗi", "error"),
            records,
            history_rows(records),
            system_status_html(),
        )


def clear_workspace():
    """Reset only user inputs and generated outputs."""
    return None, "", "", time_badge("Sẵn sàng", "idle"), image_summary(None)


def _create_legacy_demo() -> gr.Blocks:
    """Build the branded Gradio interface."""
    with gr.Blocks(
        title=APP_TITLE,
        analytics_enabled=False,
    ) as demo:
        history_state = gr.State([])

        with gr.Column(elem_id="vqa-app", elem_classes="app-shell"):
            gr.HTML(
                """
                <header class="product-header">
                  <h1>Hỏi đáp hình ảnh tiếng Việt</h1>
                  <p>Demo Qwen2-VL-2B được tinh gọn bằng chưng cất tri thức cho bài toán hỏi đáp hình ảnh.</p>
                </header>
                """
            )

            with gr.Row(equal_height=False, elem_classes="workspace"):
                with gr.Column(scale=5, elem_classes=["panel", "input-panel"]):
                    gr.HTML(
                        """
                        <div class="panel-heading">
                          <h2>Ảnh đầu vào</h2>
                        </div>
                        """
                    )
                    image = gr.Image(
                        type="pil",
                        show_label=False,
                        height=320,
                        sources=["upload", "clipboard"],
                        elem_classes="image-uploader",
                    )
                    image_meta = gr.HTML(
                        image_summary(None),
                        elem_classes="image-meta-output",
                    )

                with gr.Column(scale=6, elem_classes=["panel", "query-panel"]):
                    gr.HTML(
                        """
                        <div class="panel-heading">
                          <h2>Câu hỏi</h2>
                        </div>
                        """
                    )
                    question = gr.Textbox(
                        show_label=False,
                        placeholder="Ví dụ: Có những vật thể nào trong ảnh và chúng đang ở đâu?",
                        lines=3,
                    )
                    with gr.Accordion(
                        "Tùy chọn",
                        open=False,
                        elem_classes="options-drawer",
                    ):
                        model_choice = gr.Dropdown(
                            choices=list(ADAPTERS),
                            value=STUDENT_LABEL,
                            label="Mô hình",
                        )
                        response_style = gr.Radio(
                            choices=["Mô tả chi tiết", "Cân bằng", "Trả lời ngắn"],
                            value="Cân bằng",
                            label="Cách trả lời",
                        )
                        max_new_tokens = gr.Slider(
                            minimum=16,
                            maximum=160,
                            value=96,
                            step=8,
                            label="Số token sinh tối đa",
                        )

                    with gr.Row():
                        submit = gr.Button(
                            "Phân tích ảnh",
                            variant="primary",
                            elem_classes="primary-action",
                            scale=4,
                        )
                        clear = gr.Button("Xóa", scale=1)

                    with gr.Column(elem_classes="answer-card"):
                        with gr.Row(elem_classes="answer-header"):
                            gr.Markdown("### Kết quả", elem_classes="answer-title")
                            timing = gr.HTML(
                                time_badge("Sẵn sàng", "idle"),
                                elem_classes="time-chip-output",
                            )
                        result = gr.Textbox(
                            lines=5,
                            show_label=False,
                            placeholder="Câu trả lời sẽ xuất hiện tại đây.",
                            buttons=["copy"],
                            interactive=False,
                            elem_classes="result-box",
                        )

            # Preserve session records and device state as callback outputs without
            # exposing administrator controls in the public-facing demo.
            with gr.Column(visible=False):
                history_table = gr.Dataframe(
                    headers=HISTORY_HEADERS,
                    datatype=["str", "str", "str", "str", "number"],
                    value=[],
                    interactive=False,
                )
                system_status = gr.HTML(system_status_html())

        inputs = [
            image,
            question,
            model_choice,
            response_style,
            max_new_tokens,
            history_state,
        ]
        answer_outputs = [result, timing, history_state, history_table, system_status]
        submit.click(
            fn=answer,
            inputs=inputs,
            outputs=answer_outputs,
            show_progress="hidden",
        )
        question.submit(
            fn=answer,
            inputs=inputs,
            outputs=answer_outputs,
            show_progress="hidden",
        )
        image.change(
            fn=image_summary,
            inputs=image,
            outputs=image_meta,
            queue=False,
        )
        clear.click(
            fn=clear_workspace,
            outputs=[image, question, result, timing, image_meta],
            queue=False,
        )
    return demo


CHAT_CSS = """
html, body {
  background: #ffffff !important;
}

.gradio-container {
  width: 100% !important;
  max-width: none !important;
  min-height: 100vh !important;
  margin: 0 !important;
  padding: 0 !important;
  color: #202123 !important;
  background: #ffffff !important;
}

.gradio-container > .main.fillable {
  width: 100% !important;
  max-width: none !important;
  margin: 0 !important;
  padding: 0 !important;
}

.chat-shell {
  height: 100vh !important;
  min-height: 100vh !important;
  gap: 0 !important;
  flex-wrap: nowrap !important;
  align-items: stretch !important;
  background: #ffffff !important;
}

.chat-sidebar {
  flex: 0 0 258px !important;
  width: 258px !important;
  min-width: 258px !important;
  height: 100vh !important;
  min-height: 0 !important;
  overflow-y: auto !important;
  padding: 18px 14px !important;
  border-right: 1px solid #e7e7e7 !important;
  background: #f7f7f8 !important;
}

.brand-lockup {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 2px 6px 16px;
}

.brand-mark {
  display: grid;
  width: 34px;
  height: 34px;
  place-items: center;
  border-radius: 10px;
  color: #ffffff;
  background: #202123;
  font-size: .8rem;
  font-weight: 800;
  letter-spacing: -.03em;
}

.brand-name {
  color: #202123;
  font-size: .94rem;
  font-weight: 700;
  letter-spacing: -.01em;
}

.brand-caption {
  margin-top: 1px;
  color: #77777d;
  font-size: .72rem;
}

.new-chat-button {
  min-height: 42px !important;
  margin-bottom: 14px !important;
  border: 1px solid #d9d9dc !important;
  border-radius: 10px !important;
  color: #202123 !important;
  background: #ffffff !important;
  box-shadow: none !important;
  font-weight: 650 !important;
  text-align: left !important;
}

.new-chat-button:hover {
  background: #f0f0f1 !important;
}

.settings-label {
  padding: 10px 4px 8px;
  color: #77777d;
  font-size: .7rem;
  font-weight: 750;
  letter-spacing: .08em;
  text-transform: uppercase;
}

.chat-sidebar .block,
.chat-sidebar .form {
  border-color: #dedee1 !important;
  border-radius: 10px !important;
  box-shadow: none !important;
}

.sidebar-note {
  margin-top: 14px;
  padding: 10px 4px 0;
  border-top: 1px solid #e2e2e4;
  color: #77777d;
  font-size: .73rem;
  line-height: 1.5;
}

.chat-main {
  flex-direction: column !important;
  flex-wrap: nowrap !important;
  min-width: 0 !important;
  height: 100vh !important;
  min-height: 0 !important;
  overflow: hidden !important;
  background: #ffffff !important;
}

.chat-topbar {
  display: flex;
  flex: 0 0 62px !important;
  position: sticky;
  z-index: 20;
  top: 0;
  align-items: center;
  justify-content: space-between;
  height: 62px;
  width: 100% !important;
  padding: 0 24px;
  border-bottom: 1px solid #ececee;
  background: rgba(255, 255, 255, .94);
  backdrop-filter: blur(12px);
}

.chat-topbar-title {
  color: #202123;
  font-size: .96rem;
  font-weight: 700;
}

.model-pill {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  color: #606066;
  font-size: .75rem;
  font-weight: 600;
}

.model-pill::before {
  width: 7px;
  height: 7px;
  border-radius: 999px;
  background: #10a37f;
  content: "";
}

.chat-stage {
  flex: 1 1 auto !important;
  width: min(100%, 880px) !important;
  height: calc(100vh - 62px) !important;
  min-height: 0 !important;
  margin: 0 auto !important;
  padding: 18px 24px 26px !important;
  gap: 10px !important;
  overflow-y: auto !important;
}

#chat-thread {
  height: clamp(280px, calc(100vh - 330px), 510px) !important;
  min-height: 280px !important;
  border: 0 !important;
  border-radius: 0 !important;
  background: #ffffff !important;
  box-shadow: none !important;
}

#chat-thread .message {
  max-width: 82% !important;
  border-radius: 18px !important;
  box-shadow: none !important;
  font-size: .96rem !important;
  line-height: 1.62 !important;
}

#chat-thread .message.user {
  border: 1px solid #e8e8e8 !important;
  background: #f4f4f4 !important;
}

#chat-thread .message.bot,
#chat-thread .message.assistant {
  max-width: 92% !important;
  border-color: transparent !important;
  background: #ffffff !important;
}

.suggestion-row {
  gap: 8px !important;
  margin-top: 2px !important;
}

.suggestion-button {
  min-height: 36px !important;
  border: 1px solid #dfdfe2 !important;
  border-radius: 999px !important;
  color: #55555b !important;
  background: #ffffff !important;
  box-shadow: none !important;
  font-size: .78rem !important;
  font-weight: 550 !important;
}

.suggestion-button:hover {
  color: #202123 !important;
  background: #f7f7f8 !important;
}

.chat-composer {
  position: sticky !important;
  z-index: 15;
  bottom: 14px;
  gap: 10px !important;
  align-items: center !important;
  padding: 10px !important;
  border: 1px solid #d9d9dd !important;
  border-radius: 24px !important;
  background: #ffffff !important;
  box-shadow: 0 7px 24px rgba(0, 0, 0, .08) !important;
}

#composer-image {
  flex: 0 0 92px !important;
  width: 92px !important;
  min-width: 92px !important;
  height: 82px !important;
  min-height: 82px !important;
  overflow: hidden !important;
  border: 1px dashed #cfd0d4 !important;
  border-radius: 15px !important;
  background: #f8f8f8 !important;
}

#composer-image .upload-container {
  min-height: 80px !important;
  padding: 4px !important;
}

#composer-image .upload-container svg {
  width: 20px !important;
  height: 20px !important;
}

#composer-image .upload-container .wrap {
  font-size: 0 !important;
}

#composer-image .upload-container .wrap::after {
  display: block;
  margin-top: 4px;
  color: #6f7075;
  content: "Thêm ảnh";
  font-size: .72rem;
  font-weight: 600;
}

#composer-image .upload-container .or {
  display: none !important;
}

.chat-input {
  min-width: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}

.chat-input textarea {
  min-height: 58px !important;
  padding: 10px 4px !important;
  border: 0 !important;
  color: #202123 !important;
  background: transparent !important;
  box-shadow: none !important;
  font-size: .96rem !important;
  line-height: 1.45 !important;
}

.send-button {
  flex: 0 0 58px !important;
  width: 58px !important;
  min-width: 58px !important;
  min-height: 46px !important;
  border: 0 !important;
  border-radius: 16px !important;
  color: #ffffff !important;
  background: #202123 !important;
  box-shadow: none !important;
  font-weight: 700 !important;
}

.send-button:hover {
  background: #343437 !important;
}

.composer-meta {
  align-items: center !important;
  min-height: 22px !important;
  padding: 0 6px !important;
}

.composer-meta .image-meta,
.composer-meta .time-chip {
  margin: 0 !important;
  color: #77777d !important;
  font-size: .72rem !important;
}

.privacy-note {
  margin: 2px 0 0;
  color: #9a9a9f;
  font-size: .68rem;
  text-align: center;
}

@media (max-width: 820px) {
  .chat-shell { flex-direction: column !important; }
  .chat-sidebar {
    width: 100% !important;
    min-width: 0 !important;
    min-height: auto !important;
    padding: 10px 12px !important;
    border-right: 0 !important;
    border-bottom: 1px solid #e7e7e7 !important;
  }
  .brand-lockup { padding-bottom: 8px; }
  .chat-sidebar .settings-label,
  .chat-sidebar .sidebar-settings,
  .chat-sidebar .sidebar-note { display: none !important; }
  .new-chat-button { margin-bottom: 0 !important; }
  .chat-topbar { height: 54px; padding: 0 14px; }
  .chat-stage { padding: 10px 12px 18px !important; }
  #chat-thread { min-height: 430px !important; }
  .suggestion-row { display: none !important; }
  .chat-composer { bottom: 8px; border-radius: 20px !important; }
  #composer-image {
    flex-basis: 70px !important;
    width: 70px !important;
    min-width: 70px !important;
    height: 70px !important;
    min-height: 70px !important;
  }
}
"""


PRODUCT_CSS = """
:root {
  --vqa-canvas: #f7f7f5;
  --vqa-surface: #ffffff;
  --vqa-surface-muted: #f1f1ef;
  --vqa-ink: #18181b;
  --vqa-muted: #71717a;
  --vqa-line: #e4e4e7;
  --vqa-accent: #3454d1;
  --vqa-accent-hover: #2945b8;
}

html, body {
  overflow: hidden !important;
  background: var(--vqa-canvas) !important;
}

.gradio-container,
.gradio-container > .main.fillable,
.gradio-container main.contain {
  width: 100% !important;
  max-width: none !important;
  margin: 0 !important;
  padding: 0 !important;
}

.product-shell {
  width: 100% !important;
  height: 100vh !important;
  min-height: 0 !important;
  gap: 0 !important;
  flex-direction: column !important;
  flex-wrap: nowrap !important;
  color: var(--vqa-ink) !important;
  background: var(--vqa-canvas) !important;
  font-family: Inter, "Segoe UI", system-ui, sans-serif !important;
}

.product-header {
  flex: 0 0 68px !important;
  width: 100% !important;
  height: 68px !important;
  min-height: 68px !important;
  margin: 0 !important;
  padding: 0 24px !important;
  border-bottom: 1px solid var(--vqa-line) !important;
  background: rgba(255, 255, 255, .96) !important;
  box-shadow: none !important;
}

.product-header-inner {
  width: min(100%, 1080px) !important;
  height: 68px !important;
  margin: 0 auto !important;
  gap: 12px !important;
  align-items: center !important;
  flex-wrap: nowrap !important;
}

.product-brand {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 11px;
}

.product-mark {
  display: grid;
  width: 32px;
  height: 32px;
  flex: 0 0 32px;
  place-items: center;
  border-radius: 9px;
  color: #fff;
  background: var(--vqa-ink);
  font-size: 12px;
  font-weight: 750;
  letter-spacing: -.04em;
}

.product-name {
  color: var(--vqa-ink);
  font-size: 15px;
  font-weight: 700;
  letter-spacing: -.015em;
}

.product-caption {
  margin-top: 1px;
  overflow: hidden;
  color: var(--vqa-muted);
  font-size: 11.5px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.header-spacer {
  flex: 1 1 auto !important;
  min-width: 12px !important;
}

.header-new-chat {
  flex: 0 0 auto !important;
  width: auto !important;
  min-width: 72px !important;
  min-height: 38px !important;
  padding: 0 14px !important;
  border: 1px solid var(--vqa-line) !important;
  border-radius: 10px !important;
  color: var(--vqa-ink) !important;
  background: var(--vqa-surface) !important;
  box-shadow: none !important;
  font-size: 13px !important;
  font-weight: 600 !important;
}

.header-new-chat:hover {
  background: var(--vqa-surface-muted) !important;
}

.header-model {
  flex: 0 0 248px !important;
  width: 248px !important;
  min-width: 248px !important;
  min-height: 38px !important;
  margin: 0 !important;
  border: 1px solid var(--vqa-line) !important;
  border-radius: 10px !important;
  background: var(--vqa-surface) !important;
  box-shadow: none !important;
}

.header-model .wrap,
.header-model input {
  min-height: 36px !important;
  color: var(--vqa-ink) !important;
  background: transparent !important;
  font-size: 12.5px !important;
}

.conversation-main {
  flex: 1 1 auto !important;
  width: 100% !important;
  height: calc(100vh - 68px) !important;
  min-height: 0 !important;
  overflow: hidden !important;
  background: var(--vqa-canvas) !important;
}

.conversation-body {
  width: min(100%, 900px) !important;
  height: 100% !important;
  min-height: 0 !important;
  margin: 0 auto !important;
  padding: 16px 24px 10px !important;
  gap: 9px !important;
  flex-direction: column !important;
  flex-wrap: nowrap !important;
  overflow-y: auto !important;
}

.conversation-kicker {
  display: flex;
  align-items: center;
  justify-content: space-between;
  min-height: 20px;
  color: var(--vqa-muted);
  font-size: 11.5px;
}

.runtime-status {
  display: inline-flex;
  align-items: center;
  gap: 7px;
}

.runtime-status::before {
  width: 7px;
  height: 7px;
  border-radius: 999px;
  background: #22a06b;
  content: "";
}

#chat-thread {
  flex: 0 0 auto !important;
  width: 100% !important;
  height: clamp(250px, calc(100vh - 372px), 590px) !important;
  min-height: 250px !important;
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}

#chat-thread .message {
  max-width: 82% !important;
  padding: 10px 13px !important;
  border: 0 !important;
  border-radius: 14px !important;
  box-shadow: none !important;
  font-size: 14.5px !important;
  line-height: 1.58 !important;
}

#chat-thread .message.user {
  background: #ececea !important;
}

#chat-thread .message.bot,
#chat-thread .message.assistant {
  max-width: 94% !important;
  padding-left: 0 !important;
  background: transparent !important;
}

.advanced-panel {
  flex: 0 0 auto !important;
  margin: 0 !important;
  border: 1px solid var(--vqa-line) !important;
  border-radius: 10px !important;
  background: var(--vqa-surface) !important;
  box-shadow: none !important;
}

.advanced-panel > .label-wrap {
  min-height: 38px !important;
  padding: 8px 12px !important;
  color: var(--vqa-muted) !important;
  font-size: 12.5px !important;
  font-weight: 600 !important;
}

.advanced-controls {
  gap: 16px !important;
  padding: 0 10px 10px !important;
  align-items: end !important;
  flex-wrap: nowrap !important;
}

.advanced-controls .block,
.advanced-controls .form {
  border: 0 !important;
  box-shadow: none !important;
}

.suggestion-row {
  flex: 0 0 auto !important;
  gap: 8px !important;
  margin: 0 !important;
  flex-wrap: nowrap !important;
}

.suggestion-button {
  min-height: 34px !important;
  border: 1px solid var(--vqa-line) !important;
  border-radius: 999px !important;
  color: var(--vqa-muted) !important;
  background: var(--vqa-surface) !important;
  box-shadow: none !important;
  font-size: 12.5px !important;
  font-weight: 550 !important;
}

.suggestion-button:hover {
  color: var(--vqa-ink) !important;
  background: var(--vqa-surface-muted) !important;
}

.chat-composer {
  flex: 0 0 auto !important;
  position: sticky !important;
  z-index: 10;
  bottom: 8px;
  min-height: 88px !important;
  gap: 10px !important;
  align-items: center !important;
  padding: 9px !important;
  border: 1px solid #d9d9dc !important;
  border-radius: 18px !important;
  background: var(--vqa-surface) !important;
  box-shadow: 0 10px 30px rgba(24, 24, 27, .08) !important;
  flex-wrap: nowrap !important;
}

#composer-image {
  flex: 0 0 76px !important;
  width: 76px !important;
  min-width: 76px !important;
  height: 68px !important;
  min-height: 68px !important;
  overflow: hidden !important;
  border: 1px dashed #c9c9ce !important;
  border-radius: 12px !important;
  background: var(--vqa-surface-muted) !important;
}

#composer-image .upload-container {
  min-height: 66px !important;
  padding: 3px !important;
}

#composer-image .upload-container svg {
  width: 18px !important;
  height: 18px !important;
}

#composer-image .upload-container .wrap {
  font-size: 0 !important;
}

#composer-image .upload-container .wrap::after {
  display: block;
  margin-top: 3px;
  color: var(--vqa-muted);
  content: "Thêm ảnh";
  font-size: 10.5px;
  font-weight: 600;
}

#composer-image .upload-container .or {
  display: none !important;
}

.chat-input {
  min-width: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}

.chat-input textarea {
  min-height: 54px !important;
  padding: 8px 2px !important;
  border: 0 !important;
  color: var(--vqa-ink) !important;
  background: transparent !important;
  box-shadow: none !important;
  font-size: 14.5px !important;
  line-height: 1.45 !important;
}

.send-button {
  flex: 0 0 58px !important;
  width: 58px !important;
  min-width: 58px !important;
  min-height: 44px !important;
  border: 0 !important;
  border-radius: 12px !important;
  color: #fff !important;
  background: var(--vqa-accent) !important;
  box-shadow: none !important;
  font-size: 13px !important;
  font-weight: 700 !important;
}

.send-button:hover {
  background: var(--vqa-accent-hover) !important;
}

.composer-meta {
  flex: 0 0 20px !important;
  min-height: 20px !important;
  padding: 0 4px !important;
  align-items: center !important;
}

.composer-meta .image-meta,
.composer-meta .time-chip {
  margin: 0 !important;
  color: var(--vqa-muted) !important;
  font-size: 11px !important;
}

.privacy-note {
  margin: 0;
  color: #929298;
  font-size: 10.5px;
  text-align: center;
}

@media (max-width: 720px) {
  .product-header { padding: 0 12px !important; }
  .product-caption { display: none; }
  .header-new-chat { min-width: 58px !important; padding: 0 10px !important; }
  .header-model { flex-basis: 170px !important; width: 170px !important; min-width: 170px !important; }
  .conversation-body { padding: 10px 12px 8px !important; }
  #chat-thread { height: clamp(270px, calc(100vh - 330px), 520px) !important; }
  .suggestion-row { display: none !important; }
  .advanced-panel { display: none !important; }
  #composer-image { flex-basis: 62px !important; width: 62px !important; min-width: 62px !important; }
}
"""


GPT_CSS = """
:root {
  --gpt-sidebar: #f7f7f8;
  --gpt-surface: #ffffff;
  --gpt-hover: #ececf1;
  --gpt-ink: #202123;
  --gpt-muted: #6e6e80;
  --gpt-line: #e5e5e5;
  --gpt-action: #111827;
}

html, body {
  overflow: hidden !important;
  background: var(--gpt-surface) !important;
}

.gradio-container,
.gradio-container > .main.fillable,
.gradio-container main.contain {
  width: 100% !important;
  max-width: none !important;
  height: 100vh !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  color: var(--gpt-ink) !important;
  background: var(--gpt-surface) !important;
  font-family: Inter, "Segoe UI", system-ui, sans-serif !important;
}

.gpt-shell {
  width: 100% !important;
  height: 100vh !important;
  min-height: 0 !important;
  margin: 0 !important;
  gap: 0 !important;
  flex-wrap: nowrap !important;
  overflow: hidden !important;
  background: var(--gpt-surface) !important;
}

.gpt-sidebar {
  flex: 0 0 260px !important;
  width: 260px !important;
  min-width: 260px !important;
  height: 100vh !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 12px 10px !important;
  gap: 10px !important;
  overflow: hidden !important;
  border-right: 1px solid #eeeeef !important;
  background: var(--gpt-sidebar) !important;
  box-shadow: none !important;
}

.sidebar-brand {
  display: flex;
  height: 42px;
  align-items: center;
  gap: 10px;
  padding: 0 9px;
  color: var(--gpt-ink);
  font-size: 14px;
  font-weight: 650;
  letter-spacing: -.01em;
}

.sidebar-mark {
  display: grid;
  width: 29px;
  height: 29px;
  place-items: center;
  border-radius: 8px;
  color: #fff;
  background: #111827;
  font-size: 10px;
  font-weight: 750;
}

.sidebar-new-chat {
  width: 100% !important;
  min-height: 44px !important;
  margin: 0 !important;
  padding: 0 12px !important;
  justify-content: flex-start !important;
  border: 1px solid var(--gpt-line) !important;
  border-radius: 10px !important;
  color: var(--gpt-ink) !important;
  background: var(--gpt-surface) !important;
  box-shadow: none !important;
  font-size: 13.5px !important;
  font-weight: 550 !important;
  text-align: left !important;
}

.sidebar-new-chat:hover {
  background: var(--gpt-hover) !important;
}

.sidebar-options {
  margin: 0 !important;
  border: 1px solid var(--gpt-line) !important;
  border-radius: 10px !important;
  background: var(--gpt-surface) !important;
  box-shadow: none !important;
}

.sidebar-options > .label-wrap {
  min-height: 38px !important;
  padding: 8px 11px !important;
  color: var(--gpt-ink) !important;
  font-size: 12.5px !important;
  font-weight: 550 !important;
}

.sidebar-options > .wrap {
  padding: 0 10px 10px !important;
}

.sidebar-options .block,
.sidebar-options .form {
  border: 0 !important;
  box-shadow: none !important;
}

.sidebar-options label,
.sidebar-options .label-wrap span {
  color: var(--gpt-muted) !important;
  font-size: 11.5px !important;
}

.sidebar-history-wrap {
  flex: 1 1 auto !important;
  min-height: 0 !important;
  overflow: hidden !important;
}

.history-panel {
  height: 100%;
  overflow-y: auto;
  padding: 10px 2px 18px;
  scrollbar-width: thin;
}

.history-heading {
  padding: 7px 9px;
  color: #8a8a95;
  font-size: 11px;
  font-weight: 650;
}

.history-item {
  overflow: hidden;
  margin: 1px 0;
  padding: 9px;
  border-radius: 8px;
  color: #3f3f46;
  font-size: 13px;
  line-height: 1.35;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.history-item:first-of-type {
  color: var(--gpt-ink);
  background: #ececf1;
}

.history-empty {
  padding: 10px 9px;
  color: #92929d;
  font-size: 12.5px;
  line-height: 1.5;
}

.gpt-main {
  flex: 1 1 auto !important;
  width: auto !important;
  min-width: 0 !important;
  height: 100vh !important;
  min-height: 0 !important;
  margin: 0 !important;
  gap: 0 !important;
  flex-wrap: nowrap !important;
  overflow: clip !important;
  background: var(--gpt-surface) !important;
}

.gpt-topbar {
  flex: 0 0 58px !important;
  width: 100% !important;
  min-height: 58px !important;
  margin: 0 !important;
  padding: 8px 18px !important;
  align-items: center !important;
  flex-wrap: nowrap !important;
  background: rgba(255,255,255,.96) !important;
}

.topbar-title {
  min-width: 140px;
  color: var(--gpt-ink);
  font-size: 15px;
  font-weight: 650;
  letter-spacing: -.01em;
}

.topbar-spacer {
  flex: 1 1 auto !important;
}

.topbar-model-picker {
  flex: 0 0 auto !important;
  width: auto !important;
  min-width: 286px !important;
  min-height: 42px !important;
  margin: 0 !important;
  padding: 3px 6px 3px 11px !important;
  gap: 3px !important;
  align-items: center !important;
  border: 1px solid #e8e8ec !important;
  border-radius: 12px !important;
  background: #fff !important;
  box-shadow: 0 1px 2px rgba(0, 0, 0, .03) !important;
}

.model-picker-label {
  flex: 0 0 auto;
  color: #858591;
  font-size: 11px;
  font-weight: 650;
  letter-spacing: .01em;
  white-space: nowrap;
}

.topbar-model {
  flex: 0 0 218px !important;
  width: 218px !important;
  min-width: 218px !important;
  height: 40px !important;
  min-height: 40px !important;
  max-height: 40px !important;
  margin: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}

.topbar-model input,
.topbar-model button {
  min-height: 38px !important;
  border: 0 !important;
  border-radius: 9px !important;
  background: transparent !important;
  font-size: 13px !important;
}

.topbar-model:hover input,
.topbar-model:hover button {
  background: #f5f5f5 !important;
}

.gpt-stage {
  flex: 1 1 auto !important;
  width: min(100%, 900px) !important;
  min-height: 0 !important;
  margin: 0 !important;
  align-self: center !important;
  padding: 0 22px 12px !important;
  gap: 0 !important;
  flex-wrap: nowrap !important;
  overflow: clip !important;
}

#chat-thread {
  flex: 1 1 auto !important;
  width: 100% !important;
  height: auto !important;
  min-height: 0 !important;
  margin: 0 !important;
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}

#chat-thread .bubble-wrap {
  padding-left: 6px !important;
  padding-right: 6px !important;
}

#chat-thread .message {
  max-width: 82% !important;
  border: 0 !important;
  box-shadow: none !important;
  font-size: 14.5px !important;
  line-height: 1.62 !important;
}

#chat-thread .message.user {
  border-radius: 18px !important;
  color: var(--gpt-ink) !important;
  background: #f4f4f4 !important;
}

#chat-thread .message.bot,
#chat-thread .message.assistant {
  color: var(--gpt-ink) !important;
  background: transparent !important;
}

#chat-thread .message.user img {
  display: block !important;
  max-width: 260px !important;
  max-height: 220px !important;
  margin-bottom: 8px !important;
  border-radius: 12px !important;
  object-fit: contain !important;
}

#chat-thread .top-panel button[aria-label="Clear"] {
  display: none !important;
}

#chat-thread .message-buttons {
  opacity: 0 !important;
  transition: opacity .15s ease !important;
}

#chat-thread .message-wrap:hover .message-buttons {
  opacity: 1 !important;
}

.composer-area {
  flex: 0 0 auto !important;
  width: 100% !important;
  margin: 0 !important;
  gap: 6px !important;
}

#chat-composer.chat-composer-gpt {
  width: 100% !important;
  min-height: 64px !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: visible !important;
  border: 1px solid #dedee3 !important;
  border-radius: 26px !important;
  background: var(--gpt-surface) !important;
  box-shadow: 0 7px 24px rgba(0,0,0,.08) !important;
}

#chat-composer .full-container,
#chat-composer .input-container,
#chat-composer .input-wrapper,
#chat-composer .input-row,
#chat-composer .textarea-wrapper {
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}

#chat-composer .input-wrapper {
  min-height: 62px !important;
  padding: 8px 9px !important;
  gap: 7px !important;
}

#chat-composer .input-row {
  align-items: flex-end !important;
  gap: 7px !important;
}

#chat-composer textarea {
  min-height: 44px !important;
  max-height: 122px !important;
  padding: 11px 5px 9px !important;
  border: 0 !important;
  color: var(--gpt-ink) !important;
  background: transparent !important;
  box-shadow: none !important;
  font-size: 15px !important;
  line-height: 1.45 !important;
  overflow-y: auto !important;
  scrollbar-width: none !important;
  resize: none !important;
}

#chat-composer textarea::-webkit-scrollbar {
  width: 0 !important;
  height: 0 !important;
}

#chat-composer textarea::placeholder {
  color: #92929d !important;
}

#chat-composer [data-testid="upload-button"],
#chat-composer [data-testid="submit-button"] {
  flex: 0 0 42px !important;
  width: 42px !important;
  min-width: 42px !important;
  height: 42px !important;
  min-height: 42px !important;
  margin: 0 !important;
  padding: 0 !important;
  border: 0 !important;
  border-radius: 50% !important;
  box-shadow: none !important;
}

#chat-composer [data-testid="upload-button"] {
  color: #35353b !important;
  background: #f4f4f5 !important;
}

#chat-composer [data-testid="upload-button"]:hover {
  background: #e9e9ec !important;
}

#chat-composer [data-testid="submit-button"] {
  color: #fff !important;
  background: var(--gpt-action) !important;
  font-size: 20px !important;
}

#chat-composer [data-testid="submit-button"]:hover {
  background: #000 !important;
}

/* Keep the attachment preview inside the composer. */
#chat-composer .thumbnails {
  display: flex !important;
  min-height: 72px !important;
  padding: 3px 3px 0 !important;
  gap: 8px !important;
}

#chat-composer .thumbnail-wrapper,
#chat-composer .thumbnail-item {
  position: relative !important;
  width: 72px !important;
  min-width: 72px !important;
  height: 72px !important;
  min-height: 72px !important;
  overflow: visible !important;
  border: 0 !important;
  border-radius: 13px !important;
  background: transparent !important;
}

#chat-composer .thumbnail-image {
  width: 68px !important;
  height: 68px !important;
  border-radius: 13px !important;
  object-fit: cover !important;
  box-shadow: 0 0 0 1px rgba(17,24,39,.09) !important;
}

#chat-composer .delete-button {
  position: absolute !important;
  top: -6px !important;
  right: -3px !important;
  z-index: 3 !important;
  display: grid !important;
  place-items: center !important;
  width: 22px !important;
  min-width: 22px !important;
  height: 22px !important;
  min-height: 22px !important;
  padding: 5px !important;
  border: 2px solid #fff !important;
  border-radius: 50% !important;
  color: #fff !important;
  background: #25252a !important;
  box-shadow: 0 2px 7px rgba(0,0,0,.22) !important;
  opacity: 1 !important;
}

.composer-style {
  flex: 0 0 112px !important;
  width: 112px !important;
  min-width: 112px !important;
  height: 40px !important;
  min-height: 40px !important;
  max-height: 40px !important;
  margin: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}

.composer-style > div {
  width: 112px !important;
  height: 40px !important;
  min-height: 40px !important;
  max-height: 40px !important;
}

.composer-style input,
.composer-style button {
  min-height: 40px !important;
  border: 0 !important;
  border-radius: 20px !important;
  color: #6b6b75 !important;
  background: transparent !important;
  box-shadow: none !important;
  font-size: 12.5px !important;
}

.composer-style:hover input,
.composer-style:hover button {
  background: #f4f4f5 !important;
}

.composer-note {
  min-height: 17px !important;
  margin: 0 !important;
  text-align: center;
  color: #9a9aa4;
  font-size: 10.5px;
}

.composer-status {
  min-height: 20px !important;
  margin: 0 10px !important;
}

.composer-status .time-chip {
  padding: 0 !important;
  border: 0 !important;
  color: #8b8b96 !important;
  background: transparent !important;
  font-size: 10.5px !important;
}

.composer-status .time-chip.idle,
.composer-status .time-chip.success {
  display: none !important;
}

.composer-status .time-chip.error {
  display: inline-block !important;
  color: #b42318 !important;
}

/* Gradio 6 can mount several queue progress overlays for one submit event.
   The app already reports model loading/inference time in the chat message;
   keep those useful timings and remove only Gradio's duplicated overlay. */
.progress-text,
.wrap:has(> .progress-text) {
  display: none !important;
}

footer {
  display: none !important;
}

.gpt-hidden {
  display: none !important;
}

@media (max-width: 760px) {
  .gpt-sidebar {
    display: none !important;
  }
  .gpt-topbar {
    padding: 7px 12px !important;
  }
  .topbar-title {
    min-width: 82px !important;
  }
  .topbar-model-picker {
    flex: 1 1 auto !important;
    min-width: 0 !important;
    padding-left: 7px !important;
  }
  .model-picker-label {
    display: none !important;
  }
  .topbar-model {
    flex: 1 1 auto !important;
    width: auto !important;
    min-width: 0 !important;
  }
  .gpt-stage {
    padding: 0 10px 10px !important;
  }
  #chat-composer.chat-composer-gpt {
    border-radius: 22px !important;
  }
  .composer-style {
    display: none !important;
  }
  #chat-thread .message {
    max-width: 92% !important;
  }
}
"""


PASTE_IMAGE_JS = """
(() => {
  // Gradio mounts the file input after the page loads.  Bind at the window
  // level so Ctrl+V keeps working even when the composer is re-rendered.
  if (window.__vivqaPasteBound) return;
  window.__vivqaPasteBound = true;

  const pasteImage = (event) => {
    const items = Array.from(event.clipboardData?.items || []);
    const imageItem = items.find(
      (item) => item.type && item.type.toLowerCase().startsWith('image/')
    );
    if (!imageItem) return;

    const file = imageItem.getAsFile?.();
    const input = document.querySelector('#chat-composer input[type="file"]');
    if (!file || !input) return;

    // Some embedded browsers expose clipboard data but hide DataTransfer.
    // In that case leave the text paste untouched and let the user use +.
    if (typeof window.DataTransfer !== 'function') {
      return;
    }

    try {
      const transfer = new window.DataTransfer();
      transfer.items.add(file);
      input.files = transfer.files;
      input.dispatchEvent(new Event('input', { bubbles: true, composed: true }));
      input.dispatchEvent(new Event('change', { bubbles: true, composed: true }));
      event.preventDefault();
      event.stopPropagation();
    } catch (error) {
      console.warn('ViVQA: không thể dán ảnh trực tiếp', error);
    }
  };

  window.addEventListener('paste', pasteImage, true);
})();
"""


def initial_chat() -> list[dict[str, str]]:
    """Return a fresh welcome conversation for the public demo."""
    return [
        {
            "role": "assistant",
            "content": (
                "**Xin chào, mình có thể giúp bạn hiểu nội dung trong ảnh.**\n\n"
                "Hãy đính kèm một hình ảnh, nhập câu hỏi bằng tiếng Việt rồi nhấn **Gửi**. "
                "Bạn có thể hỏi về vật thể, màu sắc, vị trí, hành động hoặc nội dung tổng quát."
            ),
        }
    ]


def answer_chat(
    image,
    question: str,
    model_choice: str,
    response_style: str,
    max_new_tokens: int,
    history: list[dict] | None,
    conversation: list[dict] | None,
) -> tuple[list[dict], str, str, list[dict], list[list], str]:
    """Run inference and append the exchange to a ChatGPT-style conversation."""
    messages = list(conversation or product_initial_chat())
    clean_question = (question or "").strip()
    if clean_question:
        messages.append({"role": "user", "content": clean_question})

    result, timing, records, rows, status = answer(
        image,
        question,
        model_choice,
        response_style,
        max_new_tokens,
        history,
    )
    messages.append({"role": "assistant", "content": result})
    return messages, "", timing, records, rows, status


def reset_chat():
    """Start a clean conversation and clear the current image and session data."""
    return (
        None,
        "",
        initial_chat(),
        time_badge("Sẵn sàng", "idle"),
        image_summary(None),
        [],
        [],
        system_status_html(),
    )


def _create_demo_v1() -> gr.Blocks:
    """Build a restrained conversational interface inspired by modern AI chat apps."""
    with gr.Blocks(title=APP_TITLE, analytics_enabled=False) as demo:
        history_state = gr.State([])

        with gr.Row(elem_classes="chat-shell"):
            with gr.Column(scale=1, min_width=258, elem_classes="chat-sidebar"):
                gr.HTML(
                    """
                    <div class="brand-lockup">
                      <div class="brand-mark">VQ</div>
                      <div>
                        <div class="brand-name">ViVQA Assistant</div>
                        <div class="brand-caption">Vietnamese Visual Question Answering</div>
                      </div>
                    </div>
                    """
                )
                new_chat = gr.Button(
                    "+  Cuộc trò chuyện mới",
                    elem_classes="new-chat-button",
                )
                gr.HTML('<div class="settings-label">Thiết lập</div>')
                with gr.Column(elem_classes=["sidebar-settings"]):
                    model_choice = gr.Dropdown(
                        choices=list(ADAPTERS),
                        value=STUDENT_LABEL,
                        label="Mô hình",
                    )
                    response_style = gr.Radio(
                        choices=["Mô tả chi tiết", "Cân bằng", "Trả lời ngắn"],
                        value="Cân bằng",
                        label="Cách trả lời",
                    )
                    max_new_tokens = gr.Slider(
                        minimum=16,
                        maximum=160,
                        value=96,
                        step=8,
                        label="Độ dài tối đa",
                    )
                gr.HTML(
                    """
                    <div class="sidebar-note">
                      Student 2B phù hợp cho Colab T4. Khi đổi model, hệ thống sẽ giải phóng model cũ trước khi nạp model mới.
                    </div>
                    """
                )

            with gr.Column(scale=4, elem_classes="chat-main"):
                gr.HTML(
                    """
                    <header class="chat-topbar">
                      <div class="chat-topbar-title">Hỏi đáp hình ảnh tiếng Việt</div>
                      <div class="model-pill">Qwen2-VL · Knowledge Distillation</div>
                    </header>
                    """
                )

                with gr.Column(elem_classes="chat-stage"):
                    chatbot = gr.Chatbot(
                        value=initial_chat(),
                        height=510,
                        buttons=["copy"],
                        render_markdown=True,
                        show_label=False,
                        elem_id="chat-thread",
                    )

                    with gr.Row(elem_classes="suggestion-row"):
                        suggestion_1 = gr.Button(
                            "Mô tả bức ảnh",
                            elem_classes="suggestion-button",
                        )
                        suggestion_2 = gr.Button(
                            "Có những vật thể nào?",
                            elem_classes="suggestion-button",
                        )
                        suggestion_3 = gr.Button(
                            "Chuyện gì đang xảy ra?",
                            elem_classes="suggestion-button",
                        )

                    with gr.Row(elem_classes="chat-composer"):
                        image = gr.Image(
                            type="pil",
                            show_label=False,
                            height=82,
                            width=92,
                            min_width=92,
                            sources=["upload", "clipboard"],
                            buttons=[],
                            placeholder="Thêm ảnh",
                            elem_id="composer-image",
                        )
                        question = gr.Textbox(
                            show_label=False,
                            placeholder="Hỏi bất kỳ điều gì về hình ảnh...",
                            lines=2,
                            container=False,
                            scale=8,
                            elem_classes="chat-input",
                        )
                        submit = gr.Button(
                            "Gửi",
                            variant="primary",
                            scale=0,
                            min_width=58,
                            elem_classes="send-button",
                        )

                    with gr.Row(elem_classes="composer-meta"):
                        image_meta = gr.HTML(
                            image_summary(None),
                            elem_classes="image-meta-output",
                        )
                        timing = gr.HTML(
                            time_badge("Sẵn sàng", "idle"),
                            elem_classes="time-chip-output",
                        )
                    gr.HTML(
                        '<div class="privacy-note">Câu trả lời được tạo bởi mô hình thị giác–ngôn ngữ và có thể chưa hoàn toàn chính xác.</div>'
                    )

                with gr.Column(visible=False):
                    history_table = gr.Dataframe(
                        headers=HISTORY_HEADERS,
                        datatype=["str", "str", "str", "str", "number"],
                        value=[],
                        interactive=False,
                    )
                    system_status = gr.HTML(system_status_html())

        inputs = [
            image,
            question,
            model_choice,
            response_style,
            max_new_tokens,
            history_state,
            chatbot,
        ]
        outputs = [
            chatbot,
            question,
            timing,
            history_state,
            history_table,
            system_status,
        ]
        submit.click(
            fn=answer_chat,
            inputs=inputs,
            outputs=outputs,
            show_progress="hidden",
        )
        question.submit(
            fn=answer_chat,
            inputs=inputs,
            outputs=outputs,
            show_progress="hidden",
        )
        image.change(fn=image_summary, inputs=image, outputs=image_meta, queue=False)
        new_chat.click(
            fn=reset_chat,
            outputs=[
                image,
                question,
                chatbot,
                timing,
                image_meta,
                history_state,
                history_table,
                system_status,
            ],
            queue=False,
        )
        suggestion_1.click(lambda: "Hãy mô tả bức ảnh này.", outputs=question, queue=False)
        suggestion_2.click(
            lambda: "Trong ảnh có những vật thể nào?",
            outputs=question,
            queue=False,
        )
        suggestion_3.click(
            lambda: "Chuyện gì đang xảy ra trong ảnh?",
            outputs=question,
            queue=False,
        )
    return demo


def product_initial_chat() -> list[dict[str, str]]:
    """Return an empty conversation for a new ViVQA session."""
    return []


def _message_text(content) -> str:
    """Extract readable text from a Gradio messages-format content value."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, (list, tuple)):
        parts = [item for item in (_message_text(value) for value in content) if item]
        return " ".join(parts).strip()
    return ""


def conversation_context(conversation: list[dict] | None) -> str:
    """Keep a small text context so follow-up questions behave conversationally."""
    turns: list[str] = []
    for message in list(conversation or []):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        text = _message_text(message.get("content"))
        if role not in {"user", "assistant"} or not text:
            continue
        if role == "assistant" and "\n\n*" in text:
            answer_text, timing_text = text.rsplit("\n\n*", 1)
            if "suy luận" in timing_text:
                text = answer_text.strip()
        turns.append(f"{'Người dùng' if role == 'user' else 'Trợ lý'}: {text}")
    return "\n".join(turns[-4:])[-1800:]


def composer_value(value) -> tuple[str, str | None]:
    """Extract text and the first uploaded image path from the composer."""
    if isinstance(value, str):
        return value.strip(), None
    if not isinstance(value, dict):
        return "", None

    text = str(value.get("text") or "").strip()
    files = list(value.get("files") or [])
    if not files:
        return text, None

    first_file = files[0]
    if isinstance(first_file, dict):
        path = first_file.get("path")
    else:
        path = getattr(first_file, "path", first_file)
    return text, str(path) if path else None


def open_composer_image(image_path: str | None) -> Image.Image | None:
    """Load an uploaded image without keeping an open file handle."""
    if not image_path:
        return None
    with Image.open(image_path) as source:
        return source.convert("RGB").copy()


def chat_user_content(image_path: str | None, question: str):
    """Render a sent attachment and its question in the same user turn."""
    content: list = []
    if image_path:
        content.append({"path": image_path, "alt_text": "Ảnh đã đính kèm"})
    if question:
        content.append(question)
    if not content:
        return ""
    return content if len(content) > 1 else content[0]


def sidebar_history_html(items: list[str] | None) -> str:
    """Render a compact, session-local question history for the sidebar."""
    titles = list(items or [])
    if not titles:
        body = '<div class="history-empty">Chưa có câu hỏi.</div>'
    else:
        rows = []
        for title in titles[:20]:
            clean_title = " ".join(str(title).split())
            short_title = clean_title if len(clean_title) <= 38 else f"{clean_title[:37]}…"
            rows.append(f'<div class="history-item" title="{html.escape(clean_title)}">{html.escape(short_title)}</div>')
        body = "".join(rows)
    return f'<div class="history-panel"><div class="history-heading">Câu hỏi đã gửi</div>{body}</div>'


def answer_chat_gpt(
    composer,
    model_choice: str,
    response_style: str,
    max_new_tokens: int,
    active_image_path: str | None,
    history: list[dict] | None,
    conversation: list[dict] | None,
    question_titles: list[str] | None,
) -> tuple[
    list[dict], dict, str, str | None, list[dict], list[list], str, list[str], str
]:
    """Run inference, update the conversation, and retain sidebar question titles."""
    messages = list(conversation or product_initial_chat())
    clean_question, attached_image_path = composer_value(composer)
    selected_image_path = attached_image_path or active_image_path
    records = list(history or [])
    titles = list(question_titles or [])

    if attached_image_path and not clean_question:
        clean_question = "Mô tả chi tiết hình ảnh này."

    if not clean_question:
        return (
            messages,
            composer if isinstance(composer, dict) else {"text": "", "files": []},
            time_badge("Nhập câu hỏi trước", "error"),
            active_image_path,
            records,
            history_rows(records),
            system_status_html(),
            titles,
            sidebar_history_html(titles),
        )
    if not selected_image_path:
        return (
            messages,
            {"text": clean_question, "files": []},
            time_badge("Đính kèm ảnh trước", "error"),
            active_image_path,
            records,
            history_rows(records),
            system_status_html(),
            titles,
            sidebar_history_html(titles),
        )

    try:
        image = open_composer_image(selected_image_path)
    except Exception as exc:
        return (
            messages,
            {
                "text": clean_question,
                "files": [attached_image_path] if attached_image_path else [],
            },
            time_badge(f"Không thể đọc ảnh: {exc}", "error"),
            active_image_path,
            records,
            history_rows(records),
            system_status_html(),
            titles,
            sidebar_history_html(titles),
        )

    context = conversation_context(messages)
    previous_record_count = len(records)

    result, timing, records, rows, status = answer(
        image,
        clean_question,
        model_choice,
        response_style,
        max_new_tokens,
        history,
        conversation_context=context,
    )
    messages.append(
        {
            "role": "user",
            "content": chat_user_content(attached_image_path, clean_question),
        }
    )

    latest = (
        records[-1]
        if len(records) > previous_record_count
        and records[-1].get("question") == clean_question
        else None
    )
    if latest:
        load_seconds = float(latest.get("load_seconds", 0) or 0)
        inference_seconds = float(latest.get("inference_seconds", 0) or 0)
        timing_text = f"{model_choice} · suy luận {inference_seconds:.1f} giây"
        if load_seconds:
            timing_text = f"{model_choice} · nạp {load_seconds:.1f} giây · suy luận {inference_seconds:.1f} giây"
        assistant_content = f"{result}\n\n*{timing_text}*"
    else:
        assistant_content = result
    messages.append({"role": "assistant", "content": assistant_content})

    titles = [clean_question, *[item for item in titles if item != clean_question]][:20]
    return (
        messages,
        {"text": "", "files": []},
        timing,
        selected_image_path,
        records,
        rows,
        status,
        titles,
        sidebar_history_html(titles),
    )


def reset_product_chat():
    """Reset the product workspace without reintroducing legacy UI copy."""
    return (
        {"text": "", "files": []},
        product_initial_chat(),
        time_badge("Sẵn sàng", "idle"),
        None,
        [],
        [],
        system_status_html(),
        [],
        sidebar_history_html([]),
    )


def create_demo() -> gr.Blocks:
    """Build the conversational Vietnamese visual question answering demo."""
    with gr.Blocks(
        title=APP_TITLE,
        analytics_enabled=False,
        fill_height=True,
        fill_width=True,
    ) as demo:
        history_state = gr.State([])
        question_titles_state = gr.State([])
        active_image_path_state = gr.State(None)

        with gr.Row(elem_classes="gpt-shell"):
            with gr.Column(elem_classes="gpt-sidebar"):
                gr.HTML(
                    """
                    <div class="sidebar-brand">
                      <span class="sidebar-mark">VQ</span>
                      <span>ViVQA</span>
                    </div>
                    """
                )
                new_chat = gr.Button(
                    "↻  Bắt đầu lại",
                    elem_classes="sidebar-new-chat",
                )
                with gr.Accordion(
                    "Tùy chọn trả lời",
                    open=False,
                    elem_classes="sidebar-options",
                ):
                    response_style = gr.Dropdown(
                        choices=["Mô tả chi tiết", "Cân bằng", "Trả lời ngắn"],
                        value="Cân bằng",
                        label="Cách trả lời",
                    )
                    max_new_tokens = gr.Slider(
                        minimum=16,
                        maximum=160,
                        value=96,
                        step=8,
                        label="Độ dài tối đa",
                    )
                history_html = gr.HTML(
                    sidebar_history_html([]),
                    elem_classes="sidebar-history-wrap",
                )

            with gr.Column(elem_classes="gpt-main"):
                with gr.Row(elem_classes="gpt-topbar"):
                    gr.HTML('<div class="topbar-title">ViVQA</div>')
                    gr.HTML("<div></div>", elem_classes="topbar-spacer")
                    with gr.Row(elem_classes="topbar-model-picker"):
                        gr.HTML('<span class="model-picker-label">Mô hình</span>')
                        model_choice = gr.Dropdown(
                            choices=list(ADAPTERS),
                            value=STUDENT_LABEL,
                            label="Mô hình",
                            show_label=False,
                            container=False,
                            min_width=218,
                            elem_classes="topbar-model",
                        )

                with gr.Column(elem_classes="gpt-stage"):
                    chatbot = gr.Chatbot(
                        value=product_initial_chat(),
                        height=460,
                        buttons=None,
                        render_markdown=True,
                        group_consecutive_messages=True,
                        allow_file_downloads=False,
                        show_label=False,
                        elem_id="chat-thread",
                    )

                    with gr.Column(elem_classes="composer-area"):
                        timing = gr.HTML(
                            time_badge("Sẵn sàng", "idle"),
                            elem_classes="composer-status",
                        )
                        composer = gr.MultimodalTextbox(
                            sources=["upload"],
                            file_types=["image"],
                            file_count="single",
                            lines=1,
                            max_lines=5,
                            placeholder="Đính kèm ảnh và đặt câu hỏi",
                            show_label=False,
                            container=False,
                            interactive=True,
                            submit_btn="↑",
                            stop_btn=False,
                            elem_id="chat-composer",
                            elem_classes="chat-composer-gpt",
                        )
                        gr.HTML(
                            '<div class="composer-note">Ctrl+V để dán ảnh · Enter để gửi</div>'
                        )

                with gr.Column(visible=False, elem_classes="gpt-hidden"):
                    history_table = gr.Dataframe(
                        headers=HISTORY_HEADERS,
                        datatype=["str", "str", "str", "str", "number"],
                        value=[],
                        interactive=False,
                    )
                    system_status = gr.HTML(system_status_html())

        inputs = [
            composer,
            model_choice,
            response_style,
            max_new_tokens,
            active_image_path_state,
            history_state,
            chatbot,
            question_titles_state,
        ]
        outputs = [
            chatbot,
            composer,
            timing,
            active_image_path_state,
            history_state,
            history_table,
            system_status,
            question_titles_state,
            history_html,
        ]
        composer.submit(
            fn=answer_chat_gpt,
            inputs=inputs,
            outputs=outputs,
            show_progress="hidden",
        )
        new_chat.click(
            fn=reset_product_chat,
            outputs=[
                composer,
                chatbot,
                timing,
                active_image_path_state,
                history_state,
                history_table,
                system_status,
                question_titles_state,
                history_html,
            ],
            queue=False,
        )
    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Vietnamese VQA web demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a temporary public Gradio URL")
    parser.add_argument("--check", action="store_true", help="Show model sources and exit")
    parser.add_argument(
        "--download-model",
        nargs="?",
        choices=("2b", "7b", "all"),
        const="all",
        help="Download Qwen2-VL base model(s) into models/ and exit",
    )
    parser.add_argument(
        "--download-adapter",
        nargs="?",
        choices=("2b", "7b", "all"),
        const="all",
        help="Download fine-tuned adapter(s) from Ouyangxuelili on Hugging Face",
    )
    return parser.parse_args()


def download_resources(kind: str, selection: str) -> None:
    """Download base models or adapters for offline use."""
    from huggingface_hub import snapshot_download

    if kind == "model":
        repositories = {
            "2b": ("Qwen/Qwen2-VL-2B-Instruct", ROOT / "models" / "Qwen2-VL-2B-Instruct"),
            "7b": ("Qwen/Qwen2-VL-7B-Instruct", ROOT / "models" / "Qwen2-VL-7B-Instruct"),
        }
    else:
        repositories = {
            "2b": (ADAPTERS[STUDENT_LABEL]["repo"], ADAPTERS[STUDENT_LABEL]["path"]),
            "7b": (ADAPTERS[TEACHER_LABEL]["repo"], ADAPTERS[TEACHER_LABEL]["path"]),
        }

    selected = repositories if selection == "all" else {selection: repositories[selection]}
    for repository, destination in selected.values():
        downloaded = snapshot_download(
            repo_id=str(repository),
            local_dir=Path(destination),
            max_workers=4,
        )
        print(f"Downloaded {repository} to: {downloaded}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    for message in adapter_status():
        print(message)

    if args.download_model:
        download_resources("model", args.download_model)
    elif args.download_adapter:
        download_resources("adapter", args.download_adapter)
    elif not args.check:
        theme = gr.themes.Base(
            primary_hue="slate",
            secondary_hue="slate",
            neutral_hue="slate",
            radius_size="sm",
        )
        create_demo().queue(default_concurrency_limit=1, max_size=8).launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            show_error=False,
            theme=theme,
            js=PASTE_IMAGE_JS,
            css=GPT_CSS,
        )
