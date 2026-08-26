"""Branded Gradio demo for Vietnamese VQA Distillation."""

from __future__ import annotations

import argparse
import base64
import gc
import html
import io
import re
import sys
import threading
import time
import uuid
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

STUDENT_LABEL = "2B · Student"
TEACHER_LABEL = "7B · Teacher"

EMPTY_CHAT_EXAMPLES = [
    {
        "text": "Mô tả nội dung chính của ảnh",
        "display_text": "Mô tả nội dung",
    },
    {
        "text": "Trong ảnh có những đối tượng nào?",
        "display_text": "Nhận biết đối tượng",
    },
    {
        "text": "Đọc phần chữ xuất hiện trong ảnh",
        "display_text": "Đọc chữ trong ảnh",
    },
]

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

    if model_choice not in ADAPTERS:
        valid = ", ".join(ADAPTERS)
        raise ValueError(f"Model không hợp lệ: {model_choice!r}. Chọn một trong: {valid}")

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


def running_badge(message: str, started_at: float | None = None) -> str:
    """Render the live stage and elapsed time next to the composer."""
    safe_message = html.escape(message)
    started_ms = int((started_at or time.time()) * 1000)
    return (
        '<span class="time-chip running" role="status" aria-live="polite">'
        '<span class="status-pulse" aria-hidden="true"></span>'
        f'<span>{safe_message}</span>'
        f'<span class="live-elapsed" data-start="{started_ms}">0.0 giây</span>'
        '</span>'
    )


def pending_message_html(message: str) -> str:
    """Render a quiet typing indicator while detailed status stays by the composer."""
    safe_message = html.escape(message)
    return (
        f'<div class="assistant-typing" role="status" aria-label="{safe_message}">'
        '<span></span><span></span><span></span>'
        '</div>'
    )


def friendly_inference_error(exc: Exception) -> str:
    """Translate backend failures into short, useful Vietnamese UI messages."""
    detail = str(exc).lower()
    if "not enough free ram" in detail or "gb required" in detail:
        return (
            "Máy này không đủ bộ nhớ để chạy mô hình. "
            "Hãy chạy demo trên Colab với GPU T4."
        )
    if "out of memory" in detail or "cuda" in detail and "memory" in detail:
        return (
            "GPU không đủ bộ nhớ. Hãy chọn 2B · Student hoặc khởi động lại phiên Colab."
        )
    if "no such file" in detail or "not found" in detail:
        return "Không tìm thấy tệp mô hình cần thiết. Hãy kiểm tra lại thư mục trên Drive."
    return "Không thể xử lý ảnh lúc này. Hãy thử lại hoặc khởi động lại phiên Colab."


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


def _needs_conversation_context(question: str) -> bool:
    """Use previous turns only for questions that clearly refer back to them."""
    normalized = " ".join(question.lower().split())
    references = (
        "còn ",
        "vậy ",
        "nó ",
        "họ ",
        "người đó",
        "vật đó",
        "con đó",
        "bên kia",
        "ở đó",
        "như vậy",
    )
    return len(normalized) <= 120 and any(token in normalized for token in references)


def build_model_prompt(
    question: str,
    model_choice: str,
    response_style: str,
    context: str | None = None,
) -> str:
    """Build a short prompt compatible with the fine-tuned VQA adapters."""
    clean_question = " ".join(question.strip().split())
    prompt = clean_question

    if context and _needs_conversation_context(clean_question):
        prompt = f"Ngữ cảnh gần nhất:\n{context}\nCâu hỏi: {clean_question}"

    # The distilled Student was trained mostly on concise VQA labels. Long
    # meta-instructions make it repeat the instruction instead of describing
    # the image, so each model gets a deliberately short instruction.
    if response_style == "Mô tả chi tiết":
        if model_choice == STUDENT_LABEL:
            return f"{prompt}\nMô tả ảnh bằng tiếng Việt."
        return f"{prompt}\nMô tả rõ các chi tiết nhìn thấy trong ảnh bằng tiếng Việt."
    if response_style == "Trả lời ngắn":
        return f"{prompt}\nTrả lời ngắn gọn bằng tiếng Việt."
    return prompt


def looks_like_prompt_echo(answer_text: str) -> bool:
    """Detect the instruction-echo failure seen in the distilled model."""
    normalized = re.sub(r"\s+", " ", answer_text.lower()).strip(" .,:;!?")
    echo_fragments = (
        "mô tả đầy đủ ảnh",
        "mô tả ảnh bằng tiếng việt",
        "mô tả rõ các chi tiết",
        "chỉ nêu những điều",
        "trả lời ngắn gọn bằng tiếng việt",
        "câu hỏi hiện tại",
    )
    return not normalized or any(fragment in normalized for fragment in echo_fragments)


def answer(
    image,
    question: str,
    model_choice: str,
    response_style: str,
    max_new_tokens: int,
    history: list[dict] | None,
    conversation_context: str | None = None,
    preloaded_engine: VQAInference | None = None,
    preloaded_load_seconds: float = 0.0,
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
        if preloaded_engine is None:
            engine, load_seconds = get_inference(model_choice)
        else:
            engine = preloaded_engine
            load_seconds = float(preloaded_load_seconds)
        prompt = build_model_prompt(
            question,
            model_choice,
            response_style,
            conversation_context,
        )
        min_new_tokens = 0

        inference_started = time.perf_counter()
        prediction = engine.predict_single(
            image.convert("RGB"),
            prompt,
            int(max_new_tokens),
            min_new_tokens,
        )
        retried_prompt_echo = False
        if looks_like_prompt_echo(prediction):
            retried_prompt_echo = True
            prediction = engine.predict_single(
                image.convert("RGB"),
                question.strip(),
                min(int(max_new_tokens), 96),
                0,
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
                "retried_prompt_echo": retried_prompt_echo,
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
        print(f"[ViVQA] Inference error: {exc}", file=sys.stderr, flush=True)
        return (
            friendly_inference_error(exc),
            time_badge("", "idle"),
            records,
            history_rows(records),
            system_status_html(),
        )


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

.gpt-sidebar {
  margin: 0 !important;
  padding: 12px 10px !important;
  gap: 10px !important;
  border-right: 1px solid #eeeeef !important;
  background: var(--gpt-sidebar) !important;
  box-shadow: none !important;
  transition: left .2s ease !important;
}

.gpt-sidebar .sidebar-content {
  padding: 4px 20px 12px 10px !important;
  scrollbar-width: thin;
  scrollbar-color: #c8c8ce transparent;
}

.gpt-sidebar .sidebar-content > .column {
  gap: 8px !important;
}

.gpt-sidebar .sidebar-content > .column > .block:first-child .html-container {
  padding: 0 !important;
}

.gpt-sidebar .sidebar-content::-webkit-scrollbar,
.conversation-selector::-webkit-scrollbar {
  width: 6px;
}

.gpt-sidebar .sidebar-content::-webkit-scrollbar-track,
.conversation-selector::-webkit-scrollbar-track {
  background: transparent;
}

.gpt-sidebar .sidebar-content::-webkit-scrollbar-thumb,
.conversation-selector::-webkit-scrollbar-thumb {
  border-radius: 999px;
  background: #c8c8ce;
}

.gpt-sidebar .sidebar-content::-webkit-scrollbar-thumb:hover,
.conversation-selector::-webkit-scrollbar-thumb:hover {
  background: #aaaab2;
}

html:not(.vivqa-sidebar-open) .gpt-sidebar .toggle-button {
  transform: translateX(12px) !important;
}

html:not(.vivqa-sidebar-open) body .gradio-container .gpt-main .gpt-topbar {
  padding-left: 40px !important;
}

.sidebar-brand {
  display: flex;
  height: 36px;
  align-items: center;
  justify-content: space-between;
  padding: 0 8px;
  color: var(--gpt-ink);
  font-size: 14px;
  font-weight: 650;
  letter-spacing: -.01em;
}

.sidebar-mark {
  display: grid;
  width: 28px;
  height: 28px;
  place-items: center;
  border-radius: 8px;
  color: var(--gpt-ink) !important;
  background: transparent;
  font-size: 10px;
  font-weight: 750;
}

.sidebar-mark svg {
  width: 19px;
  height: 19px;
  fill: none;
  color: var(--gpt-ink) !important;
  stroke: var(--gpt-ink) !important;
  stroke-width: 1.7;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.sidebar-mark circle {
  fill: var(--gpt-ink) !important;
  stroke: none;
}

.sidebar-new-chat {
  width: 100% !important;
  min-height: 44px !important;
  margin: 0 !important;
  padding: 0 12px !important;
  justify-content: flex-start !important;
  border: 0 !important;
  border-radius: 10px !important;
  color: var(--gpt-ink) !important;
  background: transparent !important;
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
  border: 0 !important;
  border-radius: 10px !important;
  background: transparent !important;
  box-shadow: none !important;
}

.sidebar-options > .label-wrap {
  min-height: 38px !important;
  padding: 8px 11px !important;
  color: var(--gpt-ink) !important;
  font-size: 12.5px !important;
  font-weight: 550 !important;
}

.sidebar-options > .label-wrap:hover {
  background: var(--gpt-hover) !important;
}

.sidebar-options > .wrap {
  padding: 0 10px 10px !important;
}

.sidebar-options .block,
.sidebar-options .form {
  border: 0 !important;
  box-shadow: none !important;
}

.sidebar-options label {
  color: var(--gpt-muted) !important;
  font-size: 11.5px !important;
}

.sidebar-options > .label-wrap span {
  color: var(--gpt-ink) !important;
  font-size: 12.5px !important;
  font-weight: 550 !important;
}

.sidebar-history-wrap {
  flex: 1 1 auto !important;
  min-height: 0 !important;
  overflow: hidden !important;
}

.conversation-selector {
  flex: 1 1 auto !important;
  min-height: 0 !important;
  overflow-y: auto !important;
  padding: 0 2px !important;
  border: 0 !important;
  background: transparent !important;
  scrollbar-width: thin;
  scrollbar-color: #c8c8ce transparent;
}

.conversation-selector > div,
.conversation-selector .wrap {
  gap: 2px !important;
}

.conversation-selector label {
  display: block !important;
  width: 100% !important;
  min-width: 0 !important;
  min-height: 38px !important;
  margin: 0 !important;
  padding: 9px 10px !important;
  overflow: hidden !important;
  border: 0 !important;
  border-radius: 9px !important;
  color: #3f3f46 !important;
  background: transparent !important;
  font-size: 13px !important;
  line-height: 1.35 !important;
  text-overflow: ellipsis !important;
  white-space: nowrap !important;
  cursor: pointer !important;
}

.conversation-selector label:hover {
  background: #eeeeF0 !important;
}

.conversation-selector label:has(input:checked) {
  color: var(--gpt-ink) !important;
  background: #e7e7ea !important;
  font-weight: 560 !important;
}

.conversation-selector input[type="radio"] {
  position: absolute !important;
  width: 1px !important;
  height: 1px !important;
  margin: -1px !important;
  padding: 0 !important;
  overflow: hidden !important;
  clip: rect(0 0 0 0) !important;
  clip-path: inset(50%) !important;
  white-space: nowrap !important;
}

.history-heading {
  padding: 7px 9px 5px;
  color: #8a8a95;
  font-size: 11px;
  font-weight: 650;
  white-space: nowrap;
}

.sidebar-history-heading {
  min-height: 34px !important;
  margin: 8px 0 0 !important;
  align-items: center !important;
}

.sidebar-history-heading:not(:has(+ .conversation-selector label)) {
  display: none !important;
}

.sidebar-history-heading > .html-container {
  flex: 1 1 auto !important;
  min-width: 0 !important;
}

.sidebar-rename-menu {
  flex: 0 0 30px !important;
  width: 30px !important;
  min-width: 30px !important;
  min-height: 30px !important;
  margin: 0 4px 0 0 !important;
  padding: 0 !important;
  border: 0 !important;
  border-radius: 8px !important;
  color: #777781 !important;
  background: transparent !important;
  box-shadow: none !important;
  font-size: 18px !important;
}

.sidebar-rename-menu:hover {
  color: var(--gpt-ink) !important;
  background: #ececef !important;
}

.sidebar-rename-panel {
  margin: 4px 2px !important;
  padding: 9px !important;
  border: 1px solid var(--gpt-line) !important;
  border-radius: 11px !important;
  background: #fff !important;
  box-shadow: 0 8px 24px rgba(0, 0, 0, .06) !important;
}

.sidebar-rename-panel:not(:has(input)) {
  display: none !important;
}

.sidebar-rename-panel .block {
  overflow: visible !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}

.sidebar-rename-panel input,
.sidebar-rename-panel textarea {
  min-height: 36px !important;
  padding: 8px 9px !important;
  border: 1px solid #d4d4d8 !important;
  border-radius: 8px !important;
  color: var(--gpt-ink) !important;
  background: #fff !important;
  font-size: 12.5px !important;
}

.sidebar-rename-panel .input-container {
  border: 0 !important;
  box-shadow: none !important;
}

.sidebar-rename-panel input:focus,
.sidebar-rename-panel input:focus-visible,
.sidebar-rename-panel textarea:focus,
.sidebar-rename-panel textarea:focus-visible {
  border-color: #9a9aa3 !important;
  outline: none !important;
  box-shadow: 0 0 0 2px rgba(113, 113, 122, .16) !important;
}

.sidebar-rename-actions {
  display: flex !important;
  flex-wrap: nowrap !important;
  margin-top: 7px !important;
  gap: 6px !important;
}

.sidebar-rename-actions > * {
  flex: 1 1 0 !important;
  width: auto !important;
  min-width: 0 !important;
}

.sidebar-rename-actions button {
  width: 100% !important;
  min-height: 32px !important;
  padding: 0 10px !important;
  border-radius: 8px !important;
  font-size: 12px !important;
}

.sidebar-new-chat:focus-visible,
.sidebar-options > .label-wrap:focus-visible,
.sidebar-rename-menu:focus-visible,
.gpt-sidebar .toggle-button:focus-visible,
.conversation-selector label:has(input:focus-visible),
.topbar-model:focus-within {
  outline: 2px solid #73737f !important;
  outline-offset: 2px !important;
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
  transition: margin-left .2s ease, width .2s ease !important;
}

.vivqa-sidebar-open .gpt-main {
  width: calc(100% - 260px) !important;
  margin-left: 260px !important;
}

.gpt-topbar {
  flex: 0 0 58px !important;
  width: 100% !important;
  min-height: 58px !important;
  margin: 0 !important;
  padding: 8px 18px !important;
  gap: 8px !important;
  align-items: center !important;
  flex-wrap: nowrap !important;
  background: rgba(255,255,255,.96) !important;
}

.topbar-identity-wrap {
  flex: 0 0 auto !important;
  width: auto !important;
  min-width: 0 !important;
}

.topbar-identity {
  display: flex;
  align-items: center;
  gap: 7px;
}

.topbar-title {
  min-width: 0;
  color: var(--gpt-ink);
  font-size: 16px;
  font-weight: 680;
  letter-spacing: -.02em;
}

.topbar-spacer-wrap {
  flex: 1 1 auto !important;
  min-width: 16px !important;
}

.topbar-model-picker {
  flex: 0 0 auto !important;
  width: auto !important;
  min-width: 164px !important;
  min-height: 38px !important;
  margin: 0 !important;
  padding: 0 4px !important;
  gap: 0 !important;
  align-items: center !important;
  border: 0 !important;
  border-radius: 10px !important;
  background: transparent !important;
  box-shadow: none !important;
}

.topbar-model {
  flex: 0 0 156px !important;
  width: 156px !important;
  min-width: 156px !important;
  height: 38px !important;
  min-height: 38px !important;
  max-height: 38px !important;
  margin: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}

.topbar-model input,
.topbar-model button {
  min-height: 36px !important;
  border: 0 !important;
  border-radius: 9px !important;
  background: transparent !important;
  font-size: 13.5px !important;
  font-weight: 550 !important;
}

.topbar-model:hover input,
.topbar-model:hover button {
  background: #f5f5f5 !important;
}

.topbar-model-status {
  flex: 0 0 auto !important;
  margin-left: 1px !important;
}

.mobile-new-chat {
  display: none !important;
}

.model-readiness {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  color: #7b7b85;
  font-size: 10.5px;
  white-space: nowrap;
}

.model-readiness span {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #a1a1aa;
}

.model-readiness.waiting {
  display: none;
}

.model-readiness.ready span {
  background: #16a34a;
}

.model-readiness.loading span {
  background: #111827;
  animation: vivqa-pulse 1.1s ease-in-out infinite;
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

.assistant-typing {
  display: inline-flex;
  min-height: 24px;
  align-items: center;
  gap: 4px;
}

.assistant-typing span {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #565660;
  animation: vivqa-typing 1.15s ease-in-out infinite;
}

.assistant-typing span:nth-child(2) {
  animation-delay: .14s;
}

.assistant-typing span:nth-child(3) {
  animation-delay: .28s;
}

#chat-thread [class*="example"] {
  border-color: #e5e5e8 !important;
  border-radius: 12px !important;
  background: #fff !important;
  box-shadow: none !important;
}

#chat-thread [class*="example"]:hover {
  border-color: #d7d7dc !important;
  background: #f7f7f8 !important;
}

#chat-thread .message.user img {
  display: block !important;
  max-width: 260px !important;
  max-height: 220px !important;
  margin-bottom: 8px !important;
  border-radius: 12px !important;
  object-fit: contain !important;
}

#chat-thread .message.user button[aria-label="Fullscreen"] {
  display: none !important;
}

#chat-thread .top-panel button[aria-label="Clear"] {
  display: none !important;
}

#chat-thread .message-buttons {
  display: flex !important;
  margin-top: 3px !important;
  opacity: 0;
  transition: opacity .14s ease;
}

#chat-thread .message:hover .message-buttons,
#chat-thread .message:focus-within .message-buttons {
  opacity: 1;
}

#chat-thread .message.user .message-buttons {
  display: none !important;
}

#chat-thread .message-buttons button {
  width: 30px !important;
  min-width: 30px !important;
  height: 30px !important;
  min-height: 30px !important;
  padding: 6px !important;
  border: 0 !important;
  border-radius: 8px !important;
  color: #74747e !important;
  background: transparent !important;
}

#chat-thread .message-buttons button:hover {
  color: var(--gpt-ink) !important;
  background: #f1f1f3 !important;
}

#chat-thread .placeholder,
#chat-thread .empty,
#chat-thread [class*="placeholder"] {
  color: #777782 !important;
  font-size: 15px !important;
  font-weight: 500 !important;
  letter-spacing: -.01em !important;
}

.composer-area {
  flex: 0 0 auto !important;
  width: 100% !important;
  margin: 0 !important;
  gap: 6px !important;
}

.active-image-strip {
  width: 100% !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 8px !important;
  align-items: center !important;
  flex-wrap: nowrap !important;
  gap: 8px !important;
}

.active-image-preview {
  flex: 0 0 auto !important;
  width: fit-content !important;
  max-width: calc(100% - 38px) !important;
  min-width: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
}

.active-image-preview .html-container {
  width: fit-content !important;
  padding: 0 !important;
}

.pinned-image {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  max-width: 100%;
  padding: 5px 9px 5px 5px;
  border: 1px solid #e4e4e7;
  border-radius: 12px;
  color: #575761;
  background: #f7f7f8;
}

.pinned-image img {
  width: 42px;
  height: 42px;
  flex: 0 0 42px;
  border: 1px solid #e3e3e7;
  border-radius: 10px;
  object-fit: cover;
}

.pinned-image strong,
.pinned-image span {
  display: block;
}

.pinned-image strong {
  margin-bottom: 2px;
  color: #3f3f46;
  font-size: 11.5px;
  font-weight: 600;
}

.pinned-image span {
  color: #8b8b94;
  font-size: 10.5px;
}

.clear-active-image {
  flex: 0 0 30px !important;
  width: 30px !important;
  min-width: 30px !important;
  max-width: 30px !important;
  height: 30px !important;
  min-height: 30px !important;
  margin-left: 0 !important;
  padding: 0 !important;
  border: 0 !important;
  border-radius: 50% !important;
  color: #666670 !important;
  background: #f1f1f3 !important;
  box-shadow: none !important;
  font-size: 17px !important;
  line-height: 1 !important;
}

.clear-active-image:hover {
  color: var(--gpt-ink) !important;
  background: #e7e7ea !important;
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
  color: #7c7c87 !important;
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

#chat-composer [data-testid="submit-button"]:focus-visible,
#chat-composer [data-testid="upload-button"]:focus-visible {
  outline: 2px solid #73737f !important;
  outline-offset: 2px !important;
}

#chat-composer [data-testid="submit-button"]:disabled,
.topbar-model input:disabled,
.topbar-model button:disabled {
  cursor: not-allowed !important;
  opacity: .48 !important;
}

#chat-composer .stop-button {
  display: none !important;
  flex: 0 0 42px !important;
  width: 42px !important;
  min-width: 42px !important;
  height: 42px !important;
  min-height: 42px !important;
  margin: 0 !important;
  padding: 0 !important;
  border: 0 !important;
  border-radius: 50% !important;
  color: #fff !important;
  background: #111827 !important;
  box-shadow: none !important;
}

#chat-composer .stop-button.vivqa-running {
  display: inline-flex !important;
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

.composer-status .time-chip.running {
  display: inline-flex !important;
  align-items: center !important;
  gap: 7px !important;
  color: #555560 !important;
}

.status-pulse {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #111827;
  animation: vivqa-pulse 1.1s ease-in-out infinite;
}

.live-elapsed {
  color: #898994;
  font-variant-numeric: tabular-nums;
}

@keyframes vivqa-pulse {
  0%, 100% { opacity: .35; transform: scale(.82); }
  50% { opacity: 1; transform: scale(1); }
}

@keyframes vivqa-typing {
  0%, 60%, 100% { opacity: .3; transform: translateY(0); }
  30% { opacity: 1; transform: translateY(-2px); }
}

/* Hide duplicate queue overlays on output components. The composer keeps its
   own queue indicator so a queued request never looks frozen. */
.sidebar-history-wrap > .wrap.center.full.translucent,
#chat-thread > .wrap.default.minimal.translucent,
.composer-status > .wrap.center.full.translucent {
  display: none !important;
  visibility: hidden !important;
  pointer-events: none !important;
}

footer {
  display: none !important;
}

.gpt-hidden {
  display: none !important;
}

@media (max-width: 760px) {
  .vivqa-sidebar-open .gpt-main {
    width: 100% !important;
    margin-left: 0 !important;
  }
  .gpt-topbar {
    padding: 7px 12px !important;
    gap: 7px !important;
  }
  .topbar-identity-wrap {
    flex: 0 0 auto !important;
    min-width: 0 !important;
  }
  .topbar-title {
    min-width: 0 !important;
    font-size: 13px !important;
  }
  .topbar-spacer-wrap {
    flex: 1 1 auto !important;
    display: block !important;
  }
  .mobile-new-chat {
    display: inline-flex !important;
    flex: 0 0 36px !important;
    width: 36px !important;
    min-width: 36px !important;
    min-height: 36px !important;
    margin: 0 !important;
    padding: 0 !important;
    border: 1px solid #e4e4e7 !important;
    border-radius: 10px !important;
    background: #fff !important;
    box-shadow: none !important;
    font-size: 20px !important;
  }
  .topbar-model-picker {
    flex: 0 1 150px !important;
    min-width: 120px !important;
    padding: 0 !important;
  }
  .topbar-model-status {
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
  #chat-thread .message-buttons {
    opacity: 1;
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

  // Remove only obsolete ViVQA keys. Gradio otherwise attempts to decrypt
  // these values with the new schema on every reload and logs noisy errors.
  [
    'vivqa_conversations_v1',
    'vivqa_active_conversation_v1',
    'vivqa_conversations_v2',
    'vivqa_active_conversation_v2',
    'vivqa_conversations_v3',
    'vivqa_active_conversation_v3',
    'vivqa_sidebar_collapsed',
  ].forEach((key) => window.localStorage.removeItem(key));

  document.documentElement.classList.add('vivqa-sidebar-open');

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

  // The server yields a status before model loading. Keep the elapsed time
  // moving in the browser even while Python is busy loading or generating.
  window.setInterval(() => {
    const now = Date.now();
    const timers = document.querySelectorAll('.live-elapsed[data-start]');
    timers.forEach((node) => {
      const started = Number(node.dataset.start || now);
      const elapsed = Math.max(0, (now - started) / 1000);
      node.textContent = `${elapsed.toFixed(1)} giây`;
    });
    const running = timers.length > 0;
    document.querySelectorAll('#chat-composer .stop-button').forEach((button) => {
      button.classList.toggle('vivqa-running', running);
      button.setAttribute('title', 'Dừng tạo câu trả lời');
      button.setAttribute('aria-label', 'Dừng tạo câu trả lời');
    });
    document.querySelectorAll('#chat-composer [data-testid="submit-button"]').forEach((button) => {
      const composerRoot = document.querySelector('#chat-composer');
      const textarea = composerRoot?.querySelector('textarea');
      const fileInput = composerRoot?.querySelector('input[type="file"]');
      const hasQuestion = Boolean(textarea?.value?.trim());
      const hasNewImage = Boolean(
        fileInput?.files?.length ||
        composerRoot?.querySelector('.thumbnail-wrapper, .thumbnail-item')
      );
      const hasPinnedImage = Boolean(
        document.querySelector('.active-image-strip .pinned-image')
      );
      const canSend = hasQuestion && (hasNewImage || hasPinnedImage);
      button.style.display = running ? 'none' : '';
      button.disabled = running || !canSend;
      button.setAttribute('aria-disabled', String(running || !canSend));
      button.setAttribute('aria-label', 'Gửi câu hỏi');
      button.setAttribute(
        'title',
        canSend ? 'Gửi câu hỏi' : 'Thêm ảnh và nhập câu hỏi để gửi'
      );
    });
    document.querySelectorAll('.topbar-model input, .topbar-model button').forEach((control) => {
      control.disabled = running;
      control.setAttribute('aria-disabled', String(running));
      control.setAttribute('title', 'Chọn mô hình trả lời');
    });
    document.querySelectorAll('#chat-composer [data-testid="upload-button"]').forEach((button) => {
      button.setAttribute('title', 'Tải lên hoặc dán ảnh');
      button.setAttribute('aria-label', 'Tải lên hoặc dán ảnh');
    });
    document.querySelectorAll('.sidebar-rename-menu button, .sidebar-rename-menu').forEach((button) => {
      button.setAttribute('title', 'Đổi tên cuộc trò chuyện');
      button.setAttribute('aria-label', 'Đổi tên cuộc trò chuyện');
    });
    document.querySelectorAll('.clear-active-image button, .clear-active-image').forEach((button) => {
      button.setAttribute('title', 'Bỏ ảnh đang dùng');
      button.setAttribute('aria-label', 'Bỏ ảnh đang dùng');
    });
    document.querySelectorAll('.gpt-sidebar .toggle-button').forEach((button) => {
      const isOpen = document.documentElement.classList.contains('vivqa-sidebar-open');
      const label = isOpen ? 'Thu gọn thanh bên' : 'Mở thanh bên';
      button.setAttribute('title', label);
      button.setAttribute('aria-label', label);
    });
    document.querySelectorAll('#chat-thread button[aria-label="Copy message"]').forEach((button) => {
      button.setAttribute('title', 'Sao chép câu trả lời');
      button.setAttribute('aria-label', 'Sao chép câu trả lời');
    });
    document.querySelectorAll('#chat-thread button[aria-label="Retry"]').forEach((button) => {
      button.setAttribute('title', 'Thử lại');
      button.setAttribute('aria-label', 'Thử lại');
    });
    document.querySelectorAll('#chat-thread button[aria-label="Clear"]').forEach((button) => {
      button.setAttribute('title', 'Xóa nội dung');
      button.setAttribute('aria-label', 'Xóa nội dung');
    });
  }, 250);
})();
"""


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


def new_conversation() -> dict:
    """Create a serializable browser-persisted conversation."""
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    return {
        "id": uuid.uuid4().hex,
        "title": "Cuộc trò chuyện mới",
        "created_at": now,
        "updated_at": now,
        "messages": product_initial_chat(),
        "active_image_path": None,
        "records": [],
        "model_choice": STUDENT_LABEL,
        "response_style": "Cân bằng",
        "max_new_tokens": 96,
    }


def sanitize_saved_messages(messages: list[dict] | None) -> list[dict]:
    """Remove stale temporary image references after a Colab/runtime restart."""
    sanitized: list[dict] = []
    for message in list(messages or []):
        if not isinstance(message, dict):
            continue
        clean_message = dict(message)
        content = clean_message.get("content")
        if isinstance(content, list):
            clean_content = []
            for part in content:
                if isinstance(part, dict) and part.get("path"):
                    if not Path(str(part["path"])).is_file():
                        continue
                clean_content.append(part)
            clean_message["content"] = clean_content
        sanitized.append(clean_message)
    return sanitized


def normalize_conversations(conversations: list[dict] | None) -> list[dict]:
    """Discard malformed browser state and cap its size."""
    valid: list[dict] = []
    for item in list(conversations or []):
        if not isinstance(item, dict) or not item.get("id"):
            continue
        item = dict(item)
        item["messages"] = sanitize_saved_messages(item.get("messages"))
        item["records"] = list(item.get("records") or [])[-100:]
        item["title"] = str(item.get("title") or "Cuộc trò chuyện mới")
        item["model_choice"] = (
            item.get("model_choice") if item.get("model_choice") in ADAPTERS else STUDENT_LABEL
        )
        item["response_style"] = (
            item.get("response_style")
            if item.get("response_style") in {"Mô tả chi tiết", "Cân bằng", "Trả lời ngắn"}
            else "Cân bằng"
        )
        item["max_new_tokens"] = int(item.get("max_new_tokens") or 96)
        active_image_path = item.get("active_image_path")
        if active_image_path and not Path(str(active_image_path)).is_file():
            item["active_image_path"] = None
        valid.append(item)
    return valid[:30]


def conversation_choices(conversations: list[dict]) -> list[tuple[str, str]]:
    """Return label/value pairs for the sidebar conversation picker."""
    choices: list[tuple[str, str]] = []
    for item in conversations:
        if not item.get("messages") and not item.get("records"):
            continue
        title = " ".join(str(item.get("title") or "Cuộc trò chuyện mới").split())
        short = title if len(title) <= 34 else f"{title[:33]}…"
        choices.append((short, str(item["id"])))
    return choices


def conversation_picker_update(conversations: list[dict], active_id: str):
    choices = conversation_choices(conversations)
    visible_ids = {value for _label, value in choices}
    return gr.update(
        choices=choices,
        value=active_id if active_id in visible_ids else None,
    )


def find_conversation(conversations: list[dict], conversation_id: str | None) -> dict:
    for item in conversations:
        if item.get("id") == conversation_id:
            return item
    return conversations[0]


def save_conversation(conversations: list[dict], updated: dict) -> list[dict]:
    """Upsert one conversation and keep recently updated chats first."""
    updated = dict(updated)
    updated["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    return [updated, *[item for item in conversations if item.get("id") != updated["id"]]][:30]


def title_from_question(question: str) -> str:
    title = " ".join(question.split()).strip()
    return title if len(title) <= 48 else f"{title[:47]}…"


def model_readiness_html(model_choice: str, loading: bool = False) -> str:
    if loading:
        text, state = "Đang nạp", "loading"
    elif model_choice == _active_model_choice and model_choice in _inference_engines:
        text, state = "Đang dùng", "ready"
    else:
        text, state = "Chưa nạp", "waiting"
    return (
        f'<div class="model-readiness {state}" aria-label="Trạng thái mô hình"><span></span>'
        f'{html.escape(text)}</div>'
    )


def populate_example(event: gr.SelectData):
    """Place one empty-state suggestion into the composer without submitting it."""
    value = event.value if isinstance(event.value, dict) else {}
    return {"text": str(value.get("text") or ""), "files": []}


def clear_composer_notice(value, current_notice: str):
    """Clear a validation message once its missing input has been supplied."""
    question, image_path = composer_value(value)
    notice = str(current_notice or "")
    resolved_missing_image = bool(image_path) and "Đính kèm ảnh trước" in notice
    resolved_missing_question = bool(question) and "Nhập câu hỏi trước" in notice
    if resolved_missing_image or resolved_missing_question:
        return time_badge("Sẵn sàng", "idle")
    return gr.skip()


def open_rename_panel(conversations: list[dict] | None, active_id: str | None):
    """Open the compact rename form for a started conversation."""
    sessions = normalize_conversations(conversations)
    if not sessions:
        return gr.update(visible=False), ""
    active = find_conversation(sessions, active_id)
    if not active.get("messages") and not active.get("records"):
        return gr.update(visible=False), ""
    return gr.update(visible=True), str(active.get("title") or "")


def close_rename_panel():
    return gr.update(visible=False), ""


def rename_active_conversation(
    title: str,
    conversations: list[dict] | None,
    active_id: str | None,
):
    """Rename the selected conversation and refresh its sidebar entry."""
    sessions = normalize_conversations(conversations)
    if not sessions:
        sessions = [new_conversation()]
    active = dict(find_conversation(sessions, active_id))
    clean_title = " ".join(str(title or "").split()).strip()
    if clean_title:
        active["title"] = clean_title if len(clean_title) <= 48 else f"{clean_title[:47]}…"
        sessions = save_conversation(sessions, active)
    active_id = str(active["id"])
    return (
        sessions,
        conversation_picker_update(sessions, active_id),
        gr.update(visible=False),
        "",
    )


def _message_image_path(content) -> str | None:
    if not isinstance(content, (list, tuple)):
        return None
    for item in content:
        if isinstance(item, dict) and item.get("path"):
            return str(item["path"])
    return None


def cancel_chat_request(
    conversation: list[dict] | None,
    conversations: list[dict] | None,
    active_id: str | None,
    model_choice: str,
):
    """Replace an in-progress row with a stable cancelled state."""
    messages = list(conversation or [])
    if messages and messages[-1].get("role") == "assistant":
        content = str(messages[-1].get("content") or "")
        if "assistant-typing" in content or "Đang xử lý" in content:
            messages[-1] = {"role": "assistant", "content": "_Đã dừng yêu cầu._"}

    sessions = normalize_conversations(conversations)
    if not sessions:
        sessions = [new_conversation()]
    active = dict(find_conversation(sessions, active_id))
    active["messages"] = messages
    sessions = save_conversation(sessions, active)
    active_id = str(active["id"])
    return (
        messages,
        time_badge("", "idle"),
        sessions,
        conversation_picker_update(sessions, active_id),
        model_readiness_html(model_choice),
    )


def active_image_html(image_path: str | None) -> str:
    """Render a small clean thumbnail without Gradio's image toolbar."""
    if not image_path or not Path(str(image_path)).is_file():
        return ""
    try:
        with Image.open(image_path) as source:
            preview = source.convert("RGB")
            preview.thumbnail((96, 96))
            buffer = io.BytesIO()
            preview.save(buffer, format="JPEG", quality=82, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        return ""
    return (
        '<div class="pinned-image">'
        f'<img src="data:image/jpeg;base64,{encoded}" alt="Ảnh đang dùng">'
        '<div><strong>Ảnh đang dùng</strong><span>Câu hỏi tiếp theo vẫn dùng ảnh này</span></div>'
        '</div>'
    )


def image_component_update(image_path: str | None):
    value = active_image_html(image_path)
    return gr.update(value=value, visible=bool(value))


def image_clear_button_update(image_path: str | None):
    return gr.update(visible=bool(image_path))


def conversation_view(conversations: list[dict] | None, active_id: str | None):
    """Load a saved conversation into every visible control."""
    sessions = normalize_conversations(conversations)
    if not sessions:
        sessions = [new_conversation()]
    conversation = find_conversation(sessions, active_id)
    active_id = str(conversation["id"])
    image_path = conversation.get("active_image_path")
    records = list(conversation.get("records") or [])
    model_choice = str(conversation.get("model_choice") or STUDENT_LABEL)
    response_style = str(conversation.get("response_style") or "Cân bằng")
    max_new_tokens = int(conversation.get("max_new_tokens") or 96)
    return (
        {"text": "", "files": []},
        list(conversation.get("messages") or []),
        time_badge("Sẵn sàng", "idle"),
        image_path,
        image_component_update(image_path),
        image_clear_button_update(image_path),
        records,
        history_rows(records),
        system_status_html(),
        sessions,
        active_id,
        conversation_picker_update(sessions, active_id),
        model_choice,
        response_style,
        max_new_tokens,
        model_readiness_html(model_choice),
    )


def create_product_chat(conversations: list[dict] | None):
    """Start a new chat without deleting older conversations."""
    sessions = normalize_conversations(conversations)
    conversation = new_conversation()
    sessions = [conversation, *sessions][:30]
    return conversation_view(sessions, conversation["id"])


def delete_product_chat(conversations: list[dict] | None, active_id: str | None):
    """Delete only the selected conversation and select the next one."""
    sessions = [
        item for item in normalize_conversations(conversations) if item.get("id") != active_id
    ]
    if not sessions:
        sessions = [new_conversation()]
    return conversation_view(sessions, sessions[0]["id"])


def clear_active_image(conversations: list[dict] | None, active_id: str | None):
    """Detach the pinned image while preserving the current chat."""
    sessions = normalize_conversations(conversations)
    if not sessions:
        sessions = [new_conversation()]
    conversation = dict(find_conversation(sessions, active_id))
    conversation["active_image_path"] = None
    sessions = save_conversation(sessions, conversation)
    return None, image_component_update(None), image_clear_button_update(None), sessions


def answer_chat_gpt(
    composer,
    model_choice: str,
    response_style: str,
    max_new_tokens: int,
    active_image_path: str | None,
    history: list[dict] | None,
    conversation: list[dict] | None,
    conversations: list[dict] | None,
    active_conversation_id: str | None,
):
    """Yield an immediate pending state, then run one visual-answer request."""
    messages = list(conversation or product_initial_chat())
    clean_question, attached_image_path = composer_value(composer)
    selected_image_path = attached_image_path or active_image_path
    records = list(history or [])
    sessions = normalize_conversations(conversations)
    if not sessions:
        sessions = [new_conversation()]
    active = dict(find_conversation(sessions, active_conversation_id))
    active_id = str(active["id"])

    if attached_image_path and not clean_question:
        clean_question = "Mô tả hình ảnh này."

    if not clean_question:
        yield (
            messages,
            composer if isinstance(composer, dict) else {"text": "", "files": []},
            time_badge("Nhập câu hỏi trước", "error"),
            active_image_path,
            image_component_update(active_image_path),
            image_clear_button_update(active_image_path),
            records,
            history_rows(records),
            system_status_html(),
            sessions,
            active_id,
            conversation_picker_update(sessions, active_id),
            model_readiness_html(model_choice),
        )
        return
    if not selected_image_path:
        yield (
            messages,
            {"text": clean_question, "files": []},
            time_badge("Đính kèm ảnh trước", "error"),
            active_image_path,
            image_component_update(active_image_path),
            image_clear_button_update(active_image_path),
            records,
            history_rows(records),
            system_status_html(),
            sessions,
            active_id,
            conversation_picker_update(sessions, active_id),
            model_readiness_html(model_choice),
        )
        return

    request_started = time.time()
    context = conversation_context(messages)
    messages.append(
        {
            "role": "user",
            "content": chat_user_content(attached_image_path, clean_question),
        }
    )
    needs_model_load = not (
        model_choice == _active_model_choice and model_choice in _inference_engines
    )
    loading_text = (
        f"Đang nạp {model_choice}"
        if needs_model_load
        else "Đang phân tích ảnh"
    )
    messages.append(
        {
            "role": "assistant",
            "content": pending_message_html(loading_text),
        }
    )
    active.update(
        {
            "messages": messages,
            "active_image_path": selected_image_path,
            "records": records,
            "model_choice": model_choice,
            "response_style": response_style,
            "max_new_tokens": int(max_new_tokens),
        }
    )
    if active.get("title") == "Cuộc trò chuyện mới":
        active["title"] = title_from_question(clean_question)
    sessions = save_conversation(sessions, active)

    # This first yield makes the sent message and live timer visible before
    # the potentially long model-loading call starts.
    yield (
        messages,
        {"text": "", "files": []},
        running_badge(loading_text, request_started),
        selected_image_path,
        image_component_update(selected_image_path),
        image_clear_button_update(selected_image_path),
        records,
        history_rows(records),
        system_status_html(),
        sessions,
        active_id,
        conversation_picker_update(sessions, active_id),
        model_readiness_html(model_choice, loading=needs_model_load),
    )

    try:
        image = open_composer_image(selected_image_path)
        engine, load_seconds = get_inference(model_choice)
        messages = list(messages)
        messages[-1] = {
            "role": "assistant",
            "content": pending_message_html(
                "Đang phân tích ảnh và tạo câu trả lời",
            ),
        }
        yield (
            messages,
            {"text": "", "files": []},
            running_badge(
                "Đang phân tích ảnh và tạo câu trả lời",
                request_started,
            ),
            selected_image_path,
            image_component_update(selected_image_path),
            image_clear_button_update(selected_image_path),
            records,
            history_rows(records),
            system_status_html(),
            sessions,
            active_id,
            conversation_picker_update(sessions, active_id),
            model_readiness_html(model_choice),
        )
        previous_record_count = len(records)
        result, timing, records, rows, status = answer(
            image,
            clean_question,
            model_choice,
            response_style,
            max_new_tokens,
            records,
            conversation_context=context,
            preloaded_engine=engine,
            preloaded_load_seconds=load_seconds,
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
            total_seconds = time.time() - request_started
            timing_parts = [f"suy luận {inference_seconds:.1f} giây"]
            if load_seconds:
                timing_parts.insert(0, f"nạp {load_seconds:.1f} giây")
            timing_parts.append(f"tổng {total_seconds:.1f} giây")
            if latest.get("retried_prompt_echo"):
                timing_parts.append("đã thử lại câu lệnh")
            assistant_content = f"{result}\n\n*{model_choice} · {' · '.join(timing_parts)}*"
        else:
            assistant_content = result
        messages = list(messages)
        messages[-1] = {"role": "assistant", "content": assistant_content}
    except Exception as exc:
        print(f"[ViVQA] Chat request failed: {exc}", file=sys.stderr, flush=True)
        timing = time_badge("", "idle")
        status = system_status_html()
        rows = history_rows(records)
        messages = list(messages)
        messages[-1] = {
            "role": "assistant",
            "content": friendly_inference_error(exc),
        }

    active.update(
        {
            "messages": messages,
            "active_image_path": selected_image_path,
            "records": records,
            "model_choice": model_choice,
            "response_style": response_style,
            "max_new_tokens": int(max_new_tokens),
        }
    )
    sessions = save_conversation(sessions, active)
    yield (
        messages,
        {"text": "", "files": []},
        timing,
        selected_image_path,
        image_component_update(selected_image_path),
        image_clear_button_update(selected_image_path),
        records,
        rows,
        status,
        sessions,
        active_id,
        conversation_picker_update(sessions, active_id),
        model_readiness_html(model_choice),
    )


def retry_chat_gpt(
    retry_data: gr.RetryData,
    model_choice: str,
    response_style: str,
    max_new_tokens: int,
    active_image_path: str | None,
    history: list[dict] | None,
    conversation: list[dict] | None,
    conversations: list[dict] | None,
    active_conversation_id: str | None,
):
    """Regenerate one answer while keeping the selected model and image."""
    sessions = normalize_conversations(conversations)
    if not sessions:
        sessions = [new_conversation()]
    active = dict(find_conversation(sessions, active_conversation_id))

    # Gradio may optimistically remove the retried turn from the visible
    # Chatbot before this callback runs. The browser-persisted conversation is
    # therefore the reliable source for the original question and attachment.
    saved_messages = list(active.get("messages") or [])
    messages = saved_messages or list(conversation or [])
    raw_index = retry_data.index[0] if isinstance(retry_data.index, tuple) else retry_data.index
    user_index = min(max(int(raw_index), 0), max(len(messages) - 1, 0))
    while user_index >= 0 and messages[user_index].get("role") != "user":
        user_index -= 1

    if user_index < 0:
        return

    user_content = messages[user_index].get("content")
    question = _message_text(user_content)
    retry_image_path = _message_image_path(user_content) or active_image_path
    previous_messages = messages[:user_index]
    completed_turns = sum(
        1 for message in previous_messages if message.get("role") == "user"
    )
    previous_records = list(history or [])[:completed_turns]

    active.update(
        {
            "messages": previous_messages,
            "records": previous_records,
            "active_image_path": retry_image_path,
        }
    )
    sessions = save_conversation(sessions, active)

    yield from answer_chat_gpt(
        {"text": question, "files": []},
        model_choice,
        response_style,
        max_new_tokens,
        retry_image_path,
        previous_records,
        previous_messages,
        sessions,
        str(active["id"]),
    )


def create_demo() -> gr.Blocks:
    """Build the conversational Vietnamese visual question answering demo."""
    with gr.Blocks(
        title=APP_TITLE,
        analytics_enabled=False,
        fill_height=True,
        fill_width=True,
    ) as demo:
        initial_conversation = new_conversation()
        history_state = gr.State([])
        conversations_state = gr.BrowserState(
            [initial_conversation],
            storage_key="vivqa_conversations_v4",
            secret="vivqa-conversations-v4-stable-key",
        )
        active_conversation_id_state = gr.BrowserState(
            initial_conversation["id"],
            storage_key="vivqa_active_conversation_v4",
            secret="vivqa-active-chat-v4-stable-key",
        )
        active_image_path_state = gr.State(None)

        # Keep Sidebar and the main column as direct Blocks children. Wrapping
        # them in a Row prevents Gradio's native collapse control from working.
        if True:
            with gr.Sidebar(
                open=True,
                width=260,
                elem_classes="gpt-sidebar",
            ) as sidebar_panel:
                gr.HTML(
                    """
                    <div class="sidebar-brand">
                      <span class="sidebar-mark" aria-label="ViVQA">
                        <svg viewBox="0 0 24 24" aria-hidden="true">
                          <path d="M5 6.5A2.5 2.5 0 0 1 7.5 4h9A2.5 2.5 0 0 1 19 6.5v6A2.5 2.5 0 0 1 16.5 15H12l-3.8 3v-3h-.7A2.5 2.5 0 0 1 5 12.5z"/>
                          <circle cx="9" cy="9.5" r="1"/>
                          <circle cx="12" cy="9.5" r="1"/>
                          <circle cx="15" cy="9.5" r="1"/>
                        </svg>
                      </span>
                    </div>
                    """
                )
                new_chat = gr.Button(
                    "＋  Cuộc trò chuyện mới",
                    elem_classes="sidebar-new-chat",
                )
                with gr.Accordion(
                    "Điều chỉnh câu trả lời",
                    open=False,
                    elem_classes="sidebar-options",
                ):
                    response_style = gr.Dropdown(
                        choices=["Mô tả chi tiết", "Cân bằng", "Trả lời ngắn"],
                        value="Cân bằng",
                        label="Cách trả lời",
                    )
                    max_new_tokens = gr.Dropdown(
                        choices=[48, 64, 96, 128, 160],
                        value=96,
                        label="Độ dài tối đa",
                    )
                with gr.Row(elem_classes="sidebar-history-heading"):
                    gr.HTML('<div class="history-heading">Cuộc trò chuyện</div>')
                    rename_menu = gr.Button(
                        "⋯",
                        elem_classes="sidebar-rename-menu",
                    )
                conversation_selector = gr.Radio(
                    choices=[],
                    value=None,
                    show_label=False,
                    container=False,
                    elem_classes="conversation-selector",
                )
                with gr.Group(
                    visible=False,
                    elem_classes="sidebar-rename-panel",
                ) as rename_panel:
                    rename_input = gr.Textbox(
                        placeholder="Tên cuộc trò chuyện",
                        show_label=False,
                        container=False,
                        max_lines=1,
                    )
                    with gr.Row(elem_classes="sidebar-rename-actions"):
                        rename_cancel = gr.Button("Hủy")
                        rename_save = gr.Button("Lưu", variant="primary")

            with gr.Column(elem_classes="gpt-main"):
                with gr.Row(elem_classes="gpt-topbar"):
                    gr.HTML(
                        """
                        <div class="topbar-identity">
                          <span class="topbar-title">ViVQA</span>
                        </div>
                        """,
                        elem_classes="topbar-identity-wrap",
                    )
                    with gr.Row(elem_classes="topbar-model-picker"):
                        model_choice = gr.Dropdown(
                            choices=list(ADAPTERS),
                            value=STUDENT_LABEL,
                            label="Mô hình",
                            show_label=False,
                            container=False,
                            min_width=150,
                            elem_classes="topbar-model",
                        )
                    model_status = gr.HTML(
                        model_readiness_html(STUDENT_LABEL),
                        elem_classes="topbar-model-status",
                    )
                    gr.HTML("<div></div>", elem_classes="topbar-spacer-wrap")
                    mobile_new_chat = gr.Button(
                        "＋",
                        elem_classes="mobile-new-chat",
                    )

                with gr.Column(elem_classes="gpt-stage"):
                    chatbot = gr.Chatbot(
                        value=product_initial_chat(),
                        height=460,
                        buttons=["copy"],
                        render_markdown=True,
                        group_consecutive_messages=True,
                        allow_file_downloads=False,
                        show_label=False,
                        placeholder="Thêm ảnh và chọn một gợi ý để bắt đầu",
                        examples=EMPTY_CHAT_EXAMPLES,
                        elem_id="chat-thread",
                    )

                    with gr.Column(elem_classes="composer-area"):
                        with gr.Row(elem_classes="active-image-strip"):
                            active_image_preview = gr.HTML(
                                value="",
                                visible=False,
                                elem_classes="active-image-preview",
                            )
                            clear_image = gr.Button(
                                "×",
                                visible=False,
                                elem_classes="clear-active-image",
                            )
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
                            stop_btn="■",
                            elem_id="chat-composer",
                            elem_classes="chat-composer-gpt",
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
            conversations_state,
            active_conversation_id_state,
        ]
        outputs = [
            chatbot,
            composer,
            timing,
            active_image_path_state,
            active_image_preview,
            clear_image,
            history_state,
            history_table,
            system_status,
            conversations_state,
            active_conversation_id_state,
            conversation_selector,
            model_status,
        ]
        submit_event = composer.submit(
            fn=answer_chat_gpt,
            inputs=inputs,
            outputs=outputs,
            show_progress="minimal",
        )
        retry_event = chatbot.retry(
            fn=retry_chat_gpt,
            inputs=[
                model_choice,
                response_style,
                max_new_tokens,
                active_image_path_state,
                history_state,
                chatbot,
                conversations_state,
                active_conversation_id_state,
            ],
            outputs=outputs,
            show_progress="minimal",
        )
        composer.stop(
            fn=cancel_chat_request,
            inputs=[
                chatbot,
                conversations_state,
                active_conversation_id_state,
                model_choice,
            ],
            outputs=[
                chatbot,
                timing,
                conversations_state,
                conversation_selector,
                model_status,
            ],
            queue=False,
            cancels=[submit_event, retry_event],
        )
        chatbot.example_select(
            fn=populate_example,
            outputs=composer,
            queue=False,
            show_progress="hidden",
        )
        composer.input(
            fn=clear_composer_notice,
            inputs=[composer, timing],
            outputs=timing,
            queue=False,
            show_progress="hidden",
        )
        composer.change(
            fn=clear_composer_notice,
            inputs=[composer, timing],
            outputs=timing,
            queue=False,
            show_progress="hidden",
        )
        new_chat_event = new_chat.click(
            fn=create_product_chat,
            inputs=conversations_state,
            outputs=[
                composer,
                chatbot,
                timing,
                active_image_path_state,
                active_image_preview,
                clear_image,
                history_state,
                history_table,
                system_status,
                conversations_state,
                active_conversation_id_state,
                conversation_selector,
                model_choice,
                response_style,
                max_new_tokens,
                model_status,
            ],
            queue=False,
            show_progress="hidden",
            cancels=[submit_event, retry_event],
        )
        new_chat_event.then(
            fn=close_rename_panel,
            outputs=[rename_panel, rename_input],
            queue=False,
            show_progress="hidden",
        )
        mobile_new_chat_event = mobile_new_chat.click(
            fn=create_product_chat,
            inputs=conversations_state,
            outputs=[
                composer,
                chatbot,
                timing,
                active_image_path_state,
                active_image_preview,
                clear_image,
                history_state,
                history_table,
                system_status,
                conversations_state,
                active_conversation_id_state,
                conversation_selector,
                model_choice,
                response_style,
                max_new_tokens,
                model_status,
            ],
            queue=False,
            show_progress="hidden",
            cancels=[submit_event, retry_event],
        )
        mobile_new_chat_event.then(
            fn=close_rename_panel,
            outputs=[rename_panel, rename_input],
            queue=False,
            show_progress="hidden",
        )
        conversation_event = conversation_selector.input(
            fn=conversation_view,
            inputs=[conversations_state, conversation_selector],
            outputs=[
                composer,
                chatbot,
                timing,
                active_image_path_state,
                active_image_preview,
                clear_image,
                history_state,
                history_table,
                system_status,
                conversations_state,
                active_conversation_id_state,
                conversation_selector,
                model_choice,
                response_style,
                max_new_tokens,
                model_status,
            ],
            queue=False,
            show_progress="hidden",
            cancels=[submit_event, retry_event],
        )
        conversation_event.then(
            fn=close_rename_panel,
            outputs=[rename_panel, rename_input],
            queue=False,
            show_progress="hidden",
        )
        clear_image.click(
            fn=clear_active_image,
            inputs=[conversations_state, active_conversation_id_state],
            outputs=[
                active_image_path_state,
                active_image_preview,
                clear_image,
                conversations_state,
            ],
            queue=False,
            show_progress="hidden",
            cancels=[submit_event, retry_event],
        )
        rename_menu.click(
            fn=open_rename_panel,
            inputs=[conversations_state, active_conversation_id_state],
            outputs=[rename_panel, rename_input],
            queue=False,
            show_progress="hidden",
            js="""
            (conversations, activeId) => {
              const openedOptions = document.querySelector(
                '.sidebar-options .label-wrap.open'
              );
              if (openedOptions) openedOptions.click();
              return [conversations, activeId];
            }
            """,
        )
        rename_cancel.click(
            fn=close_rename_panel,
            outputs=[rename_panel, rename_input],
            queue=False,
            show_progress="hidden",
        )
        rename_save.click(
            fn=rename_active_conversation,
            inputs=[
                rename_input,
                conversations_state,
                active_conversation_id_state,
            ],
            outputs=[
                conversations_state,
                conversation_selector,
                rename_panel,
                rename_input,
            ],
            queue=False,
            show_progress="hidden",
        )
        rename_input.submit(
            fn=rename_active_conversation,
            inputs=[
                rename_input,
                conversations_state,
                active_conversation_id_state,
            ],
            outputs=[
                conversations_state,
                conversation_selector,
                rename_panel,
                rename_input,
            ],
            queue=False,
            show_progress="hidden",
        )
        model_choice.change(
            fn=model_readiness_html,
            inputs=model_choice,
            outputs=model_status,
            queue=False,
            show_progress="hidden",
        )
        sidebar_panel.expand(
            fn=None,
            js="() => document.documentElement.classList.add('vivqa-sidebar-open')",
            queue=False,
        )
        sidebar_panel.collapse(
            fn=None,
            js="() => document.documentElement.classList.remove('vivqa-sidebar-open')",
            queue=False,
        )
        demo.load(
            fn=conversation_view,
            inputs=[conversations_state, active_conversation_id_state],
            outputs=[
                composer,
                chatbot,
                timing,
                active_image_path_state,
                active_image_preview,
                clear_image,
                history_state,
                history_table,
                system_status,
                conversations_state,
                active_conversation_id_state,
                conversation_selector,
                model_choice,
                response_style,
                max_new_tokens,
                model_status,
            ],
            queue=False,
            show_progress="hidden",
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
