"""Branded Gradio demo for Vietnamese VQA Distillation."""

from __future__ import annotations

import argparse
import html
import sys
import threading
import time
from pathlib import Path

import gradio as gr

from src.models.inference import VQAInference
from src.models.model_loader import VQAModelLoader
from src.models.model_wrapper import StudentModel, TeacherModel


ROOT = Path(__file__).resolve().parent
APP_TITLE = "Vietnamese VQA Distillation"
OWNER = "OuyangXuelili"

STUDENT_LABEL = "Student 2B · nhanh và gọn"
TEACHER_LABEL = "Teacher 7B · chất lượng cao"

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


CUSTOM_CSS = """
:root {
  --brand-ink: #10213d;
  --brand-blue: #2563eb;
  --brand-cyan: #06b6d4;
  --brand-soft: #eff6ff;
}
.gradio-container {
  max-width: 1440px !important;
  margin: 0 auto !important;
  background:
    radial-gradient(circle at 8% 2%, rgba(37, 99, 235, .10), transparent 28rem),
    radial-gradient(circle at 92% 12%, rgba(6, 182, 212, .10), transparent 24rem);
}
.app-shell { padding: 8px 2px 24px; }
.hero {
  position: relative;
  overflow: hidden;
  padding: 30px 34px;
  border: 1px solid rgba(148, 163, 184, .22);
  border-radius: 24px;
  color: white;
  background: linear-gradient(125deg, #0f172a 0%, #172554 48%, #164e63 100%);
  box-shadow: 0 22px 60px rgba(15, 23, 42, .18);
}
.hero::after {
  content: "";
  position: absolute;
  width: 280px;
  height: 280px;
  right: -90px;
  top: -120px;
  border-radius: 999px;
  background: rgba(34, 211, 238, .17);
}
.hero-kicker {
  color: #67e8f9;
  font-size: .78rem;
  font-weight: 800;
  letter-spacing: .16em;
  text-transform: uppercase;
}
.hero h1 { margin: 8px 0 8px; font-size: clamp(2rem, 5vw, 3.4rem); line-height: 1.02; }
.hero p { margin: 0; max-width: 760px; color: #cbd5e1; font-size: 1.02rem; }
.hero-badges { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
.hero-badge {
  padding: 7px 11px;
  border: 1px solid rgba(255,255,255,.16);
  border-radius: 999px;
  background: rgba(255,255,255,.08);
  color: #e2e8f0;
  font-size: .82rem;
  font-weight: 650;
}
.workspace { margin-top: 18px; gap: 18px !important; }
.panel {
  border: 1px solid rgba(148, 163, 184, .24) !important;
  border-radius: 20px !important;
  background: rgba(255, 255, 255, .82) !important;
  box-shadow: 0 12px 34px rgba(15, 23, 42, .07) !important;
  backdrop-filter: blur(12px);
  padding: 8px !important;
}
.primary-action { min-height: 48px !important; font-weight: 800 !important; }
.result-box textarea { font-size: 1.02rem !important; line-height: 1.7 !important; }
.answer-card {
  margin-top: 4px;
  padding: 12px 14px 14px !important;
  border: 1px solid rgba(148, 163, 184, .24) !important;
  border-radius: 16px !important;
  background: rgba(248, 250, 252, .76) !important;
}
.answer-header { align-items: center !important; margin-bottom: 4px !important; }
.answer-title { flex: 1 1 auto !important; min-width: 160px !important; }
.answer-title h3 { margin: 0 !important; color: var(--brand-ink); font-size: .98rem !important; }
.time-chip-output {
  flex: 0 0 auto !important;
  min-width: 0 !important;
  width: auto !important;
  margin-left: auto !important;
}
.time-chip-output > div { padding: 0 !important; }
.time-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  width: max-content;
  padding: 6px 10px;
  border: 1px solid #bfdbfe;
  border-radius: 999px;
  color: #1d4ed8;
  background: #eff6ff;
  font-size: .78rem;
  font-weight: 750;
  white-space: nowrap;
}
.time-chip::before { content: "◷"; font-size: .92rem; }
.time-chip.idle { color: #64748b; border-color: #e2e8f0; background: #f8fafc; }
.time-chip.error { color: #b91c1c; border-color: #fecaca; background: #fef2f2; }
.model-note {
  padding: 12px 14px;
  border-left: 3px solid var(--brand-cyan);
  border-radius: 10px;
  background: var(--brand-soft);
  color: #334155;
  font-size: .88rem;
}
.footer-note { text-align: center; color: #64748b; font-size: .82rem; margin-top: 14px; }
.footer-note a { color: var(--brand-blue); text-decoration: none; font-weight: 700; }
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


def get_inference(model_choice: str) -> tuple[VQAInference, float]:
    """Lazily load one model and return its one-time loading duration."""
    if model_choice in _inference_engines:
        return _inference_engines[model_choice], 0.0

    with _load_lock:
        if model_choice in _inference_engines:
            return _inference_engines[model_choice], 0.0

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
        return engine, time.perf_counter() - started


def time_badge(message: str, state: str = "success") -> str:
    """Render a compact persistent timing badge for the answer card."""
    safe_message = html.escape(message)
    safe_state = state if state in {"success", "idle", "error"} else "idle"
    return f'<span class="time-chip {safe_state}">{safe_message}</span>'


def answer(
    image,
    question: str,
    model_choice: str,
    response_style: str,
    max_new_tokens: int,
) -> tuple[str, str]:
    """Answer one Vietnamese visual question from the web interface."""
    if image is None:
        return "Vui lòng tải lên một hình ảnh trước.", time_badge("Chưa thực hiện", "idle")
    if not question or not question.strip():
        return "Vui lòng nhập câu hỏi về hình ảnh.", time_badge("Chưa thực hiện", "idle")

    try:
        engine, load_seconds = get_inference(model_choice)
        prompt = question.strip()
        min_new_tokens = 0

        if response_style == "Mô tả chi tiết":
            prompt = (
                f"{prompt}\n\n"
                "Hãy trả lời bằng tiếng Việt thành một đoạn mô tả rõ ràng từ 3 đến 5 câu. "
                "Nêu các đối tượng chính, số lượng, màu sắc, vị trí tương đối, hành động và "
                "bối cảnh có thể quan sát trực tiếp. Không suy đoán thông tin ngoài hình ảnh."
            )
            min_new_tokens = min(40, max(12, int(max_new_tokens) // 3))
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
        return prediction.strip(), time_badge(" · ".join(timing_parts))
    except Exception as exc:
        safe_error = html.escape(str(exc))
        return f"Không thể chạy model: {safe_error}", time_badge("Đã xảy ra lỗi", "error")


def clear_workspace():
    """Reset only user inputs and generated outputs."""
    return None, "", "", time_badge("Sẵn sàng", "idle")


def create_demo() -> gr.Blocks:
    """Build the branded Gradio interface."""
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan",
        neutral_hue="slate",
        radius_size="lg",
    )

    with gr.Blocks(
        title=APP_TITLE,
        theme=theme,
        css=CUSTOM_CSS,
        analytics_enabled=False,
    ) as demo:
        with gr.Column(elem_classes="app-shell"):
            gr.HTML(
                """
                <section class="hero">
                  <div class="hero-kicker">Vietnamese vision-language intelligence</div>
                  <h1>Vietnamese VQA Distillation</h1>
                  <p>Đặt câu hỏi bằng tiếng Việt cho bất kỳ hình ảnh nào và so sánh sức mạnh
                  của mô hình Teacher 7B với Student 2B đã được chưng cất tri thức.</p>
                  <div class="hero-badges">
                    <span class="hero-badge">Qwen2-VL</span>
                    <span class="hero-badge">LoRA Fine-tuning</span>
                    <span class="hero-badge">Knowledge Distillation</span>
                    <span class="hero-badge">Vietnamese VQA</span>
                  </div>
                </section>
                """
            )

            with gr.Row(equal_height=True, elem_classes="workspace"):
                with gr.Column(scale=6, elem_classes="panel"):
                    image = gr.Image(
                        type="pil",
                        label="Hình ảnh đầu vào",
                        height=510,
                    )
                    gr.HTML(
                        '<div class="model-note">Mẹo: chọn ảnh rõ nét và đặt câu hỏi cụ thể '
                        'để mô hình trả lời chính xác hơn.</div>'
                    )

                with gr.Column(scale=5, elem_classes="panel"):
                    question = gr.Textbox(
                        label="Câu hỏi tiếng Việt",
                        placeholder="Ví dụ: Có những vật thể nào trong ảnh và chúng đang ở đâu?",
                        lines=3,
                    )
                    model_choice = gr.Dropdown(
                        choices=list(ADAPTERS),
                        value=STUDENT_LABEL,
                        label="Mô hình suy luận",
                    )
                    response_style = gr.Radio(
                        choices=["Mô tả chi tiết", "Cân bằng", "Trả lời ngắn"],
                        value="Mô tả chi tiết",
                        label="Độ chi tiết",
                    )

                    with gr.Accordion("Thiết lập nâng cao", open=False):
                        max_new_tokens = gr.Slider(
                            minimum=16,
                            maximum=160,
                            value=96,
                            step=8,
                            label="Số token sinh tối đa",
                        )

                    with gr.Row():
                        submit = gr.Button(
                            "Phân tích hình ảnh",
                            variant="primary",
                            elem_classes="primary-action",
                            scale=4,
                        )
                        clear = gr.Button("Làm mới", scale=1)

                    with gr.Column(elem_classes="answer-card"):
                        with gr.Row(elem_classes="answer-header"):
                            gr.Markdown("### Câu trả lời", elem_classes="answer-title")
                            timing = gr.HTML(
                                time_badge("Sẵn sàng", "idle"),
                                elem_classes="time-chip-output",
                            )
                        result = gr.Textbox(
                            lines=7,
                            show_label=False,
                            show_copy_button=True,
                            elem_classes="result-box",
                        )

                    gr.Examples(
                        examples=[
                            ["Trong ảnh có những đối tượng nào?"],
                            ["Hãy mô tả chi tiết khung cảnh trong ảnh."],
                            ["Các đối tượng đang làm gì và ở vị trí nào?"],
                            ["Màu sắc nổi bật nhất trong ảnh là gì?"],
                        ],
                        inputs=[question],
                        label="Câu hỏi gợi ý",
                    )

            gr.HTML(
                f'<div class="footer-note">Built and maintained by <strong>{OWNER}</strong> · '
                '<a href="https://github.com/OuyangXuelili/Vietnamese-VQA-Distillation" '
                'target="_blank">GitHub</a> · '
                '<a href="https://huggingface.co/Ouyangxuelili" target="_blank">Hugging Face</a></div>'
            )

        inputs = [image, question, model_choice, response_style, max_new_tokens]
        submit.click(fn=answer, inputs=inputs, outputs=[result, timing])
        question.submit(fn=answer, inputs=inputs, outputs=[result, timing])
        clear.click(
            fn=clear_workspace,
            outputs=[image, question, result, timing],
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
        create_demo().queue(default_concurrency_limit=1, max_size=8).launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            show_error=True,
        )
