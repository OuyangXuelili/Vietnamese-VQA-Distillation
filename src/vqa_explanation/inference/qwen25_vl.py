from __future__ import annotations

from pathlib import Path

from PIL import Image

from vqa_explanation.inference.prompting import SYSTEM_PROMPT, build_user_prompt


class Qwen25VLRunner:
    """Thin inference adapter; heavyweight imports stay optional for local tooling."""

    def __init__(
        self,
        model_id: str,
        revision: str,
        *,
        load_in_4bit: bool = False,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1024 * 28 * 28,
    ) -> None:
        try:
            import torch
            from transformers import (
                AutoProcessor,
                BitsAndBytesConfig,
                Qwen2_5_VLForConditionalGeneration,
            )
        except ImportError as exc:  # pragma: no cover - exercised on GPU runtime
            raise RuntimeError(
                'Install the training dependencies with: pip install -e ".[train]"'
            ) from exc

        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            revision=revision,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            revision=revision,
            device_map="auto",
            dtype=torch.bfloat16,
            quantization_config=quantization_config,
            attn_implementation="sdpa",
        )
        self.model.eval()
        self.torch = torch

    def generate(self, image_path: Path, question: str, *, max_new_tokens: int = 128) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": build_user_prompt(question)},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
        prompt_length = inputs["input_ids"].shape[1]
        completion_ids = generated[:, prompt_length:]
        return self.processor.batch_decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
