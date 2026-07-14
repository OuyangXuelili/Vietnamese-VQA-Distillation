"""Model loading utilities for the trained Vi-VQA LoRA outputs."""

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from peft import PeftModel
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ModelConfig:
    model_id: str
    model_name: str
    params: str
    purpose: str
    base_model: Optional[str] = None
    local_adapter: Optional[str] = None


MODELS = {
    "7B-base": ModelConfig(
        "7B-base",
        "unsloth/Qwen2-VL-7B-Instruct",
        "7B",
        "base_model_for_training",
        base_model="Qwen/Qwen2-VL-7B-Instruct",
    ),
    "7B-trained": ModelConfig(
        "7B-trained",
        "Ouyangxuelili/Qwen-7B-Vi-VQA",
        "7B",
        "teacher_model",
        base_model="Qwen/Qwen2-VL-7B-Instruct",
        local_adapter="outputs-7B/final_lora",
    ),
    "2B-distilled": ModelConfig(
        "2B-distilled",
        "Ouyangxuelili/Qwen-2B-ViVQA",
        "2B",
        "student_model_distilled",
        base_model="Qwen/Qwen2-VL-2B-Instruct",
        local_adapter="outputs-2B-distilled/final_lora",
    ),
}


class VQAModelLoader:
    """Load Qwen2-VL and attach the output adapter produced by training."""

    def __init__(
        self,
        device: str = "auto",
        use_auth_token: bool = False,
        load_in_4bit: Optional[bool] = None,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.device = device
        self.use_auth_token = use_auth_token
        self.load_in_4bit = (device == "cuda") if load_in_4bit is None else load_in_4bit
        self.models = {}
        self.tokenizers = {}  # Kept for compatibility; values are Qwen processors.

    def _hub_kwargs(self) -> dict:
        return {"token": True} if self.use_auth_token else {}

    @staticmethod
    def _adapter_source(config: ModelConfig) -> str:
        if config.local_adapter:
            local = PROJECT_ROOT / config.local_adapter
            if (local / "adapter_config.json").is_file():
                return str(local)
        return config.model_name

    def _model_kwargs(self, load_in_4bit: bool, max_memory=None) -> dict:
        kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            **self._hub_kwargs(),
        }
        has_bnb = importlib.util.find_spec("bitsandbytes") is not None
        if load_in_4bit:
            if self.device != "cuda":
                raise RuntimeError("4-bit loading requires a CUDA GPU")
            if not has_bnb:
                raise RuntimeError("Install bitsandbytes to use CUDA 4-bit loading")
            kwargs.update(
                device_map="auto",
                max_memory=max_memory,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                ),
            )
        elif self.device == "cuda":
            kwargs.update(device_map="auto", max_memory=max_memory, torch_dtype=torch.float16)
        else:
            kwargs.update(torch_dtype=torch.float16)
        return kwargs

    def load_model(
        self,
        model_name: str,
        load_in_4bit: Optional[bool] = None,
        load_in_8bit: bool = False,
        device_map: str = "auto",
        max_memory: Optional[dict] = None,
    ) -> Tuple[torch.nn.Module, object]:
        """Load a full Qwen2-VL checkpoint (used by the training entry point)."""
        if load_in_8bit:
            raise NotImplementedError("8-bit mode is not used by this project")
        quantized = self.load_in_4bit if load_in_4bit is None else load_in_4bit
        processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True, **self._hub_kwargs()
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, **self._model_kwargs(quantized, max_memory)
        )
        if self.device == "cpu":
            model.to("cpu")
        self.models[model_name] = model
        self.tokenizers[model_name] = processor
        return model, processor

    def load_adapter_model(self, config: ModelConfig) -> Tuple[torch.nn.Module, object]:
        adapter = self._adapter_source(config)
        base_name = config.base_model
        if not base_name:
            raise ValueError(f"No base model configured for {config.model_id}")
        local_base = PROJECT_ROOT / "models" / Path(base_name).name
        if (local_base / "config.json").is_file() and any(local_base.glob("*.safetensors")):
            base_name = str(local_base)

        if self.device == "cpu" and os.environ.get("VI_VQA_IGNORE_MEMORY_CHECK") != "1":
            try:
                import psutil

                required_gb = 5.0 if config.params == "2B" else 16.0
                available_gb = psutil.virtual_memory().available / 1024**3
                if available_gb < required_gb:
                    raise RuntimeError(
                        f"Not enough free RAM for {config.params} CPU inference: "
                        f"{available_gb:.2f} GB available, about {required_gb:.0f} GB required. "
                        "Close other applications or run on CUDA."
                    )
            except ImportError:
                pass

        print(f"Loading processor from: {adapter}")
        processor = AutoProcessor.from_pretrained(
            adapter, trust_remote_code=True, **self._hub_kwargs()
        )
        print(f"Loading base model: {base_name}")
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            base_name, **self._model_kwargs(self.load_in_4bit)
        )
        if self.device == "cpu":
            base.to("cpu")
        print(f"Attaching LoRA adapter: {adapter}")
        model = PeftModel.from_pretrained(base, adapter, **self._hub_kwargs())
        model.eval()
        self.models[config.model_id] = model
        self.tokenizers[config.model_id] = processor
        return model, processor

    def load_teacher_model(self) -> Tuple[torch.nn.Module, object]:
        return self.load_adapter_model(MODELS["7B-trained"])

    def load_student_model(self) -> Tuple[torch.nn.Module, object]:
        return self.load_adapter_model(MODELS["2B-distilled"])

    def load_base_model(self) -> Tuple[torch.nn.Module, object]:
        return self.load_model(MODELS["7B-base"].model_name)

    def get_model(self, model_name: str) -> Optional[torch.nn.Module]:
        return self.models.get(model_name)

    def get_tokenizer(self, model_name: str) -> Optional[object]:
        return self.tokenizers.get(model_name)

    def unload_model(self, model_name: str) -> None:
        self.models.pop(model_name, None)
        self.tokenizers.pop(model_name, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload_all(self) -> None:
        self.models.clear()
        self.tokenizers.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def list_loaded_models(self) -> list:
        return list(self.models.keys())

    @staticmethod
    def list_available_models() -> dict:
        return {
            model_id: {
                "name": config.model_name,
                "params": config.params,
                "purpose": config.purpose,
                "base_model": config.base_model,
                "local_adapter": config.local_adapter,
            }
            for model_id, config in MODELS.items()
        }

    @staticmethod
    def print_available_models() -> None:
        for model_id, info in VQAModelLoader.list_available_models().items():
            print(f"{model_id}: {info}")
