"""Data loading for the raw VLSP 2023 VQA repository on Hugging Face."""

import json
from typing import Dict, List, Optional
from urllib.parse import quote

from datasets import Dataset
from huggingface_hub import hf_hub_download


class VQADataLoader:
    """Load train/dev metadata without downloading every image up front."""

    def __init__(
        self,
        repo_id: str = "Ouyangxuelili/vlsp2023-vqa-dataset",
        cache_dir: str = "./data/cache",
        use_auth_token: bool = False,
        streaming: bool = False,
    ):
        self.repo_id = repo_id
        self.cache_dir = cache_dir
        self.use_auth_token = use_auth_token
        self.streaming = streaming
        self.dataset = None

    def _load_split(self, split_name: str) -> Dataset:
        filename = f"vlsp2023_{split_name}_data.json"
        path = hf_hub_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            filename=filename,
            cache_dir=self.cache_dir,
            token=True if self.use_auth_token else None,
        )
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)

        images = raw.get("images", {})
        annotations = raw.get("annotations", {})
        items = annotations.items() if isinstance(annotations, dict) else enumerate(annotations)
        repo = quote(self.repo_id, safe="/")
        records = []
        for annotation_id, annotation in items:
            image_id = annotation.get("image_id")
            filename = images.get(str(image_id))
            if not filename:
                continue
            records.append(
                {
                    "id": str(annotation_id),
                    "image_id": image_id,
                    "image": f"https://huggingface.co/datasets/{repo}/resolve/main/images/{filename}",
                    "question": annotation.get("question", ""),
                    "answer": annotation.get("answer", ""),
                }
            )
        return Dataset.from_list(records)

    def load_dataset(self) -> Dict[str, Dataset]:
        self.dataset = {
            "train": self._load_split("train"),
            "dev": self._load_split("dev"),
        }
        print(f"Dataset loaded: {self.repo_id}")
        for name, split in self.dataset.items():
            print(f"  - {name}: {len(split)} samples")
        return self.dataset

    def get_split(self, split_name: str) -> Dataset:
        if self.dataset is None:
            self.load_dataset()
        if split_name not in self.dataset:
            raise ValueError(
                f"Split {split_name!r} not found. Available: {list(self.dataset)}"
            )
        return self.dataset[split_name]

    def get_train(self) -> Dataset:
        return self.get_split("train")

    def get_dev(self) -> Dataset:
        return self.get_split("dev")

    def get_test(self) -> Dataset:
        raise ValueError("Published dataset has no test split")

    def get_sample(self, split: str = "train", index: int = 0) -> Dict:
        return self.get_split(split)[index]

    def print_sample(self, split: str = "train", index: int = 0) -> None:
        sample = self.get_sample(split, index)
        print(f"Sample from {split} (index {index})")
        for key, value in sample.items():
            print(f"{key}: {value}")

    def get_statistics(self) -> Dict[str, int]:
        if self.dataset is None:
            self.load_dataset()
        result = {name: len(split) for name, split in self.dataset.items()}
        result["total"] = sum(result.values())
        return result

    def print_statistics(self) -> None:
        for name, count in self.get_statistics().items():
            print(f"{name}: {count:,}")


def build_vqa_messages(question: str, answer: str) -> List[Dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]


def prepare_vqa_dataset(
    dataset: Dataset,
    tokenizer,
    max_samples: Optional[int] = None,
    shuffle: bool = True,
) -> Dataset:
    def process_sample(sample):
        messages = build_vqa_messages(sample["question"], sample["answer"])
        return {
            "images": [sample["image"]],
            "text": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            ),
            "question": sample["question"],
            "answer": sample["answer"],
        }

    processed = dataset.map(process_sample)
    if shuffle:
        processed = processed.shuffle(seed=42)
    if max_samples is not None:
        processed = processed.select(range(min(max_samples, len(processed))))
    return processed
