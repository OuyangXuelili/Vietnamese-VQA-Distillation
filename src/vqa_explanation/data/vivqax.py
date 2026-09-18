from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

RAW_FILES = {
    "train": "ViVQA-X_train.json",
    "validation": "ViVQA-X_val.json",
    "test": "ViVQA-X_test.json",
}


def load_raw_split(annotation_dir: str | Path, split: str) -> list[dict[str, Any]]:
    if split not in RAW_FILES:
        raise ValueError(f"unsupported split: {split}")
    source = Path(annotation_dir) / RAW_FILES[split]
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON list in {source}")
    return payload


def normalize_record(raw: dict[str, Any], split: str) -> dict[str, Any]:
    image_name = str(raw.get("image_name", "")).strip()
    if not image_name:
        raise ValueError("image_name must not be empty")
    return {
        "question_id": str(raw.get("question_id", "")).strip(),
        "image_id": str(raw.get("image_id", "")).strip(),
        "image_name": image_name,
        "image_path": f"images/{image_name}",
        "question": raw.get("question"),
        "answer": raw.get("answer"),
        "explanations": raw.get("explanation"),
        "question_type": raw.get("question_type"),
        "answer_type": raw.get("answer_type"),
        "split": split,
    }


def image_manifest(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    splits_by_image: dict[str, set[str]] = defaultdict(set)
    image_ids: dict[str, str] = {}
    for record in records:
        name = record["image_name"]
        splits_by_image[name].add(record["split"])
        image_ids[name] = record["image_id"]

    rows = []
    for name in sorted(splits_by_image):
        coco_split = "train2014" if "_train2014_" in name else "val2014"
        rows.append(
            {
                "image_name": name,
                "image_id": image_ids[name],
                "coco_split": coco_split,
                "dataset_splits": ",".join(sorted(splits_by_image[name])),
            }
        )
    return rows
