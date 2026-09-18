from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .schema import VQASample


def audit_records(
    records: Iterable[dict],
    verify_images: bool = False,
    image_root: str | Path | None = None,
) -> dict:
    counts = Counter()
    errors: list[dict] = []
    images_by_split: dict[str, set[str]] = defaultdict(set)
    seen_questions: set[str] = set()

    for index, raw in enumerate(records):
        counts["records"] += 1
        try:
            sample = VQASample.from_dict(raw)
            sample.validate_ids()
        except (TypeError, ValueError) as exc:
            errors.append({"index": index, "error": str(exc)})
            counts["invalid"] += 1
            continue
        counts[f"split_{sample.split}"] += 1
        counts["explanations"] += len(sample.explanations)
        images_by_split[sample.split].add(sample.image_id)
        if sample.question_id in seen_questions:
            counts["duplicate_question_ids"] += 1
        seen_questions.add(sample.question_id)
        image_path = Path(sample.image_path)
        if image_root is not None:
            image_path = Path(image_root) / image_path
        if verify_images and not image_path.is_file():
            counts["missing_images"] += 1

    overlaps = {}
    split_names = sorted(images_by_split)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            shared = images_by_split[left] & images_by_split[right]
            overlaps[f"{left}__{right}"] = len(shared)

    return {
        "counts": dict(sorted(counts.items())),
        "unique_images_by_split": {k: len(v) for k, v in sorted(images_by_split.items())},
        "image_overlap_by_split": overlaps,
        "errors": errors,
    }
