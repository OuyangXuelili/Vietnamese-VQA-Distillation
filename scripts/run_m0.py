from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from vqa_explanation.inference.prompting import parse_prediction
from vqa_explanation.inference.qwen25_vl import Qwen25VLRunner

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def select_records(
    records: list[dict[str, Any]], split: str, limit: int, seed: int
) -> list[dict[str, Any]]:
    selected = [record for record in records if record.get("split") == split]
    random.Random(seed).shuffle(selected)
    if limit > 0:
        selected = selected[:limit]
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen M0 pretrained baseline.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = select_records(read_jsonl(args.annotations), args.split, args.limit, args.seed)
    if not records:
        raise SystemExit(f"No records found for split={args.split!r}")

    missing = [
        str(args.image_root / record["image_path"])
        for record in records
        if not (args.image_root / record["image_path"]).is_file()
    ]
    if missing:
        raise SystemExit(f"Missing {len(missing)} selected images; first: {missing[0]}")

    runner = Qwen25VLRunner(
        args.model_id,
        args.revision,
        load_in_4bit=args.load_in_4bit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for index, record in enumerate(records, start=1):
            started = time.perf_counter()
            raw = runner.generate(
                args.image_root / record["image_path"],
                record["question"],
                max_new_tokens=args.max_new_tokens,
            )
            parsed = parse_prediction(raw)
            result = {
                "experiment_id": "M0_PRETRAINED",
                "model_id": args.model_id,
                "model_revision": args.revision,
                "seed": args.seed,
                "question_id": record["question_id"],
                "image_id": record["image_id"],
                "image_path": record["image_path"],
                "split": record["split"],
                "question": record["question"],
                "reference_answer": record["answer"],
                "reference_explanations": record["explanations"],
                "prediction_answer": parsed.answer,
                "prediction_explanation": parsed.explanation,
                "format_valid": parsed.format_valid,
                "raw_output": raw,
                "latency_seconds": round(time.perf_counter() - started, 4),
            }
            destination.write(json.dumps(result, ensure_ascii=False) + "\n")
            destination.flush()
            print(f"[{index}/{len(records)}] {record['question_id']} format={parsed.format_valid}")


if __name__ == "__main__":
    main()
