#!/usr/bin/env python3
"""Run resumable Teacher or Student prediction on a fixed VQA split."""

import argparse
import json
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from urllib.parse import urlparse

import torch

from src.data import VQADataLoader
from src.models import VQAModelLoader
from src.models.inference import VQAInference
from src.models.model_wrapper import StudentModel, TeacherModel


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", choices=("teacher", "student"), required=True)
    parser.add_argument("--split", choices=("train", "dev"), default="dev")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def atomic_save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_inference(model_type):
    loader = VQAModelLoader(device="cuda", load_in_4bit=True)
    if model_type == "teacher":
        model, processor = loader.load_teacher_model()
        wrapper = TeacherModel(model, processor, device=loader.device)
    else:
        model, processor = loader.load_student_model()
        wrapper = StudentModel(model, processor, device=loader.device)
    return VQAInference(wrapper)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation")

    rows = list(VQADataLoader().get_split(args.split))
    if args.limit is not None:
        rows = rows[: args.limit]

    images_dir = Path(args.images_dir)
    output_path = Path(args.output_file)
    predictions = []
    if args.resume and output_path.is_file():
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        predictions = saved.get("predictions", [])

    start_index = len(predictions)
    if start_index > len(rows):
        raise ValueError("Checkpoint contains more predictions than the selected split")
    if start_index:
        expected_id = str(rows[start_index - 1]["id"])
        saved_id = str(predictions[-1].get("id"))
        if saved_id != expected_id:
            raise ValueError(
                f"Checkpoint order mismatch: expected last id {expected_id}, got {saved_id}"
            )

    print(f"[{args.model_type}] Tiếp tục từ {start_index}/{len(rows)}", flush=True)
    inference = load_inference(args.model_type)
    session_start = time.time()

    def save():
        atomic_save(
            output_path,
            {
                "model": args.model_type,
                "split": args.split,
                "completed": len(predictions),
                "total": len(rows),
                "max_new_tokens": args.max_new_tokens,
                "predictions": predictions,
            },
        )
    for index in range(start_index, len(rows)):
        row = rows[index]
        image_name = Path(urlparse(row["image"]).path).name
        image_path = images_dir / image_name
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")

        started = time.perf_counter()
        answer = inference.predict_single(
            str(image_path), row["question"], args.max_new_tokens
        )
        latency = time.perf_counter() - started
        predictions.append(
            {
                "id": str(row["id"]),
                "image_path": str(image_path),
                "question": row["question"],
                "predicted_answer": answer,
                "reference_answer": row["answer"],
                "latency_s": latency,
            }
        )

        completed = index + 1
        if completed % args.checkpoint_every == 0 or completed == len(rows):
            save()
            elapsed_minutes = (time.time() - session_start) / 60
            print(
                f"[{args.model_type}] {completed}/{len(rows)} | "
                f"phiên này {elapsed_minutes:.1f} phút",
                flush=True,
            )

    save()
    print(f"[{args.model_type}] Hoàn tất: {output_path}", flush=True)


if __name__ == "__main__":
    main()
