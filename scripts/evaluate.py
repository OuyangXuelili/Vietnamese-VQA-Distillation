#!/usr/bin/env python3
"""Evaluate existing Vi-VQA outputs without retraining."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import VQADataLoader
from src.models import VQAModelLoader
from src.models.inference import ComparativeInference, VQAInference
from src.models.model_wrapper import StudentModel, TeacherModel
from src.training import MetricsCalculator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _load_wrapper(loader, model_type):
    if model_type == "teacher":
        model, processor = loader.load_teacher_model()
        return TeacherModel(model, processor, device=loader.device)
    model, processor = loader.load_student_model()
    return StudentModel(model, processor, device=loader.device)


def _load_rows(test_json=None, limit=None, split="train"):
    if test_json:
        with open(test_json, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        rows = raw if isinstance(raw, list) else raw.get("annotations", [])
        if isinstance(rows, dict):
            rows = list(rows.values())
    else:
        if split == "dev":
            logger.warning(
                "The public repository is missing dev image files (IDs 0-572); "
                "use --split train or provide --test-json with valid image paths."
            )
        rows = list(VQADataLoader().get_split(split))
    return rows[:limit] if limit is not None else rows


def _predict_rows(inference, rows, max_new_tokens=48):
    predictions = []
    references = []
    details = []
    for index, row in enumerate(rows, 1):
        image = row.get("image_path") or row.get("image")
        question = row.get("question", "")
        if not image or not question:
            continue
        logger.info("Processing %d/%d", index, len(rows))
        answer = inference.predict_single(image, question, max_new_tokens)
        reference = row.get("answer", "")
        predictions.append(answer)
        references.append(reference)
        details.append(
            {
                "image": image,
                "question": question,
                "predicted_answer": answer,
                "reference_answer": reference,
            }
        )
    return predictions, references, details


def evaluate_model(model_type, rows, device="auto", max_new_tokens=48):
    loader = VQAModelLoader(device=device)
    inference = VQAInference(_load_wrapper(loader, model_type))
    predictions, references, details = _predict_rows(
        inference, rows, max_new_tokens
    )
    metrics = MetricsCalculator.calculate_all_metrics(predictions, references)
    return {
        "model": model_type,
        "num_samples": len(predictions),
        "metrics": metrics,
        "predictions": details,
    }


def compare_models_eval(rows, device="auto", max_new_tokens=48):
    loader = VQAModelLoader(device=device)
    teacher = _load_wrapper(loader, "teacher")
    student = _load_wrapper(loader, "student")
    comparative = ComparativeInference(teacher, student)
    comparisons = []
    references = []
    for index, row in enumerate(rows, 1):
        image = row.get("image_path") or row.get("image")
        question = row.get("question", "")
        if not image or not question:
            continue
        logger.info("Comparing %d/%d", index, len(rows))
        result = comparative.compare_single(image, question, max_new_tokens)
        result["reference_answer"] = row.get("answer", "")
        comparisons.append(result)
        references.append(row.get("answer", ""))

    teacher_predictions = [item["teacher_answer"] for item in comparisons]
    student_predictions = [item["student_answer"] for item in comparisons]
    return {
        "num_samples": len(comparisons),
        "teacher_metrics": MetricsCalculator.calculate_all_metrics(
            teacher_predictions, references
        ),
        "student_metrics": MetricsCalculator.calculate_all_metrics(
            student_predictions, references
        ),
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate existing Vi-VQA outputs")
    parser.add_argument(
        "--model-type", choices=["teacher", "student", "both"], default="student"
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--test-json")
    parser.add_argument(
        "--split",
        choices=["train", "dev"],
        default="train",
        help="Hub split when --test-json is omitted; public dev images are missing",
    )
    parser.add_argument("--output-file")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()

    rows = _load_rows(args.test_json, args.limit, args.split)
    if args.model_type == "both":
        results = compare_models_eval(rows, args.device, args.max_new_tokens)
    else:
        results = evaluate_model(
            args.model_type, rows, args.device, args.max_new_tokens
        )
    print(json.dumps(results.get("metrics", results), ensure_ascii=False, indent=2))
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
