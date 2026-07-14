#!/usr/bin/env python3
"""Inference script for the trained Vi-VQA teacher/student outputs."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import VQAModelLoader
from src.models.inference import ComparativeInference, VQAInference
from src.models.model_wrapper import StudentModel, TeacherModel


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _load_wrapper(loader, model_type):
    if model_type == "teacher":
        model, processor = loader.load_teacher_model()
        return TeacherModel(model, processor, device=loader.device)
    if model_type == "student":
        model, processor = loader.load_student_model()
        return StudentModel(model, processor, device=loader.device)
    raise ValueError(f"Unknown model type: {model_type}")


def infer_single(model_type, image_path, question, device="auto", max_new_tokens=48):
    logger.info("Loading %s model", model_type)
    loader = VQAModelLoader(device=device)
    inference = VQAInference(_load_wrapper(loader, model_type))
    answer = inference.predict_from_file(image_path, question, max_new_tokens)
    result = {
        "image_path": image_path,
        "question": question,
        "predicted_answer": answer,
    }
    logger.info("Answer: %s", answer)
    return result


def infer_batch(
    model_type,
    input_json,
    output_json=None,
    device="auto",
    max_new_tokens=48,
    limit=None,
):
    loader = VQAModelLoader(device=device)
    inference = VQAInference(_load_wrapper(loader, model_type))
    predictions = inference.predict_from_json(input_json, max_new_tokens, limit)
    if output_json:
        inference.save_predictions(predictions, output_json)
    for index, prediction in enumerate(predictions[:5], 1):
        logger.info("%d. %s", index, prediction["predicted_answer"])
    return predictions


def compare_models(
    input_json,
    output_json=None,
    device="auto",
    max_new_tokens=48,
    limit=None,
):
    loader = VQAModelLoader(device=device)
    teacher = _load_wrapper(loader, "teacher")
    student = _load_wrapper(loader, "student")
    comparative = ComparativeInference(teacher, student)
    comparisons = comparative.compare_from_json(input_json, max_new_tokens, limit)
    if output_json:
        comparative.save_comparison(comparisons, output_json)
    return comparisons


def main():
    parser = argparse.ArgumentParser(description="Run inference with Vi-VQA models")
    parser.add_argument(
        "--model-type",
        choices=["teacher", "student", "both"],
        default="student",
        help="teacher (7B), student (2B), or both",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--image-path", help="Path or URL for single inference")
    parser.add_argument("--question", help="Question for single inference")
    parser.add_argument("--input-json", help="Batch input JSON")
    parser.add_argument("--output-json", help="Prediction/comparison output JSON")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--limit", type=int, help="Limit batch samples")
    args = parser.parse_args()

    if args.image_path and args.question:
        if args.model_type == "both":
            parser.error("single-image mode requires teacher or student")
        infer_single(
            args.model_type,
            args.image_path,
            args.question,
            args.device,
            args.max_new_tokens,
        )
    elif args.input_json:
        if args.model_type == "both":
            compare_models(
                args.input_json,
                args.output_json,
                args.device,
                args.max_new_tokens,
                args.limit,
            )
        else:
            infer_batch(
                args.model_type,
                args.input_json,
                args.output_json,
                args.device,
                args.max_new_tokens,
                args.limit,
            )
    else:
        parser.error("provide --image-path and --question, or --input-json")


if __name__ == "__main__":
    main()
