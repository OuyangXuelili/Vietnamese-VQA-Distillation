from __future__ import annotations

import argparse

from vqa_explanation.data.audit import audit_records
from vqa_explanation.data.io import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit normalized ViVQA-X JSONL data")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify-images", action="store_true")
    parser.add_argument("--image-root")
    args = parser.parse_args()
    report = audit_records(
        read_jsonl(args.input),
        verify_images=args.verify_images,
        image_root=args.image_root,
    )
    write_json(args.output, report)
    print(f"Audit written to {args.output}")
    print(report["counts"])


if __name__ == "__main__":
    main()
