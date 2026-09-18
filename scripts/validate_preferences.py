from __future__ import annotations

import argparse
import json

from vqa_explanation.data.io import read_jsonl
from vqa_explanation.preference.validate import validate_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate preference-pair provenance and schema")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    report = validate_pairs(read_jsonl(args.input))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["counts"].get("invalid", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

