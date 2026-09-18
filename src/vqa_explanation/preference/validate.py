from __future__ import annotations

from collections import Counter
from typing import Iterable

from .schema import PreferencePair


def validate_pairs(records: Iterable[dict]) -> dict:
    counts = Counter()
    errors = []
    seen = set()
    for index, raw in enumerate(records):
        counts["records"] += 1
        try:
            pair = PreferencePair.from_dict(raw)
        except (TypeError, ValueError) as exc:
            counts["invalid"] += 1
            errors.append({"index": index, "error": str(exc)})
            continue
        counts[f"review_{pair.review_status}"] += 1
        counts[f"error_{pair.error_type}"] += 1
        if pair.pair_id in seen:
            counts["duplicate_pair_ids"] += 1
        seen.add(pair.pair_id)
    return {"counts": dict(sorted(counts.items())), "errors": errors}

