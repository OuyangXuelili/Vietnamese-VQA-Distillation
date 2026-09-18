from __future__ import annotations

import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from PIL import Image

COCO_IMAGE_BASE_URL = "https://s3.us-east-1.amazonaws.com/images.cocodataset.org"


def select_smoke_images(
    rows: Iterable[dict[str, str]],
    quotas: dict[str, int],
    seed: int,
) -> list[dict[str, str]]:
    """Select an image-disjoint, deterministic subset for pipeline smoke tests."""
    groups: dict[str, list[dict[str, str]]] = {split: [] for split in quotas}
    for row in rows:
        split = row["dataset_splits"]
        if split in groups:
            groups[split].append(row)

    selected: list[dict[str, str]] = []
    for offset, (split, size) in enumerate(quotas.items()):
        candidates = sorted(groups[split], key=lambda item: item["image_name"])
        if len(candidates) < size:
            raise ValueError(f"not enough exclusive {split} images: {len(candidates)} < {size}")
        rng = random.Random(seed + offset)
        selected.extend(rng.sample(candidates, size))
    return sorted(selected, key=lambda item: (item["dataset_splits"], item["image_name"]))


def coco_image_url(row: dict[str, str]) -> str:
    return f"{COCO_IMAGE_BASE_URL}/{row['coco_split']}/{row['image_name']}"


def valid_jpeg(path: str | Path) -> bool:
    source = Path(path)
    if not source.is_file() or source.stat().st_size == 0:
        return False
    try:
        with Image.open(source) as image:
            if image.format != "JPEG":
                return False
            image.verify()
    except (OSError, ValueError):
        return False
    return True


def download_image(
    row: dict[str, str],
    output_dir: str | Path,
    timeout: int = 30,
    retries: int = 3,
) -> dict[str, str]:
    destination = Path(output_dir) / row["image_name"]
    if valid_jpeg(destination):
        return {"image_name": row["image_name"], "status": "skipped", "error": ""}

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(coco_image_url(row), headers={"User-Agent": "vivqa-nle/0.1"})
    last_error = ""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                temporary.write_bytes(response.read())
            if not valid_jpeg(temporary):
                raise ValueError("downloaded file is not a valid JPEG")
            temporary.replace(destination)
            return {"image_name": row["image_name"], "status": "downloaded", "error": ""}
        except Exception as exc:  # network failures need to be reported per image
            temporary.unlink(missing_ok=True)
            last_error = str(exc)
            if attempt + 1 < retries:
                time.sleep(1.5**attempt)
    return {"image_name": row["image_name"], "status": "failed", "error": last_error}


def download_images(
    rows: Iterable[dict[str, str]],
    output_dir: str | Path,
    workers: int = 8,
) -> list[dict[str, str]]:
    records = list(rows)
    results: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_image, row, output_dir): row for row in records}
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda item: item["image_name"])
