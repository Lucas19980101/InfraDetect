#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from rgb_multilabel.labels import rasterize_annotation
from rgb_multilabel.masks import decode_mask, encode_mask


def convert_one(task: tuple[str, str, bool]) -> tuple[str, str]:
    annotation_name, output_name, overwrite = task
    annotation_path, output_path = Path(annotation_name), Path(output_name)
    if output_path.exists() and not overwrite:
        try:
            cached = np.asarray(Image.open(output_path), dtype=np.uint8)
            decode_mask(cached)
            return "skipped", output_path.name
        except (OSError, ValueError):
            pass
    with annotation_path.open("r", encoding="utf-8") as handle:
        annotation = json.load(handle)
    targets, valid_mask, _ = rasterize_annotation(annotation)
    encoded = encode_mask(targets, valid_mask)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".png.tmp")
    Image.fromarray(encoded, mode="L").save(temporary, format="PNG", optimize=True)
    os.replace(temporary, output_path)
    return "written", output_path.name


def prepare_split(
    data_root: Path, cache_root: Path, split: str, workers: int, overwrite: bool
) -> dict[str, int]:
    annotation_dir = data_root / "annotations" / split
    output_dir = cache_root / split
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (str(path), str(output_dir / f"{path.stem}.png"), overwrite)
        for path in sorted(annotation_dir.glob("*.json"))
    ]
    counts = {"written": 0, "skipped": 0}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, (status, _) in enumerate(executor.map(convert_one, tasks), 1):
            counts[status] += 1
            if index % 250 == 0 or index == len(tasks):
                print(f"{split}: {index}/{len(tasks)} masks", flush=True)
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache six-channel DACL10K masks")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least one")
    result = {
        split: prepare_split(
            args.data_root.resolve(), args.cache_root.resolve(), split, args.workers, args.overwrite
        )
        for split in ("train", "validation")
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
