from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .inference import positions


SOURCES = (
    ("crack", 1, Path("crack_org")),
    ("spalling", 2, Path("spalling_org")),
)


def stable_fraction(text: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{text}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_groups(names: list[str], seed: int) -> dict[str, str]:
    shuffled = sorted(names)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    n_test = max(1, round(count * 0.1))
    n_val = max(1, round(count * 0.1))
    return {
        name: "test" if index < n_test else "val" if index < n_test + n_val else "train"
        for index, name in enumerate(shuffled)
    }


def source_pairs(root: Path) -> list[tuple[Path, Path]]:
    images = {p.stem: p for p in root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}}
    masks = {p.stem: p for p in root.iterdir() if p.suffix.lower() == ".png"}
    if images.keys() != masks.keys():
        raise ValueError(f"Unpaired originals in {root}")
    return [(images[name], masks[name]) for name in sorted(images)]


def prepare(raw_root: Path, output_root: Path, tile_size: int, stride: int,
            background_keep: float, seed: int) -> None:
    if any((output_root / kind / split).exists() and any((output_root / kind / split).iterdir())
           for kind in ("images", "masks") for split in ("train", "val", "test")):
        raise FileExistsError(f"Output is not empty: {output_root}; use a new --output directory")
    for kind in ("images", "masks"):
        for split in ("train", "val", "test"):
            (output_root / kind / split).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for source_name, class_id, relative_root in SOURCES:
        pairs = source_pairs(raw_root / relative_root)
        assignments = split_groups([image.stem for image, _ in pairs], seed + class_id)
        for image_path, mask_path in tqdm(pairs, desc=source_name):
            split = assignments[image_path.stem]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if image is None or raw_mask is None:
                raise ValueError(f"Unreadable pair: {image_path}, {mask_path}")
            if raw_mask.ndim == 3:
                raw_mask = raw_mask.max(axis=2)
            if image.shape[:2] != raw_mask.shape[:2]:
                raise ValueError(f"Shape mismatch: {image_path}, {mask_path}")
            mask = np.where(raw_mask > 0, class_id, 0).astype(np.uint8)
            height, width = mask.shape
            for y in positions(height, tile_size, stride):
                for x in positions(width, tile_size, stride):
                    tile_mask = mask[y:y + tile_size, x:x + tile_size]
                    positive = bool(np.any(tile_mask == class_id))
                    key = f"{source_name}_{image_path.stem}_{y}_{x}"
                    if split == "train" and not positive and stable_fraction(key, seed) >= background_keep:
                        continue
                    tile_image = image[y:y + tile_size, x:x + tile_size]
                    if tile_image.shape[:2] != (tile_size, tile_size):
                        raise ValueError(f"Original smaller than tile size: {image_path}")
                    image_out = output_root / "images" / split / f"{key}.jpg"
                    mask_out = output_root / "masks" / split / f"{key}.png"
                    if not cv2.imwrite(str(image_out), tile_image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        raise OSError(f"Failed to write {image_out}")
                    if not cv2.imwrite(str(mask_out), tile_mask):
                        raise OSError(f"Failed to write {mask_out}")
                    rows.append({"name": key, "split": split, "source": source_name,
                                 "original": image_path.stem, "y": y, "x": x,
                                 "positive": int(positive)})
    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    counts = Counter((row["split"], row["source"], row["positive"]) for row in rows)
    for key in sorted(counts):
        print(*key, counts[key])
    print(f"Wrote {len(rows)} paired tiles to {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-safe 1024px semantic tiles from original images")
    parser.add_argument("--raw-root", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=768)
    parser.add_argument("--background-keep", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.background_keep <= 1:
        parser.error("--background-keep must be between 0 and 1")
    prepare(args.raw_root, args.output, args.tile_size, args.stride, args.background_keep, args.seed)


if __name__ == "__main__":
    main()
