#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Convert DamCrack refined masks to the shared six-label manifest format."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np


CLASS_COLORS_BGR = {
    "crack": np.array([0, 0, 255], dtype=np.uint8),
    "spalling": np.array([255, 0, 0], dtype=np.uint8),
}
BACKGROUND_BGR = np.array([0, 0, 0], dtype=np.uint8)
MANIFEST_FIELDS = (
    "split",
    "image",
    "crack",
    "spalling",
    "exposed_rebar",
    "rust",
    "efflorescence",
    "wetspot",
)


def stable_split(stem: str, seed: str, train_fraction: float,
                 val_fraction: float) -> str:
    """Assign a stable split without depending on filesystem order."""
    digest = hashlib.sha256(f"{seed}:{stem}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < train_fraction:
        return "train"
    if value < train_fraction + val_fraction:
        return "val"
    return "test"


def relative_to_manifest(path: Path, manifest: Path) -> str:
    return Path(os.path.relpath(path.resolve(), manifest.parent.resolve())).as_posix()


def prepare(data_root: Path, output_root: Path, seed: str,
            train_fraction: float, val_fraction: float) -> dict:
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1:
        raise ValueError("Train and validation fractions must be between zero and one")
    if train_fraction + val_fraction >= 1:
        raise ValueError("Train and validation fractions must leave a test split")

    images_dir = data_root / "Images"
    masks_dir = data_root / "Refined Labels" / "Mask"
    images = {path.stem: path for path in images_dir.glob("*.jpg")}
    source_masks = {path.stem: path for path in masks_dir.glob("*.png")}
    if not images or images.keys() != source_masks.keys():
        missing_masks = sorted(images.keys() - source_masks.keys())[:10]
        missing_images = sorted(source_masks.keys() - images.keys())[:10]
        raise ValueError(
            f"Image/mask mismatch: missing masks={missing_masks}, "
            f"missing images={missing_images}"
        )

    manifest = output_root / "manifest.csv"
    class_dirs = {name: output_root / "masks" / name for name in CLASS_COLORS_BGR}
    for directory in class_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    split_counts = {"train": 0, "val": 0, "test": 0}
    positive_images = {name: 0 for name in CLASS_COLORS_BGR}
    multi_label_images = 0
    rows = []
    for stem in sorted(images, key=lambda name: int(name[1:])):
        color_mask = cv2.imread(str(source_masks[stem]), cv2.IMREAD_COLOR)
        image = cv2.imread(str(images[stem]), cv2.IMREAD_COLOR)
        if color_mask is None or image is None or color_mask.shape != image.shape:
            raise ValueError(f"Unreadable or wrong-size image/mask pair: {stem}")

        known = np.all(color_mask == BACKGROUND_BGR, axis=2)
        binary_masks = {}
        for name, color in CLASS_COLORS_BGR.items():
            selected = np.all(color_mask == color, axis=2)
            known |= selected
            binary_masks[name] = selected
        if not np.all(known):
            unknown = np.unique(color_mask[~known], axis=0)[:10].tolist()
            raise ValueError(f"Unknown BGR colors in {source_masks[stem]}: {unknown}")

        split = stable_split(stem, seed, train_fraction, val_fraction)
        split_counts[split] += 1
        active = 0
        output_masks = {}
        for name, selected in binary_masks.items():
            output_path = class_dirs[name] / f"{stem}.png"
            if not cv2.imwrite(str(output_path), selected.astype(np.uint8) * 255):
                raise OSError(f"Could not write {output_path}")
            output_masks[name] = output_path
            if selected.any():
                positive_images[name] += 1
                active += 1
        multi_label_images += active > 1
        rows.append({
            "split": split,
            "image": relative_to_manifest(images[stem], manifest),
            "crack": relative_to_manifest(output_masks["crack"], manifest),
            "spalling": relative_to_manifest(output_masks["spalling"], manifest),
            "exposed_rebar": "",
            "rust": "",
            "efflorescence": "",
            "wetspot": "",
        })

    output_root.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "data_root": str(data_root.resolve()),
        "manifest": str(manifest.resolve()),
        "images": len(rows),
        "split_counts": split_counts,
        "positive_images": positive_images,
        "multi_label_images": multi_label_images,
        "class_mapping": {
            "red_rgb_255_0_0": "crack",
            "blue_rgb_0_0_255": "spalling",
        },
        "split_warning": (
            "DamCrack does not publish patch-to-source-image provenance. "
            "This deterministic patch-level split may contain correlated crops "
            "from the same source image and must not be described as source-grouped."
        ),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Damage Segmentaion directory containing Images and Refined Labels")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", default="damcrack-v1")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = prepare(args.data_root, args.output_root, args.seed,
                     args.train_fraction, args.val_fraction)
    print(json.dumps(result, indent=2))
