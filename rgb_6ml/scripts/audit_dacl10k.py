#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from rgb_multilabel.labels import CLASS_NAMES, rasterize_annotation


LABELED_SPLITS = ("train", "validation")
ALL_SPLITS = (*LABELED_SPLITS, "testdev")


def image_stems(directory: Path) -> set[str]:
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    return {path.stem for path in directory.iterdir() if path.suffix.lower() in extensions}


def annotation_stems(directory: Path) -> set[str]:
    return {path.stem for path in directory.glob("*.json")}


def empty_matrix() -> np.ndarray:
    return np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)


def audit_labeled_split(root: Path, split: str) -> dict[str, Any]:
    image_dir = root / "images" / split
    annotation_dir = root / "annotations" / split
    images, annotations = image_stems(image_dir), annotation_stems(annotation_dir)
    missing_images = sorted(annotations - images)
    missing_annotations = sorted(images - annotations)

    image_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    pixel_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    image_cooccurrence = empty_matrix()
    source_label_images: Counter[str] = Counter()
    source_label_polygons: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    invalid: list[dict[str, str]] = []
    total_pixels = 0
    valid_pixels = 0
    ignored_pixels = 0
    selected_union_pixels = 0

    files = sorted(annotation_dir.glob("*.json"))
    for index, path in enumerate(files, 1):
        try:
            with path.open("r", encoding="utf-8") as handle:
                annotation = json.load(handle)
            expected_name = str(annotation.get("imageName", ""))
            if Path(expected_name).stem != path.stem:
                raise ValueError(f"imageName mismatch: {expected_name!r}")
            targets, valid_mask, labels = rasterize_annotation(annotation)
            height, width = targets.shape[-2:]
            dimensions[f"{width}x{height}"] += 1
            pixels = height * width
            total_pixels += pixels
            valid_pixels += int(valid_mask.sum())
            ignored_pixels += int((~valid_mask).sum())
            selected_union_pixels += int(targets.any(axis=0).sum())

            present = targets.reshape(len(CLASS_NAMES), -1).any(axis=1)
            class_pixels = targets.reshape(len(CLASS_NAMES), -1).sum(axis=1)
            image_counts += present
            pixel_counts += class_pixels
            image_cooccurrence += np.logical_and.outer(present, present)
            source_label_images.update(labels)
            source_label_polygons.update(
                str(shape["label"]) for shape in annotation.get("shapes", ())
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            invalid.append({"file": str(path), "error": str(error)})
        if index % 250 == 0 or index == len(files):
            print(f"{split}: {index}/{len(files)} annotations", flush=True)

    return {
        "split": split,
        "image_count": len(images),
        "annotation_count": len(annotations),
        "missing_images": missing_images,
        "missing_annotations": missing_annotations,
        "invalid_annotations": invalid,
        "dimensions": dict(sorted(dimensions.items())),
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "ignored_pixels": ignored_pixels,
        "ignored_pixel_fraction": ignored_pixels / total_pixels if total_pixels else None,
        "selected_union_pixels": selected_union_pixels,
        "selected_union_fraction": selected_union_pixels / total_pixels if total_pixels else None,
        "class_image_counts": dict(zip(CLASS_NAMES, image_counts.tolist())),
        "class_pixel_counts": dict(zip(CLASS_NAMES, pixel_counts.tolist())),
        "class_pixel_fractions": {
            name: int(count) / total_pixels if total_pixels else None
            for name, count in zip(CLASS_NAMES, pixel_counts)
        },
        "image_cooccurrence": image_cooccurrence.tolist(),
        "source_label_image_counts": dict(sorted(source_label_images.items())),
        "source_label_polygon_counts": dict(sorted(source_label_polygons.items())),
    }


def audit_unlabeled_split(root: Path, split: str) -> dict[str, Any]:
    image_dir = root / "images" / split
    annotation_dir = root / "annotations" / split
    return {
        "split": split,
        "image_count": len(image_stems(image_dir)),
        "annotation_count": (
            len(annotation_stems(annotation_dir)) if annotation_dir.is_dir() else 0
        ),
    }


def write_class_csv(path: Path, results: dict[str, dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("split", "class", "image_count", "pixel_count", "pixel_fraction"),
        )
        writer.writeheader()
        for split in LABELED_SPLITS:
            result = results[split]
            for name in CLASS_NAMES:
                writer.writerow(
                    {
                        "split": split,
                        "class": name,
                        "image_count": result["class_image_counts"][name],
                        "pixel_count": result["class_pixel_counts"][name],
                        "pixel_fraction": result["class_pixel_fractions"][name],
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit DACL10K for the six-channel RGB task")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.data_root.resolve()
    required = [root / "images" / split for split in ALL_SPLITS]
    required += [root / "annotations" / split for split in LABELED_SPLITS]
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing DACL10K directories: {missing}")

    results = {split: audit_labeled_split(root, split) for split in LABELED_SPLITS}
    results["testdev"] = audit_unlabeled_split(root, "testdev")
    report = {
        "data_root": str(root),
        "class_names": list(CLASS_NAMES),
        "mask_rule": "selected positive > excluded-label ignore > unannotated negative",
        "splits": results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "dacl10k_audit.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    write_class_csv(args.output_dir / "dacl10k_class_statistics.csv", results)

    problems = sum(
        len(results[split][key])
        for split in LABELED_SPLITS
        for key in ("missing_images", "missing_annotations", "invalid_annotations")
    )
    print(json.dumps({
        "report": str(report_path),
        "class_statistics": str(args.output_dir / "dacl10k_class_statistics.csv"),
        "problems": problems,
    }, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
