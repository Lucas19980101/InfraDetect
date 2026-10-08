#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from rgb_multilabel.labels import CLASS_NAMES, SOURCE_TO_CLASS


PAIR_FEATURES = tuple(combinations(range(len(CLASS_NAMES)), 2))
FEATURE_NAMES = (
    *CLASS_NAMES,
    *(f"{CLASS_NAMES[a]}+{CLASS_NAMES[b]}" for a, b in PAIR_FEATURES),
    "none_of_six",
)


def read_presence(annotation_dir: Path) -> tuple[list[dict[str, object]], np.ndarray]:
    records: list[dict[str, object]] = []
    feature_rows: list[list[bool]] = []
    for path in sorted(annotation_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            annotation = json.load(handle)
        present = np.zeros(len(CLASS_NAMES), dtype=np.bool_)
        for shape in annotation.get("shapes", ()):
            class_index = SOURCE_TO_CLASS.get(str(shape["label"]))
            if class_index is not None:
                present[class_index] = True
        pair_values = [bool(present[a] and present[b]) for a, b in PAIR_FEATURES]
        features = [*present.tolist(), *pair_values, not bool(present.any())]
        image_name = str(annotation["imageName"])
        if Path(image_name).stem != path.stem:
            raise ValueError(f"imageName mismatch in {path}: {image_name!r}")
        records.append({"image_name": image_name, "present": present.tolist()})
        feature_rows.append(features)
    return records, np.asarray(feature_rows, dtype=np.int16)


def objective(counts: np.ndarray, targets: np.ndarray) -> float:
    scale = np.maximum(targets, 1.0)
    return float(np.square((counts - targets) / scale).sum())


def stratified_selection(
    features: np.ndarray, validation_size: int, seed: int, iterations: int
) -> np.ndarray:
    """Select an exact-size validation set preserving labels and pair co-occurrence.

    A seeded swap optimizer avoids adding a dependency solely for iterative
    multilabel stratification. The same inputs always produce the same split.
    """
    sample_count = len(features)
    if not 0 < validation_size < sample_count:
        raise ValueError("validation_size must be between zero and sample count")
    rng = np.random.default_rng(seed)
    selected = np.zeros(sample_count, dtype=np.bool_)
    selected[rng.choice(sample_count, size=validation_size, replace=False)] = True
    targets = features.sum(axis=0, dtype=np.int64) * (validation_size / sample_count)
    counts = features[selected].sum(axis=0, dtype=np.int64)
    score = objective(counts, targets)
    selected_indices = np.flatnonzero(selected)
    train_indices = np.flatnonzero(~selected)

    for _ in range(iterations):
        selected_position = int(rng.integers(len(selected_indices)))
        train_position = int(rng.integers(len(train_indices)))
        old_index = int(selected_indices[selected_position])
        new_index = int(train_indices[train_position])
        candidate_counts = counts - features[old_index] + features[new_index]
        candidate_score = objective(candidate_counts, targets)
        if candidate_score < score:
            selected[old_index] = False
            selected[new_index] = True
            selected_indices[selected_position] = new_index
            train_indices[train_position] = old_index
            counts, score = candidate_counts, candidate_score
    return selected


def distribution(features: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    counts = features[mask].sum(axis=0, dtype=np.int64)
    size = int(mask.sum())
    return {
        "size": size,
        "feature_counts": dict(zip(FEATURE_NAMES, counts.tolist())),
        "feature_fractions": {
            name: int(count) / size for name, count in zip(FEATURE_NAMES, counts)
        },
    }


def write_manifest(
    path: Path,
    train_records: list[dict[str, object]],
    validation_mask: np.ndarray,
    official_validation_dir: Path,
    testdev_dir: Path,
) -> None:
    fields = ("image_name", "official_split", "experiment_split", *CLASS_NAMES)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, record in enumerate(train_records):
            row = {
                "image_name": record["image_name"],
                "official_split": "train",
                "experiment_split": "val" if validation_mask[index] else "train",
            }
            row.update(
                {name: int(value) for name, value in zip(CLASS_NAMES, record["present"])}
            )
            writer.writerow(row)
        for annotation_path in sorted(official_validation_dir.glob("*.json")):
            with annotation_path.open("r", encoding="utf-8") as annotation_file:
                annotation = json.load(annotation_file)
            present = np.zeros(len(CLASS_NAMES), dtype=np.bool_)
            for shape in annotation.get("shapes", ()):
                class_index = SOURCE_TO_CLASS.get(str(shape["label"]))
                if class_index is not None:
                    present[class_index] = True
            row = {
                "image_name": annotation["imageName"],
                "official_split": "validation",
                "experiment_split": "test",
            }
            row.update({name: int(value) for name, value in zip(CLASS_NAMES, present)})
            writer.writerow(row)
        for image_path in sorted(testdev_dir.glob("*.jpg")):
            row = {
                "image_name": image_path.name,
                "official_split": "testdev",
                "experiment_split": "unlabeled",
            }
            row.update({name: "" for name in CLASS_NAMES})
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the frozen DACL10K experiment split")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=200_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.data_root.resolve()
    records, features = read_presence(root / "annotations" / "train")
    validation_size = round(len(records) * args.validation_fraction)
    validation_mask = stratified_selection(
        features, validation_size, args.seed, args.iterations
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(
        args.output,
        records,
        validation_mask,
        root / "annotations" / "validation",
        root / "images" / "testdev",
    )
    report = {
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "optimization_iterations": args.iterations,
        "feature_names": list(FEATURE_NAMES),
        "official_train": distribution(features, np.ones(len(features), dtype=np.bool_)),
        "experiment_train": distribution(features, ~validation_mask),
        "experiment_val": distribution(features, validation_mask),
        "frozen_test_size": len(list((root / "annotations" / "validation").glob("*.json"))),
        "unlabeled_testdev_size": len(list((root / "images" / "testdev").glob("*.jpg"))),
    }
    report_path = args.output.with_suffix(".summary.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({"manifest": str(args.output), "summary": str(report_path), **{
        name: report[name]["size"] for name in ("experiment_train", "experiment_val")
    }, "frozen_test": report["frozen_test_size"], "unlabeled_testdev": report["unlabeled_testdev_size"]}, indent=2))


if __name__ == "__main__":
    main()
