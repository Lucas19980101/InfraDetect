"""Validate prepared component masks and add non-overlapping training splits."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


ALLOWED = {0, 1, 2, 255}


def finalize(dataset: Path, dom_split: Path, exclusion_file: Path | None = None,
             orientation_file: Path | None = None,
             orientation_accepted_file: Path | None = None) -> dict:
    with (dataset / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with dom_split.open(newline="", encoding="utf-8") as handle:
        dom_rows = {row["name"]: row for row in csv.DictReader(handle)}
    exclusions: dict[tuple[str, str], str] = {}
    if orientation_file is not None and orientation_file.exists():
        with orientation_file.open(newline="", encoding="utf-8") as handle:
            exclusions.update({(row["source"], row["name"]): row["reason"] for row in csv.DictReader(handle)})
    accepted: set[tuple[str, str]] = set()
    if orientation_accepted_file is not None and orientation_accepted_file.exists():
        with orientation_accepted_file.open(newline="", encoding="utf-8") as handle:
            accepted = {(row["source"], row["name"]) for row in csv.DictReader(handle)}
        for key in accepted:
            exclusions.pop(key, None)
    if exclusion_file is not None and exclusion_file.exists():
        with exclusion_file.open(newline="", encoding="utf-8") as handle:
            exclusions.update({(row["source"], row["name"]): row["reason"] for row in csv.DictReader(handle)})
    if len(rows) != 1517 or len({(row["source"], row["name"]) for row in rows}) != len(rows):
        raise ValueError("Expected 1517 unique prepared image/mask pairs")
    counts: Counter = Counter()
    pixel_counts: Counter = Counter()
    mismatches = []
    for row in rows:
        image_path, mask_path = Path(row["image"]), Path(row["mask"])
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing image or mask: {image_path}, {mask_path}")
        with Image.open(image_path) as image, Image.open(mask_path) as mask_image:
            if image.size != mask_image.size or mask_image.mode != "L":
                mismatches.append(row["name"])
            mask = np.asarray(mask_image)
        values, frequencies = np.unique(mask, return_counts=True)
        unexpected = set(values.tolist()) - ALLOWED
        if unexpected:
            raise ValueError(f"Unexpected mask IDs {unexpected}: {mask_path}")
        pixel_counts.update({int(key): int(value) for key, value in zip(values, frequencies)})
        if row["source"] == "3dom":
            split = dom_rows.get(row["name"])
            if split is None:
                raise ValueError(f"Missing original 3DOM split: {row['name']}")
            row["split"], row["group"] = split["split"], split["group"]
        else:
            row["split"], row["group"] = "train", row["source"]
        if (row["source"], row["name"]) in exclusions:
            row["split"] = "exclude"
        counts[(row["source"], row["split"])] += 1
    if mismatches:
        raise ValueError(f"Size/mode mismatches: {mismatches[:20]}")
    output = dataset / "manifest_with_splits.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("source", "name", "split", "group", "image",
                                                  "source_label", "mask", "building_surface", "opening",
                                                  "other", "ignore"))
        writer.writeheader(); writer.writerows(rows)
    result = {"pairs": len(rows), "sources_and_splits": {f"{source}/{split}": total
                                                     for (source, split), total in sorted(counts.items())},
              "pixel_counts": {str(key): value for key, value in sorted(pixel_counts.items())},
              "exclusions": {f"{source}/{name}": reason for (source, name), reason in sorted(exclusions.items())},
              "orientation_review_accepted": sorted(f"{source}/{name}" for source, name in accepted),
              "note": "Additional accepted sources are train-only; original 3DOM validation/test are unchanged. BFDD has no component labels."}
    (dataset / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/components_3class"))
    parser.add_argument("--dom-split", type=Path, default=Path("data/3dom_split.csv"))
    parser.add_argument("--exclusions", type=Path, default=Path("data/components_3class/exclusions.csv"))
    parser.add_argument("--orientation-review", type=Path,
                        default=Path("data/components_3class/hznu_orientation_review.csv"))
    parser.add_argument("--orientation-accepted", type=Path,
                        default=Path("data/components_3class/hznu_orientation_accepted.csv"))
    args = parser.parse_args()
    print(json.dumps(finalize(args.dataset, args.dom_split, args.exclusions,
                              args.orientation_review, args.orientation_accepted), indent=2))


if __name__ == "__main__":
    main()
