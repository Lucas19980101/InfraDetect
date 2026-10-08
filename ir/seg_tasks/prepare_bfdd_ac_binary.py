"""Convert BFDD LabelMe AC polygons into a grouped binary segmentation dataset."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .audit_labelme_other import shape_points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path,
                        default=Path("../collected_datasets/Dataset_1x/LabelMe"))
    parser.add_argument("--bfdd-manifest", type=Path, default=Path("data/bfdd_split.csv"))
    parser.add_argument("--assets", type=Path, default=Path("data/bfdd_ac_binary"))
    parser.add_argument("--output", type=Path, default=Path("data/bfdd_ac_binary/manifest.csv"))
    args = parser.parse_args()

    with args.bfdd_manifest.open(newline="", encoding="utf-8-sig") as handle:
        bfdd = {row["name"]: row for row in csv.DictReader(handle)}
    masks = args.assets / "masks"
    masks.mkdir(parents=True, exist_ok=True)
    # Keep whole capture groups together. Groups 0/5 validate and group 38 tests.
    split_for_group = {"0": "val", "5": "val", "38": "test"}
    rows = []
    for annotation in sorted(args.annotations.glob("*.json")):
        name = annotation.stem
        if name not in bfdd:
            raise ValueError(f"{name} is absent from {args.bfdd_manifest}")
        data = json.loads(annotation.read_text(encoding="utf-8"))
        image = (annotation.parent / data["imagePath"]).resolve()
        rgb = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if rgb is None:
            raise OSError(f"Could not read {image}")
        mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        for index, shape in enumerate(data.get("shapes", [])):
            if str(shape.get("label", "")).strip() != "other":
                raise ValueError(f"{annotation.name} shape {index}: expected label 'other'")
            cv2.fillPoly(mask, [shape_points(shape)], 1)
        if not mask.any():
            raise ValueError(f"No AC pixels in {annotation}")
        mask_path = (masks / f"{name}.png").resolve()
        if not cv2.imwrite(str(mask_path), mask):
            raise OSError(f"Could not write {mask_path}")
        group = bfdd[name]["group"].strip()
        rows.append({"source": "bfdd_ac_binary", "name": name,
                     "split": split_for_group.get(group, "train"),
                     "group": f"bfdd/{group}", "image": str(image), "mask": str(mask_path),
                     "background": int((mask == 0).sum()), "air_conditioner": int(mask.sum())})

    fields = ["source", "name", "split", "group", "image", "mask",
              "background", "air_conditioner"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    splits = Counter(row["split"] for row in rows)
    pixels = {split: {key: sum(int(row[key]) for row in rows if row["split"] == split)
                      for key in ("background", "air_conditioner")}
              for split in sorted(splits)}
    print(json.dumps({"samples": len(rows), "splits": dict(sorted(splits.items())),
                      "pixels": pixels, "manifest": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
