"""Build AC full-image plus positive/hard-negative crop training data."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def expanded_crop(x: int, y: int, w: int, h: int, image_w: int, image_h: int,
                  scale: float, minimum: int) -> tuple[int, int, int, int]:
    crop_w = min(image_w, max(minimum, int(round(w * scale))))
    crop_h = min(image_h, max(minimum, int(round(h * scale))))
    cx, cy = x + w / 2, y + h / 2
    x1 = max(0, min(image_w - crop_w, int(round(cx - crop_w / 2))))
    y1 = max(0, min(image_h - crop_h, int(round(cy - crop_h / 2))))
    return x1, y1, x1 + crop_w, y1 + crop_h


def components(mask: np.ndarray, minimum_area: int) -> list[tuple[int, int, int, int, int]]:
    count, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    values = [tuple(map(int, stats[index])) for index in range(1, count)
              if int(stats[index, cv2.CC_STAT_AREA]) >= minimum_area]
    return sorted(values, key=lambda value: value[4], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path,
                        default=Path("data/bfdd_ac_binary/manifest.csv"))
    parser.add_argument("--component-masks", type=Path,
                        default=Path("runs/components_bfdd_adapted/bfdd_rgb_audit"))
    parser.add_argument("--assets", type=Path, default=Path("data/bfdd_ac_patches"))
    parser.add_argument("--output", type=Path, default=Path("data/bfdd_ac_patches/manifest.csv"))
    parser.add_argument("--minimum-crop-size", type=int, default=160)
    parser.add_argument("--positive-scale", type=float, default=1.6)
    parser.add_argument("--negative-scale", type=float, default=1.3)
    parser.add_argument("--negative-crops-per-image", type=int, default=2)
    args = parser.parse_args()

    with args.base_manifest.open(newline="", encoding="utf-8") as handle:
        base = list(csv.DictReader(handle))
    image_dir, mask_dir = args.assets / "images", args.assets / "masks"
    image_dir.mkdir(parents=True, exist_ok=True); mask_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    def append(source: str, name: str, split: str, group: str,
               image_path: Path, mask_path: Path, mask: np.ndarray) -> None:
        rows.append({"source": source, "name": name, "split": split, "group": group,
                     "image": str(image_path.resolve()), "mask": str(mask_path.resolve()),
                     "background": int((mask == 0).sum()),
                     "air_conditioner": int((mask == 1).sum())})

    for row in base:
        image_path, full_mask_path = Path(row["image"]), Path(row["mask"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(full_mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None or image.shape[:2] != mask.shape:
            raise ValueError(f"Invalid full pair: {image_path}, {full_mask_path}")
        append("bfdd_ac_full", row["name"], row["split"], row["group"],
               image_path, full_mask_path, mask)
        if row["split"] != "train":
            continue

        positive_boxes = components(mask, minimum_area=20)
        for index, (x, y, w, h, _) in enumerate(positive_boxes):
            x1, y1, x2, y2 = expanded_crop(x, y, w, h, image.shape[1], image.shape[0],
                                            args.positive_scale, args.minimum_crop_size)
            crop_image, crop_mask = image[y1:y2, x1:x2], mask[y1:y2, x1:x2]
            name = f"{row['name']}__pos{index:02d}"
            out_image, out_mask = image_dir / f"{name}.png", mask_dir / f"{name}.png"
            cv2.imwrite(str(out_image), crop_image); cv2.imwrite(str(out_mask), crop_mask)
            append("bfdd_ac_positive_crop", name, "train", row["group"],
                   out_image, out_mask, crop_mask)

        component_path = args.component_masks / f"{row['name']}_component_ids.png"
        component = cv2.imread(str(component_path), cv2.IMREAD_UNCHANGED)
        if component is None or component.shape != mask.shape:
            raise ValueError(f"Missing component pseudo-mask for hard negatives: {component_path}")
        negatives = components(component == 1, minimum_area=500)
        made = 0
        for x, y, w, h, _ in negatives:
            if made >= args.negative_crops_per_image:
                break
            x1, y1, x2, y2 = expanded_crop(x, y, w, h, image.shape[1], image.shape[0],
                                            args.negative_scale, args.minimum_crop_size)
            if mask[y1:y2, x1:x2].any():
                continue
            crop_image = image[y1:y2, x1:x2]
            crop_mask = np.zeros(crop_image.shape[:2], dtype=np.uint8)
            name = f"{row['name']}__neg{made:02d}"
            out_image, out_mask = image_dir / f"{name}.png", mask_dir / f"{name}.png"
            cv2.imwrite(str(out_image), crop_image); cv2.imwrite(str(out_mask), crop_mask)
            append("bfdd_ac_negative_crop", name, "train", row["group"],
                   out_image, out_mask, crop_mask)
            made += 1

    fields = ["source", "name", "split", "group", "image", "mask",
              "background", "air_conditioner"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    report = {"rows": len(rows), "by_source": dict(Counter(row["source"] for row in rows)),
              "by_split": dict(Counter(row["split"] for row in rows)),
              "manifest": str(args.output.resolve())}
    (args.assets / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
