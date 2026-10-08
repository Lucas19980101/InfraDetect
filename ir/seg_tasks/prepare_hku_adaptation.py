"""Convert reviewed HKU LabelMe annotations into component and AC masks."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .audit_labelme_other import shape_points


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_shapes(path: Path, allowed: set[str]) -> list[dict]:
    if not path.exists():
        return []
    shapes = json.loads(path.read_text(encoding="utf-8")).get("shapes", [])
    unknown = sorted({str(shape.get("label", "")).strip() for shape in shapes} - allowed)
    if unknown:
        raise ValueError(f"Unexpected labels in {path}: {unknown}")
    return shapes


def fill(mask: np.ndarray, shapes: list[dict], label: str, value: int) -> None:
    for shape in shapes:
        if str(shape.get("label", "")).strip() == label:
            cv2.fillPoly(mask, [shape_points(shape)], value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/hku_adaptation"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/hku_adaptation/prepared"))
    parser.add_argument("--component-base", type=Path,
                        default=Path("data/components_3class/manifest_bfdd_adaptation.csv"))
    parser.add_argument("--ac-base", type=Path,
                        default=Path("data/bfdd_ac_patches/manifest.csv"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    component_rows, ac_rows = [], []
    for split in ("train", "val"):
        ids = read_ids(args.root / f"{split}_ids.txt")
        component_dir = args.root / "component_annotations" / split
        ac_dir = args.root / "ac_annotations" / split
        component_out = args.output / "component_masks" / split
        ac_out = args.output / "ac_masks" / split
        component_out.mkdir(parents=True, exist_ok=True)
        ac_out.mkdir(parents=True, exist_ok=True)
        for name in ids:
            image_path = component_dir / f"{name}.png"
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(image_path)
            component_json = component_dir / f"{name}.json"
            ac_json = ac_dir / f"{name}.json"
            component_shapes = load_shapes(component_json, {"opening", "other"})
            ac_shapes = load_shapes(ac_json, {"air_conditioner"})

            # Unmarked pixels are reviewed building surface. Openings are 1;
            # other objects and independently annotated AC units are 2.
            component = np.zeros(image.shape[:2], np.uint8)
            fill(component, component_shapes, "opening", 1)
            fill(component, component_shapes, "other", 2)
            fill(component, ac_shapes, "air_conditioner", 2)
            ac = np.zeros(image.shape[:2], np.uint8)
            fill(ac, ac_shapes, "air_conditioner", 1)
            component_path = component_out / f"{name}.png"
            ac_path = ac_out / f"{name}.png"
            if not cv2.imwrite(str(component_path), component):
                raise OSError(component_path)
            if not cv2.imwrite(str(ac_path), ac):
                raise OSError(ac_path)
            counts = np.bincount(component.ravel(), minlength=3)
            component_rows.append({
                "source": "hku_adapt", "name": name, "split": split,
                "group": f"hku_{name}", "image": str(image_path.resolve()),
                "source_label": str(component_json.resolve()) if component_json.exists() else "",
                "mask": str(component_path.resolve()),
                "building_surface": int(counts[0]), "opening": int(counts[1]),
                "other": int(counts[2]), "ignore": 0,
            })
            ac_rows.append({
                "source": "hku_ac", "name": name, "split": split,
                "group": f"hku_{name}", "image": str(image_path.resolve()),
                "source_label": str(ac_json.resolve()) if ac_json.exists() else "",
                "mask": str(ac_path.resolve()), "background": int((ac == 0).sum()),
                "air_conditioner": int(ac.sum()),
            })

    for filename, rows in (("component_manifest.csv", component_rows),
                           ("ac_manifest.csv", ac_rows)):
        with (args.output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def merge(base_path: Path, added: list[dict], output_name: str) -> None:
        with base_path.open(newline="", encoding="utf-8") as handle:
            base = list(csv.DictReader(handle))
        fields = list(base[0]) + [key for key in added[0] if key not in base[0]]
        with (args.output / output_name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(base)
            writer.writerows(added)

    merge(args.component_base, component_rows, "component_manifest_combined.csv")
    merge(args.ac_base, ac_rows, "ac_manifest_combined.csv")
    summary = {
        "component_samples": len(component_rows),
        "ac_samples": len(ac_rows),
        "splits": {split: sum(row["split"] == split for row in component_rows)
                   for split in ("train", "val")},
        "component_pixels": {key: sum(row[key] for row in component_rows)
                             for key in ("building_surface", "opening", "other")},
        "ac_positive_samples": sum(row["air_conditioner"] > 0 for row in ac_rows),
        "ac_pixels": sum(row["air_conditioner"] for row in ac_rows),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
