"""Merge LabelMe AC exclusions into component pseudo-masks and write a manifest."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from .audit_labelme_other import shape_points
from .inspect_pair import load_model, predict, save_png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path,
                        default=Path("data/components_3class/manifest_bfdd_adaptation.csv"))
    parser.add_argument("--bfdd-manifest", type=Path, default=Path("data/bfdd_split.csv"))
    parser.add_argument("--bfdd-root", type=Path,
                        default=Path("../collected_datasets/Dataset_1x"))
    parser.add_argument("--annotations", type=Path,
                        default=Path("../collected_datasets/Dataset_1x/LabelMe"))
    parser.add_argument("--pseudo-masks", type=Path,
                        default=Path("runs/components_bfdd_adapted/bfdd_rgb_audit"))
    parser.add_argument("--component-config", type=Path,
                        default=Path("configs/components_bfdd_adapted.yaml"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("runs/components_bfdd_adapted/best.pt"))
    parser.add_argument("--assets", type=Path, default=Path("runs/components_bfdd_ac/assets"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/components_3class/manifest_bfdd_ac.csv"))
    args = parser.parse_args()

    with args.base.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    if fields is None:
        raise ValueError(f"Empty base manifest: {args.base}")
    with args.bfdd_manifest.open(newline="", encoding="utf-8-sig") as handle:
        bfdd = {row["name"]: row for row in csv.DictReader(handle)}

    annotations = sorted(args.annotations.glob("*.json"))
    names = {path.stem for path in annotations}
    missing = names - set(bfdd)
    if missing:
        raise ValueError(f"Annotations missing from BFDD manifest: {sorted(missing)}")
    # Replace any earlier pseudo-label row for the same image instead of duplicating it.
    rows = [row for row in rows if not (row["source"] == "bfdd_rgb" and row["name"] in names)]
    masks_dir, panels_dir = args.assets / "masks", args.assets / "panels"
    masks_dir.mkdir(parents=True, exist_ok=True); panels_dir.mkdir(parents=True, exist_ok=True)
    args.pseudo_masks.mkdir(parents=True, exist_ok=True)

    missing_pseudo = [path.stem for path in annotations
                      if not (args.pseudo_masks / f"{path.stem}_component_ids.png").exists()]
    model = cfg = device = None
    if missing_pseudo:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, cfg = load_model(args.component_config, args.checkpoint, "components", device)
        print(f"Generating {len(missing_pseudo)} missing pseudo-masks on {device}")

    # AC validation uses groups 0 and 5; group 38 remains the component test group.
    split_for_group = {"0": "val", "5": "val", "38": "test"}
    records = []
    for annotation in annotations:
        name = annotation.stem
        data = json.loads(annotation.read_text(encoding="utf-8"))
        image = (annotation.parent / data["imagePath"]).resolve()
        pseudo_path = args.pseudo_masks / f"{name}_component_ids.png"
        pseudo = cv2.imread(str(pseudo_path), cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if rgb is None:
            raise ValueError(f"Could not read image for {name}")
        if pseudo is None and model is not None and cfg is not None and device is not None:
            pseudo = predict(model, rgb, tuple(cfg["data"]["image_size"]), device)
            save_png(pseudo_path, pseudo)
        if pseudo is None or pseudo.shape != rgb.shape[:2]:
            raise ValueError(f"Missing or size-mismatched image/pseudo-mask for {name}")
        invalid = set(map(int, np.unique(pseudo))) - {0, 1, 2, 255}
        if invalid:
            raise ValueError(f"Invalid pseudo-mask values for {name}: {sorted(invalid)}")
        other = np.zeros(pseudo.shape, dtype=np.uint8)
        for index, shape in enumerate(data.get("shapes", [])):
            if str(shape.get("label", "")).strip() != "other":
                raise ValueError(f"{annotation.name} shape {index} is not labelled other")
            cv2.fillPoly(other, [shape_points(shape)], 1)
        if not other.any():
            raise ValueError(f"No labelled pixels in {annotation}")
        merged = pseudo.copy(); merged[other > 0] = 2
        mask_path = (masks_dir / f"{name}.png").resolve()
        cv2.imwrite(str(mask_path), merged)

        before, after = rgb.copy(), rgb.copy()
        old_other, new_other = pseudo == 2, merged == 2
        before[old_other] = cv2.addWeighted(rgb, .45, np.full_like(rgb, (0, 255, 255)), .55, 0)[old_other]
        after[new_other] = cv2.addWeighted(rgb, .45, np.full_like(rgb, (0, 255, 255)), .55, 0)[new_other]
        panel = np.hstack([before, after])
        cv2.putText(panel, "before", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
        cv2.putText(panel, "after LabelMe other", (rgb.shape[1] + 12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
        cv2.imwrite(str(panels_dir / f"{name}.png"), panel)

        group = bfdd[name]["group"].strip()
        split = split_for_group.get(group, "train")
        counts = {label: int((merged == value).sum()) for value, label in
                  ((0, "building_surface"), (1, "opening"), (2, "other"), (255, "ignore"))}
        rows.append({"source": "bfdd_ac", "name": name, "split": split,
                     "group": f"bfdd/{group}", "image": str(image),
                     "source_label": str(annotation.resolve()), "mask": str(mask_path), **counts})
        records.append({"name": name, "group": group, "split": split,
                        "new_other_pixels": int(((other > 0) & (pseudo != 2)).sum()), **counts})

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    (args.assets / "summary.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    counts = Counter(row["split"] for row in records)
    print(json.dumps({"annotations": len(records), "splits": dict(sorted(counts.items())),
                      "manifest": str(args.output.resolve()),
                      "panels": str(panels_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
