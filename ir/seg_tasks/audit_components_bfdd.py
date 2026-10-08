"""Sample BFDD validation RGB images to audit 3DOM component transfer."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from .data import COMPONENT_COLORS_RGB
from .inspect_pair import load_model, overlay, predict, save_png
from .train import config_from_file


def sample_rows(manifest: Path, count: int, split: str = "val") -> list[dict[str, str]]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    if not rows:
        raise ValueError(f"Manifest contains no {split} images")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    chosen: list[dict[str, str]] = []
    # Spread the fixed audit sample over capture groups, then across time in each group.
    for group in sorted(groups):
        values = groups[group]
        order = sorted(range(len(values)), key=lambda i: (abs(i - (len(values) - 1) / 2), i))
        groups[group] = [values[i] for i in order]
    while len(chosen) < min(count, len(rows)):
        progressed = False
        for group in sorted(groups):
            if groups[group] and len(chosen) < count:
                chosen.append(groups[group].pop(0))
                progressed = True
        if not progressed:
            break
    return chosen


def find_rgb(root: Path, name: str) -> Path:
    matches = [path for path in (root / "RGB").glob(f"{name}.*")
               if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one RGB image for {name}, found {matches}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/bfdd_split.csv"))
    parser.add_argument("--bfdd-root", type=Path, default=Path("../collected_datasets/Dataset_1x"))
    parser.add_argument("--component-config", type=Path, default=Path("configs/components_3dom.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/components_3dom/best.pt"))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--all", action="store_true", help="Process every image in the selected split")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output", type=Path, default=Path("runs/component_bfdd_audit"))
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    rows = sample_rows(args.manifest, 10**9 if args.all else args.count, args.split)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.component_config, args.checkpoint, "components", device)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    tiles = []
    names = cfg["data"]["class_names"]
    palette = (np.asarray(((0, 0, 255), (0, 255, 0), (255, 255, 0)), dtype=np.uint8)
               if names == ["building_surface", "opening", "other"]
               else np.asarray(COMPONENT_COLORS_RGB, dtype=np.uint8))
    for row in rows:
        name = row["name"]
        image_path = find_rgb(args.bfdd_root, name)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Could not read {image_path}")
        mask = predict(model, image, tuple(cfg["data"]["image_size"]), device)
        layer = overlay(image, mask, palette)
        save_png(args.output / f"{name}_component_ids.png", mask)
        save_png(args.output / f"{name}_overlay.png", layer)
        fractions = {label: round(float(np.mean(mask == index)), 4) for index, label in enumerate(names)}
        summary.append({"name": name, "group": row["group"], "rgb": str(image_path.resolve()),
                        "predicted_area_fraction": fractions})
        thumb_w, thumb_h = 320, 256
        left = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        right = cv2.resize(layer, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        tile = np.concatenate((left, right), axis=1)
        cv2.putText(tile, f"{name} group={row['group']}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    columns, per_page = 2, 20
    for page, start in enumerate(range(0, len(tiles), per_page), 1):
        page_tiles = tiles[start:start + per_page]
        blank = np.zeros_like(page_tiles[0])
        while len(page_tiles) % columns:
            page_tiles.append(blank)
        montage = np.vstack([np.hstack(page_tiles[i:i + columns])
                             for i in range(0, len(page_tiles), columns)])
        filename = "contact_sheet.jpg" if len(tiles) <= per_page else f"contact_sheet_{page:03d}.jpg"
        save_png(args.output / filename, montage)
    result = {"split": args.split, "checkpoint": str(args.checkpoint.resolve()), "samples": summary,
              "note": "Fractions describe predictions only; no BFDD component ground truth is available."}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {len(summary)} validation samples to {args.output}")


if __name__ == "__main__":
    main()
