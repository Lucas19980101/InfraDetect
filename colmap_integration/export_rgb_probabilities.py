#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Export six-class RGB probability maps for COLMAP 2D-to-3D fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from rgb_multilabel.data import MEAN, STD
from rgb_multilabel.evaluate_building import load_model
from rgb_multilabel.labels import CLASS_NAMES

from compare_rgb_models import model_logits


def predict_image(model: torch.nn.Module, bgr: np.ndarray, size: int,
                  tile_size: int, device: torch.device) -> np.ndarray:
    """Return [C,H,W] probabilities aligned to the original image."""
    mean = np.asarray(MEAN, dtype=np.float32)
    std = np.asarray(STD, dtype=np.float32)

    def predict_crop(crop: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(cv2.resize(crop, (size, size)), cv2.COLOR_BGR2RGB)
        normalized = (rgb.astype(np.float32) / 255 - mean) / std
        tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))
        return model_logits(model, tensor[None].to(device)).sigmoid()[0].cpu().numpy()

    height, width = bgr.shape[:2]
    if not tile_size:
        small = predict_crop(bgr)
        return np.stack([cv2.resize(layer, (width, height), interpolation=cv2.INTER_LINEAR)
                         for layer in small])
    if tile_size > min(width, height) or tile_size < 1:
        raise ValueError("tile_size must fit inside each image")
    stride = max(1, tile_size * 3 // 4)
    x_starts = sorted(set(range(0, width - tile_size + 1, stride)) | {width - tile_size})
    y_starts = sorted(set(range(0, height - tile_size + 1, stride)) | {height - tile_size})
    totals = np.zeros((len(CLASS_NAMES), height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.uint8)
    for y in y_starts:
        for x in x_starts:
            layers = predict_crop(bgr[y:y + tile_size, x:x + tile_size])
            for index, layer in enumerate(layers):
                totals[index, y:y + tile_size, x:x + tile_size] += cv2.resize(
                    layer, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR
                )
            counts[y:y + tile_size, x:x + tile_size] += 1
    if not np.all(counts):
        raise RuntimeError("Tiling left pixels uncovered")
    return totals / counts[None]


def run(args: argparse.Namespace) -> dict:
    if args.limit_images < 0 or args.tile_size < 0:
        raise ValueError("Image limit and tile size must be nonnegative")
    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    size = int(checkpoint["config"]["data"]["crop_size"])
    images = sorted(path for path in args.images.iterdir()
                    if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    if not images:
        raise ValueError(f"No images in {args.images}")
    selected = images[:args.limit_images] if args.limit_images else images
    output = args.output / "probabilities_512"
    output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "summary.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        if (previous.get("checkpoint") != str(args.checkpoint.resolve())
                or previous.get("tile_size") != args.tile_size
                or previous.get("input_images") != str(args.images.resolve())):
            raise ValueError("Output folder belongs to a different checkpoint, tile size, or image set")
    with torch.inference_mode():
        for number, path in enumerate(selected, 1):
            targets = [output / f"{path.name}.{name}.png" for name in CLASS_NAMES]
            if all(target.is_file() for target in targets):
                print(f"[{number}/{len(selected)}] {path.name}: cached", flush=True)
                continue
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Could not decode {path}")
            probabilities = predict_image(model, bgr, size, args.tile_size, device)
            for index, target in enumerate(targets):
                resized = cv2.resize(probabilities[index], (size, size), interpolation=cv2.INTER_AREA)
                encoded = np.rint(np.clip(resized, 0, 1) * 65535).astype(np.uint16)
                if not cv2.imwrite(str(target), encoded):
                    raise OSError(f"Could not write {target}")
            print(f"[{number}/{len(selected)}] {path.name}: six maps", flush=True)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "architecture": checkpoint["config"]["model"]["architecture"],
        "input_images": str(args.images.resolve()),
        "images_total": len(images),
        "images_selected": len(selected),
        "tile_size": args.tile_size,
        "saved_probability_resolution": [size, size],
        "classes": list(CLASS_NAMES),
        "thresholds": {name: float(checkpoint["thresholds"][name]) for name in CLASS_NAMES},
        "complete": all((output / f"{path.name}.{name}.png").is_file()
                        for path in images for name in CLASS_NAMES),
    }
    summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=Path("../collected_datasets/south-building/images"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--limit-images", type=int, default=0)
    report = run(parser.parse_args())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
