#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Project RGB crack/spalling probabilities onto an existing COLMAP sparse model."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch

from rgb_multilabel.data import MEAN, STD
from rgb_multilabel.evaluate_building import load_model


CLASSES = ("crack", "spalling")
COLORS = np.asarray(((255, 55, 55), (255, 175, 35)), dtype=np.float32)


def sample_probabilities(
    probabilities: np.ndarray, coordinates: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinearly sample [C,H,W] maps at original-image pixel coordinates."""
    map_height, map_width = probabilities.shape[-2:]
    valid = (
        (coordinates[:, 0] >= 0)
        & (coordinates[:, 0] < width)
        & (coordinates[:, 1] >= 0)
        & (coordinates[:, 1] < height)
    )
    indices = np.flatnonzero(valid)
    if not len(indices):
        return indices, np.empty((0, probabilities.shape[0]), dtype=np.float32)
    xy = coordinates[indices]
    x = np.clip((xy[:, 0] + 0.5) * map_width / width - 0.5, 0, map_width - 1)
    y = np.clip((xy[:, 1] + 0.5) * map_height / height - 0.5, 0, map_height - 1)
    x0, y0 = np.floor(x).astype(np.intp), np.floor(y).astype(np.intp)
    x1, y1 = np.minimum(x0 + 1, map_width - 1), np.minimum(y0 + 1, map_height - 1)
    dx, dy = x - x0, y - y0
    values = (
        probabilities[:, y0, x0] * ((1 - dx) * (1 - dy))
        + probabilities[:, y0, x1] * (dx * (1 - dy))
        + probabilities[:, y1, x0] * ((1 - dx) * dy)
        + probabilities[:, y1, x1] * (dx * dy)
    )
    return indices, values.T.astype(np.float32)


def predict_probabilities(
    model: torch.nn.Module,
    bgr: np.ndarray,
    size: int,
    tile_size: int,
    device: torch.device,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    def predict(crop: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(cv2.resize(crop, (size, size)), cv2.COLOR_BGR2RGB)
        normalized = (rgb.astype(np.float32) / 255.0 - mean) / std
        tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))
        return model(tensor[None].to(device)).sigmoid()[0, :2].cpu().numpy()

    height, width = bgr.shape[:2]
    if not tile_size:
        return predict(bgr)
    if tile_size > min(width, height):
        raise ValueError("tile_size cannot exceed the smallest image dimension")
    stride = tile_size * 3 // 4
    x_starts = sorted(set(range(0, width - tile_size + 1, stride)) | {width - tile_size})
    y_starts = sorted(set(range(0, height - tile_size + 1, stride)) | {height - tile_size})
    sum_probabilities = np.zeros((2, height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.uint8)
    for y in y_starts:
        for x in x_starts:
            tile = bgr[y : y + tile_size, x : x + tile_size]
            probabilities = predict(tile)
            for index in range(2):
                full_tile = cv2.resize(probabilities[index], (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
                sum_probabilities[index, y : y + tile_size, x : x + tile_size] += full_tile
            counts[y : y + tile_size, x : x + tile_size] += 1
    if not np.all(counts):
        raise RuntimeError("Tiling left pixels uncovered")
    return sum_probabilities / counts[None]


def classify_votes(votes: np.ndarray, thresholds: np.ndarray, min_views: int) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(votes, axis=0)
    positives = (votes >= thresholds).sum(axis=0)
    selected = (len(votes) >= min_views) & (positives >= min_views) & (median >= thresholds)
    return median, selected


def write_ply(path: Path, rows: list[tuple[np.ndarray, np.ndarray]]) -> None:
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(rows)}\n")
        for field in ("x", "y", "z"):
            handle.write(f"property float {field}\n")
        for field in ("red", "green", "blue"):
            handle.write(f"property uchar {field}\n")
        handle.write("end_header\n")
        for xyz, color in rows:
            handle.write(
                f"{xyz[0]:.8g} {xyz[1]:.8g} {xyz[2]:.8g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def overlay_preview(image: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    width = min(image.shape[1], 1024)
    height = round(image.shape[0] * width / image.shape[1])
    preview = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    for index, color in enumerate(COLORS):
        probability = cv2.resize(probabilities[index], (width, height), interpolation=cv2.INTER_LINEAR)
        mask = probability >= thresholds[index]
        preview[mask] = (0.55 * preview[mask] + 0.45 * color[::-1]).astype(np.uint8)
    return preview


def run(args: argparse.Namespace) -> None:
    if args.min_views < 2:
        raise ValueError("min_views must be at least 2")
    image_dir = args.dataset / "images"
    reconstruction_dir = args.dataset / "sparse"
    if not image_dir.is_dir() or not reconstruction_dir.is_dir():
        raise FileNotFoundError(f"Expected images/ and sparse/ under {args.dataset}")
    if args.output.resolve() == args.dataset.resolve() or args.output.resolve().is_relative_to(args.dataset.resolve()):
        raise ValueError("Output must be outside the source dataset")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, checkpoint = load_model(args.checkpoint, device)
    size = int(checkpoint["config"]["data"]["crop_size"])
    thresholds = np.asarray([checkpoint["thresholds"][name] for name in CLASSES], dtype=np.float32)
    reconstruction = pycolmap.Reconstruction(reconstruction_dir)
    images = sorted(reconstruction.images.values(), key=lambda image: image.name)
    if not images:
        raise ValueError("The sparse model has no registered images")
    for image in images:
        if not (image_dir / image.name).is_file():
            raise FileNotFoundError(image_dir / image.name)

    args.output.mkdir(parents=True, exist_ok=True)
    probability_dir = args.output / "probabilities_512"
    preview_dir = args.output / "previews"
    observation_dir = args.output / "observations"
    probability_dir.mkdir(exist_ok=True)
    preview_dir.mkdir(exist_ok=True)
    observation_dir.mkdir(exist_ok=True)
    mean = np.asarray(MEAN, dtype=np.float32)
    std = np.asarray(STD, dtype=np.float32)

    with torch.inference_mode():
        selected_images = images[args.start : args.start + args.limit if args.limit else None]
        for number, image in enumerate(selected_images, args.start + 1):
            observation_path = observation_dir / f"{image.name}.npz"
            if observation_path.is_file():
                print(f"[{number}/{len(images)}] {image.name}: cached", flush=True)
                continue
            bgr = cv2.imread(str(image_dir / image.name), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Could not read {image.name}")
            height, width = bgr.shape[:2]
            camera = reconstruction.cameras[image.camera_id]
            if (width, height) != (camera.width, camera.height):
                raise ValueError(f"Camera/image size mismatch for {image.name}")
            probabilities = predict_probabilities(model, bgr, size, args.tile_size, device, mean, std)
            # uint16 retains sub-threshold confidence while keeping the output compact.
            encoded = np.rint(cv2.resize(probabilities.transpose(1, 2, 0), (size, size)) * 65535).astype(np.uint16)
            cv2.imwrite(str(probability_dir / f"{image.name}.crack.png"), encoded[:, :, 0])
            cv2.imwrite(str(probability_dir / f"{image.name}.spalling.png"), encoded[:, :, 1])
            cv2.imwrite(str(preview_dir / f"{image.name}.png"), overlay_preview(bgr, probabilities, thresholds))

            point_ids = np.asarray([point.point3D_id for point in image.points2D if point.has_point3D()], dtype=np.uint64)
            coordinates = np.asarray([point.xy for point in image.points2D if point.has_point3D()], dtype=np.float64).reshape(-1, 2)
            indices, values = sample_probabilities(probabilities, coordinates, width, height)
            np.savez_compressed(observation_path, point_ids=point_ids[indices], values=values)
            print(f"[{number}/{len(images)}] {image.name}: {len(indices)} observations", flush=True)

    missing = [image.name for image in images if not (observation_dir / f"{image.name}.npz").is_file()]
    if missing:
        print(f"Batch complete; {len(missing)} images remain. Resume with --start and --limit.")
        return
    votes: dict[int, list[np.ndarray]] = defaultdict(list)
    for image in images:
        with np.load(observation_dir / f"{image.name}.npz") as observations:
            for point_id, value in zip(observations["point_ids"], observations["values"]):
                votes[int(point_id)].append(value)
    rows = []
    ply_rows = []
    selected_count = {name: 0 for name in CLASSES}
    for point_id, point in sorted(reconstruction.points3D.items()):
        point_votes = np.asarray(votes.get(point_id, []), dtype=np.float32).reshape(-1, 2)
        if len(point_votes):
            median, selected = classify_votes(point_votes, thresholds, args.min_views)
            positive_counts = (point_votes >= thresholds).sum(axis=0)
        else:
            median = np.zeros(2, dtype=np.float32)
            selected = np.zeros(2, dtype=bool)
            positive_counts = np.zeros(2, dtype=int)
        for index, name in enumerate(CLASSES):
            selected_count[name] += int(selected[index])
        color = np.asarray(point.color, dtype=np.uint8)
        if selected.any():
            color = COLORS[int(np.argmax(np.where(selected, median, -1)))].astype(np.uint8)
        ply_rows.append((point.xyz, color))
        rows.append((
            point_id, *point.xyz, len(point_votes), *median, *positive_counts,
            int(selected[0]), int(selected[1]), float(point.error),
        ))

    with (args.output / "points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("point3D_id", "x", "y", "z", "valid_views", "crack_median", "spalling_median", "crack_positive_views", "spalling_positive_views", "crack", "spalling", "reprojection_error"))
        writer.writerows(rows)
    write_ply(args.output / "defects_sparse.ply", ply_rows)
    summary = {
        "dataset": str(args.dataset.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "architecture": checkpoint["config"]["model"]["architecture"],
        "images": len(images),
        "points3D": len(rows),
        "device": str(device),
        "saved_probability_resolution": [size, size],
        "tile_size": args.tile_size,
        "thresholds": dict(zip(CLASSES, map(float, thresholds))),
        "min_views": args.min_views,
        "selected_points": selected_count,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("../collected_datasets/south-building"))
    parser.add_argument("--checkpoint", type=Path, default=Path("../RGB/6ML/runs/segformer_b0_6ml_building/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("benchmark/reconstruction/runs/south-building-rgb"))
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--tile-size", type=int, default=0, help="Original-image tile size; 0 resizes whole image")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--start", type=int, default=0, help="First image index in sorted order")
    parser.add_argument("--limit", type=int, default=0, help="Maximum images this invocation processes; 0 means all")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
