#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Fuse per-image defect probabilities into a COLMAP dense point cloud.

Probability PNGs are aligned with the *original* (distorted) photographs. The
geometric depth maps and dense reconstruction are aligned with undistorted
photographs. Each 3D point is projected into both camera models; a depth test
determines whether the point is visible before its original-image probability
is counted as a vote.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap


ALL_CLASSES = (
    "crack", "spalling", "exposed_rebar", "rust", "efflorescence", "wetspot"
)
PALETTE = np.asarray(
    ((255, 55, 55), (255, 175, 35), (240, 50, 210),
     (190, 85, 20), (245, 245, 90), (45, 125, 255)),
    dtype=np.uint8,
)
PLY_FIELDS = (
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
)


def read_colmap_ply(path: Path, limit: int = 0) -> np.memmap:
    """Memory-map the exact binary PLY layout emitted by stereo_fusion."""
    with path.open("rb") as handle:
        header = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"PLY header has no end_header: {path}")
            header.append(line.decode("ascii").strip())
            if header[-1] == "end_header":
                break
        offset = handle.tell()
    if header[:2] != ["ply", "format binary_little_endian 1.0"]:
        raise ValueError("Only COLMAP binary little-endian PLY is supported")
    vertex_line = next((line for line in header if line.startswith("element vertex ")), None)
    if vertex_line is None or any(line.startswith("element ") and not line.startswith("element vertex ") for line in header):
        raise ValueError("PLY must contain only vertices")
    expected = [f"property {'float' if kind == '<f4' else 'uchar'} {name}" for name, kind in PLY_FIELDS]
    actual = [line for line in header if line.startswith("property ")]
    if actual != expected:
        raise ValueError(f"Unexpected PLY fields: {actual}")
    count = int(vertex_line.rsplit(" ", 1)[1])
    if limit:
        count = min(count, limit)
    return np.memmap(path, dtype=np.dtype(list(PLY_FIELDS)), mode="r", offset=offset, shape=(count,))


def project_pinhole(camera: pycolmap.Camera, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame XYZ using an undistorted PINHOLE camera."""
    if str(camera.model) != "PINHOLE":
        raise ValueError(f"Dense camera must be PINHOLE, got {camera.model}")
    fx, fy, cx, cy = camera.params
    z = xyz[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * xyz[:, 0] / z + cx
        v = fy * xyz[:, 1] / z + cy
    return u, v


def project_original(camera: pycolmap.Camera, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized projection for common COLMAP perspective camera models."""
    params = camera.params
    model = str(camera.model)
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fx = fy = params[0]
        cx, cy = params[1:3]
        distortion = params[3:]
    elif model in ("PINHOLE", "OPENCV"):
        fx, fy, cx, cy = params[:4]
        distortion = params[4:]
    else:
        raise ValueError(f"Original camera model {model} is not implemented")
    with np.errstate(divide="ignore", invalid="ignore"):
        x = xyz[:, 0] / xyz[:, 2]
        y = xyz[:, 1] / xyz[:, 2]
    r2 = x * x + y * y
    if model == "SIMPLE_RADIAL":
        scale = 1 + distortion[0] * r2
        x, y = x * scale, y * scale
    elif model == "RADIAL":
        scale = 1 + distortion[0] * r2 + distortion[1] * r2 * r2
        x, y = x * scale, y * scale
    elif model == "OPENCV":
        k1, k2, p1, p2 = distortion
        scale = 1 + k1 * r2 + k2 * r2 * r2
        x, y = (x * scale + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                y * scale + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)
    return fx * x + cx, fy * y + cy


def sample_probability(map_image: np.ndarray, u: np.ndarray, v: np.ndarray,
                       image_width: int, image_height: int) -> np.ndarray:
    """Bilinear sample a uint16 probability map scaled from the source photo."""
    map_height, map_width = map_image.shape
    x = np.clip((u + 0.5) * map_width / image_width - 0.5, 0, map_width - 1)
    y = np.clip((v + 0.5) * map_height / image_height - 0.5, 0, map_height - 1)
    x0 = np.floor(x).astype(np.intp)
    y0 = np.floor(y).astype(np.intp)
    x1 = np.minimum(x0 + 1, map_width - 1)
    y1 = np.minimum(y0 + 1, map_height - 1)
    dx, dy = x - x0, y - y0
    values = ((1 - dx) * (1 - dy) * map_image[y0, x0]
              + dx * (1 - dy) * map_image[y0, x1]
              + (1 - dx) * dy * map_image[y1, x0]
              + dx * dy * map_image[y1, x1])
    return (values / 65535.0).astype(np.float32)


def visible_indices(points: np.ndarray, pose: np.ndarray, camera: pycolmap.Camera,
                    depth: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """Return point indexes visible in a geometric depth map and camera XYZ."""
    xyz = points @ pose[:3, :3].T + pose[:3, 3]
    u, v = project_pinhole(camera, xyz)
    height, width = depth.shape
    in_frame = (np.isfinite(u) & np.isfinite(v) & (xyz[:, 2] > 0)
                & (u >= 0) & (v >= 0) & (u < width) & (v < height))
    indexes = np.flatnonzero(in_frame)
    if not len(indexes):
        return indexes, xyz[indexes]
    # Depth maps are defined at integer pixel centers; nearest-neighbor avoids
    # interpolating across foreground/background discontinuities.
    ix = np.clip(np.rint(u[indexes]).astype(np.intp), 0, width - 1)
    iy = np.clip(np.rint(v[indexes]).astype(np.intp), 0, height - 1)
    expected = depth[iy, ix]
    good = (np.isfinite(expected) & (expected > 0)
            & (np.abs(xyz[indexes, 2] - expected) <= tolerance * expected))
    return indexes[good], xyz[indexes[good]]


def fuse_labels(probability_sum: np.ndarray, positive_sum: np.ndarray, positive_count: np.ndarray,
                valid_count: np.ndarray, thresholds: np.ndarray, min_views: int,
                min_fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Retain every accepted class; choose one primary class for display."""
    support_scores = np.divide(positive_sum, positive_count,
                               out=np.zeros_like(positive_sum), where=positive_count > 0)
    mean_probabilities = np.divide(probability_sum, valid_count[:, None],
                                   out=np.zeros_like(probability_sum), where=valid_count[:, None] > 0)
    fraction = np.divide(positive_count, valid_count[:, None],
                         out=np.zeros_like(support_scores), where=valid_count[:, None] > 0)
    accepted = ((positive_count >= min_views) & (fraction >= min_fraction)
                & (support_scores >= thresholds[None, :]))
    priority = np.where(accepted, support_scores * fraction, -1.0)
    label = np.where(accepted.any(axis=1), priority.argmax(axis=1) + 1, 0).astype(np.uint8)
    return label, accepted, mean_probabilities, support_scores


def write_labeled_ply(path: Path, source: np.memmap, labels: np.ndarray,
                      accepted: np.ndarray, mean_probabilities: np.ndarray, support_scores: np.ndarray,
                      positive_count: np.ndarray, valid_count: np.ndarray,
                      classes: tuple[str, ...], chunk_size: int) -> None:
    dtype = np.dtype([*PLY_FIELDS, ("label", "u1"), ("label_bits", "u1"),
                      ("valid_views", "<u2"),
                      *((f"is_{name}", "u1") for name in classes),
                      *((f"prob_{name}", "<f4") for name in classes),
                      *((f"support_{name}", "<f4") for name in classes),
                      *((f"positive_views_{name}", "<u2") for name in classes)])
    with path.open("wb") as handle:
        lines = ["ply", "format binary_little_endian 1.0", f"element vertex {len(source)}"]
        for name, kind in PLY_FIELDS:
            lines.append(f"property {'float' if kind == '<f4' else 'uchar'} {name}")
        lines.extend(("property uchar label", "property uchar label_bits",
                      "property ushort valid_views"))
        lines.extend(f"property uchar is_{name}" for name in classes)
        lines.extend(f"property float prob_{name}" for name in classes)
        lines.extend(f"property float support_{name}" for name in classes)
        lines.extend(f"property ushort positive_views_{name}" for name in classes)
        lines.append("end_header")
        handle.write(("\n".join(lines) + "\n").encode("ascii"))
        for start in range(0, len(source), chunk_size):
            stop = min(start + chunk_size, len(source))
            block = np.empty(stop - start, dtype=dtype)
            for name, _ in PLY_FIELDS:
                block[name] = source[name][start:stop]
            block["label"] = labels[start:stop]
            block["label_bits"] = 0
            block["valid_views"] = valid_count[start:stop]
            for index, name in enumerate(classes):
                class_accepted = accepted[start:stop, index]
                block[f"is_{name}"] = class_accepted.astype(np.uint8)
                block["label_bits"] |= class_accepted.astype(np.uint8) << index
                block[f"prob_{name}"] = mean_probabilities[start:stop, index]
                block[f"support_{name}"] = support_scores[start:stop, index]
                block[f"positive_views_{name}"] = positive_count[start:stop, index]
            colors = np.column_stack((block["red"], block["green"], block["blue"]))
            for index, name in enumerate(classes):
                colors[labels[start:stop] == index + 1] = PALETTE[ALL_CLASSES.index(name)]
            block["red"], block["green"], block["blue"] = colors.T
            handle.write(block.tobytes())


def run(args: argparse.Namespace) -> dict:
    classes = tuple(args.classes)
    if not classes or len(set(classes)) != len(classes) or any(name not in ALL_CLASSES for name in classes):
        raise ValueError(f"Classes must be unique members of {ALL_CLASSES}")
    if args.min_views < 1 or not 0 <= args.min_fraction <= 1 or args.depth_tolerance <= 0:
        raise ValueError("Invalid fusion thresholds")
    if args.chunk_size < 1 or args.limit_points < 0 or args.limit_images < 0:
        raise ValueError("Chunk size must be positive; limits cannot be negative")
    if args.output.resolve() == args.ply.resolve():
        raise ValueError("Output must not overwrite the original dense PLY")

    original = pycolmap.Reconstruction(args.dataset / "sparse")
    dense = pycolmap.Reconstruction(args.workspace / "sparse")
    source = read_colmap_ply(args.ply, args.limit_points)
    images = sorted(dense.images.values(), key=lambda image: image.name)
    if args.limit_images:
        images = images[:args.limit_images]
    if not images:
        raise ValueError("No registered dense images")
    summary_path = args.probabilities.parent / "summary.json"
    previous = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    defaults = previous.get("thresholds", {})
    thresholds = np.asarray([
        args.threshold if args.threshold is not None else defaults.get(name, 0.5)
        for name in classes
    ], dtype=np.float32)
    if np.any((thresholds < 0) | (thresholds > 1)):
        raise ValueError("Probability thresholds must be in [0, 1]")
    count = len(source)
    probability_sum = np.zeros((count, len(classes)), dtype=np.float32)
    positive_sum = np.zeros((count, len(classes)), dtype=np.float32)
    positive_count = np.zeros((count, len(classes)), dtype=np.uint16)
    valid_count = np.zeros(count, dtype=np.uint16)
    diagnostics = []

    for ordinal, image in enumerate(images, 1):
        if image.image_id not in original.images or original.images[image.image_id].name != image.name:
            raise ValueError(f"Dense/original reconstruction mismatch: {image.name}")
        orig_image = original.images[image.image_id]
        orig_camera = original.cameras[orig_image.camera_id]
        dense_camera = dense.cameras[image.camera_id]
        depth_path = args.workspace / "stereo" / "depth_maps" / f"{image.name}.geometric.bin"
        if not depth_path.is_file():
            raise FileNotFoundError(depth_path)
        depth_map = pycolmap.DepthMap()
        depth_map.read(depth_path)
        depth = depth_map.to_array()
        if depth.shape != (dense_camera.height, dense_camera.width):
            raise ValueError(f"Depth/camera size mismatch for {image.name}")
        del depth_map
        maps = []
        for name in classes:
            path = args.probabilities / f"{image.name}.{name}.png"
            probability = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if probability is None or probability.dtype != np.uint16 or probability.ndim != 2:
                raise ValueError(f"Expected 16-bit grayscale probability PNG: {path}")
            maps.append(probability)
        pose = np.asarray(image.cam_from_world().matrix(), dtype=np.float64)
        original_pose = np.asarray(orig_image.cam_from_world().matrix(), dtype=np.float64)
        if not np.allclose(pose, original_pose, rtol=0, atol=1e-5):
            raise ValueError(f"Original/dense coordinate frames differ for {image.name}")
        seen = 0
        for start in range(0, count, args.chunk_size):
            stop = min(start + args.chunk_size, count)
            points = np.column_stack(tuple(source[field][start:stop] for field in ("x", "y", "z")))
            visible, xyz = visible_indices(points, pose, dense_camera, depth, args.depth_tolerance)
            if not len(visible):
                continue
            u, v = project_original(orig_camera, xyz)
            in_original = (np.isfinite(u) & np.isfinite(v) & (u >= 0) & (v >= 0)
                           & (u < orig_camera.width) & (v < orig_camera.height))
            absolute = start + visible[in_original]
            if not len(absolute):
                continue
            u, v = u[in_original], v[in_original]
            valid_count[absolute] += 1
            seen += len(absolute)
            for index, probability in enumerate(maps):
                values = sample_probability(probability, u, v, orig_camera.width, orig_camera.height)
                probability_sum[absolute, index] += values
                positive = values >= thresholds[index]
                selected = absolute[positive]
                positive_sum[selected, index] += values[positive]
                positive_count[selected, index] += 1
        diagnostics.append({"image": image.name, "visible_points": seen})
        print(f"[{ordinal}/{len(images)}] {image.name}: {seen} visible points", flush=True)

    labels, accepted, mean_probabilities, support_scores = fuse_labels(
        probability_sum, positive_sum, positive_count, valid_count,
        thresholds, args.min_views, args.min_fraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_labeled_ply(args.output, source, labels, accepted, mean_probabilities,
                      support_scores, positive_count, valid_count, classes, args.chunk_size)
    report = {
        "source_ply": str(args.ply.resolve()),
        "output_ply": str(args.output.resolve()),
        "probabilities": str(args.probabilities.resolve()),
        "points": count,
        "images": len(images),
        "classes": {str(index + 1): name for index, name in enumerate(classes)},
        "label_bits": {str(1 << index): name for index, name in enumerate(classes)},
        "thresholds": dict(zip(classes, map(float, thresholds))),
        "min_positive_views": args.min_views,
        "min_positive_fraction": args.min_fraction,
        "depth_tolerance_relative": args.depth_tolerance,
        "visible_points": int(np.count_nonzero(valid_count)),
        "labeled_points": {name: int(np.count_nonzero(accepted[:, index])) for index, name in enumerate(classes)},
        "primary_label_points": {name: int(np.count_nonzero(labels == index + 1)) for index, name in enumerate(classes)},
        "multi_label_points": int(np.count_nonzero(accepted.sum(axis=1) > 1)),
        "per_image": diagnostics,
        "partial_run": bool(args.limit_points or args.limit_images),
    }
    report_path = args.output.with_suffix(".summary.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("points", "images", "visible_points", "labeled_points", "multi_label_points", "partial_run")}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("../collected_datasets/south-building"))
    parser.add_argument("--workspace", type=Path, default=Path("benchmark/reconstruction/runs/south-building-dense"))
    parser.add_argument("--ply", type=Path, default=Path("benchmark/reconstruction/runs/south-building-dense/fused.ply"))
    parser.add_argument("--probabilities", type=Path, default=Path("benchmark/reconstruction/runs/south-building-rgb-tiles/probabilities_512"))
    parser.add_argument("--output", type=Path, default=Path("benchmark/reconstruction/runs/south-building-dense/defects_dense.ply"))
    parser.add_argument("--classes", nargs="+", default=["crack", "spalling"], choices=ALL_CLASSES)
    parser.add_argument("--threshold", type=float, help="Override every class threshold; default: RGB summary or 0.5")
    parser.add_argument("--min-views", type=int, default=2, help="Minimum positive views per label")
    parser.add_argument("--min-fraction", type=float, default=0.3, help="Minimum positive fraction among visible views")
    parser.add_argument("--depth-tolerance", type=float, default=0.02, help="Relative depth mismatch allowed for visibility")
    parser.add_argument("--chunk-size", type=int, default=200000)
    parser.add_argument("--limit-points", type=int, default=0, help="Smoke-test first N points")
    parser.add_argument("--limit-images", type=int, default=0, help="Smoke-test first N images")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
