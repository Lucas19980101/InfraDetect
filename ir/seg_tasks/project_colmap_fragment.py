"""Project fused RGB/IR defect states into a COLMAP dense point cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import yaml
from scipy.io import loadmat


PLY_FIELDS = (
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
)
STATUS_NAMES = ("confirmed", "uncertain", "equipment_influenced", "excluded_nonwall")
STATUS_COLORS = np.asarray(
    ((255, 40, 40), (255, 165, 35), (180, 55, 220), (145, 145, 145)),
    dtype=np.uint8,
)
DEFECT_NAMES = ("cracks", "peeling", "hollow_areas", "stains", "erosion")
DEFECT_COLORS = np.asarray(
    ((61, 61, 245), (169, 36, 191), (174, 79, 13), (36, 179, 83), (203, 253, 0)),
    dtype=np.uint8,
)


def defect_metadata(config: Path | None) -> tuple[tuple[str, ...], np.ndarray]:
    """Load non-background IR class names/colors, or use legacy BFDD metadata."""
    if config is None:
        return DEFECT_NAMES, DEFECT_COLORS
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    names = list(cfg["data"]["class_names"])
    palette = np.asarray(cfg["data"].get("palette_rgb"), dtype=np.uint8)
    if not names or names[0].lower() != "background" or palette.shape != (len(names), 3):
        raise ValueError("IR config must define Background first and one palette_rgb entry per class")
    normalized = tuple(name.lower().replace(" ", "_") for name in names[1:])
    return normalized, palette[1:]


def read_colmap_ply(path: Path, limit: int = 0) -> np.memmap:
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
    vertex = next((x for x in header if x.startswith("element vertex ")), None)
    expected = [f"property {'float' if kind == '<f4' else 'uchar'} {name}"
                for name, kind in PLY_FIELDS]
    if vertex is None or [x for x in header if x.startswith("property ")] != expected:
        raise ValueError("Unexpected COLMAP PLY layout")
    count = int(vertex.rsplit(" ", 1)[1])
    if limit:
        count = min(count, limit)
    return np.memmap(path, dtype=np.dtype(list(PLY_FIELDS)), mode="r", offset=offset,
                     shape=(count,))


def project_pinhole(camera: pycolmap.Camera, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if str(camera.model) != "PINHOLE":
        raise ValueError(f"Dense camera must be PINHOLE, got {camera.model}")
    fx, fy, cx, cy = camera.params
    with np.errstate(divide="ignore", invalid="ignore"):
        return fx * xyz[:, 0] / xyz[:, 2] + cx, fy * xyz[:, 1] / xyz[:, 2] + cy


def project_original(camera: pycolmap.Camera, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        x, y = xyz[:, 0] / xyz[:, 2], xyz[:, 1] / xyz[:, 2]
    radius2 = x * x + y * y
    if model == "SIMPLE_RADIAL":
        scale = 1 + distortion[0] * radius2
        x, y = x * scale, y * scale
    elif model == "RADIAL":
        scale = 1 + distortion[0] * radius2 + distortion[1] * radius2 * radius2
        x, y = x * scale, y * scale
    elif model == "OPENCV":
        k1, k2, p1, p2 = distortion
        scale = 1 + k1 * radius2 + k2 * radius2 * radius2
        x, y = (x * scale + 2 * p1 * x * y + p2 * (radius2 + 2 * x * x),
                y * scale + p1 * (radius2 + 2 * y * y) + 2 * p2 * x * y)
    return fx * x + cx, fy * y + cy


def visible_indices(points: np.ndarray, pose: np.ndarray, camera: pycolmap.Camera,
                    depth: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    xyz = points @ pose[:3, :3].T + pose[:3, 3]
    u, v = project_pinhole(camera, xyz)
    height, width = depth.shape
    inside = (np.isfinite(u) & np.isfinite(v) & (xyz[:, 2] > 0)
              & (u >= 0) & (v >= 0) & (u < width) & (v < height))
    indexes = np.flatnonzero(inside)
    if not len(indexes):
        return indexes, xyz[indexes]
    ix = np.clip(np.rint(u[indexes]).astype(np.intp), 0, width - 1)
    iy = np.clip(np.rint(v[indexes]).astype(np.intp), 0, height - 1)
    expected = depth[iy, ix]
    good = (np.isfinite(expected) & (expected > 0)
            & (np.abs(xyz[indexes, 2] - expected) <= tolerance * expected))
    return indexes[good], xyz[indexes[good]]


def apply_homography(u: np.ndarray, v: np.ndarray,
                     matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.column_stack((u, v, np.ones_like(u)))
    mapped = points @ matrix.T
    valid = np.isfinite(mapped).all(axis=1) & (np.abs(mapped[:, 2]) > 1e-12)
    x = np.full_like(u, np.nan, dtype=np.float64)
    y = np.full_like(v, np.nan, dtype=np.float64)
    x[valid] = mapped[valid, 0] / mapped[valid, 2]
    y[valid] = mapped[valid, 1] / mapped[valid, 2]
    return x, y, valid


def accepted_votes(counts: np.ndarray, valid_count: np.ndarray, min_views: int,
                   min_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    fraction = np.divide(counts, valid_count[:, None], out=np.zeros_like(counts, dtype=np.float32),
                         where=valid_count[:, None] > 0)
    accepted = (counts >= min_views) & (fraction >= min_fraction)
    scores = np.where(accepted, fraction, -1)
    primary = np.where(accepted.any(axis=1), scores.argmax(axis=1) + 1, 0).astype(np.uint8)
    return accepted, primary


def write_output(path: Path, source: np.memmap, primary_status: np.ndarray,
                 accepted_status: np.ndarray, status_count: np.ndarray,
                 defect_class: np.ndarray, defect_count: np.ndarray,
                 valid_count: np.ndarray, mode: str, chunk_size: int,
                 defect_names: tuple[str, ...], defect_colors: np.ndarray) -> None:
    extra = [("status", "u1"), ("status_bits", "u1"), ("defect_class", "u1"),
             ("valid_views", "<u2")]
    extra += [(f"views_{name}", "<u2") for name in STATUS_NAMES]
    extra += [(f"views_{name}", "<u2") for name in defect_names]
    dtype = np.dtype([*PLY_FIELDS, *extra])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        lines = ["ply", "format binary_little_endian 1.0", f"element vertex {len(source)}"]
        for name, kind in PLY_FIELDS:
            lines.append(f"property {'float' if kind == '<f4' else 'uchar'} {name}")
        lines += ["property uchar status", "property uchar status_bits",
                  "property uchar defect_class", "property ushort valid_views"]
        lines += [f"property ushort views_{name}" for name in STATUS_NAMES]
        lines += [f"property ushort views_{name}" for name in defect_names]
        lines.append("end_header")
        handle.write(("\n".join(lines) + "\n").encode("ascii"))
        for start in range(0, len(source), chunk_size):
            stop = min(start + chunk_size, len(source))
            block = np.empty(stop - start, dtype=dtype)
            for name, _ in PLY_FIELDS:
                block[name] = source[name][start:stop]
            block["status"] = primary_status[start:stop]
            bits = np.zeros(stop - start, dtype=np.uint8)
            for index in range(len(STATUS_NAMES)):
                bits |= accepted_status[start:stop, index].astype(np.uint8) << index
            block["status_bits"] = bits
            block["defect_class"] = defect_class[start:stop]
            block["valid_views"] = valid_count[start:stop]
            for index, name in enumerate(STATUS_NAMES):
                block[f"views_{name}"] = status_count[start:stop, index]
            for index, name in enumerate(defect_names):
                block[f"views_{name}"] = defect_count[start:stop, index]
            colors = np.column_stack((block["red"], block["green"], block["blue"]))
            if mode == "review":
                for index, color in enumerate(STATUS_COLORS, 1):
                    colors[primary_status[start:stop] == index] = color
            elif mode == "classes":
                confirmed = primary_status[start:stop] == 1
                for index, color in enumerate(defect_colors, 1):
                    colors[confirmed & (defect_class[start:stop] == index)] = color
            elif mode == "confirmed":
                colors[primary_status[start:stop] == 1] = STATUS_COLORS[0]
            else:
                raise ValueError(mode)
            block["red"], block["green"], block["blue"] = colors.T
            handle.write(block.tobytes())


def run(args: argparse.Namespace) -> dict:
    if args.output.resolve() == args.ply.parent.resolve():
        raise ValueError("Output must be a separate directory from the source PLY")
    original = pycolmap.Reconstruction(args.model)
    dense = pycolmap.Reconstruction(args.workspace / "sparse")
    source = read_colmap_ply(args.ply, args.limit_points)
    defect_names, defect_colors = defect_metadata(args.ir_config)
    images = sorted(dense.images.values(), key=lambda item: item.name)
    if args.limit_images:
        images = images[:args.limit_images]
    count = len(source)
    valid_count = np.zeros(count, dtype=np.uint16)
    status_count = np.zeros((count, 4), dtype=np.uint16)
    defect_count = np.zeros((count, len(defect_names)), dtype=np.uint16)
    diagnostics = []

    for ordinal, image in enumerate(images, 1):
        if image.image_id not in original.images or original.images[image.image_id].name != image.name:
            raise ValueError(f"Original/dense reconstruction mismatch: {image.name}")
        stem = Path(image.name).stem
        status_path = args.fusion / stem / "defect_status.png"
        defect_path = args.fusion / stem / "defect_ids.png"
        matrix_path = args.homographies / f"{stem}.mat"
        status = cv2.imread(str(status_path), cv2.IMREAD_UNCHANGED)
        defect = cv2.imread(str(defect_path), cv2.IMREAD_UNCHANGED)
        if status is None or defect is None or status.ndim != 2 or defect.shape != status.shape:
            raise ValueError(f"Missing or invalid fusion maps for {stem}")
        matrix = np.asarray(loadmat(matrix_path)["H"], dtype=np.float64)
        depth_map = pycolmap.DepthMap()
        depth_map.read(args.workspace / "stereo" / "depth_maps" / f"{image.name}.geometric.bin")
        depth = depth_map.to_array()
        dense_camera = dense.cameras[image.camera_id]
        original_image = original.images[image.image_id]
        original_camera = original.cameras[original_image.camera_id]
        pose = np.asarray(image.cam_from_world().matrix(), dtype=np.float64)
        original_pose = np.asarray(original_image.cam_from_world().matrix(), dtype=np.float64)
        if not np.allclose(pose, original_pose, atol=1e-5, rtol=0):
            raise ValueError(f"Original/dense poses differ: {image.name}")
        seen = sampled = 0
        for start in range(0, count, args.chunk_size):
            stop = min(start + args.chunk_size, count)
            points = np.column_stack(tuple(source[field][start:stop] for field in ("x", "y", "z")))
            visible, xyz = visible_indices(points, pose, dense_camera, depth, args.depth_tolerance)
            seen += len(visible)
            if not len(visible):
                continue
            u, v = project_original(original_camera, xyz)
            ir_x, ir_y, finite = apply_homography(u, v, matrix)
            inside = (finite & (ir_x >= 0) & (ir_y >= 0)
                      & (ir_x < status.shape[1]) & (ir_y < status.shape[0]))
            absolute = start + visible[inside]
            if not len(absolute):
                continue
            ix = np.rint(ir_x[inside]).astype(np.intp).clip(0, status.shape[1] - 1)
            iy = np.rint(ir_y[inside]).astype(np.intp).clip(0, status.shape[0] - 1)
            sampled_status = status[iy, ix]
            sampled_defect = defect[iy, ix]
            valid_count[absolute] += 1
            sampled += len(absolute)
            for value in range(1, 5):
                selected = absolute[sampled_status == value]
                status_count[selected, value - 1] += 1
            confirmed = sampled_status == 1
            for value in range(1, len(defect_names) + 1):
                selected = absolute[confirmed & (sampled_defect == value)]
                defect_count[selected, value - 1] += 1
        diagnostics.append({"image": image.name, "depth_visible_points": seen,
                            "ir_sampled_points": sampled})
        print(f"[{ordinal}/{len(images)}] {image.name}: visible={seen}, sampled={sampled}", flush=True)

    accepted_status, primary_status = accepted_votes(
        status_count, valid_count, args.min_views, args.min_fraction)
    confirmed_count = status_count[:, 0]
    accepted_defect, defect_class = accepted_votes(
        defect_count, confirmed_count, args.min_views, args.min_fraction)
    defect_class[primary_status != 1] = 0
    args.output.mkdir(parents=True, exist_ok=True)
    for mode, filename in (("review", "fragment_review.ply"),
                           ("classes", "fragment_classes.ply"),
                           ("confirmed", "fragment_confirmed.ply")):
        write_output(args.output / filename, source, primary_status, accepted_status,
                     status_count, defect_class, defect_count, valid_count, mode, args.chunk_size,
                     defect_names, defect_colors)
    report = {
        "source_ply": str(args.ply.resolve()),
        "points": count,
        "images": len(images),
        "min_views": args.min_views,
        "min_fraction": args.min_fraction,
        "depth_tolerance": args.depth_tolerance,
        "visible_points": int(np.count_nonzero(valid_count)),
        "status_points": {name: int(np.count_nonzero(primary_status == index))
                          for index, name in enumerate(STATUS_NAMES, 1)},
        "defect_class_points": {name: int(np.count_nonzero(defect_class == index))
                                for index, name in enumerate(defect_names, 1)},
        "ir_class_names": ["background", *defect_names],
        "multi_status_points": int(np.count_nonzero(accepted_status.sum(axis=1) > 1)),
        "per_image": diagnostics,
        "partial_run": bool(args.limit_points or args.limit_images),
    }
    (args.output / "fragment_projection.summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("points", "images", "visible_points", "status_points",
                       "defect_class_points", "multi_status_points", "partial_run")}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Original distorted sparse model")
    parser.add_argument("--workspace", type=Path, required=True, help="COLMAP dense workspace")
    parser.add_argument("--ply", type=Path, required=True, help="Source COLMAP fused.ply")
    parser.add_argument("--fusion", type=Path, required=True, help="Per-image InfraDetect fusion root")
    parser.add_argument("--homographies", type=Path, required=True, help="MFDN RGB-to-IR .mat folder")
    parser.add_argument("--ir-config", type=Path,
                        help="IR config defining class_names and palette_rgb; omit for legacy BFDD")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--min-fraction", type=float, default=0.3)
    parser.add_argument("--depth-tolerance", type=float, default=0.02)
    parser.add_argument("--chunk-size", type=int, default=200000)
    parser.add_argument("--limit-points", type=int, default=0)
    parser.add_argument("--limit-images", type=int, default=0)
    args = parser.parse_args()
    if args.min_views < 1 or not 0 <= args.min_fraction <= 1 or args.depth_tolerance <= 0:
        parser.error("Invalid fusion thresholds")
    run(args)


if __name__ == "__main__":
    main()
