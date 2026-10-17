"""Assemble registered evidence into reviewable thermal-anomaly scene maps.

The input is a scene manifest, not a dataset label file. Thermal probabilities
must come from a model appropriate for the input sensor/domain; this module
does not reinterpret BFDD defect labels as thermal-anomaly ground truth.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat


STATUS = {
    0: "none",
    1: "wall_supported_candidate",
    2: "uncertain_registration_or_boundary",
    3: "equipment_influenced",
    4: "nonwall_overlap",
}


def _path(base: Path, value: str) -> Path:
    """Resolve and check a required evidence path."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _probability(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Load and validate a probability map without changing its pixel size."""
    if path.suffix.lower() != ".npy":
        raise ValueError(f"Probability input must be a .npy float array: {path}")
    result = np.load(path, allow_pickle=False)
    if result.ndim != 2 or (shape is not None and result.shape != shape):
        raise ValueError(f"Invalid probability shape: {path}: {result.shape}")
    if not np.issubdtype(result.dtype, np.floating) or not np.isfinite(result).all():
        raise ValueError(f"Probability must contain finite floating-point values: {path}")
    if np.any((result < 0) | (result > 1)):
        raise ValueError(f"Probability must be within [0, 1]: {path}")
    return result.astype(np.float32, copy=False)


def _mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Load and validate a single-channel mask."""
    result = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if result is None or result.ndim != 2 or (shape is not None and result.shape != shape):
        raise ValueError(f"Invalid single-channel mask: {path}")
    return result


def _homography(path: Path) -> np.ndarray:
    """Read the registered RGB-to-IR transform for this scene."""
    if path.suffix.lower() == ".mat":
        matrix = loadmat(path)["H"]
    elif path.suffix.lower() == ".npy":
        matrix = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".json":
        matrix = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Homography must be .mat, .npy or .json: {path}")
    matrix = np.asarray(matrix, np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError(f"Invalid RGB-to-thermal homography: {path}")
    return matrix


def _preview(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Render a human-readable preview of the fused evidence."""
    if path.suffix.lower() == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 3 or array.shape[2] < 4:
            raise ValueError(f"Expected scene preview with [B,G,R,Thermal]: {path}")
        thermal = np.asarray(array[..., 3], dtype=np.uint8)
        image = cv2.cvtColor(thermal, cv2.COLOR_GRAY2BGR)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != shape:
        raise ValueError(f"Thermal preview dimensions differ from probability: {path}")
    return image


def fuse_scene(scene: dict, base: Path, threshold: float, edge_radius: int,
               interference_threshold: float) -> tuple[dict[str, np.ndarray], dict]:
    """Create status maps while preserving each original evidence layer."""
    required = ("id", "source_dataset", "thermal_probability", "thermal_model")
    if any(not scene.get(key) for key in required):
        raise ValueError(f"Scene requires {', '.join(required)}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", scene["id"]):
        raise ValueError("Scene id must be a safe file/directory name")
    thermal_path = _path(base, scene["thermal_probability"])
    probability = _probability(thermal_path)
    shape = probability.shape
    valid_path = _path(base, scene["valid_mask"]) if scene.get("valid_mask") else None
    valid = (_mask(valid_path, shape) > 0 if valid_path else np.ones(shape, bool))
    anomaly = (probability >= threshold) & valid
    status = np.zeros(shape, np.uint8)
    status[anomaly] = 2  # Without accepted registration, retain as uncertain.
    wall = np.zeros(shape, np.uint8)
    wall_core = np.zeros(shape, np.uint8)
    warped_components = np.full(shape, 255, np.uint8)
    registration_coverage = np.zeros(shape, np.uint8)
    equipment = np.zeros(shape, np.uint8)
    wall_probability_ir = None
    interference_probability_ir = None
    rgb_semantic = None
    rgb_semantic_coverage = None

    registered = scene.get("registration_accepted", False)
    if not isinstance(registered, bool):
        raise ValueError("registration_accepted must be true or false")
    component_path = _path(base, scene["component_ids_rgb"]) if scene.get("component_ids_rgb") else None
    matrix_path = _path(base, scene["rgb_to_thermal"]) if scene.get("rgb_to_thermal") else None
    if registered and (component_path is None or matrix_path is None):
        raise ValueError("Accepted registration requires component_ids_rgb and rgb_to_thermal")
    if registered:
        components = _mask(component_path)
        if not np.isin(components, (0, 1, 2, 255)).all():
            raise ValueError("Components must use 0=surface, 1=opening, 2=other, 255=unknown")
        matrix = _homography(matrix_path)
        height, width = shape
        warped_components = cv2.warpPerspective(
            components, matrix, (width, height), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        registration_coverage = cv2.warpPerspective(
            np.ones(components.shape, np.uint8), matrix, (width, height),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        wall = (warped_components == 0).astype(np.uint8)
        if scene.get("wall_probability_rgb"):
            wall_rgb = _probability(_path(base, scene["wall_probability_rgb"]), components.shape)
            wall_probability_ir = cv2.warpPerspective(
                wall_rgb, matrix, (width, height), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if edge_radius:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (2 * edge_radius + 1, 2 * edge_radius + 1))
            wall_core = cv2.erode(wall, kernel)
            near_wall = cv2.dilate(wall, kernel) > 0
        else:
            wall_core = wall.copy()
            near_wall = wall > 0
        known_nonwall = (warped_components == 1) | (warped_components == 2)
        status[anomaly & ~near_wall & known_nonwall] = 4
        status[anomaly & (wall_core > 0)] = 1

        if scene.get("interference_probability_rgb"):
            interference_path = _path(base, scene["interference_probability_rgb"])
            interference_rgb = _probability(interference_path, components.shape)
            interference_ir = cv2.warpPerspective(
                interference_rgb, matrix, (width, height), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            interference_probability_ir = interference_ir
            equipment = (interference_ir >= interference_threshold).astype(np.uint8)
            status[anomaly & (equipment > 0)] = 3

        if scene.get("rgb_semantic_probabilities"):
            semantic_path = _path(base, scene["rgb_semantic_probabilities"])
            with np.load(semantic_path, allow_pickle=False) as source:
                channels = np.asarray(source["probabilities"], np.float32)
                names = np.asarray(source["class_names"])
                semantic_valid = (np.asarray(source["coverage"], bool)
                                  if "coverage" in source else np.ones(components.shape, bool))
            if (channels.ndim != 3 or channels.shape[1:] != components.shape
                    or len(channels) != len(names) or not np.isfinite(channels).all()
                    or np.any((channels < 0) | (channels > 1))
                    or semantic_valid.shape != components.shape):
                raise ValueError(f"Invalid RGB semantic probabilities: {semantic_path}")
            warped = np.stack([cv2.warpPerspective(
                channel, matrix, (width, height), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0) for channel in channels])
            rgb_semantic = (warped.astype(np.float16), names)
            rgb_semantic_coverage = cv2.warpPerspective(
                semantic_valid.astype(np.uint8), matrix, (width, height),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    elif (scene.get("interference_probability_rgb") or scene.get("wall_probability_rgb")
          or scene.get("rgb_semantic_probabilities")):
        raise ValueError("RGB evidence requires accepted registration")

    maps = {
        "thermal_anomaly_probability.npy": probability,
        "thermal_anomaly_mask.png": anomaly.astype(np.uint8),
        "valid_thermal_mask.png": valid.astype(np.uint8),
        "component_ids_in_ir.png": warped_components,
        "registration_coverage_in_ir.png": registration_coverage,
        "wall_in_ir.png": wall,
        "wall_core_in_ir.png": wall_core,
        "equipment_mask_in_ir.png": equipment,
        # Names expected by the existing COLMAP projector. ID 1 is a thermal
        # anomaly, never a Hollow/Stains or other semantic defect class.
        "defect_ids.png": anomaly.astype(np.uint8),
        "defect_status.png": status,
    }
    if rgb_semantic is not None:
        maps["rgb_semantic_evidence.npz"] = rgb_semantic
        maps["rgb_semantic_coverage_in_ir.png"] = rgb_semantic_coverage * 255
    # Both maps are optional evidence. They never modify the IR probability
    # or candidate mask; the downstream reviewer may use them for ordering.
    if wall_probability_ir is not None:
        maps["wall_probability_in_ir.npy"] = wall_probability_ir.astype(np.float32)
    if interference_probability_ir is not None:
        maps["interference_probability_in_ir.npy"] = interference_probability_ir.astype(np.float32)
    preview_path = _path(base, scene["thermal_preview"]) if scene.get("thermal_preview") else None
    if preview_path is not None:
        overlay = _preview(preview_path, shape)
        colors = {1: (40, 40, 255), 2: (35, 165, 255),
                  3: (220, 55, 180), 4: (145, 145, 145)}
        for value, color in colors.items():
            selected = status == value
            overlay[selected] = (0.45 * overlay[selected] + 0.55 * np.asarray(color)).astype(np.uint8)
        maps["status_overlay.png"] = overlay
    summary = {
        "schema_version": 1,
        "scene_id": scene["id"],
        "source_dataset": scene["source_dataset"],
        "thermal_model": scene["thermal_model"],
        "thermal_input_mode": scene.get("thermal_input_mode"),
        "inputs": {key: str(_path(base, scene[key])) for key in (
            "thermal_probability", "thermal_preview", "valid_mask",
            "component_ids_rgb", "interference_probability_rgb",
            "wall_probability_rgb",
            "rgb_semantic_probabilities", "rgb_to_thermal") if scene.get(key)},
        "registration_accepted": registered,
        "status_legend": {str(key): value for key, value in STATUS.items()},
        "class_names": ["Background", "Thermal Anomaly"],
        "threshold": threshold,
        "edge_radius_ir_pixels": edge_radius,
        "interference_threshold": interference_threshold,
        "shape": list(shape),
        "valid_pixels": int(valid.sum()),
        "anomaly_pixels": int(anomaly.sum()),
        "status_pixels": {name: int((status == index).sum())
                          for index, name in STATUS.items() if index},
        "rgb_semantic_channels": ([str(name) for name in rgb_semantic[1]]
                                  if rgb_semantic is not None else []),
        "rgb_semantic_evaluated_pixels": (int(rgb_semantic_coverage.sum())
                                          if rgb_semantic_coverage is not None else None),
        "interpretation": "candidate evidence, not an engineering defect diagnosis",
    }
    return maps, summary


def run(manifest: Path, output: Path) -> list[dict]:
    """Validate inputs, execute this module's workflow, and write its outputs."""
    definition = json.loads(manifest.read_text(encoding="utf-8"))
    if definition.get("schema_version") != 1:
        raise ValueError("Expected scene manifest schema_version=1")
    settings = definition.get("fusion", {})
    threshold = float(settings.get("thermal_threshold", 0.5))
    interference_threshold = float(settings.get("interference_threshold", 0.78))
    edge_radius = int(settings.get("edge_radius_ir_pixels", 8))
    if not (0 <= threshold <= 1 and 0 <= interference_threshold <= 1 and edge_radius >= 0):
        raise ValueError("Invalid fusion thresholds or edge radius")
    scenes = definition.get("scenes", [])
    if not scenes or any("id" not in scene for scene in scenes) or len({scene["id"] for scene in scenes}) != len(scenes):
        raise ValueError("Manifest requires at least one scene with unique IDs")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty to preserve prior runs: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for scene in scenes:
        maps, summary = fuse_scene(scene, manifest.parent, threshold,
                                   edge_radius, interference_threshold)
        scene_id = scene["id"]
        scene_dir = output / scene_id
        scene_dir.mkdir()
        linked = []
        for name, value in maps.items():
            path = scene_dir / name
            source = None
            if name == "thermal_anomaly_probability.npy":
                candidate = Path(summary["inputs"]["thermal_probability"])
                original = np.load(candidate, mmap_mode="r", allow_pickle=False)
                if (original.dtype == value.dtype and original.shape == value.shape
                        and np.array_equal(original, value)):
                    source = candidate
            elif name == "rgb_semantic_evidence.npz":
                inputs = summary["inputs"]
                if (inputs.get("rgb_to_thermal") and inputs.get("rgb_semantic_probabilities")
                        and np.array_equal(_homography(Path(inputs["rgb_to_thermal"])), np.eye(3))):
                    candidate = Path(inputs["rgb_semantic_probabilities"])
                    with np.load(candidate, allow_pickle=False) as archive:
                        if (np.array_equal(archive["probabilities"], value[0])
                                and np.array_equal(archive["class_names"], value[1])):
                            source = candidate
            if source is not None:
                try:
                    os.link(source.resolve(), path)
                except OSError as error:
                    # Cross-device or restricted filesystems still get a full,
                    # independent output through the ordinary writer below.
                    if error.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES,
                                           errno.ENOTSUP, errno.EOPNOTSUPP):
                        raise
                else:
                    linked.append(name)
                    continue
            if name.endswith(".npy"):
                np.save(path, value, allow_pickle=False)
            elif name.endswith(".npz"):
                np.savez_compressed(path, probabilities=value[0], class_names=value[1])
            elif not cv2.imwrite(str(path), value):
                raise OSError(f"Could not write {path}")
        summary["storage_linked_files"] = linked
        (scene_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        summaries.append(summary)
    (output / "manifest_snapshot.json").write_text(
        json.dumps(definition, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    return summaries


def main() -> None:
    """Parse command-line arguments and run this module's workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.manifest.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
