"""Group labeled COLMAP points into reviewable 3D candidate instances.

This is geometric grouping in COLMAP coordinates. It does not assign metric
area/length or turn model predictions into verified building defects.
"""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


PLY_TYPES = {
    "char": "i1", "uchar": "u1", "short": "<i2", "ushort": "<u2",
    "int": "<i4", "uint": "<u4", "float": "<f4", "double": "<f8",
}


def read_vertex_ply(path: Path) -> np.memmap:
    """Map a binary little-endian vertex-only PLY without copying the cloud."""
    properties: list[tuple[str, str]] = []
    count = None
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        if handle.readline().strip() != b"format binary_little_endian 1.0":
            raise ValueError("Only binary little-endian PLY is supported")
        in_vertices = False
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError("PLY header has no end_header")
            line = raw.decode("ascii").strip()
            if line.startswith("element "):
                _, element, number = line.split()
                in_vertices = element == "vertex"
                if in_vertices:
                    count = int(number)
            elif line.startswith("property ") and in_vertices:
                _, kind, name = line.split()
                if kind not in PLY_TYPES:
                    raise ValueError(f"Unsupported PLY property type: {kind}")
                properties.append((name, PLY_TYPES[kind]))
            elif line == "end_header":
                offset = handle.tell()
                break
    if count is None or not {"x", "y", "z"}.issubset({name for name, _ in properties}):
        raise ValueError("PLY must contain vertices and x/y/z fields")
    dtype = np.dtype(properties)
    if path.stat().st_size < offset + count * dtype.itemsize:
        raise ValueError("PLY vertex data is truncated")
    return np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count,))


def _classes(cloud: np.memmap, summary: dict) -> tuple[str, dict[int, str]]:
    """Decode the class flags attached to a projected 3D point."""
    names = set(cloud.dtype.names or ())
    if "label" in names:
        return "label", {int(key): str(value) for key, value in summary["classes"].items()}
    if "defect_class" in names:
        class_names = summary["ir_class_names"]
        return "defect_class", {index: str(name)
                                for index, name in enumerate(class_names[1:], 1)}
    raise ValueError("PLY has no supported class field (label or defect_class)")


def _components(points: np.ndarray, radius: float) -> list[np.ndarray]:
    """Connected components under a fixed 3D distance threshold."""
    parent = np.arange(len(points), dtype=np.int32)

    def root(index: int) -> int:
        """Find the representative of one point in the union-find grouping."""
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for left, right in cKDTree(points).query_pairs(radius, output_type="ndarray"):
        a, b = root(int(left)), root(int(right))
        if a != b:
            parent[max(a, b)] = min(a, b)
    groups: dict[int, list[int]] = {}
    for index in range(len(points)):
        groups.setdefault(root(index), []).append(index)
    return [np.asarray(indices, np.intp) for indices in groups.values()]


def _color(instance_id: int) -> tuple[int, int, int]:
    """Choose a display color for a candidate class."""
    red, green, blue = colorsys.hsv_to_rgb((instance_id * 0.61803398875) % 1, 0.78, 1.0)
    return tuple(round(255 * channel) for channel in (red, green, blue))


def _write_candidate_ply(path: Path, points: np.ndarray, colors: np.ndarray,
                         instance_ids: np.ndarray, class_ids: np.ndarray) -> None:
    """Save candidate points with colors for point-cloud review."""
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                      ("instance_id", "<u4"), ("class_id", "u1")])
    vertices = np.empty(len(points), dtype=dtype)
    for column, name in enumerate(("x", "y", "z")):
        vertices[name] = points[:, column]
    for column, name in enumerate(("red", "green", "blue")):
        vertices[name] = colors[:, column]
    vertices["instance_id"] = instance_ids
    vertices["class_id"] = class_ids
    header = ["ply", "format binary_little_endian 1.0",
              f"element vertex {len(points)}",
              "property float x", "property float y", "property float z",
              "property uchar red", "property uchar green", "property uchar blue",
              "property uint instance_id", "property uchar class_id", "end_header"]
    with path.open("wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii"))
        handle.write(vertices.tobytes())


def aggregate(cloud: np.memmap, summary: dict, radius_factor: float,
              min_points: int) -> tuple[list[dict], dict[str, np.ndarray], dict]:
    """按缺陷类别与三维距离把投影候选点分成可复核的实例。

连接半径由同类点的典型最近邻间距乘 radius_factor 得到，并受场景
对角线约束，防止 COLMAP 任意尺度或稀疏区域把远处点错误连成一组。
min_points 过滤很小的点簇；分组结果只是空间候选，不是病害确诊。
    """
    if radius_factor <= 0 or min_points < 1:
        raise ValueError("radius_factor and min_points must be positive")
    class_field, names = _classes(cloud, summary)
    selected = np.flatnonzero(cloud[class_field] > 0)
    if not len(selected):
        raise ValueError("PLY contains no labeled candidate points")
    if any(value not in names for value in np.unique(cloud[class_field][selected])):
        raise ValueError("PLY class IDs disagree with projection summary")
    coords = np.column_stack(tuple(cloud[axis][selected] for axis in "xyz"))
    # COLMAP scale is arbitrary. The scene diagonal caps a class-specific
    # radius so isolated candidates do not acquire very long links.
    bounds = np.asarray([[cloud[axis].min(), cloud[axis].max()] for axis in "xyz"])
    scene_diagonal = float(np.linalg.norm(bounds[:, 1] - bounds[:, 0]))
    max_radius = scene_diagonal * 0.001
    instance_ids = np.zeros(len(selected), np.uint32)
    radii = {}
    components = []
    for class_id, class_name in sorted(names.items()):
        local = np.flatnonzero(cloud[class_field][selected] == class_id)
        if not len(local):
            radii[class_name] = None
            continue
        class_points = coords[local]
        if len(local) > 1:
            neighbors = cKDTree(class_points).query(class_points, k=2)[0][:, 1]
            typical_spacing = float(np.median(neighbors))
            radius = min(radius_factor * typical_spacing, max_radius)
            if radius <= 0:
                raise ValueError("Labeled points have zero spacing")
            groups = _components(class_points, radius)
        else:
            radius = 0.0
            groups = [np.array([0], np.intp)]
        radii[class_name] = radius
        for group in groups:
            point_rows = local[group]
            components.append((class_id, class_name, point_rows))
    components.sort(key=lambda item: (-len(item[2]), item[0], int(selected[item[2][0]])))

    instances = []
    names_in_cloud = set(cloud.dtype.names or ())
    for instance_id, (class_id, class_name, rows) in enumerate(components, 1):
        instance_ids[rows] = instance_id
        source_rows = selected[rows]
        points = coords[rows]
        safe_name = class_name.lower().replace(" ", "_")
        record = {
            "instance_id": instance_id,
            "class_id": class_id,
            "class_name": class_name,
            "point_count": len(rows),
            "review_priority": "regular" if len(rows) >= min_points else "small_cluster",
            "centroid_colmap": points.mean(axis=0).tolist(),
            "bbox_min_colmap": points.min(axis=0).tolist(),
            "bbox_max_colmap": points.max(axis=0).tolist(),
            "source_point_index_min": int(source_rows.min()),
        }
        for prefix, output_name in (("prob_", "mean_probability"),
                                    ("support_", "mean_support_fraction"),
                                    ("positive_views_", "mean_positive_views")):
            field = prefix + safe_name
            if field in names_in_cloud:
                record[output_name] = float(np.mean(cloud[field][source_rows]))
        views_field = f"views_{safe_name}"
        if views_field in names_in_cloud:
            record["mean_class_views"] = float(np.mean(cloud[views_field][source_rows]))
        if "valid_views" in names_in_cloud:
            record["mean_valid_views"] = float(np.mean(cloud["valid_views"][source_rows]))
        instances.append(record)
    colors = np.asarray([_color(int(index)) for index in instance_ids], np.uint8)
    labels = {"source_point_indices": selected.astype(np.uint32),
              "instance_ids": instance_ids,
              "class_ids": cloud[class_field][selected].astype(np.uint8),
              "points": coords.astype(np.float32), "colors": colors}
    report = {
        "source_points": len(cloud),
        "labeled_points": len(selected),
        "instance_count": len(instances),
        "reviewable_instances": sum(item["review_priority"] == "regular" for item in instances),
        "small_clusters": sum(item["review_priority"] == "small_cluster" for item in instances),
        "classes": names,
        "radius_factor": radius_factor,
        "radius_by_class_colmap": radii,
        "scene_diagonal_colmap": scene_diagonal,
        "min_points_for_regular_review": min_points,
        "metric_scale_known": False,
    }
    return instances, labels, report


def run(ply: Path, projection_summary: Path, output: Path,
        radius_factor: float = 4.0, min_points: int = 5) -> dict:
    """Validate inputs, execute this module's workflow, and write its outputs."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty to preserve prior runs: {output}")
    cloud = read_vertex_ply(ply)
    source = json.loads(projection_summary.read_text(encoding="utf-8"))
    if int(source["points"]) != len(cloud):
        raise ValueError("PLY vertex count differs from projection summary")
    instances, labels, report = aggregate(cloud, source, radius_factor, min_points)
    report.update({"source_ply": str(ply.resolve()),
                   "projection_summary": str(projection_summary.resolve())})
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "candidate_point_membership.npz",
                        source_point_indices=labels["source_point_indices"],
                        instance_ids=labels["instance_ids"], class_ids=labels["class_ids"])
    _write_candidate_ply(output / "candidate_instances.ply", labels["points"], labels["colors"],
                         labels["instance_ids"], labels["class_ids"])
    (output / "instances.json").write_text(json.dumps(instances, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    """Parse command-line arguments and run this module's workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--projection-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius-factor", type=float, default=4.0)
    parser.add_argument("--min-points", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run(args.ply, args.projection_summary, args.output,
                         args.radius_factor, args.min_points), indent=2))


if __name__ == "__main__":
    main()
