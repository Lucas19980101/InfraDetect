# SPDX-License-Identifier: BSD-3-Clause
"""本机三维候选复核服务：点云、原图/热图、叠加层及人工判断。

输入可以是 TBBR 原标注、模型多视角结果、BFDD 原标注或几何基线；各模式
调用不同的 build_*_payload 适配成共同的浏览数据。HTTP 只监听本机，
人工结果单独保存，不修改原图、mask、权重或点云。彩色三维点是候选证据，
不是已证实的工程缺陷。交互方式参考 south_building_review.py。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import secrets
import tempfile
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import cv2
import numpy as np


# 不同模式的人工选项刻意分开：几何偏离不能直接使用“确认热桥”的语义。
STATUSES = {"confirmed", "rejected", "uncertain", "unreviewed"}
GEOMETRY_STATUSES = {"suspected", "structure", "reconstruction_noise", "uncertain", "unreviewed"}
RGB_SUPPORT = {"supported", "not_supported", "uncertain"}
INTERFERENCE_JUDGEMENTS = {
    "equipment_suspected", "material_boundary_suspected", "sunlight_or_reflection_suspected",
    "other_suspected", "no_interference_evidence", "uncertain",
}
BFDD_CLASSES = {1: ("Cracks", (230, 66, 60)), 2: ("Peeling", (171, 69, 194)),
                3: ("Hollow Areas", (42, 145, 210)), 4: ("Stains", (31, 168, 102)),
                5: ("Erosion", (220, 169, 35))}


def _dense_ply_points(path: Path) -> list[list[float | int]]:
    """按 COLMAP 预期字段读取稠密 PLY，返回供网页显示的 xyz+RGB 列表。"""
    # 严格匹配字段顺序和字节序，避免把法线或颜色错当三维坐标。
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    expected = [b"property float x", b"property float y", b"property float z",
                b"property float nx", b"property float ny", b"property float nz",
                b"property uchar red", b"property uchar green", b"property uchar blue"]
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply" or handle.readline().strip() != b"format binary_little_endian 1.0":
            raise ValueError("Expected binary little-endian COLMAP PLY")
        count = None
        properties = []
        while True:
            line = handle.readline().strip()
            if not line or line == b"end_header":
                break
            if line.startswith(b"element vertex "):
                count = int(line.split()[-1])
            elif line.startswith(b"property "):
                properties.append(line)
            elif line.startswith(b"element ") and not line.startswith(b"element vertex "):
                raise ValueError("Unexpected PLY element")
        if line != b"end_header" or count is None or properties != expected:
            raise ValueError("Unexpected COLMAP PLY schema")
        vertices = np.fromfile(handle, dtype=dtype, count=count)
        if len(vertices) != count or handle.read(1):
            raise ValueError("Truncated or extended COLMAP PLY")
    if not np.isfinite(np.stack([vertices[key] for key in ("x", "y", "z")])).all():
        raise ValueError("Non-finite dense cloud coordinates")
    return [[float(v["x"]), float(v["y"]), float(v["z"]),
             int(v["red"]), int(v["green"]), int(v["blue"])] for v in vertices]


def _class_bits(value: int, names: list[str]) -> list[str]:
    """把单像素的 RGB 缺陷位编码还原成可读的类别名称。"""
    return [name for index, name in enumerate(names) if value & (1 << index)]


def group_candidates(candidates: list[dict]) -> list[dict]:
    """把三维近邻且在共享照片中也相近的候选点组成一个复核组。"""
    if not candidates:
        return []
    # 1. 从点云自身的最近邻间距估计连接门槛，不假设 COLMAP 有米制尺度。
    xyz = np.asarray([point["xyz"] for point in candidates], np.float64)
    distances = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    positive = nearest[np.isfinite(nearest) & (nearest > 1e-9)]
    threshold = float(np.median(positive) * 3) if len(positive) else 0.0
    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        """并查集查找点所属的组，同时压缩路径以减少重复查询。"""
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    # 2. 只有同一有效照片中的两个标注位置也足够近，才允许三维邻点并组。
    observations = [
        {row["scene_id"]: row for row in point["observations"] if row["annotated"]}
        for point in candidates
    ]
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            if distances[left, right] > threshold:
                continue
            common = observations[left].keys() & observations[right].keys()
            if any(np.hypot(observations[left][scene]["x"] - observations[right][scene]["x"],
                            observations[left][scene]["y"] - observations[right][scene]["y"])
                   <= 48 for scene in common):
                parent[find(right)] = find(left)
    # 3. 输出组 ID 和 RGB/设备/边界提示，保留“空间组并非病害实例”的说明。
    components: dict[int, list[dict]] = {}
    for index, point in enumerate(candidates):
        components.setdefault(find(index), []).append(point)
    groups = []
    for members in components.values():
        ids = sorted(int(point["id"]) for point in members)
        group_id = f"group_{ids[0]}"
        for point in members:
            point["group_id"] = group_id
        supporting = [row for point in members for row in point["observations"]
                      if row["annotated"]]
        exact = sorted({name for row in supporting for name in row.get("rgb_exact_classes", [])})
        nearby = sorted({name for row in supporting for name in row.get("rgb_near_classes", [])})
        status = sorted({int(row["fusion_status"]) for row in supporting})
        groups.append({
            "id": group_id, "point_ids": ids, "point_count": len(ids),
            "positive_scenes": sorted({row["scene_id"] for row in supporting}),
            "rgb_exact_classes": exact, "rgb_near_classes": nearby,
            "equipment_model_flag": 3 in status,
            "boundary_or_registration_flag": 2 in status,
            "nonwall_flag": 4 in status,
            "rgb_support_evidence": ("exact_visible_candidate" if exact else
                                     "nearby_visible_candidate" if nearby else
                                     "no_visible_candidate_at_sampled_points"),
            "interference_evidence": ("equipment_model_flag_not_confirmed_cause" if 3 in status else
                                      "boundary_or_registration_flag" if 2 in status else
                                      "nonwall_flag" if 4 in status else
                                      "no_known_interference_flag_cause_unknown"),
            "grouping_note": "3D-near and <=48 pixels in a shared positive view; "
                             "not a physical defect instance",
            "distance_threshold_model_units": threshold,
        })
    groups.sort(key=lambda row: (-row["point_count"], row["id"]))
    return groups


def _rows(manifest: Path, flight: str) -> dict[str, dict[str, str]]:
    """从 TBBR 清单只取指定航线的 train 行，建立 scene_id→照片记录。"""
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = {row["name"]: row for row in csv.DictReader(handle)
                if row["group"] == flight and row["split"] == "train"}
    if not rows:
        raise ValueError(f"No train images for {flight}")
    return rows


def _source_map(path: Path) -> dict[str, dict[str, str]]:
    """读取非 TBBR 场景的 RGB/IR 配对表，验证每张图可读取且 ID 唯一。"""
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if mapping.get("schema_version") != 1 or not isinstance(mapping.get("scenes"), list):
        raise ValueError("Expected source-map schema version 1 with scenes list")
    rows = {}
    for scene in mapping["scenes"]:
        stem = scene["scene_id"]
        if not stem or stem in rows or Path(stem).name != stem:
            raise ValueError(f"Invalid or duplicate scene ID: {stem}")
        rgb, thermal = Path(scene["rgb"]), Path(scene["thermal"])
        if not rgb.is_file() or not thermal.is_file():
            raise FileNotFoundError(f"Missing paired images: {stem}")
        rows[stem] = {"rgb": str(rgb.resolve()), "thermal": str(thermal.resolve())}
    if not rows:
        raise ValueError("Source map has no scenes")
    return rows


def build_payload(model_path: Path, manifest: Path, flight: str,
                  min_positive_views: int = 2) -> tuple[dict, dict]:
    """把 TBBR 原始热桥标注经 COLMAP 稀疏轨迹放到三维点上。"""
    import pycolmap

    if min_positive_views < 1:
        raise ValueError("min_positive_views must be positive")
    # 1. 重建的照片集合必须与 train 清单完全对齐，避免错配其他航线。
    reconstruction = pycolmap.Reconstruction(model_path)
    rows = _rows(manifest, flight)
    images = sorted(reconstruction.images.values(), key=lambda item: item.name)
    names = {Path(image.name).stem for image in images}
    if names != rows.keys():
        raise ValueError(f"Reconstruction/manifest mismatch: {len(names - rows.keys())} "
                         f"unlisted images, {len(rows.keys() - names)} unregistered images")
    # 2. 第一遍计数：同一个 3D 点有多少张有效照片标成热桥；255 是 Ignore。
    positive = Counter()
    observed = Counter()
    for image in images:
        stem = Path(image.name).stem
        mask = cv2.imread(rows[stem]["mask"], cv2.IMREAD_UNCHANGED)
        camera = reconstruction.cameras[image.camera_id]
        if mask is None or mask.shape != (camera.height, camera.width):
            raise ValueError(f"Mask and camera size differ: {stem}")
        if not np.isin(mask, (0, 1, 255)).all():
            raise ValueError(f"Unexpected mask values for {stem}")
        for point in image.points2D:
            if not point.has_point3D():
                continue
            x, y = np.rint(point.xy).astype(int)
            if 0 <= x < camera.width and 0 <= y < camera.height and mask[y, x] != 255:
                observed[point.point3D_id] += 1
                if mask[y, x] == 1:
                    positive[point.point3D_id] += 1

    candidate_ids = {point_id for point_id, count in positive.items()
                     if count >= min_positive_views}
    # 3. 第二遍只收集达门槛候选的逐图像素位置，供网页展示黄圈与原图。
    observations: dict[int, list[dict]] = {point_id: [] for point_id in candidate_ids}
    for image in images:
        stem = Path(image.name).stem
        mask = cv2.imread(rows[stem]["mask"], cv2.IMREAD_UNCHANGED)
        camera = reconstruction.cameras[image.camera_id]
        for point in image.points2D:
            point_id = point.point3D_id if point.has_point3D() else -1
            if point_id not in candidate_ids:
                continue
            x, y = np.rint(point.xy).astype(int)
            if 0 <= x < camera.width and 0 <= y < camera.height and mask[y, x] != 255:
                observations[point_id].append({
                    "image": image.name, "scene_id": stem,
                    "x": round(float(point.xy[0]), 2),
                    "y": round(float(point.xy[1]), 2),
                    "width": camera.width, "height": camera.height,
                    "annotated": bool(mask[y, x] == 1),
                })

    # 4. 候选按支持视角排序；背景点仅抽样显示，不影响候选点统计。
    points = reconstruction.points3D
    candidates = []
    for point_id in candidate_ids:
        point = points[point_id]
        candidates.append({
            "id": int(point_id), "xyz": [float(value) for value in point.xyz],
            "positive_views": int(positive[point_id]), "observed_views": int(observed[point_id]),
            "observations": sorted(observations[point_id],
                                   key=lambda item: (not item["annotated"], item["image"])),
        })
    candidates.sort(key=lambda point: (-point["positive_views"],
                                       -point["positive_views"] / point["observed_views"], point["id"]))
    stride = max(1, len(points) // 30000)
    background = [[*map(float, point.xyz), *map(int, point.color)]
                  for ordinal, (_, point) in enumerate(sorted(points.items()))
                  if ordinal % stride == 0]
    summary = {
        "flight": flight, "registered_images": len(images), "sparse_points": len(points),
        "background_display_points": len(background), "candidate_points": len(candidates),
        "min_annotated_views": min_positive_views,
        "annotation_source": "TBBRv2 thermal-bridge COCO masks rasterized in IR/data/tbbr_thermal",
        "interpretation": "Sparse points supported by per-photo annotations; not model predictions "
                          "or independently verified physical defects",
    }
    return {"background": background, "candidates": candidates, "summary": summary}, rows


def build_geometry_payload(result_dir: Path, ply_override: Path | None = None,
                           *, model_path: Path | None = None,
                           workspace: Path | None = None,
                           image_dir: Path | None = None,
                           source_map: Path | None = None,
                           homographies: Path | None = None) -> tuple[dict, dict]:
    """Show saved wall-plane scores on the same PLY point indices used by inference.

    Geometry-only review intentionally has no RGB/IR photo correspondence. A
    high plane distance can also mean a window, railing or reconstruction
    artifact, so the payload calls points candidates rather than defects.
    """

    # 1. 几何分数必须绑定到当时的 PLY；点数相同但点序不同也不可混用。
    report = json.loads((result_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("backend") != "wall_plane_v0":
        raise ValueError("Expected a wall_plane_v0 geometry report")
    ply = ply_override or Path(report["source_ply"])
    expected_hash = report.get("source_ply_sha256")
    if expected_hash is not None:
        digest = hashlib.sha256()
        with ply.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected_hash:
            raise ValueError("Geometry PLY differs from the scored source cloud")
    elif ply.resolve() != Path(report["source_ply"]).resolve():
        raise ValueError("Legacy geometry result requires the original source PLY path")
    import open3d as o3d
    cloud = o3d.io.read_point_cloud(str(ply))
    if cloud.is_empty():
        raise ValueError(f"Empty or unreadable geometry PLY: {ply}")
    xyz = np.asarray(cloud.points)
    if len(xyz) != report["point_count"]:
        raise ValueError("PLY point count differs from geometry score report")
    with np.load(result_dir / "point_scores.npz", allow_pickle=False) as saved:
        scores = saved["scores"]
        candidate = saved["candidate"]
    if len(scores) != len(xyz) or len(candidate) != len(xyz):
        raise ValueError("Geometry scores do not match PLY point order")
    if int(np.count_nonzero(candidate)) != report["candidate_count"]:
        raise ValueError("Geometry candidate count differs from report")
    # 2. 浏览器最多显示最高分 5000 个候选；NPZ 仍保存全部结果。
    indexes = np.flatnonzero(candidate)
    # Limit browser payload; preserve the strongest scores and disclose display sampling.
    if len(indexes) > 5000:
        indexes = indexes[np.argsort(scores[indexes])[-5000:]]
    indexes = indexes[np.argsort(scores[indexes])[::-1]]
    colors = np.asarray(cloud.colors)
    stride = max(1, len(xyz) // 30000)
    background = []
    for index in range(0, len(xyz), stride):
        rgb = np.rint(colors[index] * 255).astype(int).tolist() if len(colors) == len(xyz) else [145, 153, 163]
        background.append([*xyz[index].astype(float).tolist(), *rgb])
    candidates = [
        {"id": int(index), "xyz": xyz[index].astype(float).tolist(),
         "geometry_score": float(scores[index]), "positive_views": 0,
         "observed_views": 0, "observations": []}
        for index in indexes
    ]
    # 3. 只有额外提供相机和深度时才链接照片，不能靠空间接近乱配图。
    rows = (attach_geometry_photos(candidates, model_path=model_path,
                                   workspace=workspace, image_dir=image_dir,
                                   source_map=source_map, homographies=homographies)
            if model_path is not None and workspace is not None and image_dir is not None else {})
    groups = group_geometry_candidates(candidates, float(report["diagonal"]))
    summary = {"evidence_mode": "geometry_plane_candidate",
               "dense_points": len(xyz), "background_display_points": len(background),
               "candidate_points": int(report["candidate_count"]),
               "displayed_candidates": len(candidates),
               "candidate_groups": len(groups),
               "linked_images": len(rows),
               "thermal_display": "paired_image" if rows and all("thermal" in row for row in rows.values()) else "none",
               "candidate_threshold": report["candidate_threshold"],
               "source_ply": str(ply.resolve()),
               "interpretation": "Plane-deviation candidates only; not verified defects"}
    return {"background": background, "candidates": candidates,
            "groups": groups, "summary": summary}, rows


def group_geometry_candidates(candidates: list[dict], diagonal: float) -> list[dict]:
    """Connect nearby displayed candidates without inferring a defect identity.

    The radius follows candidate spacing but is capped to 1% of cloud size, so
    isolated hot spots cannot join across a large facade. The grouping is on
    the displayed subset only; any omitted candidates remain in the NPZ.
    """

    if not candidates:
        return []
    from scipy.spatial import cKDTree

    # 半径受点间距和云整体尺寸共同约束；只作用于本次显示的候选子集。
    xyz = np.asarray([point["xyz"] for point in candidates], dtype=np.float64)
    tree = cKDTree(xyz)
    if len(xyz) > 1:
        nearest = tree.query(xyz, k=2)[0][:, 1]
        positive = nearest[nearest > 1e-9]
        radius = min(float(np.median(positive) * 3), diagonal * 0.01) if len(positive) else 0.0
    else:
        radius = 0.0
    parent = list(range(len(candidates)))

    def root(index: int) -> int:
        """返回几何候选所属组的根节点，供近邻点归组。"""
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left, right in tree.query_pairs(radius):
        parent[root(right)] = root(left)
    components: dict[int, list[dict]] = {}
    for index, point in enumerate(candidates):
        components.setdefault(root(index), []).append(point)
    groups = []
    for members in components.values():
        ids = sorted(int(point["id"]) for point in members)
        group_id = f"geometry_{ids[0]}"
        for point in members:
            point["group_id"] = group_id
        scenes = sorted({row["scene_id"] for point in members
                         for row in point["observations"]})
        groups.append({"id": group_id, "point_ids": ids,
                       "point_count": len(ids),
                       "max_score": max(point["geometry_score"] for point in members),
                       "positive_scenes": scenes,
                       "distance_threshold_model_units": radius,
                       "grouping_note": "Spatially adjacent displayed points only; not one verified defect."})
    return sorted(groups, key=lambda row: (-row["max_score"], row["id"]))


def attach_geometry_photos(candidates: list[dict], *, model_path: Path,
                           workspace: Path, image_dir: Path,
                           source_map: Path | None = None,
                           homographies: Path | None = None) -> dict[str, dict[str, str]]:
    """Link points to RGB/IR only after COLMAP depth-visibility checks.

    Original cameras supply pixel coordinates. Dense-workspace depth maps
    reject points hidden by foreground geometry. Optional RGB→IR homographies
    are required before an IR marker can be shown.
    """

    import pycolmap
    from .geometry.project import (
        apply_homography, project_original, visible_indices,
    )
    from scipy.io import loadmat

    # 原模型负责像素坐标，稠密工作区的深度图负责遮挡；两者位姿要一致。
    original = pycolmap.Reconstruction(model_path)
    dense = pycolmap.Reconstruction(workspace / "sparse")
    original_by_name = {image.name: image for image in original.images.values()}
    mapped = _source_map(source_map) if source_map is not None else None
    rows: dict[str, dict[str, str]] = {}
    if not candidates:
        return rows
    xyz = np.asarray([point["xyz"] for point in candidates], dtype=np.float64)
    for image in sorted(dense.images.values(), key=lambda value: value.name):
        stem = Path(image.name).stem
        original_image = original_by_name.get(image.name)
        if original_image is None:
            raise ValueError(f"Image absent from original reconstruction: {image.name}")
        rgb_path = Path(mapped[stem]["rgb"]) if mapped is not None and stem in mapped else image_dir / image.name
        if not rgb_path.is_file():
            continue
        depth_path = workspace / "stereo" / "depth_maps" / f"{image.name}.geometric.bin"
        if not depth_path.is_file():
            continue
        pose = np.asarray(image.cam_from_world().matrix(), dtype=np.float64)
        original_pose = np.asarray(original_image.cam_from_world().matrix(), dtype=np.float64)
        if not np.allclose(pose, original_pose, atol=1e-5, rtol=0):
            raise ValueError(f"Original/dense camera poses differ: {image.name}")
        # 看不见的点不能强行画在照片里，更不能参与人工“漏检”判断。
        depth_map = pycolmap.DepthMap()
        depth_map.read(depth_path)
        indexes, camera_xyz = visible_indices(
            xyz, pose, dense.cameras[image.camera_id], depth_map.to_array(), 0.05)
        if not len(indexes):
            continue
        camera = original.cameras[original_image.camera_id]
        u, v = project_original(camera, camera_xyz)
        valid = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (v >= 0) & (u < camera.width) & (v < camera.height)
        if not valid.any():
            continue
        rows[stem] = {"rgb": str(rgb_path.resolve())}
        ir_x = ir_y = ir_valid = None
        # IR 黄圈另需该图自己的 RGB→IR 配准矩阵；无矩阵只显示 RGB。
        if mapped is not None and stem in mapped and homographies is not None:
            thermal_path = Path(mapped[stem]["thermal"])
            matrix_path = next((homographies / f"{stem}{suffix}" for suffix in (".mat", ".npy", ".json")
                                if (homographies / f"{stem}{suffix}").is_file()), None)
            if matrix_path is not None:
                if matrix_path.suffix == ".mat":
                    matrix = np.asarray(loadmat(matrix_path)["H"], dtype=np.float64)
                elif matrix_path.suffix == ".npy":
                    matrix = np.asarray(np.load(matrix_path, allow_pickle=False), dtype=np.float64)
                else:
                    matrix = np.asarray(json.loads(matrix_path.read_text(encoding="utf-8")), dtype=np.float64)
                if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
                    raise ValueError(f"Invalid RGB-to-IR homography: {matrix_path}")
                ir_x, ir_y, ir_valid = apply_homography(u, v, matrix)
                ir_image = cv2.imread(str(thermal_path), cv2.IMREAD_UNCHANGED)
                if ir_image is None:
                    raise ValueError(f"Cannot read paired thermal image: {thermal_path}")
                rows[stem]["thermal"] = str(thermal_path.resolve())
                ir_size = ir_image.shape[:2]
        for local, point_index in enumerate(indexes):
            if not valid[local]:
                continue
            observation = {"image": image.name, "scene_id": stem,
                           "x": float(u[local]), "y": float(v[local]),
                           "width": int(camera.width), "height": int(camera.height),
                           "annotated": False}
            if ir_valid is not None and ir_valid[local] and 0 <= ir_x[local] < ir_size[1] and 0 <= ir_y[local] < ir_size[0]:
                observation.update({"ir_x": float(ir_x[local]), "ir_y": float(ir_y[local]),
                                    "ir_width": int(ir_size[1]), "ir_height": int(ir_size[0])})
            candidates[int(point_index)]["observations"].append(observation)
    return rows


def build_bfdd_dense_payload(model_path: Path, dense_ply: Path,
                             dataset: Path) -> tuple[dict, dict]:
    """显示 BFDD 稠密云，并用稀疏轨迹把原始 RGB 主导标签对应到三维。"""
    import pycolmap

    # 1. 三种原文件逐张成对核对；稠密云用于背景，不冒充逐点标注真值。
    reconstruction = pycolmap.Reconstruction(model_path)
    images = sorted(reconstruction.images.values(), key=lambda item: item.name)
    rows: dict[str, dict[str, str]] = {}
    masks: dict[int, np.ndarray] = {}
    for image in images:
        stem = Path(image.name).stem
        rgb = dataset / "RGB" / image.name
        ir = dataset / "IR" / f"{stem}.png"
        label = dataset / "Label" / f"{stem}.png"
        if not all(path.is_file() for path in (rgb, ir, label)):
            raise FileNotFoundError(f"Missing BFDD RGB/IR/label pair: {stem}")
        camera = reconstruction.cameras[image.camera_id]
        mask = cv2.imread(str(label), cv2.IMREAD_UNCHANGED)
        rgb_image = cv2.imread(str(rgb), cv2.IMREAD_UNCHANGED)
        ir_image = cv2.imread(str(ir), cv2.IMREAD_UNCHANGED)
        shape = (camera.height, camera.width)
        if (mask is None or mask.ndim != 2 or mask.shape != shape or
                rgb_image is None or rgb_image.shape[:2] != shape or
                ir_image is None or ir_image.shape[:2] != shape or
                not np.isin(mask, (0, *BFDD_CLASSES)).all()):
            raise ValueError(f"BFDD pair/camera dimensions or labels differ: {stem}")
        if stem in rows:
            raise ValueError(f"Duplicate BFDD stem: {stem}")
        rows[stem] = {"rgb": str(rgb), "thermal": str(ir), "mask": str(label)}
        masks[image.image_id] = mask
    if not images:
        raise ValueError("BFDD reconstruction has no images")

    # 2. 同一稀疏点须在至少两张图命中同一 BFDD 类别，才成为展示候选。
    candidates = []
    for point_id, point in reconstruction.points3D.items():
        observations = []
        counts: Counter[int] = Counter()
        for element in point.track.elements:
            image = reconstruction.images[element.image_id]
            camera = reconstruction.cameras[image.camera_id]
            xy = image.points2D[element.point2D_idx].xy
            x, y = np.rint(xy).astype(int)
            if not (0 <= x < camera.width and 0 <= y < camera.height):
                continue
            label = int(masks[image.image_id][y, x])
            if label:
                counts[label] += 1
            observations.append({"image": image.name, "scene_id": Path(image.name).stem,
                                 "x": round(float(xy[0]), 2), "y": round(float(xy[1]), 2),
                                 "width": camera.width, "height": camera.height,
                                 "raw_label": label})
        if not counts or max(counts.values()) < 2:
            continue
        class_id, votes = min(counts.items(), key=lambda item: (-item[1], item[0]))
        for observation in observations:
            observation["annotated"] = observation["raw_label"] == class_id
        observations.sort(key=lambda row: (not row["annotated"], row["image"]))
        candidates.append({"id": int(point_id), "xyz": [float(value) for value in point.xyz],
                           "class_id": class_id, "class_name": BFDD_CLASSES[class_id][0],
                           "display_color": BFDD_CLASSES[class_id][1],
                           "positive_views": votes, "observed_views": len(observations),
                           "observations": observations})
    candidates.sort(key=lambda point: (-point["positive_views"], point["class_id"], point["id"]))
    # 3. 前景类别来自稀疏投影，背景来自完整稠密 PLY；两者来源不同。
    background = _dense_ply_points(dense_ply)
    summary = {"flight": "BFDD Dataset_1x 2025-06-24 fragment",
               "evidence_mode": "bfdd_dense_original_labels",
               "registered_images": len(images), "sparse_points": len(reconstruction.points3D),
               "dense_points": len(background), "background_display_points": len(background),
               "candidate_points": len(candidates), "min_same_class_label_views": 2,
               "candidate_class_counts": dict(Counter(point["class_name"] for point in candidates)),
               "class_colors": {name: color for name, color in BFDD_CLASSES.values()},
               "annotation_source": str((dataset / "Label").resolve()),
               "interpretation": "Dense cloud for geometry; original RGB-dominant BFDD labels "
                                 "are projected through sparse tracks. Not verified IR-visible "
                                 "defects or physical diagnosis."}
    return {"background": background, "candidates": candidates, "summary": summary}, rows


def build_model_payload(model_path: Path, manifest: Path, flight: str,
                        audit: Path, scenes: Path,
                        source_map: Path | None = None) -> tuple[dict, dict]:
    """Adapt depth-checked model evidence to the existing point/photo viewer."""
    import pycolmap

    # 1. 使用上游已做深度可见性核验的审计点，不从二维 mask 重新猜 3D 点。
    rows = _source_map(source_map) if source_map is not None else _rows(manifest, flight)
    reconstruction = pycolmap.Reconstruction(model_path)
    audit_summary = json.loads((audit / "summary.json").read_text(encoding="utf-8"))
    audit_points = json.loads((audit / "points.json").read_text(encoding="utf-8"))
    registered = {image.name: image for image in reconstruction.images.values()}
    rgb_evidence: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    candidates = []
    used_scenes = set()
    # 2. 逐点核对审计 ID、三维坐标、照片位置和 RGB/IR 对应图，拒绝旧数据混入。
    for point in audit_points:
        if not point["geometry_distinct_support"] or point["thermal_positive_views"] < 2:
            continue
        if point["point3d_id"] not in reconstruction.points3D:
            raise ValueError("Audit point missing from reconstruction")
        actual_xyz = reconstruction.points3D[point["point3d_id"]].xyz
        if not np.allclose(actual_xyz, point["xyz"], atol=1e-5, rtol=0):
            raise ValueError("Audit/reconstruction 3D coordinates differ")
        observations = []
        for observation in point["observations"]:
            image_name = observation["image"]
            stem = Path(image_name).stem
            if image_name not in registered or stem not in rows:
                raise ValueError(f"Unregistered audit observation: {image_name}")
            used_scenes.add(stem)
            camera = reconstruction.cameras[registered[image_name].camera_id]
            x, y = observation["xy"]
            if not (0 <= x < camera.width and 0 <= y < camera.height):
                raise ValueError(f"Out-of-image observation: {image_name}")
            mask_path = scenes / "scenes" / stem / "fusion" / stem / "thermal_anomaly_mask.png"
            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)
            # 同一张图的两张 RGB 位图只读一次；逐点用像素值交叉检查审计。
            if stem not in rgb_evidence:
                scene_run = scenes / "scenes" / stem
                report = json.loads((scene_run / "report.json").read_text(encoding="utf-8"))
                if report["scene"]["scene_id"] != stem or not report["scene"]["registration_accepted"]:
                    raise ValueError(f"Unaccepted RGB/IR registration: {stem}")
                exact = cv2.imread(str(scene_run / "cross_modal" / "rgb_defect_bits_in_ir.png"),
                                   cv2.IMREAD_UNCHANGED)
                near = cv2.imread(str(scene_run / "cross_modal" /
                                      "rgb_defect_near_bits_in_ir.png"), cv2.IMREAD_UNCHANGED)
                if (exact is None or near is None or exact.ndim != 2 or near.ndim != 2 or
                        exact.shape != (camera.height, camera.width) or near.shape != exact.shape):
                    raise ValueError(f"Missing or mismatched RGB evidence maps: {stem}")
                rgb_evidence[stem] = (exact, near, report["rgb"]["rgb_defect_class_names"])
            exact, near, class_names = rgb_evidence[stem]
            px, py = int(round(x)), int(round(y))
            if not (0 <= px < camera.width and 0 <= py < camera.height):
                raise ValueError(f"Rounded observation outside image: {image_name}")
            if int(near[py, px]) != int(observation["rgb_near_bits"]):
                raise ValueError(f"RGB evidence/audit mismatch: {image_name}")
            observations.append({"image": image_name, "scene_id": stem,
                                 "x": x, "y": y, "width": camera.width,
                                 "height": camera.height,
                                 "annotated": bool(observation["thermal_positive"]),
                                 "fusion_status": observation["fusion_status"],
                                 "rgb_relation": observation["rgb_relation"],
                                 "rgb_exact_classes": _class_bits(int(exact[py, px]), class_names),
                                 "rgb_near_classes": _class_bits(int(near[py, px]), class_names)})
        candidates.append({"id": point["point3d_id"], "xyz": point["xyz"],
                           "positive_views": point["thermal_positive_views"],
                           "observed_views": point["visible_views"],
                           "observations": observations})
    candidates.sort(key=lambda p: (-p["positive_views"], p["id"]))
    if source_map is not None and used_scenes != set(rows):
        raise ValueError("Source map must match the audited view set exactly")
    if source_map is not None:
        for stem in used_scenes:
            report = json.loads((scenes / "scenes" / stem / "report.json")
                                .read_text(encoding="utf-8"))
            scene = report["scene"]
            if scene["scene_id"] != stem or not scene["registration_accepted"]:
                raise ValueError(f"Unaccepted or mismatched RGB/IR registration: {stem}")
    # 3. 给连续邻点加浏览组，背景点下采样只是为了前端绘制速度。
    groups = group_candidates(candidates)
    points = reconstruction.points3D
    stride = max(1, len(points) // 30000)
    background = [[*map(float, point.xyz), *map(int, point.color)]
                  for ordinal, (_, point) in enumerate(sorted(points.items()))
                  if ordinal % stride == 0]
    summary = {"flight": flight, "registered_images": len(registered),
               "sparse_points": len(points), "background_display_points": len(background),
               "candidate_points": len(candidates), "candidate_groups": len(groups),
               "evidence_mode": "model_prediction",
               "thermal_display": "paired_image" if source_map is not None else "stack_channel_4",
               "audit": str(audit.resolve()), "scene_root": str(scenes.resolve()),
               "scope": audit_summary["scope"],
               "interpretation": "Depth-checked, multiview model candidates; not physical defects"}
    return {"background": background, "candidates": candidates, "groups": groups,
            "summary": summary}, rows


def _save_reviews(path: Path, reviews: dict) -> None:
    """先写临时 JSON 再原子替换，避免保存中断损坏人工判断。"""
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".reviews-",
                                     suffix=".json", encoding="utf-8", delete=False) as handle:
        json.dump(reviews, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def serve(model_path: Path, manifest: Path, image_dir: Path, review_dir: Path,
          flight: str, port: int, min_positive_views: int,
          model_audit: Path | None = None, scenes: Path | None = None,
          source_map: Path | None = None, bfdd_root: Path | None = None,
          dense_ply: Path | None = None,
          geometry_result: Path | None = None,
          geometry_ply: Path | None = None,
          geometry_workspace: Path | None = None,
          geometry_homographies: Path | None = None) -> None:
    """准备选定证据模式并启动仅监听 127.0.0.1 的人工复核网页。"""
    # 1. 四种来源互斥：几何、BFDD 原标签、模型候选、TBBR 原标注。
    if geometry_result is not None:
        if bfdd_root is not None or model_audit is not None:
            raise ValueError("Geometry review cannot be combined with BFDD or thermal model mode")
        payload, rows = build_geometry_payload(
            geometry_result, geometry_ply,
            model_path=model_path if geometry_workspace is not None else None,
            workspace=geometry_workspace, image_dir=image_dir,
            source_map=source_map, homographies=geometry_homographies)
    elif bfdd_root is not None:
        if dense_ply is None or model_audit is not None:
            raise ValueError("BFDD dense mode needs a PLY and cannot use model audit")
        payload, rows = build_bfdd_dense_payload(model_path, dense_ply, bfdd_root)
    elif model_audit is not None:
        if scenes is None:
            raise ValueError("Model mode requires integrated scene root")
        payload, rows = build_model_payload(model_path, manifest, flight, model_audit,
                                            scenes, source_map)
    else:
        payload, rows = build_payload(model_path, manifest, flight, min_positive_views)
    if source_map is None and bfdd_root is None and geometry_result is None and {path.stem for path in image_dir.glob("*.jpg")} != rows.keys():
        raise ValueError("RGB export and train manifest do not have identical scene IDs")
    # 2. 人工判断写到单独目录；几何模式尤其要绑定原 PLY/分数身份。
    review_dir.mkdir(parents=True, exist_ok=True)
    review_path = review_dir / "reviews.json"
    group_review_path = review_dir / "group_reviews.json"
    if geometry_result is not None:
        # Point IDs are PLY row numbers, so stale reviews from another geometry
        # run must never be shown as decisions about this cloud.
        evidence = {"result": str(geometry_result.resolve()),
                    "source_ply": payload["summary"]["source_ply"],
                    "candidate_threshold": payload["summary"]["candidate_threshold"],
                    "group_ids": [group["id"] for group in payload["groups"]]}
        identity = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
        binding = review_dir / "geometry_review_source.json"
        if binding.exists():
            previous = json.loads(binding.read_text(encoding="utf-8"))
            if previous.get("identity") != identity:
                raise ValueError("Review directory belongs to another geometry result; choose a new --review-dir")
        elif review_path.exists() or group_review_path.exists():
            raise ValueError("Existing unbound reviews: choose a new --review-dir")
        else:
            _save_reviews(binding, {"identity": identity, "evidence": evidence})
    reviews = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {}
    group_reviews = (json.loads(group_review_path.read_text(encoding="utf-8"))
                     if group_review_path.exists() else {})
    valid_group_ids = {group["id"] for group in payload.get("groups", [])}
    if model_audit is not None or geometry_result is not None:
        evidence = {"summary": payload["summary"], "groups": payload["groups"],
                    "model": str(model_path.resolve())}
        serialized = json.dumps(evidence, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(serialized.encode()).hexdigest()[:12]
        snapshot = review_dir / f"evidence_groups_{digest}.json"
        if not snapshot.exists():
            _save_reviews(snapshot, evidence)
    # 3. 页面令牌只授权本机 POST 保存，不是用于远程公开部署的认证方案。
    valid_ids = {point["id"] for point in payload["candidates"]}
    token = secrets.token_urlsafe(24)
    html = Path(__file__).with_name("tbbr_review.html").read_bytes().replace(
        b"__REVIEW_TOKEN__", token.encode())

    @lru_cache(maxsize=16)
    def photo(scene_id: str, channel: str, layers: str) -> bytes:
        """返回指定 RGB/IR 原图及所选叠加层；黄圈由前端单独绘制。"""
        row = rows[scene_id]
        if channel == "rgb":
            rgb_path = Path(row["rgb"]) if source_map is not None or bfdd_root is not None or geometry_result is not None else image_dir / f"{scene_id}.jpg"
            if not layers and rgb_path.suffix.lower() in {".jpg", ".jpeg"}:
                return rgb_path.read_bytes()
            image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        else:
            if source_map is not None or bfdd_root is not None or geometry_result is not None:
                if "thermal" not in row:
                    raise ValueError(f"No verified RGB-to-IR link for {scene_id}")
                image = cv2.imread(row["thermal"], cv2.IMREAD_UNCHANGED)
                if image is not None and image.ndim == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif image is not None and image.ndim == 3 and image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            else:
                stack = np.load(row["image"], mmap_mode="r", allow_pickle=False)
                image = cv2.cvtColor(np.asarray(stack[..., 3], dtype=np.uint8), cv2.COLOR_GRAY2BGR)
        if image is None:
            raise ValueError(f"Could not read {scene_id} {channel}")
        # t 是热/原标注 mask；r 是 RGB 缺陷；e 是设备边线，互不覆盖原文件。
        if "t" in layers and bfdd_root is not None:
            mask = cv2.imread(row["mask"], cv2.IMREAD_UNCHANGED)
            if mask is None or mask.shape != image.shape[:2]:
                raise ValueError(f"Invalid BFDD mask for {scene_id}")
            for class_id, (_, rgb_color) in BFDD_CLASSES.items():
                selected = mask == class_id
                image[selected] = (0.45 * image[selected] +
                                   0.55 * np.asarray(rgb_color[::-1])).astype(np.uint8)
        elif "t" in layers:
            mask_path = (scenes / "scenes" / scene_id / "fusion" / scene_id /
                         "thermal_anomaly_mask.png") if model_audit is not None else Path(row["mask"])
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.shape != image.shape[:2]:
                raise ValueError(f"Invalid annotation for {scene_id}")
            selected = mask > 0 if model_audit is not None else mask == 1
            image[selected] = (0.45 * image[selected] +
                               0.55 * np.asarray((0, 90, 255))).astype(np.uint8)
        if channel == "context" or "r" in layers or "e" in layers:
            scene_run = scenes / "scenes" / scene_id
            def layer(path: Path) -> np.ndarray:
                """读取一张与当前照片同尺寸的二维叠加层。"""
                mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if mask is None or mask.ndim != 2 or mask.shape != image.shape[:2]:
                    raise ValueError(f"Invalid evidence layer: {path}")
                return mask
            if "r" in layers:
                rgb = layer(scene_run / "cross_modal" / "rgb_defect_bits_in_ir.png") > 0
                image[rgb] = (0.48 * image[rgb] +
                              0.52 * np.asarray((255, 225, 0))).astype(np.uint8)
            if "e" in layers:
                equipment = layer(scene_run / "fusion" / scene_id / "equipment_mask_in_ir.png") > 0
                equipment_edge = cv2.morphologyEx(
                    equipment.astype(np.uint8), cv2.MORPH_GRADIENT,
                    np.ones((5, 5), np.uint8)) > 0
                image[equipment_edge] = (0.48 * image[equipment_edge] +
                                         0.52 * np.asarray((190, 0, 210))).astype(np.uint8)
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not success:
            raise OSError(f"Could not encode {scene_id}")
        return encoded.tobytes()

    class Handler(BaseHTTPRequestHandler):
        """只开放页面、只读证据 API 和明确的人工保存 API。"""
        def send_bytes(self, body: bytes, content_type: str,
                       status: HTTPStatus = HTTPStatus.OK) -> None:
            """发送带类型与长度的响应，并禁止浏览器缓存人工复核状态。"""
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            """返回页面、点云 JSON 或经参数白名单检查后的照片图层。"""
            request = urlsplit(self.path)
            if request.path == "/":
                self.send_bytes(html, "text/html; charset=utf-8")
            elif request.path == "/api/data":
                self.send_bytes(json.dumps({**payload, "reviews": reviews,
                                            "group_reviews": group_reviews},
                                           ensure_ascii=False).encode(),
                                "application/json; charset=utf-8")
            elif request.path.startswith("/api/photo/"):
                scene_id = unquote(request.path.removeprefix("/api/photo/"))
                options = parse_qs(request.query)
                channel = options.get("channel", ["rgb"])[0]
                annotated = options.get("annotation", ["0"])[0]
                layer_options = options.get("layers")
                layers = layer_options[0] if layer_options else ("t" if annotated == "1" else "")
                if channel == "context" and layer_options is None:
                    layers = "tre"
                allowed_channels = {"rgb", "thermal", "context"} if model_audit else (
                    {"rgb", "thermal"} if geometry_result is None or "thermal" in rows.get(scene_id, {}) else {"rgb"})
                if (scene_id not in rows or channel not in allowed_channels or annotated not in {"0", "1"}
                        or len(layers) > 3 or len(set(layers)) != len(layers)
                        or not set(layers) <= ({"t", "r", "e"} if model_audit else
                                               set() if geometry_result is not None else {"t"})):
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                try:
                    self.send_bytes(photo(scene_id, channel, layers), "image/jpeg")
                except (OSError, ValueError):
                    self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            """验证令牌、候选 ID、选项和长度后保存人工判断。"""
            if self.path not in {"/api/review", "/api/group-review"} or self.headers.get("X-Review-Token") != token:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(length))
                if self.path == "/api/review":
                    point_id = int(request["point_id"])
                    status = request["status"]
                    allowed_statuses = GEOMETRY_STATUSES if geometry_result is not None else STATUSES
                    if point_id not in valid_ids or status not in allowed_statuses:
                        raise ValueError("Invalid point or status")
                else:
                    if model_audit is None and geometry_result is None:
                        raise ValueError("Group review requires model mode")
                    group_id = request["group_id"]
                    notes = request.get("notes", "")
                    if geometry_result is not None:
                        group_status = request["status"]
                        if group_id not in valid_group_ids or group_status not in GEOMETRY_STATUSES or not isinstance(notes, str) or len(notes) > 1000:
                            raise ValueError("Invalid geometry group review")
                    else:
                        rgb_support = request["rgb_support"]
                        interference = request["interference_judgement"]
                        if (group_id not in valid_group_ids or rgb_support not in RGB_SUPPORT or
                                interference not in INTERFERENCE_JUDGEMENTS or
                                not isinstance(notes, str) or len(notes) > 1000):
                            raise ValueError("Invalid group review")
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            # 点与组的复核分开存；撤销点判断会删除该点记录而非写入真值。
            if self.path == "/api/group-review":
                group_reviews[group_id] = ({"status": group_status, "notes": notes,
                                            "review_kind": "geometry_candidate_human_review_not_ground_truth",
                                            "updated_at": datetime.now(timezone.utc).isoformat()}
                                           if geometry_result is not None else
                                           {"rgb_support": rgb_support, "interference_judgement": interference,
                                            "notes": notes, "review_kind": "human_visual_assessment_not_physical_truth",
                                            "updated_at": datetime.now(timezone.utc).isoformat()})
                _save_reviews(group_review_path, group_reviews)
                response = {"group_id": group_id, **group_reviews[group_id]}
            else:
                if status == "unreviewed":
                    reviews.pop(str(point_id), None)
                else:
                    reviews[str(point_id)] = {"status": status,
                                              "updated_at": datetime.now(timezone.utc).isoformat()}
                _save_reviews(review_path, reviews)
                response = {"point_id": point_id, "status": status}
            self.send_bytes(json.dumps(response).encode(), "application/json")

    # 4. 只绑定回环地址；关闭服务不会删除已保存的 JSON。
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"{('BFDD dense' if bfdd_root is not None else 'TBBR')} review: http://127.0.0.1:{port}/", flush=True)
    print(f"3D candidates: {payload['summary']['candidate_points']}", flush=True)
    print(f"Human reviews: {review_path}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    """解析数据模式、路径和端口；校验互斥参数后启动本机服务。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=Path("runs/tbbr_colmap_flug100_sparse_v1/sparse/0"))
    parser.add_argument("--manifest", type=Path, default=Path("data/tbbr_thermal/manifest.csv"))
    parser.add_argument("--images", type=Path,
                        default=Path("runs/tbbr_colmap_flug100_rgb_v1/images"))
    parser.add_argument("--review-dir", type=Path,
                        help="Separate review directory; defaults depend on evidence mode")
    parser.add_argument("--flight", default="Flug1_100")
    parser.add_argument("--min-positive-views", type=int, default=2)
    parser.add_argument("--model-audit", type=Path,
                        help="Use depth-checked model-candidate points instead of COCO labels")
    parser.add_argument("--scenes", type=Path,
                        help="Integrated 2D scene root required with --model-audit")
    parser.add_argument("--source-map", type=Path,
                        help="Optional dataset-neutral RGB/thermal scene map for model mode")
    parser.add_argument("--bfdd-dense", action="store_true",
                        help="Review BFDD original labels on the existing 27-view dense cloud")
    parser.add_argument("--geometry-result", type=Path,
                        help="Show geometry-baseline result on the 3D viewer")
    parser.add_argument("--geometry-ply", type=Path,
                        help="PLY used by geometry-baseline; defaults to report source_ply")
    parser.add_argument("--geometry-model", type=Path,
                        help="Original COLMAP sparse model; requires --geometry-workspace")
    parser.add_argument("--geometry-workspace", type=Path,
                        help="Dense COLMAP workspace with sparse/ and stereo/depth_maps/")
    parser.add_argument("--geometry-homographies", type=Path,
                        help="RGB-to-IR matrices; use with --source-map for paired IR markers")
    parser.add_argument("--bfdd-root", type=Path, default=Path("../collected_datasets/Dataset_1x"))
    parser.add_argument("--bfdd-cloud", type=Path, default=Path(
        "../colmap/benchmark/reconstruction/runs/dataset1x-20250624-sift-affine/dense-8/fused.ply"))
    parser.add_argument("--bfdd-model", type=Path, default=Path(
        "../colmap/benchmark/reconstruction/runs/dataset1x-20250624-sift-affine/sparse/8"))
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    # 不允许把一种模式的照片/模型参数混到另一种模式，避免证据错配。
    if args.source_map is not None and args.model_audit is None and args.geometry_result is None:
        parser.error("--source-map requires --model-audit or --geometry-result")
    if args.bfdd_dense and (args.model_audit is not None or args.source_map is not None or args.scenes is not None):
        parser.error("--bfdd-dense cannot be combined with model audit or integrated scenes")
    if args.geometry_ply is not None and args.geometry_result is None:
        parser.error("--geometry-ply requires --geometry-result")
    if args.geometry_result is not None and (args.bfdd_dense or args.model_audit is not None):
        parser.error("--geometry-result cannot be combined with BFDD or model audit")
    if (args.geometry_model is None) != (args.geometry_workspace is None):
        parser.error("--geometry-model and --geometry-workspace must be provided together")
    if any(value is not None for value in (args.geometry_model, args.geometry_workspace,
                                            args.geometry_homographies)) and args.geometry_result is None:
        parser.error("Geometry camera inputs require --geometry-result")
    if args.geometry_homographies is not None and (args.source_map is None or args.geometry_model is None):
        parser.error("--geometry-homographies requires --source-map and --geometry-model")
    if args.geometry_result is not None and args.source_map is not None and args.geometry_model is None:
        parser.error("Geometry --source-map requires --geometry-model and --geometry-workspace")
    # 默认复核目录按来源分开；用户也可显式提供独立输出目录。
    review_dir = args.review_dir or Path(
        "runs/geometry_plane_review_v0" if args.geometry_result is not None else
        "runs/bfdd_dense_27view_review_v1" if args.bfdd_dense else
        "runs/tbbr_colmap_flug100_model_review_v1" if args.model_audit else
        "runs/tbbr_colmap_flug100_review_v1")
    serve(args.bfdd_model if args.bfdd_dense else args.geometry_model or args.model,
          args.manifest, args.images, review_dir,
          args.flight, args.port, args.min_positive_views, args.model_audit, args.scenes,
          args.source_map, args.bfdd_root if args.bfdd_dense else None,
          args.bfdd_cloud if args.bfdd_dense else None,
          args.geometry_result, args.geometry_ply, args.geometry_workspace,
          args.geometry_homographies)


if __name__ == "__main__":
    main()
