"""让同一个三维候选点回到多张原照片中，检查证据是否相互支持。

输入：已投影的候选点 PLY、COLMAP 相机/深度、每张图的 IR 与 RGB 对应图。
处理：逐视角检查遮挡、投影、配准及热异常，再按同一个 3D 点汇总。
输出：点级、候选组级核验 JSON 和可追溯的观测 NPZ。视角数量不足或
相机位置太相近时，不会冒称为独立多视角支持；支持也不等于物理确诊。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def maximum_view_angle_deg(rays: np.ndarray) -> float:
    """计算同一 3D 点的最大视线夹角，区分不同角度与相邻航线帧。"""
    if len(rays) < 2:
        return 0.0
    unit = rays / np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
    cosine = np.clip(unit @ unit.T, -1.0, 1.0)
    return float(np.degrees(np.arccos(np.min(cosine))))


def summarize_observations(candidate_count: int, point_rows: np.ndarray,
                           image_ordinals: np.ndarray, thermal_positive: np.ndarray,
                           correspondence: np.ndarray, rgb_near_bits: np.ndarray,
                           status: np.ndarray,
                           rays: np.ndarray, min_views: int, min_fraction: float,
                           min_angle_deg: float) -> list[dict]:
    """对每个候选 3D 点汇总不同照片中的有效支持。

    先按点索引聚集可见观测；同一照片无论重复出现多少次只算一个视角。
    除支持视角数外，还要求支持比例和视线夹角达到阈值，以免几张几乎
    同位置的连续帧被误当作独立证据。未被观测到与已观测但阴性分开计数。
    """
    if min_views < 2 or not 0 <= min_fraction <= 1 or not 0 <= min_angle_deg <= 180:
        raise ValueError("Invalid multiview acceptance thresholds")
    if not (len(point_rows) == len(image_ordinals) == len(thermal_positive)
            == len(correspondence) == len(rgb_near_bits) == len(status) == len(rays)):
        raise ValueError("Observation columns differ in length")
    if np.any((point_rows < 0) | (point_rows >= candidate_count)):
        raise ValueError("Observation points are outside candidate range")
    # 1. 把所有照片中的观测按“原三维候选点”归拢，而非按二维像素合并。
    order = np.argsort(point_rows, kind="stable")
    sorted_rows = point_rows[order]
    boundaries = np.searchsorted(sorted_rows, np.arange(candidate_count + 1))
    checks = []
    for row in range(candidate_count):
        local = order[boundaries[row]:boundaries[row + 1]]
        positive = local[thermal_positive[local] > 0]
        # 同一照片重复观测只算一次，阴性观测仍进入支持比例的分母。
        visible_images = np.unique(image_ordinals[local])
        positive_images = np.unique(image_ordinals[positive])
        angle = maximum_view_angle_deg(rays[positive])
        fraction = len(positive_images) / len(visible_images) if len(visible_images) else 0.0
        # 数量、比例和角度三道门共同限制“独立支持”的含义。
        supported = (len(positive_images) >= min_views and fraction >= min_fraction
                     and angle >= min_angle_deg)
        checks.append({
            "candidate_row": row,
            "visible_views": int(len(visible_images)),
            "thermal_positive_views": int(len(positive_images)),
            "thermal_positive_fraction": float(fraction),
            "max_positive_view_angle_deg": angle,
            "geometry_independent_support": bool(supported),
            "rgb_colocated_views": int(np.count_nonzero(correspondence[positive] == 3)),
            "rgb_nearby_views": int(np.count_nonzero(rgb_near_bits[positive] > 0)),
            "wall_supported_views": int(np.count_nonzero(status[positive] == 1)),
            "equipment_influenced_views": int(np.count_nonzero(status[positive] == 3)),
            "nonwall_overlap_views": int(np.count_nonzero(status[positive] == 4)),
            "supporting_image_ordinals": positive_images.astype(int).tolist(),
        })
    return checks


def summarize_instances(checks: list[dict], membership: dict[int, int],
                        image_names: list[str]) -> list[dict]:
    """按上游组 ID 聚合点级检查；一组仍只是复核单位，不是确诊病害。"""
    # 先建立 group -> point checks，再统计每张照片真正支持多少个点。
    groups: dict[int, list[dict]] = {}
    for check in checks:
        ident = membership.get(check["source_point_index"])
        if ident is not None:
            groups.setdefault(ident, []).append(check)
    result = []
    for ident, points in sorted(groups.items()):
        per_image: dict[int, int] = {}
        for point in points:
            for ordinal in point["supporting_image_ordinals"]:
                per_image[ordinal] = per_image.get(ordinal, 0) + 1
        sources = [{"image": image_names[ordinal], "thermal_positive_points": count}
                   for ordinal, count in per_image.items()]
        sources.sort(key=lambda item: (-item["thermal_positive_points"], item["image"]))
        result.append({
            "instance_id": ident, "candidate_points": len(points),
            "independently_supported_points": sum(point["geometry_independent_support"] for point in points),
            "rgb_colocated_point_views": sum(point["rgb_colocated_views"] for point in points),
            "rgb_nearby_point_views": sum(point["rgb_nearby_views"] for point in points),
            "source_images": sources,
            "interpretation": "same 3D points observed in multiple images, subject to manual review",
        })
    return result


def _read_image(path: Path) -> np.ndarray:
    """读取必需的单通道证据图；尺寸在逐场景检查时与 IR 图核对。"""
    result = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if result is None or result.ndim != 2:
        raise ValueError(f"Missing single-channel map: {path}")
    return result


def run(*, projected_ply: Path, projection_summary: Path, original_model: Path,
        dense_workspace: Path, fusion_root: Path, correspondence_root: Path,
        output: Path, membership_file: Path | None = None, min_views: int = 2,
        min_fraction: float = 0.3, min_angle_deg: float = 5.0) -> dict:
    """把已投影的 3D 候选重新放回各原始照片核对证据与遮挡。

    要求投影摘要完整、原始/稠密 COLMAP 模型的图像身份与位姿一致，
    并逐图读取深度、IR 有效区、RGB↔IR 配准和二维对应图。只把可见点
    的观测送入多视角统计；否则背面或被遮挡的点可能被错算作漏检。
    输出是跨视角支持证据，不是自动确诊。
    """
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty: {output}")
    import pycolmap
    from .aggregate import read_vertex_ply
    from ..evidence.fusion import _homography
    from .project import (apply_homography, project_original, visible_indices)

    # 1. 核对投影结果确为完整热异常任务；点云顺序必须与摘要一致。
    source = json.loads(projection_summary.read_text(encoding="utf-8"))
    if source.get("evidence_task") != "thermal_anomaly" or source.get("partial_run"):
        raise ValueError("Require complete thermal-anomaly 3D projection")
    cloud = read_vertex_ply(projected_ply)
    if len(cloud) != source["points"]:
        raise ValueError("Projected PLY/summary point count mismatch")
    indices = np.flatnonzero(cloud["defect_class"] > 0)
    if not len(indices):
        raise ValueError("Projected PLY has no thermal candidate points")
    points = np.column_stack(tuple(cloud[field][indices] for field in "xyz"))
    # 2. 原稀疏模型提供原图像素坐标；稠密模型提供深度遮挡信息。
    original = pycolmap.Reconstruction(original_model)
    dense = pycolmap.Reconstruction(dense_workspace / "sparse")
    images = sorted(dense.images.values(), key=lambda item: item.name)
    if len(images) != source["images"]:
        raise ValueError("Image count differs from original projection")
    names = [image.name for image in images]
    point_parts, image_parts, thermal_parts, correspondence_parts, near_parts = [], [], [], [], []
    status_parts, ray_parts = [], []
    # 3. 对每张照片做可见性、原图投影、RGB→IR 变换和有效区检查。
    for ordinal, image in enumerate(images):
        original_image = original.images[image.image_id]
        if original_image.name != image.name:
            raise ValueError(f"COLMAP image identity mismatch: {image.name}")
        stem = Path(image.name).stem
        fusion = fusion_root / stem
        corresponding = correspondence_root / stem
        fusion_summary = json.loads((fusion / "summary.json").read_text(encoding="utf-8"))
        if not fusion_summary.get("registration_accepted"):
            raise ValueError(f"Unregistered scene cannot enter 3D comparison: {stem}")
        thermal = _read_image(fusion / "thermal_anomaly_mask.png")
        valid = _read_image(fusion / "valid_thermal_mask.png")
        coverage = _read_image(fusion / "registration_coverage_in_ir.png")
        status = _read_image(fusion / "defect_status.png")
        relation = _read_image(corresponding / "correspondence_state.png")
        near = _read_image(corresponding / "rgb_defect_near_bits_in_ir.png")
        if any(item.shape != thermal.shape for item in (valid, coverage, status, relation, near)):
            raise ValueError(f"2D maps differ in size: {stem}")
        matrix = _homography(Path(fusion_summary["inputs"]["rgb_to_thermal"]))
        depth_map = pycolmap.DepthMap()
        depth_map.read(dense_workspace / "stereo" / "depth_maps" / f"{image.name}.geometric.bin")
        depth = depth_map.to_array()
        dense_camera = dense.cameras[image.camera_id]
        original_camera = original.cameras[original_image.camera_id]
        if depth.shape != (dense_camera.height, dense_camera.width):
            raise ValueError(f"Depth size differs from dense camera: {image.name}")
        pose = np.asarray(image.cam_from_world().matrix(), dtype=np.float64)
        original_pose = np.asarray(original_image.cam_from_world().matrix(), dtype=np.float64)
        if not np.allclose(pose, original_pose, atol=1e-5, rtol=0):
            raise ValueError(f"Original/dense poses differ: {image.name}")
        # 背面或被别的构件遮住的点不应被当作“该图没有检测到”。
        visible, xyz = visible_indices(points, pose, dense_camera, depth,
                                       source["depth_tolerance"])
        del depth, depth_map
        if not len(visible):
            continue
        u, v = project_original(original_camera, xyz)
        x, y, finite = apply_homography(u, v, matrix)
        inside = (finite & (x >= 0) & (y >= 0)
                  & (x < thermal.shape[1]) & (y < thermal.shape[0]))
        rows = visible[inside]
        ix = np.rint(x[inside]).astype(np.intp).clip(0, thermal.shape[1] - 1)
        iy = np.rint(y[inside]).astype(np.intp).clip(0, thermal.shape[0] - 1)
        # 配准外/ROI 外/无效热像素不投票；这些点不是阴性证据。
        keep = (valid[iy, ix] > 0) & (coverage[iy, ix] > 0) & (relation[iy, ix] != 255)
        rows, ix, iy = rows[keep], ix[keep], iy[keep]
        if not len(rows):
            continue
        camera_center = -pose[:3, :3].T @ pose[:3, 3]
        rays = points[rows] - camera_center
        rays = rays / np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
        point_parts.append(rows.astype(np.uint32))
        image_parts.append(np.full(len(rows), ordinal, np.uint16))
        thermal_parts.append((thermal[iy, ix] > 0).astype(np.uint8))
        correspondence_parts.append(relation[iy, ix].astype(np.uint8))
        near_parts.append(near[iy, ix].astype(np.uint8))
        status_parts.append(status[iy, ix].astype(np.uint8))
        ray_parts.append(rays.astype(np.float32))
        print(f"[{ordinal + 1}/{len(images)}] {image.name}: {len(rows)} valid candidate observations", flush=True)
    if not point_parts:
        raise ValueError("No candidates were visible with valid RGB/thermal registration")
    # 4. 把分照片的观测合并，再按同一 3D 点和不同相机视角统计。
    row = np.concatenate(point_parts)
    ordinal = np.concatenate(image_parts)
    thermal = np.concatenate(thermal_parts)
    relation = np.concatenate(correspondence_parts)
    near = np.concatenate(near_parts)
    status = np.concatenate(status_parts)
    rays = np.concatenate(ray_parts)
    checks = summarize_observations(len(indices), row, ordinal, thermal, relation, near, status,
                                    rays, min_views, min_fraction, min_angle_deg)
    for check in checks:
        check["source_point_index"] = int(indices[check["candidate_row"]])
        check["supporting_images"] = [names[index] for index in check["supporting_image_ordinals"]]
    # 可选 membership 把相邻候选点归组，但不改变单点的观测和判定。
    membership = {}
    if membership_file is not None:
        with np.load(membership_file, allow_pickle=False) as source_membership:
            membership = dict(zip(source_membership["source_point_indices"].astype(int),
                                  source_membership["instance_ids"].astype(int)))
    instances = summarize_instances(checks, membership, names)
    report = {
        "candidate_points": len(indices), "images_checked": len(images),
        "valid_point_image_observations": len(row),
        "independently_supported_points": sum(item["geometry_independent_support"] for item in checks),
        "min_positive_views": min_views, "min_positive_fraction": min_fraction,
        "min_view_angle_deg": min_angle_deg,
        "source_projected_ply": str(projected_ply.resolve()),
        "source_projection_summary": str(projection_summary.resolve()),
        "instance_count": len(instances),
        "interpretation": "thermal candidate corroboration across geometrically distinct views",
    }
    # 5. JSON 方便人工阅读；NPZ 保留逐点逐图的原始计数供复算。
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "point_checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    (output / "instance_checks.json").write_text(json.dumps(instances, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(output / "point_observations.npz",
                        source_point_indices=indices[row].astype(np.uint32),
                        image_ordinals=ordinal, thermal_positive=thermal,
                        correspondence_state=relation, rgb_near_bits=near,
                        fusion_status=status,
                        viewing_rays=rays)
    return report


def main() -> None:
    """解析投影、COLMAP、二维证据路径及多视角门槛并调用 run。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projected-ply", type=Path, required=True)
    parser.add_argument("--projection-summary", type=Path, required=True)
    parser.add_argument("--original-model", type=Path, required=True)
    parser.add_argument("--dense-workspace", type=Path, required=True)
    parser.add_argument("--fusion-root", type=Path, required=True)
    parser.add_argument("--correspondence-root", type=Path, required=True)
    parser.add_argument("--membership-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--min-fraction", type=float, default=0.3)
    parser.add_argument("--min-angle-deg", type=float, default=5.0)
    args = parser.parse_args()
    print(json.dumps(run(projected_ply=args.projected_ply,
                         projection_summary=args.projection_summary,
                         original_model=args.original_model,
                         dense_workspace=args.dense_workspace,
                         fusion_root=args.fusion_root,
                         correspondence_root=args.correspondence_root,
                         output=args.output, membership_file=args.membership_file,
                         min_views=args.min_views, min_fraction=args.min_fraction,
                         min_angle_deg=args.min_angle_deg), indent=2))


if __name__ == "__main__":
    main()
