"""可复现的墙面几何基线：拟合主要平面，给每个点一个偏离分数。

这是从工作区 building-anomaly 的平面距离实验提炼的推理入口，不是
PointCore/PASDF 的预训练模型。只适用于已经裁出的、以墙面为主的点云；
窗、管道、栏杆和另一面墙也会偏离平面，必须留给人工复核。
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np


def score_wall_points(points: np.ndarray, *, fit_fraction: float = 0.002,
                      candidate_fraction: float = 0.006,
                      fit_limit: int = 50000, iterations: int = 400,
                      seed: int = 0) -> dict[str, object]:
    """Fit a dominant plane on a bounded sample, then score all original points.

    Threshold fractions refer to the point cloud bounding-box diagonal, not
    metres: COLMAP reconstructions usually have an arbitrary scale. The output
    score is an absolute point-to-plane distance in that same arbitrary unit.
    """

    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 3 or not np.isfinite(xyz).all():
        raise ValueError("Expected at least three finite XYZ points")
    if not (0 < fit_fraction < candidate_fraction) or fit_limit < 3 or iterations < 1:
        raise ValueError("Require 0 < fit_fraction < candidate_fraction, fit_limit >= 3")
    diagonal = float(np.linalg.norm(np.ptp(xyz, axis=0)))
    if diagonal <= 0:
        raise ValueError("Point cloud has no spatial extent")
    fit_threshold = diagonal * fit_fraction
    candidate_threshold = diagonal * candidate_fraction
    rng = np.random.default_rng(seed)
    sample = xyz[rng.choice(len(xyz), min(len(xyz), fit_limit), replace=False)]
    best = np.zeros(len(sample), dtype=bool)
    for _ in range(iterations):
        triangle = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        magnitude = np.linalg.norm(normal)
        if magnitude <= 1e-12:
            continue
        normal /= magnitude
        mask = np.abs((sample - triangle[0]) @ normal) <= fit_threshold
        if mask.sum() > best.sum():
            best = mask
    if best.sum() < 3:
        raise ValueError("No dominant plane found")
    centre = sample[best].mean(axis=0)
    _, _, basis = np.linalg.svd(sample[best] - centre, full_matrices=False)
    normal = basis[-1]
    distance = np.abs((xyz - centre) @ normal)
    return {
        "scores": distance.astype(np.float32),
        "candidate": distance > candidate_threshold,
        "plane": [*[float(value) for value in normal], float(-centre @ normal)],
        "diagonal": diagonal,
        "fit_threshold": fit_threshold,
        "candidate_threshold": candidate_threshold,
        "fit_inlier_fraction": float(best.mean()),
        "sample_count": len(sample),
    }


def run_wall_baseline(*, ply: Path, output: Path, fit_fraction: float = 0.002,
                      candidate_fraction: float = 0.006,
                      fit_limit: int = 50000, iterations: int = 400) -> dict[str, object]:
    """Read a PLY, save scores in source point order, and write provenance."""

    if not ply.is_file():
        raise FileNotFoundError(ply)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Choose a new empty output directory: {output}")
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError("geometry-baseline needs optional dependency open3d") from error
    cloud = o3d.io.read_point_cloud(str(ply))
    if cloud.is_empty():
        raise ValueError(f"Empty or unreadable point cloud: {ply}")
    result = score_wall_points(np.asarray(cloud.points), fit_fraction=fit_fraction,
                               candidate_fraction=candidate_fraction,
                               fit_limit=fit_limit, iterations=iterations)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "point_scores.npz", scores=result["scores"],
                        candidate=result["candidate"])
    report = {key: value for key, value in result.items()
              if key not in ("scores", "candidate")}
    # Point scores are indexed by the input PLY's vertex order. A hash prevents
    # a different cloud with the same point count from receiving false markers.
    digest = hashlib.sha256()
    with ply.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    report.update({"backend": "wall_plane_v0", "source_ply": str(ply.resolve()),
                   "source_ply_sha256": digest.hexdigest(),
                   "point_count": len(cloud.points),
                   "candidate_count": int(result["candidate"].sum()),
                   "scores_file": "point_scores.npz",
                   "warning": "几何候选，不是缺陷确认；仅适合单一主墙面 ROI。"})
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
    return report
