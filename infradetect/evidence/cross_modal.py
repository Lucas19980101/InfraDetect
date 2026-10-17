"""比较一张图里已配准的 RGB 可见缺陷与 IR 热异常的位置关系。

输入：融合后的 IR 坐标热概率/有效区、RGB 六类概率及阈值。
处理：逐类阈值化、计算精确重合和指定像素半径内的邻近关系。
输出：对应状态图、RGB 类别位图、区域统计 JSON。空间共址不是同因诊断。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _thresholds(summary: dict, names: list[str]) -> dict[str, float]:
    """取得六类 RGB 缺陷各自的判定阈值，而不是套用统一 0.5。

    优先使用本次 RGB 推理摘要里的阈值；旧摘要缺失时从 checkpoint
    读取。类别名必须完全一致，且每个值都应是 0～1 的有限概率。
    """
    # 优先尊重本次推理记录；兼容旧摘要时才从同一权重文件读阈值。
    values = summary.get("rgb_defect_thresholds")
    if values is None:
        import torch

        checkpoint = Path(summary["rgb_defect_checkpoint"])
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        values = state["thresholds"]
    if set(values) != set(names):
        raise ValueError("RGB class names differ from checkpoint thresholds")
    result = {name: float(values[name]) for name in names}
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in result.values()):
        raise ValueError("RGB thresholds must be finite probabilities")
    return result


def compare(thermal: np.ndarray, rgb: np.ndarray, names: list[str],
            thresholds: dict[str, float], valid: np.ndarray, status: np.ndarray,
            radius: int) -> tuple[dict[str, np.ndarray], dict]:
    """在 IR 原像素网格比较热异常与已配准的 RGB 缺陷证据。

    thermal 为 H×W，rgb 为 C×H×W；valid 同时限制两条分支的统计范围。
    每类 RGB 用其专属阈值得到掩码，radius 仅用于“附近有 RGB 证据”
    的容差判断，不会把附近像素改写成真正的精确重叠。返回像素状态图、
    类别位图和逐热区域的 exact/nearby/thermal_only 统计。
    共址只表示空间对应，不证明两种现象有相同病因。
    """
    if thermal.ndim != 2 or rgb.ndim != 3 or rgb.shape[1:] != thermal.shape:
        raise ValueError("Thermal and RGB evidence sizes differ")
    if len(names) != len(rgb) or len(names) > 8 or len(set(names)) != len(names):
        raise ValueError("Expected one to eight distinct RGB defect classes")
    if valid.shape != thermal.shape or status.shape != thermal.shape or radius < 0:
        raise ValueError("Invalid validity mask, status or correspondence radius")
    if not np.isfinite(thermal).all() or not np.isfinite(rgb).all():
        raise ValueError("Probabilities must be finite")
    if any(name not in thresholds for name in names):
        raise ValueError("Missing RGB class threshold")
    # 1. 所有比较都限制在 IR 有效且 RGB 已覆盖的像素，未配准处不判阴性。
    anomaly = (status > 0) & valid
    rgb_bits = np.zeros(thermal.shape, np.uint8)
    near_bits = np.zeros(thermal.shape, np.uint8)
    rgb_masks = {}
    near_masks = {}
    # 2. “附近”通过像素膨胀得到，保留原始 mask 供精确重合单独统计。
    if radius:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    for index, name in enumerate(names):
        mask = (rgb[index] >= thresholds[name]) & valid
        rgb_masks[name] = mask
        rgb_bits[mask] |= 1 << index
        near_masks[name] = (cv2.dilate(mask.astype(np.uint8), kernel) > 0 if radius else mask) & valid
        near_bits[near_masks[name]] |= 1 << index
    rgb_any = rgb_bits > 0
    # 3. 状态码：0 都无、1 仅热、2 仅 RGB、3 共址、255 未评估。
    state = np.full(thermal.shape, 255, np.uint8)
    state[valid] = anomaly[valid].astype(np.uint8) + 2 * rgb_any[valid].astype(np.uint8)
    # 4. 每个相连热区域分别统计六类 RGB 的 exact/near，不用整图结果替代局部证据。
    count, labels, stats, centers = cv2.connectedComponentsWithStats(anomaly.astype(np.uint8), 8)
    regions = []
    for label in range(1, count):
        left, top, width, height, area = map(int, stats[label])
        component = labels[top:top + height, left:left + width] == label
        context = status[top:top + height, left:left + width][component]
        classes = {}
        for name in names:
            exact = int(np.count_nonzero(rgb_masks[name][top:top + height, left:left + width][component]))
            nearby = int(np.count_nonzero(near_masks[name][top:top + height, left:left + width][component]))
            classes[name] = {"exact_pixels": exact, "within_radius_pixels": nearby}
        exact_any = any(entry["exact_pixels"] > 0 for entry in classes.values())
        near_any = any(entry["within_radius_pixels"] > 0 for entry in classes.values())
        regions.append({
            "region_id": label, "thermal_pixels": area,
            "bbox_xywh": [left, top, width, height],
            "centroid_xy": centers[label].tolist(),
            "rgb_relation": ("co_located" if exact_any else "nearby" if near_any else "thermal_only"),
            "rgb_classes": classes,
            "context_status_pixels": {str(value): int(np.count_nonzero(context == value))
                                      for value in range(1, 5)},
        })
    # 5. 汇总像素数和区域数，明确这些数字是空间关系而非工程确诊。
    summary = {
        "registered_valid_pixels": int(valid.sum()),
        "thermal_region_count": len(regions),
        "relation_counts": {relation: sum(item["rgb_relation"] == relation for item in regions)
                            for relation in ("co_located", "nearby", "thermal_only")},
        "pixel_counts": {"thermal_only": int(np.count_nonzero(state == 1)),
                         "rgb_only": int(np.count_nonzero(state == 2)),
                         "co_located": int(np.count_nonzero(state == 3))},
        "per_rgb_class": {
            name: {"rgb_positive_pixels": int(np.count_nonzero(rgb_masks[name])),
                   "exact_thermal_overlap_pixels": int(np.count_nonzero(anomaly & rgb_masks[name])),
                   "within_radius_thermal_pixels": int(np.count_nonzero(anomaly & near_masks[name]))}
            for name in names},
        "correspondence_radius_ir_pixels": radius,
        "rgb_thresholds": thresholds,
        "map_legend": {"0": "neither", "1": "thermal_only", "2": "rgb_only",
                       "3": "co_located", "255": "invalid_or_unregistered"},
        "interpretation": "registered spatial evidence, not same-cause defect confirmation",
        "regions": regions,
    }
    return {"correspondence_state.png": state,
            "rgb_defect_bits_in_ir.png": rgb_bits,
            "rgb_defect_near_bits_in_ir.png": near_bits}, summary


def run(fusion_scene: Path, rgb_summary: Path, output: Path,
        radius: int = 8) -> dict:
    """读取单场景融合图和 RGB 摘要，校验坐标后保存对应图及统计。

    此处消费的是已投影到 IR 像素坐标的证据；若配准或尺寸不一致应报错，
    而不能通过全图 resize 偷偷对齐。输出用于人工复核及后续三维核对。
    """
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty: {output}")
    # 1. 先确认配准被接受；否则 RGB 和 IR 不能逐像素相互支持或否定。
    source = json.loads((fusion_scene / "summary.json").read_text(encoding="utf-8"))
    if not source.get("registration_accepted"):
        raise ValueError("RGB/thermal correspondence requires accepted registration")
    rgb_info = json.loads(rgb_summary.read_text(encoding="utf-8"))
    # 2. RGB 概率已由上游投到 IR 坐标；这里不 resize 或重新猜测配对。
    with np.load(fusion_scene / "rgb_semantic_evidence.npz", allow_pickle=False) as archive:
        rgb = np.asarray(archive["probabilities"], np.float32)
        names = [str(item) for item in archive["class_names"]]
    thresholds = _thresholds(rgb_info, names)
    thermal = np.load(fusion_scene / "thermal_anomaly_probability.npy", mmap_mode="r", allow_pickle=False)
    valid_thermal = cv2.imread(str(fusion_scene / "valid_thermal_mask.png"), cv2.IMREAD_UNCHANGED)
    coverage = cv2.imread(str(fusion_scene / "registration_coverage_in_ir.png"), cv2.IMREAD_UNCHANGED)
    semantic_coverage_path = fusion_scene / "rgb_semantic_coverage_in_ir.png"
    semantic_coverage = (cv2.imread(str(semantic_coverage_path), cv2.IMREAD_UNCHANGED)
                         if semantic_coverage_path.is_file() else None)
    status = cv2.imread(str(fusion_scene / "defect_status.png"), cv2.IMREAD_UNCHANGED)
    if any(item is None for item in (valid_thermal, coverage, status)):
        raise ValueError("Fusion scene is missing validity or status maps")
    if semantic_coverage_path.is_file() and (semantic_coverage is None or semantic_coverage.shape != thermal.shape):
        raise ValueError("Invalid RGB semantic ROI coverage map")
    # ROI 覆盖与配准覆盖取交集；ROI 外是未评估，不是 RGB 模型的阴性。
    valid = (valid_thermal > 0) & (coverage > 0)
    if semantic_coverage is not None:
        valid &= semantic_coverage > 0
    # 3. 计算对应关系，并记录热分支是否本来就用了 RGB 输入，避免双算独立证据。
    maps, report = compare(thermal, rgb, names, thresholds,
                           valid, status, radius)
    thermal_input_mode = source.get("thermal_input_mode")
    if thermal_input_mode is None:
        companion = fusion_scene.parent.parent / "thermal_primary" / "summary.json"
        if companion.is_file():
            thermal_input_mode = json.loads(companion.read_text(encoding="utf-8")).get("input_mode")
    report.update({"scene_id": source["scene_id"], "source_fusion": str(fusion_scene.resolve()),
                   "source_rgb_summary": str(rgb_summary.resolve()),
                   "thermal_model_input_mode": thermal_input_mode,
                   "thermal_prediction_uses_rgb": thermal_input_mode == "rgb_thermal",
                   "rgb_semantic_roi_coverage_applied": semantic_coverage is not None})
    # 4. 保存数值图与人眼可读 overlay；不同文件供下游程序或审核界面消费。
    output.mkdir(parents=True, exist_ok=True)
    for name, image in maps.items():
        if not cv2.imwrite(str(output / name), image):
            raise OSError(f"Could not write {output / name}")
    preview_path = source.get("inputs", {}).get("thermal_preview")
    if preview_path:
        path = Path(preview_path)
        if path.suffix.lower() == ".npy":
            stack = np.load(path, mmap_mode="r", allow_pickle=False)
            preview = cv2.cvtColor(np.asarray(stack[..., 3], np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            preview = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if preview is None or preview.shape[:2] != thermal.shape:
            raise ValueError("Thermal preview differs from correspondence map")
        state = maps["correspondence_state.png"]
        for value, color in ((1, (35, 165, 255)), (2, (255, 170, 35)), (3, (40, 210, 40))):
            selected = state == value
            preview[selected] = (0.45 * preview[selected] + 0.55 * np.asarray(color)).astype(np.uint8)
        if not cv2.imwrite(str(output / "correspondence_overlay.png"), preview):
            raise OSError("Could not write correspondence overlay")
    (output / "summary.json").write_text(json.dumps({k: v for k, v in report.items() if k != "regions"}, indent=2) + "\n", encoding="utf-8")
    (output / "regions.json").write_text(json.dumps(report["regions"], indent=2) + "\n", encoding="utf-8")
    return {key: report[key] for key in ("scene_id", "thermal_region_count", "relation_counts", "pixel_counts")}


def main() -> None:
    """接收融合场景和 RGB 摘要路径，运行一次跨模态空间对照。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-scene", type=Path, required=True)
    parser.add_argument("--rgb-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(run(args.fusion_scene, args.rgb_summary, args.output, args.radius), indent=2))


if __name__ == "__main__":
    main()
