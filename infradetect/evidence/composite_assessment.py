"""把两条热分支的候选并成“待复核区域”，再附上 RGB 与干扰上下文。

输入：pipeline 已保存的 report、热概率、墙面/设备/RGB 对应图。
处理：取候选并集、分连通区域、统计每区证据、给人工排序提示。
输出：candidate_assessment_overlay.png 与 summary.json。这里既不是新
训练出的缺陷分类器，也没有定标缺陷概率；低优先级候选仍会保留。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    """读取一张单通道 mask，并拒绝缺失或尺寸不匹配的文件。"""
    result = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if result is None or result.ndim != 2 or (shape is not None and result.shape != shape):
        raise ValueError(f"Missing or mismatched single-channel map: {path}")
    return result


def _optional_probability(path: Path, shape: tuple[int, int]) -> np.ndarray | None:
    """读取可选的 0～1 连续证据图；旧运行没有该图时返回 None。"""
    if not path.is_file():
        return None
    result = np.load(path, allow_pickle=False)
    if (result.shape != shape or not np.issubdtype(result.dtype, np.floating)
            or not np.isfinite(result).all() or np.any((result < 0) | (result > 1))):
        raise ValueError(f"Invalid probability layer: {path}")
    return result


def assess(source: Path, output: Path) -> dict:
    """按主分支与 IR-only 分支的候选并集生成可复核区域。

    两条热分支报出的区域都保留，再附加墙面、空调、RGB 可见缺陷、
    配准覆盖及图像边界等上下文。墙面与干扰连续概率若存在，只调整
    人工复核顺序；不会删掉候选或改写原始 IR 概率。输出 attention 是
    解释性提示，review_priority 未在目标建筑上定标，绝非缺陷概率。
    """
    if output.exists():
        raise FileExistsError(output)
    # 1. 找到本次运行使用的场景和 IR 像素尺寸，防止不同图的证据混合。
    report = json.loads((source / "report.json").read_text(encoding="utf-8"))
    scene = report["scene"]
    scene_id = scene["scene_id"]
    shape = tuple(scene["shape"])
    primary_dir = source / "fusion" / scene_id
    diagnostic_dir = source / "fusion_diagnostic" / scene_id
    # 2. 候选来自两条热分支。valid 限定可评估像素；墙、设备和 RGB 是上下文。
    primary = _mask(primary_dir / "thermal_anomaly_mask.png", shape) > 0
    diagnostic = _mask(diagnostic_dir / "thermal_anomaly_mask.png", shape) > 0
    valid = _mask(primary_dir / "valid_thermal_mask.png", shape) > 0
    if np.any((primary | diagnostic) & ~valid):
        raise ValueError("Candidate outside valid thermal area")
    equipment = _mask(primary_dir / "equipment_mask_in_ir.png", shape) > 0
    wall_core = _mask(primary_dir / "wall_core_in_ir.png", shape) > 0
    registration = _mask(primary_dir / "registration_coverage_in_ir.png", shape) > 0
    rgb_exact = _mask(source / "cross_modal" / "rgb_defect_bits_in_ir.png", shape) > 0
    rgb_near = _mask(source / "cross_modal" / "rgb_defect_near_bits_in_ir.png", shape) > 0
    # 连续概率用于“先看哪块”的提示；缺图时仍能给出区域和离散证据。
    primary_probability = _optional_probability(primary_dir / "thermal_anomaly_probability.npy", shape)
    diagnostic_probability = _optional_probability(diagnostic_dir / "thermal_anomaly_probability.npy", shape)
    wall_probability = _optional_probability(primary_dir / "wall_probability_in_ir.npy", shape)
    interference_probability = _optional_probability(
        primary_dir / "interference_probability_in_ir.npy", shape)
    if report["primary_thermal"]["input_mode"] != "rgb_thermal":
        raise ValueError("Expected RGB+IR primary model")
    if report["diagnostic_thermal"]["input_mode"] != "thermal":
        raise ValueError("Expected independent IR-only diagnostic model")
    # 3. 并集后按相连像素分块。每块只表示一个可查看区域，不等于同一病害。
    union = (primary | diagnostic).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(union, 8)
    regions = []
    for index in range(1, count):
        component = labels == index
        area = int(stats[index, cv2.CC_STAT_AREA])
        primary_count = int(np.count_nonzero(primary & component))
        diagnostic_count = int(np.count_nonzero(diagnostic & component))
        overlap = int(np.count_nonzero(primary & diagnostic & component))
        equipment_count = int(np.count_nonzero(equipment & component))
        wall_count = int(np.count_nonzero(wall_core & component))
        registered_count = int(np.count_nonzero(registration & component))
        rgb_exact_count = int(np.count_nonzero(rgb_exact & component))
        rgb_near_count = int(np.count_nonzero(rgb_near & component))
        # 两条热分支是否一致，与 RGB 的可见缺陷是否共址是两件不同的事。
        if primary_count and diagnostic_count:
            agreement = "both_thermal_branches"
        elif primary_count:
            agreement = "rgb_thermal_primary_only"
        else:
            agreement = "independent_ir_only"
        # 干扰和墙面线索只影响复核提示，绝不把热候选自动删除或判阴性。
        if equipment_count:
            attention = "equipment_influenced_manual_review"
        elif registered_count < area or wall_count * 2 < area:
            attention = "registration_or_nonwall_manual_review"
        elif agreement == "both_thermal_branches":
            attention = "two_branch_thermal_candidate_manual_review"
        else:
            attention = "single_branch_thermal_candidate_manual_review"
        x, y, width, height = map(int, stats[index, :4])
        touches_image_boundary = (x == 0 or y == 0 or
                                  x + width == shape[1] or y + height == shape[0])
        # Review priority is deliberately bounded and uncalibrated. Low wall
        # confidence or equipment evidence never deletes an IR candidate.
        raw_ir_score = wall_score = interference_score = review_priority = None
        if (primary_probability is not None and diagnostic_probability is not None
                and wall_probability is not None):
            raw_ir_score = max(float(primary_probability[component].mean()),
                               float(diagnostic_probability[component].mean()))
            wall_score = float(wall_probability[component].mean())
            interference_score = (float(interference_probability[component].mean())
                                  if interference_probability is not None else None)
            review_priority = (raw_ir_score * (0.8 + 0.2 * wall_score)
                               * (1.0 - 0.25 * (interference_score or 0.0)))
        # 保留足够的原始计数，组员可以追溯为何出现该提示/排名。
        regions.append({
            "region_id": index, "bbox_xywh": [x, y, width, height],
            "touches_image_boundary": touches_image_boundary,
            "centroid_xy": [float(centroids[index, 0]), float(centroids[index, 1])],
            "union_pixels": area, "primary_rgb_ir_pixels": primary_count,
            "independent_ir_pixels": diagnostic_count, "both_pixels": overlap,
            "air_conditioner_influence_pixels": equipment_count,
            "wall_core_pixels": wall_count, "registered_pixels": registered_count,
            "rgb_visible_defect_exact_pixels": rgb_exact_count,
            "rgb_visible_defect_near_pixels": rgb_near_count,
            "thermal_branch_agreement": agreement,
            "raw_ir_score": raw_ir_score,
            "wall_support_mean": wall_score,
            "interference_mean": interference_score,
            "review_priority": review_priority,
            "anti_interference_evidence": ("air_conditioner_model_flagged" if equipment_count
                                           else "not_flagged_by_air_conditioner_model"),
            "attention": attention,
            "diagnosis": "unverified_thermal_anomaly_candidate",
        })
    # 4. 只给具备连续概率证据的区域排名；None 表示无法计算，不是低风险。
    ranked = sorted((region for region in regions if region["review_priority"] is not None),
                    key=lambda region: region["review_priority"], reverse=True)
    for rank, region in enumerate(ranked, 1):
        region["review_rank"] = rank
    # 5. 以原热图做可视化底图，轮廓颜色表示热分支来源，不表示真值类别。
    if scene.get("source_format") == "aligned_png_pair_adapter":
        provenance = json.loads(Path(scene["input_provenance"]).read_text(encoding="utf-8"))
        preview = _mask(Path(provenance["source_ir"]), shape)
    else:
        stack = np.load(scene["image"], mmap_mode="r", allow_pickle=False)
        preview = np.asarray(stack[..., 3], np.uint8)
    overlay = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
    colors = {"both_thermal_branches": (0, 220, 220),
              "rgb_thermal_primary_only": (0, 130, 255),
              "independent_ir_only": (255, 150, 40)}
    for region in regions:
        index = region["region_id"]
        selected = (labels == index).astype(np.uint8)
        contours, _ = cv2.findContours(selected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = colors[region["thermal_branch_agreement"]]
        cv2.drawContours(overlay, contours, -1, color, 2)
        x, y, _, _ = region["bbox_xywh"]
        cv2.putText(overlay, str(index), (x + 2, max(18, y - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    output.mkdir(parents=True)
    if not cv2.imwrite(str(output / "candidate_assessment_overlay.png"), overlay):
        raise OSError("Could not save assessment overlay")
    # 6. JSON 明列计算公式、限制和每区证据，供界面与人工复核读取。
    summary = {
        "scene_id": scene_id, "candidate_regions": len(regions),
        "source_dataset": scene["source_dataset"],
        "primary_input_mode": "rgb_thermal", "diagnostic_input_mode": "thermal",
        "anti_interference_scope": "air_conditioner_model_plus_surface_opening_context_only",
        "three_dimensional_support": report["three_dimensional_projection"],
        "calibrated_defect_probability": None,
        "review_priority_formula": "raw_ir * (0.8 + 0.2 * wall) * (1 - 0.25 * interference)",
        "review_priority_calibrated": False,
        "ranked_regions": len(ranked),
        "engineering_diagnosis": "not_available",
        "region_assessments": regions,
        "legend": {"yellow": "both thermal branches", "orange": "RGB+IR primary only",
                   "blue": "independent IR only"},
        "limits": [
            "No Hollow/Stains diagnosis or causal attribution",
            "Air-conditioner model does not cover sunlight, reflection or all thermal interference",
            "RGB visible-defect co-location is spatial evidence, not the same-cause proof",
            "Primary RGB+IR branch is not independent of the RGB visible-defect model",
            "Attention categories are uncalibrated rules; never use as accuracy scores",
            "Image-boundary contact is context evidence, not grounds for automatic suppression",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    """从命令行读取一个已完成的二维运行目录并生成综合评判文件。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess(args.source_run, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "region_assessments"},
                     indent=2))


if __name__ == "__main__":
    main()
