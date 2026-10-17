"""Export independent RGB evidence for the thermal-anomaly fusion manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from ..inference.rgb_predict import (load_model, load_rgb_defect_model,
                             predict_native_class_probabilities, predict_rgb_defects)


def run(rgb_path: Path, output: Path, component_config: Path,
        component_checkpoint: Path, ac_config: Path | None = None,
        ac_checkpoint: Path | None = None, rgb_defect_checkpoint: Path | None = None,
        rgb_defect_source: Path | None = None, *,
        rgb_patch_size: int = 512, rgb_stride: int = 384,
        rgb_roi_mask: Path | None = None) -> dict:
    """从原尺寸 RGB 图导出构件、空调和可选六类缺陷证据。

    四通道起的场景输入只取前 3 个 BGR 通道；普通图像按 OpenCV 的 BGR
    顺序读取。各模型以原像素滑窗运行，RGB 缺陷分支可按同尺寸 ROI
    跳过不关注区域，但跳过处必须由覆盖图标为未评估。输出概率、类别图、
    patch 坐标及 summary，供后续 IR 配准和融合使用。
    """
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty to preserve prior runs: {output}")
    if (ac_config is None) != (ac_checkpoint is None):
        raise ValueError("AC config and checkpoint must be supplied together")
    if rgb_path.suffix.lower() == ".npy":
        array = np.load(rgb_path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 3 or array.shape[2] < 4:
            raise ValueError("Scene NumPy image must contain [B,G,R,Thermal] channels")
        image = np.asarray(array[..., :3], dtype=np.uint8)
    else:
        image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(rgb_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    component_model, component_cfg = load_model(
        component_config, component_checkpoint, "components", device)
    if component_cfg["data"]["class_names"] != ["building_surface", "opening", "other"]:
        raise ValueError("Component model must use building_surface/opening/other IDs 0/1/2")
    component_probabilities, component_patches = predict_native_class_probabilities(
        component_model, image, tuple(component_cfg["data"]["image_size"]),
        device, image_id=rgb_path.stem)
    component_ids = component_probabilities.argmax(0).astype(np.uint8)
    result = {
        "rgb": str(rgb_path.resolve()),
        "shape": list(image.shape[:2]),
        "device": str(device),
        "component_checkpoint": str(component_checkpoint.resolve()),
        "component_class_names": component_cfg["data"]["class_names"],
        "component_inference": "native_pixel_overlapping_patches_mean",
    }
    ac_probability = None
    ac_patches = None
    if ac_config is not None:
        ac_model, ac_cfg = load_model(ac_config, ac_checkpoint, "components", device)
        if ac_cfg["data"]["class_names"] != ["background", "air_conditioner"]:
            raise ValueError("AC model must use background/air_conditioner IDs 0/1")
        ac_probabilities, ac_patches = predict_native_class_probabilities(
            ac_model, image, tuple(ac_cfg["data"]["image_size"]),
            device, image_id=rgb_path.stem)
        ac_probability = ac_probabilities[1]
        result["ac_checkpoint"] = str(ac_checkpoint.resolve())
        result["ac_inference"] = "native_pixel_overlapping_patches_mean"
    rgb_semantic = None
    rgb_coverage = None
    rgb_patches = None
    if rgb_defect_checkpoint is not None:
        if rgb_defect_source is None:
            raise ValueError("RGB defect source is required with its checkpoint")
        roi = None
        if rgb_roi_mask is not None:
            roi = cv2.imread(str(rgb_roi_mask), cv2.IMREAD_GRAYSCALE)
            if roi is None or roi.shape != image.shape[:2]:
                raise ValueError("RGB ROI must match the original RGB image")
            roi = roi > 0
        defect_model, defect_state = load_rgb_defect_model(
            rgb_defect_checkpoint, rgb_defect_source, device)
        probabilities, _, names, rgb_coverage, rgb_patches = predict_rgb_defects(
            defect_model, defect_state, image, device, image_id=rgb_path.stem,
            patch_size=rgb_patch_size, stride=rgb_stride, roi=roi,
            return_coverage=True)
        rgb_semantic = (probabilities.astype(np.float16), np.asarray(names))
        result["rgb_defect_checkpoint"] = str(rgb_defect_checkpoint.resolve())
        result["rgb_defect_class_names"] = names
        result["rgb_defect_thresholds"] = {
            name: float(defect_state["thresholds"][name]) for name in names
        }
        result["rgb_defect_tiling"] = {
            "mode": "native_pixel_overlapping_patches",
            "patch_size": rgb_patch_size, "stride": rgb_stride,
            "blend": "mean", "global_resize": False,
            "roi_mask": str(rgb_roi_mask.resolve()) if rgb_roi_mask else None,
            "evaluated_pixels": int(rgb_coverage.sum()),
            "total_pixels": int(rgb_coverage.size),
            "patches_evaluated": sum(item["evaluated"] for item in rgb_patches),
            "patches_total": len(rgb_patches),
        }
    output.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output / "component_ids_rgb.png"), component_ids):
        raise OSError("Could not write component IDs")
    # Keep continuous wall confidence. The discrete component ID is useful
    # for display, but must not become a hard exclusion rule for IR candidates.
    np.save(output / "wall_probability_rgb.npy", component_probabilities[0].astype(np.float32),
            allow_pickle=False)
    (output / "component_patches.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in component_patches), encoding="utf-8")
    if ac_probability is not None:
        np.save(output / "interference_probability_rgb.npy", ac_probability.astype(np.float32),
                allow_pickle=False)
        (output / "ac_patches.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in ac_patches), encoding="utf-8")
    if rgb_semantic is not None:
        np.savez_compressed(output / "rgb_semantic_probabilities.npz",
                            probabilities=rgb_semantic[0], class_names=rgb_semantic[1],
                            coverage=rgb_coverage)
        if not cv2.imwrite(str(output / "rgb_semantic_coverage_rgb.png"),
                           rgb_coverage.astype(np.uint8) * 255):
            raise OSError("Could not write RGB semantic coverage")
        (output / "rgb_defect_patches.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in rgb_patches), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    """Parse command-line arguments and run this module's workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True,
                        help="BGR photo or scene [B,G,R,Thermal,...] .npy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--component-config", type=Path,
                        default=Path("models/rgb/facade_components/config.yaml"))
    parser.add_argument("--component-checkpoint", type=Path,
                        default=Path("models/rgb/facade_components/checkpoint.pt"))
    parser.add_argument("--ac-config", type=Path)
    parser.add_argument("--ac-checkpoint", type=Path)
    parser.add_argument("--rgb-defect-checkpoint", type=Path)
    parser.add_argument("--rgb-defect-source", type=Path,
                        default=Path("infradetect/inference"))
    parser.add_argument("--rgb-patch-size", type=int, default=512)
    parser.add_argument("--rgb-stride", type=int, default=384)
    parser.add_argument("--rgb-roi-mask", type=Path,
                        help="Original-pixel facade/wall ROI; omit for ROI-off A/B run")
    args = parser.parse_args()
    print(json.dumps(run(args.rgb, args.output, args.component_config,
                         args.component_checkpoint, args.ac_config, args.ac_checkpoint,
                         args.rgb_defect_checkpoint, args.rgb_defect_source,
                         rgb_patch_size=args.rgb_patch_size, rgb_stride=args.rgb_stride,
                         rgb_roi_mask=args.rgb_roi_mask), indent=2))


if __name__ == "__main__":
    main()
