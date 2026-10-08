from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def partition_defects(defect_mask_ir: np.ndarray, component_mask_rgb: np.ndarray,
                      rgb_to_ir: np.ndarray, ir_size: tuple[int, int], edge_radius: int = 8,
                      valid_registration: bool = True) -> dict[str, np.ndarray]:
    """Split defects into confirmed wall-core, uncertain edge, and rejected regions."""
    height, width = ir_size
    if defect_mask_ir.shape != (height, width):
        raise ValueError("Defect mask does not match requested IR size")
    if component_mask_rgb.ndim != 2 or rgb_to_ir.shape != (3, 3):
        raise ValueError("Expected 2-D component mask and 3x3 homography")
    if edge_radius < 0:
        raise ValueError("edge_radius must be non-negative")
    zeros = np.zeros((height, width), dtype=np.uint8)
    if not valid_registration:
        return {"warped_components": np.full((height, width), 255, np.uint8),
                "wall": zeros, "wall_core": zeros, "uncertain_band": zeros,
                "hard_reject": zeros, "confirmed": defect_mask_ir.copy(),
                "uncertain": np.zeros_like(defect_mask_ir),
                "rejected": np.zeros_like(defect_mask_ir), "lenient": defect_mask_ir.copy()}
    warped = cv2.warpPerspective(component_mask_rgb, rgb_to_ir.astype(np.float64), (width, height),
                                 flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=255).astype(np.uint8)
    wall = (warped == 0).astype(np.uint8)
    if edge_radius:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (2 * edge_radius + 1, 2 * edge_radius + 1))
        core = cv2.erode(wall, kernel)
        near_wall = cv2.dilate(wall, kernel)
    else:
        core = wall.copy()
        near_wall = wall.copy()
    uncertain_band = ((near_wall > 0) & (core == 0)).astype(np.uint8)
    hard_reject = (near_wall == 0).astype(np.uint8)
    confirmed = np.where(core > 0, defect_mask_ir, 0).astype(defect_mask_ir.dtype)
    uncertain = np.where(uncertain_band > 0, defect_mask_ir, 0).astype(defect_mask_ir.dtype)
    rejected = np.where(hard_reject > 0, defect_mask_ir, 0).astype(defect_mask_ir.dtype)
    lenient = np.where(near_wall > 0, defect_mask_ir, 0).astype(defect_mask_ir.dtype)
    return {"warped_components": warped, "wall": wall, "wall_core": core,
            "uncertain_band": uncertain_band, "hard_reject": hard_reject,
            "confirmed": confirmed, "uncertain": uncertain, "rejected": rejected,
            "lenient": lenient}


def filter_defects(defect_mask_ir: np.ndarray, component_mask_rgb: np.ndarray,
                   rgb_to_ir: np.ndarray, ir_size: tuple[int, int],
                   valid_registration: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return filtered defects and an IR-space wall mask.

    ``rgb_to_ir`` is a 3x3 homography mapping RGB pixels to IR pixels. Callers
    must establish and validate it; MFDN's matches alone are not a homography.
    Uncertain registration returns the unfiltered defect result.
    """
    height, width = ir_size
    if defect_mask_ir.shape != (height, width):
        raise ValueError("Defect mask does not match requested IR size")
    if component_mask_rgb.ndim != 2 or rgb_to_ir.shape != (3, 3):
        raise ValueError("Expected 2-D component mask and 3x3 homography")
    if not valid_registration:
        return defect_mask_ir.copy(), np.zeros_like(defect_mask_ir, dtype=np.uint8)
    zones = partition_defects(defect_mask_ir, component_mask_rgb, rgb_to_ir, ir_size,
                              edge_radius=0, valid_registration=valid_registration)
    return zones["confirmed"], zones["wall"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter IR defects using registered RGB facade classes")
    parser.add_argument("--defect-mask", type=Path, required=True)
    parser.add_argument("--component-mask", type=Path, required=True)
    parser.add_argument("--rgb-to-ir", type=Path, required=True,
                        help="MFDN .mat file containing H or a 3x3 .npy file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-output", type=Path)
    parser.add_argument("--uncertain-output", type=Path)
    parser.add_argument("--edge-radius", type=int, default=8)
    args = parser.parse_args()
    defect = cv2.imread(str(args.defect_mask), cv2.IMREAD_UNCHANGED)
    component = cv2.imread(str(args.component_mask), cv2.IMREAD_UNCHANGED)
    if defect is None or component is None or defect.ndim != 2 or component.ndim != 2:
        raise ValueError("Both inputs must be single-channel class-index PNG masks")
    if args.rgb_to_ir.suffix.lower() == ".mat":
        from scipy.io import loadmat
        matrix = loadmat(args.rgb_to_ir)["H"]
    elif args.rgb_to_ir.suffix.lower() == ".npy":
        matrix = np.load(args.rgb_to_ir)
    else:
        raise ValueError("Homography must be .mat or .npy")
    zones = partition_defects(defect, component, matrix, defect.shape, args.edge_radius)
    filtered, wall = zones["lenient"], zones["wall"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), filtered):
        raise OSError(f"Could not write {args.output}")
    if args.wall_output:
        args.wall_output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.wall_output), wall):
            raise OSError(f"Could not write {args.wall_output}")
    if args.uncertain_output:
        args.uncertain_output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.uncertain_output), zones["uncertain"]):
            raise OSError(f"Could not write {args.uncertain_output}")


if __name__ == "__main__":
    main()
