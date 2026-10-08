"""Inspect one paired RGB/IR scene without changing either trained model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from .data import BFDD_PALETTE_RGB, COMPONENT_COLORS_RGB, restore_mask
from .filter import partition_defects
from .models import build_model
from .train import config_from_file


def load_model(config_path: Path, checkpoint_path: Path, task: str, device: torch.device):
    cfg = config_from_file(config_path)
    if cfg["task"] != task:
        raise ValueError(f"Expected {task} config: {config_path}")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if state["config"]["task"] != task or state["config"]["data"]["class_names"] != cfg["data"]["class_names"]:
        raise ValueError(f"Checkpoint task/classes differ from {config_path}")
    model = build_model(task, cfg["model"]["classes"], pretrained=False).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, cfg


def prepare(image: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    height, width = size
    source_h, source_w = image.shape[:2]
    scale = min(height / source_h, width / source_w)
    resized_h, resized_w = max(1, round(source_h * scale)), max(1, round(source_w * scale))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    top, left = (height - resized_h) // 2, (width - resized_w) // 2
    canvas = cv2.copyMakeBorder(resized, top, height - resized_h - top, left, width - resized_w - left,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array((0.485, 0.456, 0.406), np.float32)) / np.array((0.229, 0.224, 0.225), np.float32)
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0)


@torch.no_grad()
def predict(model, image: np.ndarray, size: tuple[int, int], device: torch.device) -> np.ndarray:
    logits = model(prepare(image, size).to(device))
    mask = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    return restore_mask(mask, image.shape[:2])


@torch.no_grad()
def predict_probability(model, image: np.ndarray, size: tuple[int, int], device: torch.device,
                        class_index: int) -> np.ndarray:
    logits = model(prepare(image, size).to(device))
    probability = logits.softmax(1)[0, class_index].cpu().numpy().astype(np.float32)
    height, width = image.shape[:2]
    canvas_h, canvas_w = probability.shape
    scale = min(canvas_h / height, canvas_w / width)
    resized_h, resized_w = max(1, round(height * scale)), max(1, round(width * scale))
    top, left = (canvas_h - resized_h) // 2, (canvas_w - resized_w) // 2
    cropped = probability[top:top + resized_h, left:left + resized_w]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


def load_rgb_defect_model(checkpoint_path: Path, source_path: Path,
                          device: torch.device):
    """Load the independent six-channel RGB multilabel model."""
    sys.path.insert(0, str(source_path.resolve()))
    from rgb_multilabel.models import build_model as build_rgb_defect_model

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_rgb_defect_model(state["config"]["model"], pretrained=False).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


@torch.no_grad()
def predict_rgb_defects(model, state: dict, image: np.ndarray,
                        device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return probabilities and thresholded masks at the original RGB size."""
    names = list(state["config"]["data"]["class_names"])
    size = int(state["config"]["data"]["crop_size"])
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array((0.485, 0.456, 0.406), np.float32)) / np.array(
        (0.229, 0.224, 0.225), np.float32)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).to(device)
    try:
        logits = model(tensor)
    except TypeError as error:
        # SMP before 0.5 expects unpacked encoder features. This permits use of
        # the existing DeepLabs+ environment without altering its packages.
        if "conv2d() received an invalid combination" not in str(error):
            raise
        features = model.net.encoder(tensor)
        logits = model.net.segmentation_head(model.net.decoder(*features))
    small = logits.sigmoid()[0].cpu().numpy().astype(np.float32)
    height, width = image.shape[:2]
    probabilities = np.stack([
        cv2.resize(channel, (width, height), interpolation=cv2.INTER_LINEAR)
        for channel in small
    ])
    thresholds = np.asarray([state["thresholds"][name] for name in names], np.float32)
    return probabilities, probabilities >= thresholds[:, None, None], names


def rgb_defect_overlay(image: np.ndarray, masks: np.ndarray, names: list[str]) -> np.ndarray:
    """Overlay only structural RGB evidence used by the first fusion stage."""
    colors = {"crack": (0, 0, 255), "spalling": (0, 165, 255)}  # BGR
    result = image.copy()
    for index, name in enumerate(names):
        if name not in colors:
            continue
        selected = masks[index]
        layer = np.full_like(image, colors[name])
        blended = cv2.addWeighted(image, 0.45, layer, 0.55, 0)
        result[selected] = blended[selected]
    return result


def multimodal_status_overlay(image: np.ndarray, status: np.ndarray) -> np.ndarray:
    """Overlay IR-only, RGB-only, agreement, uncertain and influenced evidence."""
    result = image.copy()
    colors = {
        1: (255, 0, 0),       # IR only: blue
        2: (0, 255, 255),     # RGB only: yellow
        3: (0, 0, 255),       # agreement: red
        4: (0, 165, 255),     # uncertain: orange
        5: (180, 0, 180),     # equipment/non-wall influenced: purple
    }
    for value, color in colors.items():
        selected = status == value
        layer = np.full_like(image, color)
        blended = cv2.addWeighted(image, 0.45, layer, 0.55, 0)
        result[selected] = blended[selected]
    return result


def hysteresis_mask(probability: np.ndarray, high: float, low: float,
                    min_seed_area: int = 500, bridge_radius: int = 2,
                    dilation_radius: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grow low-confidence pixels only from sufficiently large high-confidence seeds."""
    if not 0 <= low <= high <= 1:
        raise ValueError("Expected 0 <= low <= high <= 1")
    if min_seed_area < 0 or bridge_radius < 0 or dilation_radius < 0:
        raise ValueError("Hysteresis morphology parameters must be non-negative")
    raw_seed = (probability >= high).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw_seed, connectivity=8)
    seed = np.zeros_like(raw_seed)
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= min_seed_area:
            seed[labels == index] = 1
    candidate = (probability >= low).astype(np.uint8)
    if bridge_radius:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (2 * bridge_radius + 1, 2 * bridge_radius + 1))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
    count, labels = cv2.connectedComponents(candidate, connectivity=8)
    selected = np.zeros_like(candidate)
    touching = np.unique(labels[seed > 0])
    for index in touching:
        if index != 0:
            selected[labels == index] = 1
    if dilation_radius:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (2 * dilation_radius + 1, 2 * dilation_radius + 1))
        selected = cv2.dilate(selected, kernel)
    return seed, candidate, selected


def filter_binary_components(mask: np.ndarray, min_area: int = 0,
                             drop_border: bool = False,
                             drop_side_border: bool = False,
                             drop_narrow_side_border_width: int = 0) -> np.ndarray:
    """Remove small components and, optionally, components touching the image border."""
    if mask.ndim != 2 or min_area < 0 or drop_narrow_side_border_width < 0:
        raise ValueError("Expected a 2-D mask and non-negative filter settings")
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = binary.shape
    result = np.zeros_like(binary)
    for index in range(1, count):
        x, y, w, h, area = map(int, stats[index])
        touches = x == 0 or y == 0 or x + w == width or y + h == height
        touches_side = x == 0 or x + w == width
        narrow_side_artifact = (touches_side and drop_narrow_side_border_width > 0
                                and w <= drop_narrow_side_border_width)
        if (area >= min_area and not (drop_border and touches)
                and not (drop_side_border and touches_side)
                and not narrow_side_artifact):
            result[labels == index] = 1
    return result


def load_homography(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".mat":
        from scipy.io import loadmat
        matrix = loadmat(path)["H"]
    elif path.suffix.lower() == ".npy":
        matrix = np.load(path, allow_pickle=False)
    else:
        raise ValueError("Homography must be a .mat or .npy file")
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("Homography must be a finite, nonsingular 3x3 RGB-to-IR matrix")
    return matrix


def overlay(image: np.ndarray, mask: np.ndarray, palette_rgb: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    color = cv2.cvtColor(palette_rgb[mask], cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(image, 0.55, color, 0.45, 0)
    if valid is None:
        return blended
    return np.where(valid[..., None], blended, image)


def component_palette(names: list[str]) -> np.ndarray:
    if names == ["building_surface", "opening", "other"]:
        return np.array(((0, 0, 255), (0, 255, 0), (255, 255, 0)), np.uint8)
    return np.array(COMPONENT_COLORS_RGB, np.uint8)


def ir_palette_and_names(cfg: dict) -> tuple[np.ndarray, list[str]]:
    """Return config-specific IR display metadata with legacy BFDD fallback."""
    names = list(cfg["data"]["class_names"])
    palette = np.asarray(cfg["data"].get("palette_rgb", BFDD_PALETTE_RGB), dtype=np.uint8)
    if palette.shape != (len(names), 3):
        raise ValueError("IR palette_rgb must contain one RGB triplet per class")
    return palette, names


def map_ir_truth(truth: np.ndarray, cfg: dict) -> np.ndarray:
    """Map source BFDD IDs into the model label space, preserving ignore=255."""
    raw_map = cfg["data"].get("label_map")
    if not raw_map:
        classes = int(cfg["model"]["classes"])
        if np.any((truth >= classes) & (truth != 255)):
            raise ValueError("IR truth contains IDs outside the configured model classes")
        return truth.copy()
    mapped = np.full(truth.shape, 255, dtype=np.uint8)
    for old, new in raw_map.items():
        mapped[truth == int(old)] = int(new)
    return mapped


def fusion_overlay(image: np.ndarray, confirmed: np.ndarray, uncertain: np.ndarray,
                   rejected: np.ndarray) -> np.ndarray:
    result = image.copy()
    colors = ((rejected != 0, (128, 128, 128)),
              (uncertain != 0, (0, 165, 255)),
              (confirmed != 0, (0, 0, 255)))  # BGR: gray, orange, red
    for selected, color in colors:
        layer = np.full_like(image, color)
        blended = cv2.addWeighted(image, 0.45, layer, 0.55, 0)
        result[selected] = blended[selected]
    return result


def status_overlay(image: np.ndarray, status: np.ndarray) -> np.ndarray:
    """Overlay defect disposition: confirmed, uncertain, equipment, excluded."""
    result = image.copy()
    colors = ((status == 4, (128, 128, 128)),  # excluded: gray
              (status == 3, (180, 0, 180)),    # equipment influenced: purple
              (status == 2, (0, 165, 255)),    # uncertain: orange
              (status == 1, (0, 0, 255)))      # confirmed: red
    for selected, color in colors:
        layer = np.full_like(image, color)
        blended = cv2.addWeighted(image, 0.45, layer, 0.55, 0)
        result[selected] = blended[selected]
    return result


def linear_artifact_candidates(defect: np.ndarray, evidence: np.ndarray,
                               confirmed: np.ndarray, min_length: int = 20,
                               max_gap: int = 6, thickness: int = 5) -> np.ndarray:
    """Find straight defect responses supported by nearby interference evidence."""
    if defect.shape != evidence.shape or defect.shape != confirmed.shape:
        raise ValueError("Linear-artifact inputs must have identical shapes")
    if min_length < 1 or max_gap < 0 or thickness < 1:
        raise ValueError("Invalid linear-artifact geometry settings")
    binary = ((defect != 0) & (confirmed != 0)).astype(np.uint8) * 255
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, threshold=max(8, min_length // 2),
                            minLineLength=min_length, maxLineGap=max_gap)
    line_mask = np.zeros_like(binary)
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            cv2.line(line_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness)
    return ((line_mask > 0) & (binary > 0) & (evidence > 0)).astype(np.uint8)


def compare_defects(before: np.ndarray, after: np.ndarray, truth: np.ndarray,
                    classes: int | None = None) -> dict:
    if before.shape != truth.shape or after.shape != truth.shape:
        raise ValueError("IR truth and predictions must have identical dimensions")
    if classes is not None and np.any((truth >= classes) & (truth != 255)):
        raise ValueError("IR truth contains class IDs outside the prediction label space")
    valid = truth != 255
    removed = valid & (before != 0) & (after == 0)
    return {
        "true_defect_pixels": int(np.count_nonzero(valid & (truth != 0))),
        "predicted_defect_pixels_before": int(np.count_nonzero(valid & (before != 0))),
        "predicted_defect_pixels_after": int(np.count_nonzero(valid & (after != 0))),
        "removed_correct_class_pixels": int(np.count_nonzero(removed & (before == truth))),
        "removed_wrong_class_or_background_pixels": int(np.count_nonzero(removed & (before != truth))),
        "binary_true_positive_before": int(np.count_nonzero(valid & (truth != 0) & (before != 0))),
        "binary_true_positive_after": int(np.count_nonzero(valid & (truth != 0) & (after != 0))),
        "binary_false_positive_before": int(np.count_nonzero(valid & (truth == 0) & (before != 0))),
        "binary_false_positive_after": int(np.count_nonzero(valid & (truth == 0) & (after != 0))),
    }


def save_png(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", type=Path, required=True, help="RGB photo from the same scene as --ir")
    parser.add_argument("--ir", type=Path, required=True, help="Pseudo-color IR image")
    parser.add_argument("--component-config", type=Path, default=Path("configs/components_hku_adapted.yaml"))
    parser.add_argument("--component-checkpoint", type=Path, default=Path("runs/components_hku_adapted/best.pt"))
    parser.add_argument("--ir-config", type=Path, default=Path("configs/ir_bfdd_weighted.yaml"))
    parser.add_argument("--ir-checkpoint", type=Path, default=Path("runs/ir_bfdd_weighted/best.pt"))
    parser.add_argument("--rgb-defect-checkpoint", type=Path,
                        help="Optional frozen six-channel RGB multilabel checkpoint")
    parser.add_argument("--rgb-defect-source", type=Path,
                        default=Path("../RGB/6ML/src"),
                        help="Path containing the rgb_multilabel package")
    parser.add_argument("--rgb-crack-radius", type=int, default=5,
                        help="IR-pixel agreement radius for thin RGB cracks")
    parser.add_argument("--rgb-spalling-radius", type=int, default=3,
                        help="IR-pixel agreement radius for RGB spalling")
    parser.add_argument("--ac-config", type=Path, help="Optional binary RGB air-conditioner config")
    parser.add_argument("--ac-checkpoint", type=Path, help="Optional binary RGB air-conditioner checkpoint")
    parser.add_argument("--ac-threshold", type=float, default=0.78)
    parser.add_argument("--ac-low-threshold", type=float,
                        help="Enable connected low-confidence growth from high-confidence AC seeds")
    parser.add_argument("--ac-min-seed-area", type=int, default=500)
    parser.add_argument("--ac-bridge-radius", type=int, default=2)
    parser.add_argument("--ac-dilation-radius", type=int, default=2)
    parser.add_argument("--ac-min-component-area", type=int, default=0)
    parser.add_argument("--ac-drop-border-components", action="store_true")
    parser.add_argument("--ac-drop-side-border-components", action="store_true")
    parser.add_argument("--ac-drop-narrow-side-border-width", type=int, default=0,
                        help="Drop only side-touching AC components no wider than this many RGB pixels")
    parser.add_argument("--ac-local-low-threshold", type=float,
                        help="Add lower-confidence pixels only within a fixed radius of filtered AC cores")
    parser.add_argument("--ac-local-radius", type=int, default=24)
    parser.add_argument("--rgb-to-ir", type=Path, help="Validated RGB-pixel to IR-pixel homography (.mat H or .npy)")
    parser.add_argument("--ir-label", type=Path, help="Optional original-size single-channel BFDD label PNG")
    parser.add_argument("--edge-radius", type=int, default=8,
                        help="IR-pixel radius of the uncertain component boundary band")
    parser.add_argument("--review-linear-artifacts", action="store_true",
                        help="Move supported thin straight responses from confirmed to uncertain")
    parser.add_argument("--line-ac-threshold", type=float, default=0.45)
    parser.add_argument("--line-proximity-radius", type=int, default=16)
    parser.add_argument("--line-min-length", type=int, default=20)
    parser.add_argument("--line-max-gap", type=int, default=6)
    parser.add_argument("--line-thickness", type=int, default=5)
    parser.add_argument("--ac-influence-radius", type=int, default=12,
                        help="IR-pixel radius around AC masks marked equipment-influenced")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.ac_config is None) != (args.ac_checkpoint is None):
        parser.error("--ac-config and --ac-checkpoint must be provided together")
    if not 0 <= args.ac_threshold <= 1:
        parser.error("--ac-threshold must be between 0 and 1")
    if args.ac_low_threshold is not None and not 0 <= args.ac_low_threshold <= args.ac_threshold:
        parser.error("--ac-low-threshold must be between 0 and --ac-threshold")
    if args.ac_min_component_area < 0:
        parser.error("--ac-min-component-area must be non-negative")
    if args.ac_drop_narrow_side_border_width < 0:
        parser.error("--ac-drop-narrow-side-border-width must be non-negative")
    if (args.ac_local_low_threshold is not None and
            not 0 <= args.ac_local_low_threshold <= args.ac_threshold):
        parser.error("--ac-local-low-threshold must be between 0 and --ac-threshold")
    if args.ac_local_radius < 0:
        parser.error("--ac-local-radius must be non-negative")
    if not 0 <= args.line_ac_threshold <= 1 or args.line_proximity_radius < 0:
        parser.error("Invalid linear-artifact evidence settings")
    if args.ac_influence_radius < 0:
        parser.error("--ac-influence-radius must be non-negative")
    if args.rgb_crack_radius < 0 or args.rgb_spalling_radius < 0:
        parser.error("RGB defect agreement radii must be non-negative")
    rgb = cv2.imread(str(args.rgb), cv2.IMREAD_COLOR)
    ir = cv2.imread(str(args.ir), cv2.IMREAD_COLOR)
    if rgb is None or ir is None:
        raise FileNotFoundError("Could not read RGB or IR input")
    truth = None
    if args.ir_label:
        truth = cv2.imread(str(args.ir_label), cv2.IMREAD_UNCHANGED)
        if truth is None or truth.ndim != 2 or truth.shape != ir.shape[:2]:
            raise ValueError("IR label must be a single-channel mask with the IR image size")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    component_model, component_cfg = load_model(args.component_config, args.component_checkpoint, "components", device)
    ir_model, ir_cfg = load_model(args.ir_config, args.ir_checkpoint, "ir", device)
    ir_palette, ir_class_names = ir_palette_and_names(ir_cfg)
    if truth is not None:
        truth = map_ir_truth(truth, ir_cfg)
    component = predict(component_model, rgb, tuple(component_cfg["data"]["image_size"]), device)
    defect = predict(ir_model, ir, tuple(ir_cfg["data"]["image_size"]), device)
    rgb_defect_probabilities = rgb_defect_masks = rgb_defect_state = None
    rgb_defect_names = None
    if args.rgb_defect_checkpoint:
        rgb_defect_model, rgb_defect_state = load_rgb_defect_model(
            args.rgb_defect_checkpoint, args.rgb_defect_source, device)
        rgb_defect_probabilities, rgb_defect_masks, rgb_defect_names = predict_rgb_defects(
            rgb_defect_model, rgb_defect_state, rgb, device)
    ac_probability = ac_mask = ac_incremental = ac_seed = ac_candidate = None
    ac_local_candidate = ac_local_vicinity = None
    if args.ac_checkpoint:
        ac_model, ac_cfg = load_model(args.ac_config, args.ac_checkpoint, "components", device)
        if ac_cfg["data"]["class_names"] != ["background", "air_conditioner"]:
            raise ValueError("AC model must use [background, air_conditioner] classes")
        ac_probability = predict_probability(ac_model, rgb, tuple(ac_cfg["data"]["image_size"]),
                                             device, class_index=1)
        if args.ac_low_threshold is None:
            ac_mask = (ac_probability >= args.ac_threshold).astype(np.uint8)
        else:
            ac_seed, ac_candidate, ac_mask = hysteresis_mask(
                ac_probability, args.ac_threshold, args.ac_low_threshold,
                args.ac_min_seed_area, args.ac_bridge_radius, args.ac_dilation_radius)
        ac_mask = filter_binary_components(ac_mask, args.ac_min_component_area,
                                           args.ac_drop_border_components,
                                           args.ac_drop_side_border_components,
                                           args.ac_drop_narrow_side_border_width)
        if args.ac_local_low_threshold is not None:
            ac_local_candidate = (ac_probability >= args.ac_local_low_threshold).astype(np.uint8)
            if args.ac_local_radius:
                radius = args.ac_local_radius
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                   (2 * radius + 1, 2 * radius + 1))
                ac_local_vicinity = cv2.dilate(ac_mask, kernel)
            else:
                ac_local_vicinity = ac_mask.copy()
            ac_mask |= ac_local_candidate & ac_local_vicinity
            ac_mask = filter_binary_components(ac_mask, args.ac_min_component_area,
                                               args.ac_drop_border_components,
                                               args.ac_drop_side_border_components,
                                               args.ac_drop_narrow_side_border_width)
        # The AC branch only needs to add exclusions where the component model
        # still says wall. Opening/other pixels are already rejected upstream.
        ac_incremental = (ac_mask & (component == 0)).astype(np.uint8)
    args.output.mkdir(parents=True, exist_ok=True)
    save_png(args.output / "component_ids.png", component)
    save_png(args.output / "defect_ids.png", defect)
    save_png(args.output / "component_overlay.png",
             overlay(rgb, component, component_palette(component_cfg["data"]["class_names"])))
    save_png(args.output / "defect_overlay.png", overlay(ir, defect, ir_palette, defect != 0))
    result = {"rgb": str(args.rgb.resolve()), "ir": str(args.ir.resolve()), "device": str(device),
              "component_checkpoint": str(args.component_checkpoint.resolve()),
              "ir_checkpoint": str(args.ir_checkpoint.resolve()),
              "ir_class_names": ir_class_names, "registration_applied": False}
    if rgb_defect_masks is not None and rgb_defect_names is not None:
        crack_index = rgb_defect_names.index("crack")
        spalling_index = rgb_defect_names.index("spalling")
        structural = (rgb_defect_masks[crack_index].astype(np.uint8)
                      | (rgb_defect_masks[spalling_index].astype(np.uint8) << 1))
        save_png(args.output / "rgb_defect_mask.png", structural)
        save_png(args.output / "rgb_defect_overlay.png",
                 rgb_defect_overlay(rgb, rgb_defect_masks, rgb_defect_names))
        np.savez_compressed(
            args.output / "rgb_defect_probabilities.npz",
            probabilities=rgb_defect_probabilities.astype(np.float16),
            class_names=np.asarray(rgb_defect_names),
            thresholds=np.asarray([rgb_defect_state["thresholds"][name]
                                   for name in rgb_defect_names], np.float32),
        )
        result["rgb_defect_checkpoint"] = str(args.rgb_defect_checkpoint.resolve())
        result["rgb_defect_thresholds"] = {
            name: float(rgb_defect_state["thresholds"][name])
            for name in rgb_defect_names
        }
        result["rgb_defect_pixels"] = {
            "crack": int(rgb_defect_masks[crack_index].sum()),
            "spalling": int(rgb_defect_masks[spalling_index].sum()),
        }
    if ac_mask is not None and ac_probability is not None:
        save_png(args.output / "ac_probability.png", np.rint(ac_probability * 255).astype(np.uint8))
        save_png(args.output / "ac_mask.png", ac_mask * 255)
        save_png(args.output / "ac_incremental_wall_mask.png", ac_incremental * 255)
        if ac_seed is not None and ac_candidate is not None:
            save_png(args.output / "ac_seed_mask.png", ac_seed * 255)
            save_png(args.output / "ac_candidate_mask.png", ac_candidate * 255)
        if ac_local_candidate is not None and ac_local_vicinity is not None:
            save_png(args.output / "ac_local_candidate_mask.png", ac_local_candidate * 255)
            save_png(args.output / "ac_local_vicinity_mask.png", ac_local_vicinity * 255)
        ac_overlay = rgb.copy()
        tint = np.full_like(rgb, (0, 255, 255))
        selected = ac_mask > 0
        ac_overlay[selected] = cv2.addWeighted(rgb, .45, tint, .55, 0)[selected]
        save_png(args.output / "ac_overlay.png", ac_overlay)
        result["ac_checkpoint"] = str(args.ac_checkpoint.resolve())
        result["ac_threshold"] = args.ac_threshold
        result["ac_min_component_area"] = args.ac_min_component_area
        result["ac_drop_border_components"] = args.ac_drop_border_components
        result["ac_drop_side_border_components"] = args.ac_drop_side_border_components
        result["ac_drop_narrow_side_border_width"] = args.ac_drop_narrow_side_border_width
        if args.ac_local_low_threshold is not None:
            result["ac_local_low_threshold"] = args.ac_local_low_threshold
            result["ac_local_radius"] = args.ac_local_radius
        if args.ac_low_threshold is not None:
            result["ac_low_threshold"] = args.ac_low_threshold
            result["ac_min_seed_area"] = args.ac_min_seed_area
            result["ac_bridge_radius"] = args.ac_bridge_radius
            result["ac_dilation_radius"] = args.ac_dilation_radius
        result["ac_rgb_coverage_fraction"] = float(ac_mask.mean())
        result["ac_incremental_wall_coverage_fraction"] = float(ac_incremental.mean())
    if truth is not None:
        result["unfiltered"] = compare_defects(defect, defect, truth, len(ir_class_names))
        save_png(args.output / "label_overlay.png", overlay(ir, np.where(truth == 255, 0, truth).astype(np.uint8),
                                                                 ir_palette, (truth != 0) & (truth != 255)))
    if args.rgb_to_ir:
        matrix = load_homography(args.rgb_to_ir)
        zones = partition_defects(defect, component, matrix, ir.shape[:2], args.edge_radius)
        ac_in_ir = None
        ac_equipment_in_ir = None
        if ac_mask is not None:
            ac_equipment_in_ir = cv2.warpPerspective(
                ac_mask, matrix.astype(np.float64), (ir.shape[1], ir.shape[0]),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                borderValue=0).astype(np.uint8)
            save_png(args.output / "ac_mask_full_in_ir.png", ac_equipment_in_ir * 255)
        if ac_incremental is not None:
            ac_in_ir = cv2.warpPerspective(ac_incremental, matrix.astype(np.float64),
                                           (ir.shape[1], ir.shape[0]), flags=cv2.INTER_NEAREST,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.uint8)
            selected = ac_in_ir > 0
            zones["wall"][selected] = 0
            zones["wall_core"][selected] = 0
            zones["uncertain_band"][selected] = 0
            zones["hard_reject"][selected] = 1
            zones["confirmed"][selected] = 0
            zones["uncertain"][selected] = 0
            zones["rejected"][selected] = defect[selected]
            zones["lenient"][selected] = 0
            save_png(args.output / "ac_mask_in_ir.png", ac_in_ir * 255)
            result["ac_ir_coverage_fraction"] = float(ac_in_ir.mean())
        if args.review_linear_artifacts:
            nonwall = (zones["warped_components"] != 0) & (zones["warped_components"] != 255)
            if args.line_proximity_radius:
                radius = args.line_proximity_radius
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
                evidence = cv2.dilate(nonwall.astype(np.uint8), kernel)
            else:
                evidence = nonwall.astype(np.uint8)
            if ac_probability is not None:
                ac_probability_ir = cv2.warpPerspective(
                    ac_probability, matrix.astype(np.float64), (ir.shape[1], ir.shape[0]),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                evidence |= (ac_probability_ir >= args.line_ac_threshold).astype(np.uint8)
                save_png(args.output / "ac_line_evidence_in_ir.png",
                         np.rint(np.clip(ac_probability_ir, 0, 1) * 255).astype(np.uint8))
            artifact = linear_artifact_candidates(
                defect, evidence, zones["confirmed"], args.line_min_length,
                args.line_max_gap, args.line_thickness)
            selected = artifact > 0
            zones["confirmed"][selected] = 0
            zones["uncertain"][selected] = defect[selected]
            save_png(args.output / "line_artifact_evidence.png", evidence * 255)
            save_png(args.output / "line_artifact_candidates.png", artifact * 255)
            result["linear_artifact_review"] = {
                "pixels_moved_to_uncertain": int(selected.sum()),
                "ac_probability_threshold": args.line_ac_threshold,
                "component_proximity_radius": args.line_proximity_radius,
                "min_length": args.line_min_length,
                "max_gap": args.line_max_gap,
                "thickness": args.line_thickness,
            }
        equipment_zone = np.zeros(ir.shape[:2], dtype=np.uint8)
        if ac_equipment_in_ir is not None:
            if args.ac_influence_radius:
                radius = args.ac_influence_radius
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                   (2 * radius + 1, 2 * radius + 1))
                equipment_zone = cv2.dilate(ac_equipment_in_ir, kernel)
            else:
                equipment_zone = ac_equipment_in_ir.copy()
        status = np.zeros(ir.shape[:2], dtype=np.uint8)
        status[zones["confirmed"] != 0] = 1
        status[zones["uncertain"] != 0] = 2
        status[zones["rejected"] != 0] = 4
        equipment_defect = (equipment_zone > 0) & (defect != 0)
        status[equipment_defect] = 3
        excluded_defect = (status == 4)
        save_png(args.output / "ac_influence_zone_in_ir.png", equipment_zone * 255)
        save_png(args.output / "equipment_defect_ids.png",
                 np.where(equipment_defect, defect, 0).astype(defect.dtype))
        save_png(args.output / "excluded_defect_ids.png",
                 np.where(excluded_defect, defect, 0).astype(defect.dtype))
        save_png(args.output / "defect_status.png", status)
        save_png(args.output / "final_status_overlay.png", status_overlay(ir, status))
        result["defect_status_legend"] = {
            "0": "none", "1": "wall_supported_candidate", "2": "uncertain",
            "3": "equipment_influenced", "4": "excluded_nonwall"}
        result["defect_status_semantics_version"] = 2
        result["ac_influence_radius_ir_pixels"] = args.ac_influence_radius
        result["status_pixels"] = {
            "wall_supported_candidate": int(np.count_nonzero(status == 1)),
            "uncertain": int(np.count_nonzero(status == 2)),
            "equipment_influenced": int(np.count_nonzero(status == 3)),
            "excluded_nonwall": int(np.count_nonzero(status == 4)),
        }
        if rgb_defect_masks is not None and rgb_defect_names is not None:
            crack_index = rgb_defect_names.index("crack")
            spalling_index = rgb_defect_names.index("spalling")
            crack_ir = cv2.warpPerspective(
                rgb_defect_masks[crack_index].astype(np.uint8), matrix.astype(np.float64),
                (ir.shape[1], ir.shape[0]), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            spalling_ir = cv2.warpPerspective(
                rgb_defect_masks[spalling_index].astype(np.uint8), matrix.astype(np.float64),
                (ir.shape[1], ir.shape[0]), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            save_png(args.output / "rgb_crack_in_ir.png", crack_ir * 255)
            save_png(args.output / "rgb_spalling_in_ir.png", spalling_ir * 255)
            crack_support = crack_ir
            if args.rgb_crack_radius:
                radius = args.rgb_crack_radius
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                   (2 * radius + 1, 2 * radius + 1))
                crack_support = cv2.dilate(crack_support, kernel)
            spalling_support = spalling_ir
            if args.rgb_spalling_radius:
                radius = args.rgb_spalling_radius
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                   (2 * radius + 1, 2 * radius + 1))
                spalling_support = cv2.dilate(spalling_support, kernel)
            rgb_support = (crack_support > 0) | (spalling_support > 0)
            rgb_structural = (crack_ir > 0) | (spalling_ir > 0)
            valid_rgb = rgb_structural & (zones["wall"] > 0)
            ir_primary = status == 1
            agreement = ir_primary & rgb_support
            multimodal = np.zeros(ir.shape[:2], dtype=np.uint8)
            multimodal[ir_primary] = 1
            multimodal[valid_rgb & (defect == 0)] = 2
            multimodal[agreement] = 3
            multimodal[status == 2] = 4
            influenced = ((status == 3) | (status == 4)) & ((defect != 0) | rgb_support)
            multimodal[influenced] = 5
            packed_rgb_ir = crack_ir.astype(np.uint8) | (spalling_ir.astype(np.uint8) << 1)
            save_png(args.output / "rgb_defect_in_ir.png", packed_rgb_ir)
            save_png(args.output / "rgb_defect_support_in_ir.png", rgb_support.astype(np.uint8) * 255)
            save_png(args.output / "multimodal_status.png", multimodal)
            save_png(args.output / "multimodal_status_overlay.png",
                     multimodal_status_overlay(ir, multimodal))
            result["multimodal_status_legend"] = {
                "0": "none", "1": "ir_only", "2": "rgb_only",
                "3": "rgb_ir_agree", "4": "uncertain",
                "5": "equipment_or_nonwall_influenced",
            }
            result["rgb_defect_agreement_radius_ir_pixels"] = {
                "crack": args.rgb_crack_radius,
                "spalling": args.rgb_spalling_radius,
            }
            result["multimodal_status_pixels"] = {
                "ir_only": int(np.count_nonzero(multimodal == 1)),
                "rgb_only": int(np.count_nonzero(multimodal == 2)),
                "rgb_ir_agree": int(np.count_nonzero(multimodal == 3)),
                "uncertain": int(np.count_nonzero(multimodal == 4)),
                "equipment_or_nonwall_influenced": int(np.count_nonzero(multimodal == 5)),
            }
        warped_rgb = cv2.warpPerspective(rgb, matrix, (ir.shape[1], ir.shape[0]))
        checker = ir.copy()
        tile = 64
        yy, xx = np.indices(ir.shape[:2])
        use_rgb = ((yy // tile + xx // tile) % 2) == 0
        checker[use_rgb] = warped_rgb[use_rgb]
        save_png(args.output / "registration_checker.png", checker)
        save_png(args.output / "component_ids_in_ir.png", zones["warped_components"])
        save_png(args.output / "wall_in_ir.png", zones["wall"] * 255)
        save_png(args.output / "wall_core_in_ir.png", zones["wall_core"] * 255)
        save_png(args.output / "uncertain_band_in_ir.png", zones["uncertain_band"] * 255)
        save_png(args.output / "hard_reject_in_ir.png", zones["hard_reject"] * 255)
        save_png(args.output / "confirmed_defect_ids.png", zones["confirmed"])
        # Preferred semantic name; the legacy filename above remains for
        # downstream compatibility with existing projection runs.
        save_png(args.output / "wall_supported_candidate_ids.png", zones["confirmed"])
        save_png(args.output / "uncertain_defect_ids.png", zones["uncertain"])
        save_png(args.output / "rejected_defect_ids.png", zones["rejected"])
        save_png(args.output / "filtered_defect_ids.png", zones["lenient"])
        save_png(args.output / "filtered_overlay.png",
                 overlay(ir, zones["lenient"], ir_palette, zones["lenient"] != 0))
        save_png(args.output / "fusion_overlay.png",
                 fusion_overlay(ir, zones["confirmed"], zones["uncertain"], zones["rejected"]))
        result["registration_applied"] = True
        result["rgb_to_ir"] = str(args.rgb_to_ir.resolve())
        result["edge_radius_ir_pixels"] = args.edge_radius
        result["wall_coverage_fraction"] = float(zones["wall"].mean())
        result["wall_core_fraction"] = float(zones["wall_core"].mean())
        result["uncertain_band_fraction"] = float(zones["uncertain_band"].mean())
        result["defect_pixels"] = {
            "unfiltered": int(np.count_nonzero(defect)),
            "wall_supported_candidate": int(np.count_nonzero(zones["confirmed"])),
            "uncertain": int(np.count_nonzero(zones["uncertain"])),
            "rejected": int(np.count_nonzero(zones["rejected"])),
        }
        if truth is not None:
            result["wall_supported_candidate_comparison"] = compare_defects(
                defect, zones["confirmed"], truth, len(ir_class_names))
            result["lenient_comparison"] = compare_defects(
                defect, zones["lenient"], truth, len(ir_class_names))
    (args.output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
