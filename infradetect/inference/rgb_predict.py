"""Load RGB checkpoints and predict component, equipment, and defect evidence.

Only the functions needed by the integrated inference exporter are included.
Training datasets, optimizers, metrics, and label-rasterization code are absent.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .config import config_from_file
from .thermal_context_architectures import build_model
from .rgb_defect_architectures import build_model as build_rgb_defect_model
from .tiling import predict_tiled


def load_model(config_path: Path, checkpoint_path: Path, task: str,
               device: torch.device):
    """Load a component or equipment model after checking task/class IDs."""

    cfg = config_from_file(config_path)
    if cfg["task"] != task:
        raise ValueError(f"Expected {task} config: {config_path}")
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if (state["config"]["task"] != task or
            state["config"]["data"]["class_names"] != cfg["data"]["class_names"]):
        raise ValueError(f"Checkpoint task/classes differ from {config_path}")
    model = build_model(task, cfg["model"]["classes"], pretrained=False).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def predict_native_class_probabilities(model, image: np.ndarray,
                                       size: tuple[int, int], device: torch.device,
                                       image_id: str = "image") -> tuple[np.ndarray, list[dict]]:
    """Mean-blend component/equipment patch probabilities at original pixels."""

    patch_h, patch_w = map(int, size)
    stride = (max(1, round(0.75 * patch_h)), max(1, round(0.75 * patch_w)))

    def predict_patch(crop: np.ndarray) -> np.ndarray:
        """Normalize one BGR crop using the model's ImageNet RGB convention."""

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - np.array((0.485, 0.456, 0.406), np.float32)) / np.array(
            (0.229, 0.224, 0.225), np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).to(device)
        logits = model(tensor)
        if logits.shape[-2:] != crop.shape[:2]:
            raise ValueError("Component/AC model changed patch output size")
        return logits.softmax(1)[0].cpu().numpy().astype(np.float32)

    probabilities, _, patches = predict_tiled(
        image, predict_patch, image_id=image_id,
        patch_size=(patch_h, patch_w), stride=stride)
    return probabilities, patches


def load_rgb_defect_model(checkpoint_path: Path, source_path: Path,
                          device: torch.device):
    """Load six-class RGB weights; source_path remains for manifest compatibility.

    The network code now lives in this package, so inference does not mutate
    sys.path. The preflight still checks that the declared source directory exists.
    """

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = build_rgb_defect_model(state["config"]["model"], pretrained=False).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


@torch.no_grad()
def predict_rgb_defects(model, state: dict, image: np.ndarray,
                        device: torch.device, *, image_id: str = "image",
                        patch_size: int = 512, stride: int = 384,
                        roi: np.ndarray | None = None,
                        return_coverage: bool = False):
    """Return full-size RGB probabilities and explicit evaluated-pixel coverage."""

    names = list(state["config"]["data"]["class_names"])

    def predict_patch(crop: np.ndarray) -> np.ndarray:
        """Predict one crop; support both current and older SMP decoder APIs."""

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - np.array((0.485, 0.456, 0.406), np.float32)) / np.array(
            (0.229, 0.224, 0.225), np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).to(device)
        try:
            logits = model(tensor)
        except TypeError as error:
            # Older segmentation-models-pytorch versions require unpacked features.
            if "conv2d() received an invalid combination" not in str(error):
                raise
            features = model.net.encoder(tensor)
            logits = model.net.segmentation_head(model.net.decoder(*features))
        return logits.sigmoid()[0].cpu().numpy().astype(np.float32)

    probabilities, coverage, patches = predict_tiled(
        image, predict_patch, image_id=image_id, patch_size=patch_size,
        stride=stride, roi=roi)
    thresholds = np.asarray([state["thresholds"][name] for name in names], np.float32)
    masks = (probabilities >= thresholds[:, None, None]) & coverage[None]
    if return_coverage:
        return probabilities, masks, names, coverage, patches
    return probabilities, masks, names
