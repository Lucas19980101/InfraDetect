"""Full-resolution sliding-window thermal inference, without evaluation/training."""

from __future__ import annotations

import numpy as np
import torch


def starts(length: int, window: int, overlap: float) -> list[int]:
    """Cover an axis fully, including its far edge, with overlapping crops."""

    if window >= length:
        return [0]
    stride = max(1, round(window * (1.0 - overlap)))
    positions = list(range(0, length - window + 1, stride))
    if positions[-1] != length - window:
        positions.append(length - window)
    return positions


@torch.no_grad()
def predict_full(model: torch.nn.Module, row: dict[str, str], cfg: dict,
                 device: torch.device) -> torch.Tensor:
    """Mean-blend patch logits at the image's original H×W coordinates."""

    data = cfg["data"]
    array = np.load(row["image"], mmap_mode="r", allow_pickle=False)
    thermal = np.asarray(array[..., int(data.get("thermal_channel", 3))], dtype=np.float32)
    rgb = (np.asarray(array[..., [2, 1, 0]], dtype=np.float32)
           if data.get("input_mode") == "rgb_thermal" else None)
    height, width = thermal.shape
    patch_h, patch_w = map(int, data["image_size"])
    overlap = float(data.get("tile_overlap", 0.25))
    mean, std = float(data["normalization"]["mean"]), float(data["normalization"]["std"])
    rgb_mean = np.asarray(data.get("rgb_normalization", {}).get(
        "mean", [123.675, 116.28, 103.53]), np.float32)
    rgb_std = np.asarray(data.get("rgb_normalization", {}).get(
        "std", [58.395, 57.12, 57.375]), np.float32)
    use_amp = device.type == "cuda" and bool(cfg["train"].get("mixed_precision", False))
    summed = torch.zeros((cfg["model"]["classes"], height, width), dtype=torch.float32)
    counts = torch.zeros((height, width), dtype=torch.float32)
    for top in starts(height, patch_h, overlap):
        for left in starts(width, patch_w, overlap):
            patch = thermal[top:top + patch_h, left:left + patch_w]
            image = np.repeat(((patch - mean) / std)[None, None], 3, axis=1)
            if rgb is not None:
                rgb_patch = rgb[top:top + patch_h, left:left + patch_w]
                rgb_input = (rgb_patch - rgb_mean) / rgb_std
                image = np.concatenate((rgb_input.transpose(2, 0, 1)[None], image), axis=1)
            tensor = torch.from_numpy(np.ascontiguousarray(image)).to(device)
            with torch.autocast(device.type, enabled=use_amp):
                logits = model(tensor)[0].float().cpu()
            ph, pw = patch.shape
            summed[:, top:top + ph, left:left + pw] += logits[:, :ph, :pw]
            counts[top:top + ph, left:left + pw] += 1
    return summed / counts.clamp_min(1).unsqueeze(0)
