from __future__ import annotations

import cv2
import numpy as np
import torch


def positions(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile: return [0]
    values = list(range(0, length - tile + 1, stride))
    if values[-1] != length - tile: values.append(length - tile)
    return values


def blend_window(size: int) -> torch.Tensor:
    axis = torch.hann_window(size, periodic=False).clamp_min(1e-3)
    return axis[:, None] * axis[None, :]


def normalize_rgb(tile: np.ndarray) -> torch.Tensor:
    value = torch.from_numpy(np.ascontiguousarray(tile.transpose(2, 0, 1))).float() / 255
    mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
    return (value - mean) / std


@torch.no_grad()
def sliding_window_predict(model, image_rgb: np.ndarray, tile_size: int, stride: int, classes: int,
                           device: torch.device, batch_size: int = 4, return_features: bool = False):
    height, width = image_rgb.shape[:2]
    pad_h, pad_w = max(0, tile_size - height), max(0, tile_size - width)
    padded = cv2.copyMakeBorder(image_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    ph, pw = padded.shape[:2]; ys, xs = positions(ph, tile_size, stride), positions(pw, tile_size, stride)
    logits_sum = torch.zeros(classes, ph, pw); weights = torch.zeros(ph, pw)
    window = blend_window(tile_size); coordinates = [(y, x) for y in ys for x in xs]
    decoder_tiles = []
    model.eval()
    for start in range(0, len(coordinates), batch_size):
        coords = coordinates[start:start + batch_size]
        batch = torch.stack([normalize_rgb(padded[y:y + tile_size, x:x + tile_size]) for y, x in coords]).to(device)
        output = model(batch, return_features=return_features)
        batch_logits = output["logits"] if return_features else output
        for index, (y, x) in enumerate(coords):
            logits_sum[:, y:y + tile_size, x:x + tile_size] += batch_logits[index].cpu() * window
            weights[y:y + tile_size, x:x + tile_size] += window
            if return_features: decoder_tiles.append({"y": y, "x": x, "feature": output["decoder"][index].cpu()})
    logits = (logits_sum / weights.clamp_min(1e-8))[..., :height, :width]
    return (logits, decoder_tiles) if return_features else logits

