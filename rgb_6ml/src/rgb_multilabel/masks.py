from __future__ import annotations

import numpy as np

from .labels import CLASS_NAMES


VALID_BIT = 1 << len(CLASS_NAMES)


def encode_mask(targets: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Pack six binary targets and the valid mask into one uint8 image."""
    expected_channels = len(CLASS_NAMES)
    if targets.ndim != 3 or targets.shape[0] != expected_channels:
        raise ValueError(f"Expected targets [{expected_channels},H,W], got {targets.shape}")
    if valid_mask.shape != targets.shape[1:]:
        raise ValueError(
            f"valid_mask shape {valid_mask.shape} does not match targets {targets.shape[1:]}"
        )
    encoded = np.zeros(valid_mask.shape, dtype=np.uint8)
    for index in range(expected_channels):
        encoded |= targets[index].astype(np.uint8) << index
    encoded |= valid_mask.astype(np.uint8) * VALID_BIT
    return encoded


def decode_mask(encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpack a uint8 cache image into boolean targets and valid mask."""
    if encoded.ndim != 2 or encoded.dtype != np.uint8:
        raise ValueError(f"Expected a 2D uint8 mask, got {encoded.shape} {encoded.dtype}")
    targets = np.stack(
        [((encoded >> index) & 1).astype(np.bool_) for index in range(len(CLASS_NAMES))]
    )
    valid_mask = (encoded & VALID_BIT).astype(np.bool_)
    return targets, valid_mask
