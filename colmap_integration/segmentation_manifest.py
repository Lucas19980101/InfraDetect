#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""A simple six-class image/mask manifest for external inspection datasets."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from rgb_multilabel.data import build_transforms
from rgb_multilabel.labels import CLASS_NAMES


class SegmentationManifestDataset(Dataset):
    """Load masks by class, preserving missing classes as unsupervised."""

    def __init__(self, manifest: Path, split: str, image_size: int):
        self.manifest = manifest.resolve()
        self.transform = build_transforms(image_size, training=False)
        with self.manifest.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not {"split", "image"}.issubset(reader.fieldnames):
                raise ValueError("Manifest requires split and image columns")
            self.rows = [row for row in reader if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No {split} rows in {manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def _path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.manifest.parent / path

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image_path = self._path(row["image"])
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not read image {image_path}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = bgr.shape[:2]
        target = np.zeros((height, width, len(CLASS_NAMES)), dtype=np.uint8)
        valid = np.zeros_like(target)
        for class_index, name in enumerate(CLASS_NAMES):
            mask_name = (row.get(name) or "").strip()
            if not mask_name:
                continue
            mask_path = self._path(mask_name)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != (height, width):
                raise ValueError(f"Missing or wrong-size mask {mask_path}")
            target[..., class_index] = mask > 0
            valid_name = (row.get(f"valid_{name}") or "").strip()
            if valid_name:
                valid_path = self._path(valid_name)
                valid_mask = cv2.imread(str(valid_path), cv2.IMREAD_GRAYSCALE)
                if valid_mask is None or valid_mask.shape != (height, width):
                    raise ValueError(f"Missing or wrong-size validity mask {valid_path}")
                valid[..., class_index] = valid_mask > 0
            else:
                valid[..., class_index] = 1
        combined = np.concatenate((target, valid), axis=2)
        transformed = self.transform(image=image, mask=combined)
        mask = transformed["mask"]
        count = len(CLASS_NAMES)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(
                transformed["image"].transpose(2, 0, 1))).float(),
            "target": torch.from_numpy(np.ascontiguousarray(
                mask[..., :count].transpose(2, 0, 1))).float(),
            "valid_mask": torch.from_numpy(np.ascontiguousarray(
                mask[..., count:].transpose(2, 0, 1))).bool(),
            "name": image_path.stem,
        }
