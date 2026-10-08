# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .data import build_transforms
from .labels import CLASS_NAMES


class PartialLabelManifestDataset(Dataset):
    """Multi-label masks where blank class paths mean unsupervised classes."""

    def __init__(
        self,
        manifest: str | Path,
        splits: str | list[str],
        image_size: int,
        training: bool,
        source: str,
    ):
        self.manifest = Path(manifest).resolve()
        selected = {splits} if isinstance(splits, str) else set(splits)
        self.transform = build_transforms(image_size, training)
        self.source = source
        with self.manifest.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"split", "image", *CLASS_NAMES}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"Manifest requires columns {sorted(required)}")
            self.rows = [row for row in reader if row["split"] in selected]
        if not self.rows:
            raise ValueError(f"No rows for splits {sorted(selected)} in {self.manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def _path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.manifest.parent / path

    def __getitem__(self, index: int) -> dict[str, object]:
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
            mask_name = row[name].strip()
            if not mask_name:
                continue
            mask_path = self._path(mask_name)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != (height, width):
                raise ValueError(f"Missing or wrong-size mask {mask_path}")
            target[..., class_index] = mask > 0
            valid_name = (row.get(f"valid_{name}") or "").strip()
            if valid_name:
                validity = cv2.imread(str(self._path(valid_name)), cv2.IMREAD_GRAYSCALE)
                if validity is None or validity.shape != (height, width):
                    raise ValueError(f"Missing or wrong-size validity mask for {mask_path}")
                valid[..., class_index] = validity > 0
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
