from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .data import build_transforms
from .labels import CLASS_NAMES


SOURCE_CLASS = {"crack": 0, "spalling": 1}
SOURCE_VALUE = {"crack": 1, "spalling": 2}


class BuildingPartialLabelDataset(Dataset):
    """Current-building tiles with source-aware partial class supervision."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int,
        training: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = build_transforms(image_size, training)
        with (self.root / "manifest.csv").open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        self.samples: list[tuple[Path, Path, str, str]] = []
        for row in rows:
            source = row["source"]
            if source not in SOURCE_CLASS:
                raise ValueError(f"Unknown building label source: {source!r}")
            name = row["name"]
            image = self.root / "images" / split / f"{name}.jpg"
            mask = self.root / "masks" / split / f"{name}.png"
            if not image.is_file() or not mask.is_file():
                raise FileNotFoundError(f"Missing image/mask pair for {name}")
            self.samples.append((image, mask, name, source))
        if not self.samples:
            raise ValueError(f"No building samples for split {split!r}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_path, mask_path, name, source = self.samples[index]
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if bgr is None or mask is None or mask.ndim != 2:
            raise ValueError(f"Could not read building pair {image_path}, {mask_path}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        targets = np.zeros((*mask.shape, len(CLASS_NAMES)), dtype=np.uint8)
        supervised = np.zeros_like(targets)
        class_index = SOURCE_CLASS[source]
        pixel_valid = mask != 255
        targets[..., class_index] = mask == SOURCE_VALUE[source]
        supervised[..., class_index] = pixel_valid
        combined = np.concatenate((targets, supervised), axis=2)
        transformed = self.transform(image=image, mask=combined)
        transformed_mask = transformed["mask"]
        class_count = len(CLASS_NAMES)
        return {
            "image": torch.from_numpy(
                np.ascontiguousarray(transformed["image"].transpose(2, 0, 1))
            ).float(),
            "target": torch.from_numpy(
                np.ascontiguousarray(transformed_mask[..., :class_count].transpose(2, 0, 1))
            ).float(),
            "valid_mask": torch.from_numpy(
                np.ascontiguousarray(transformed_mask[..., class_count:].transpose(2, 0, 1))
            ).bool(),
            "name": name,
            "source": source,
        }
