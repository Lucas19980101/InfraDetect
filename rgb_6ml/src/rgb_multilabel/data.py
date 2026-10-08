from __future__ import annotations

import csv
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .labels import CLASS_NAMES
from .masks import decode_mask


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_transforms(crop_size: int, training: bool) -> A.Compose:
    # DACL10K images have widely varying sizes and aspect ratios. Both models
    # receive the same complete image resized to a fixed tensor, matching the
    # public toolkit protocol and avoiding biased center-crop evaluation.
    transforms: list[A.BasicTransform] = [A.Resize(crop_size, crop_size)]
    if training:
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(0.12, 0.12, p=0.3),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 3)),
                        A.GaussNoise(std_range=(0.01, 0.04)),
                    ],
                    p=0.15,
                ),
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(-0.03, 0.03),
                    rotate=(-5, 5),
                    p=0.2,
                ),
            ]
        )
    transforms.append(A.Normalize(mean=MEAN, std=STD))
    return A.Compose(transforms)


class DACL10KSixLabelDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        cache_root: str | Path,
        manifest_path: str | Path,
        split: str,
        crop_size: int,
        training: bool = False,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported labeled split: {split}")
        self.data_root = Path(data_root)
        self.cache_root = Path(cache_root)
        self.transform = build_transforms(crop_size, training)
        with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row["experiment_split"] == split]
        if not rows:
            raise ValueError(f"Manifest contains no samples for split {split!r}")
        self.samples: list[tuple[Path, Path, str]] = []
        for row in rows:
            official_split = row["official_split"]
            image_path = self.data_root / "images" / official_split / row["image_name"]
            mask_path = self.cache_root / official_split / f"{Path(row['image_name']).stem}.png"
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"Missing cached mask {mask_path}; run scripts/prepare_dacl10k_masks.py"
                )
            self.samples.append((image_path, mask_path, Path(row["image_name"]).stem))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_path, mask_path, name = self.samples[index]
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        encoded = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if bgr is None or encoded is None:
            raise ValueError(f"Could not read {image_path} or {mask_path}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        targets, valid_mask = decode_mask(encoded)
        if image.shape[:2] != encoded.shape:
            raise ValueError(
                f"Image/mask shape mismatch for {name}: {image.shape[:2]} vs {encoded.shape}"
            )
        combined_mask = np.concatenate(
            (targets.transpose(1, 2, 0), valid_mask[..., None]), axis=2
        ).astype(np.uint8)
        transformed = self.transform(image=image, mask=combined_mask)
        transformed_mask = transformed["mask"]
        tensor = torch.from_numpy(
            np.ascontiguousarray(transformed["image"].transpose(2, 0, 1))
        ).float()
        target_tensor = torch.from_numpy(
            np.ascontiguousarray(transformed_mask[..., : len(CLASS_NAMES)].transpose(2, 0, 1))
        ).float()
        valid_tensor = torch.from_numpy(
            np.ascontiguousarray(transformed_mask[..., len(CLASS_NAMES)][None])
        ).bool()
        return {
            "image": tensor,
            "target": target_tensor,
            "valid_mask": valid_tensor,
            "name": name,
        }
