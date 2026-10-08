from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def build_transforms(tile_size: int, training: bool) -> A.Compose:
    transforms: list[A.BasicTransform] = []
    if training:
        transforms.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(0.12, 0.12, p=0.3),
            A.OneOf([A.GaussianBlur(blur_limit=(3, 3)), A.GaussNoise(std_range=(0.01, 0.04))], p=0.15),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.03, 0.03), rotate=(-5, 5), p=0.2),
        ])
    transforms.extend([
        A.PadIfNeeded(tile_size, tile_size, border_mode=cv2.BORDER_REFLECT_101),
        A.RandomCrop(tile_size, tile_size) if training else A.CenterCrop(tile_size, tile_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return A.Compose(transforms)


class SemanticDataset(Dataset):
    def __init__(self, root: str | Path, split: str, tile_size: int, training: bool = False):
        root = Path(root)
        self.image_dir = root / "images" / split
        self.mask_dir = root / "masks" / split
        if not self.image_dir.is_dir() or not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Expected {self.image_dir} and {self.mask_dir}")
        self.images = sorted(p for p in self.image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        self.pairs = [(p, self._mask_for(p)) for p in self.images]
        self.transform = build_transforms(tile_size, training)

    def _mask_for(self, image: Path) -> Path:
        mask = self.mask_dir / f"{image.stem}.png"
        if not mask.exists():
            raise FileNotFoundError(f"Missing mask for {image.name}: {mask}")
        return mask

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_path, mask_path = self.pairs[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            raise ValueError(f"Could not read {image_path} or {mask_path}")
        if mask.ndim != 2:
            raise ValueError(f"Mask must be single-channel: {mask_path}")
        result = self.transform(image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB), mask=mask)
        tensor = torch.from_numpy(np.ascontiguousarray(result["image"].transpose(2, 0, 1))).float()
        return {"image": tensor, "mask": torch.from_numpy(result["mask"].astype(np.int64)), "name": image_path.stem}

