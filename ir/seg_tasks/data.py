from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


COMPONENT_NAMES = ("facade", "window", "obstacle", "sky", "door")
COMPONENT_COLORS_RGB = ((0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255))
BFDD_NAMES = ("Background", "Cracks", "Peeling", "Hollow Areas", "Stains", "Erosion")
BFDD_PALETTE_RGB = np.array(((0, 0, 0), (61, 61, 245), (169, 36, 191),
                             (174, 79, 13), (36, 179, 83), (203, 253, 0)), dtype=np.uint8)
IRREGULAR_3DOM_GRAY_LABELS = {
    "Navona_image123": "Navona_image123_lpng.png",
    "Navona_image147_undist": "Navona_image147_l_colored_undist.png",
    "Navona_image148_undist": "Navona_image148_l_colored_undist.png",
    "Navona_image149_undist": "Navona_image149_l_colored_undist.png",
    "Navona_image151_undist": "Navona_image151_l_colored_undist.png",
    "Navona_image152_undist": "Navona_image152_l_colored_undist.png",
    "image115i": "image115_l.png",
}


def component_pairs(root: Path) -> list[tuple[Path, Path]]:
    """Pair only exact 3DOM stems; irregular names require explicit review."""
    images = {p.stem: p for p in (root / "data").iterdir() if p.suffix.lower() == ".jpg"}
    labels = {p.stem[:-2]: p for p in (root / "labels_gray").iterdir()
              if p.suffix.lower() == ".png" and p.stem.endswith("_l")}
    for name, filename in IRREGULAR_3DOM_GRAY_LABELS.items():
        labels[name] = root / "labels_gray" / filename
    if missing := images.keys() - labels.keys():
        raise ValueError(f"Unpaired 3DOM images: {sorted(missing)}")
    return [(images[name], labels[name]) for name in sorted(images)]


def load_component_mask(path: Path, classes: int = 5) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Cannot read component mask: {path}")
    if raw.ndim == 2:
        values = np.unique(raw)
        if np.any((values >= classes) & (values != 255)):
            raise ValueError(f"Unexpected component label values {values.tolist()}: {path}")
        return raw.astype(np.uint8)
    if raw.ndim != 3 or raw.shape[2] != 3:
        raise ValueError(f"Expected indexed or three identical gray channels: {path}")
    if not (np.array_equal(raw[..., 0], raw[..., 1]) and np.array_equal(raw[..., 0], raw[..., 2])):
        raise ValueError(f"Gray label channels differ: {path}")
    gray = raw[..., 0]
    if np.any(gray > 5):
        raise ValueError(f"Unexpected 3DOM label value: {path}")
    result = gray.astype(np.uint8) - np.uint8(1)
    result[gray == 0] = 255
    return result


def load_ir_mask(path: Path, classes: int,
                 label_map: dict[int, int] | None = None) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None or mask.ndim != 2:
        raise ValueError(f"Expected single-channel IR class-index PNG: {path}")
    if label_map is not None:
        source = mask.copy()
        mask = np.full(source.shape, 255, dtype=np.uint8)
        for old, new in label_map.items():
            mask[source == old] = new
    values = np.unique(mask)
    if np.any((values >= classes) & (values != 255)):
        raise ValueError(f"Unexpected IR label values {values.tolist()}: {path}")
    return mask


def read_manifest(path: Path, split: str, root: Path, task: str,
                  include_sources: set[str] | None = None) -> list[tuple[Path, Path]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    if include_sources is not None:
        rows = [row for row in rows if row.get("source") in include_sources]
    if task == "ir":
        return [(root / "IR" / f"{row['name']}.png", root / "Label" / f"{row['name']}.png") for row in rows]
    if rows and all(row.get("image") and row.get("mask") for row in rows):
        return [(Path(row["image"]), Path(row["mask"])) for row in rows]
    available = {image.stem: (image, mask) for image, mask in component_pairs(root)}
    return [available[row["name"]] for row in rows]


def resize_and_pad(image: np.ndarray, mask: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    scale = min(height / h, width / w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    top, left = (height - new_h) // 2, (width - new_w) // 2
    image = cv2.copyMakeBorder(image, top, height - new_h - top, left, width - new_w - left,
                               cv2.BORDER_CONSTANT, value=(0, 0, 0))
    mask = cv2.copyMakeBorder(mask, top, height - new_h - top, left, width - new_w - left,
                              cv2.BORDER_CONSTANT, value=255)
    return image, mask


def restore_mask(mask: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
    height, width = original_size
    canvas_h, canvas_w = mask.shape
    scale = min(canvas_h / height, canvas_w / width)
    resized_h, resized_w = max(1, round(height * scale)), max(1, round(width * scale))
    top, left = (canvas_h - resized_h) // 2, (canvas_w - resized_w) // 2
    cropped = mask[top:top + resized_h, left:left + resized_w]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_NEAREST)


class SegmentationDataset(Dataset):
    def __init__(self, root: Path, manifest: Path, split: str, task: str,
                 image_size: tuple[int, int], classes: int, augment: bool = False,
                 include_sources: set[str] | None = None,
                 rotation_degrees: float = 0.0, rotation_probability: float = 0.5,
                 label_map: dict[int, int] | None = None):
        if task not in {"ir", "components"}:
            raise ValueError(task)
        self.task, self.size, self.classes, self.augment = task, image_size, classes, augment
        self.rotation_degrees = float(rotation_degrees)
        self.rotation_probability = float(rotation_probability)
        self.label_map = label_map
        if self.rotation_degrees < 0 or not 0 <= self.rotation_probability <= 1:
            raise ValueError("Invalid rotation augmentation settings")
        self.pairs = read_manifest(manifest, split, root, task, include_sources)
        with manifest.open(newline="", encoding="utf-8") as handle:
            selected_rows = [row for row in csv.DictReader(handle)
                             if row["split"] == split and
                             (include_sources is None or row.get("source") in include_sources)]
        self.sources = [row.get("source", task) for row in selected_rows]
        if len(self.sources) != len(self.pairs):
            raise ValueError("Manifest source metadata and resolved pairs differ")
        if not self.pairs:
            raise ValueError(f"Empty {task}/{split} manifest")
        for image, mask in self.pairs:
            if not image.is_file() or not mask.is_file():
                raise FileNotFoundError(f"Missing image or mask: {image}, {mask}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_path, mask_path = self.pairs[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = (load_ir_mask(mask_path, self.classes, self.label_map) if self.task == "ir"
                else load_component_mask(mask_path, self.classes))
        if image is not None and self.task == "components" and image.shape[:2] == mask.shape[::-1]:
            # 51 landscape 3DOM photos have portrait masks. Boundary overlays
            # and an all-pair edge check confirm counterclockwise mask rotation.
            mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if image is None or image.shape[:2] != mask.shape:
            raise ValueError(f"Image/mask size mismatch: {image_path}, {mask_path}")
        original_size = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, mask = resize_and_pad(image, mask, *self.size)
        if self.augment:
            if np.random.random() < 0.5:
                image, mask = np.flip(image, 1), np.flip(mask, 1)
            if self.rotation_degrees and np.random.random() < self.rotation_probability:
                angle = np.random.uniform(-self.rotation_degrees, self.rotation_degrees)
                center = ((image.shape[1] - 1) / 2, (image.shape[0] - 1) / 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=(0, 0, 0))
                mask = cv2.warpAffine(mask, matrix, (mask.shape[1], mask.shape[0]),
                                      flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=255)
        # Both inputs are encoded as 8-bit RGB images. These statistics are for
        # ImageNet encoder initialization, not a claim that IR is visible light.
        mean = np.array((0.485, 0.456, 0.406), np.float32)
        std = np.array((0.229, 0.224, 0.225), np.float32)
        image = (image.astype(np.float32) / 255.0 - mean) / std
        return {"image": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
                "mask": torch.from_numpy(np.ascontiguousarray(mask.astype(np.int64))),
                "name": image_path.stem, "original_size": np.asarray(original_size, dtype=np.int32)}
