from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


CLASS_NAMES = (
    "crack",
    "spalling",
    "exposed_rebar",
    "rust",
    "efflorescence",
    "wetspot",
)

DACL10K_MAPPING = {
    "crack": frozenset(("Crack", "ACrack")),
    "spalling": frozenset(("Spalling",)),
    "exposed_rebar": frozenset(("ExposedRebars",)),
    "rust": frozenset(("Rust",)),
    "efflorescence": frozenset(("Efflorescence",)),
    "wetspot": frozenset(("Wetspot",)),
}

SOURCE_TO_CLASS = {
    source: class_index
    for class_index, class_name in enumerate(CLASS_NAMES)
    for source in DACL10K_MAPPING[class_name]
}


def _draw_shapes(
    shapes: Iterable[Mapping[str, Any]], width: int, height: int
) -> tuple[np.ndarray, np.ndarray, set[str]]:
    """Rasterize selected classes and all non-selected labels.

    Pillow's polygon rasterizer is used to match the public DACL10K toolkit.
    """
    selected_images = [Image.new("1", (width, height), 0) for _ in CLASS_NAMES]
    selected_draws = [ImageDraw.Draw(image) for image in selected_images]
    other_image = Image.new("1", (width, height), 0)
    other_draw = ImageDraw.Draw(other_image)
    labels: set[str] = set()

    for shape in shapes:
        label = str(shape["label"])
        points = shape.get("points", ())
        if len(points) < 3:
            raise ValueError(f"Label {label!r} has fewer than three polygon points")
        polygon = [(float(point[0]), float(point[1])) for point in points]
        labels.add(label)
        class_index = SOURCE_TO_CLASS.get(label)
        if class_index is None:
            other_draw.polygon(polygon, outline=1, fill=1)
        else:
            selected_draws[class_index].polygon(polygon, outline=1, fill=1)

    targets = np.stack(
        [np.asarray(image, dtype=np.bool_) for image in selected_images], axis=0
    )
    other = np.asarray(other_image, dtype=np.bool_)
    return targets, other, labels


def rasterize_annotation(
    annotation: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, set[str]]:
    """Return ``targets[6,H,W]``, ``valid_mask[H,W]``, and source labels.

    Pixels positive for a selected class remain supervised even when another
    DACL10K label overlaps them. Pixels containing only excluded labels are
    ignored. Unannotated pixels are valid negative background.
    """
    width = int(annotation["imageWidth"])
    height = int(annotation["imageHeight"])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid annotation dimensions: {width}x{height}")
    targets, other, labels = _draw_shapes(
        annotation.get("shapes", ()), width=width, height=height
    )
    selected_positive = targets.any(axis=0)
    valid_mask = selected_positive | ~other
    return targets, valid_mask, labels
