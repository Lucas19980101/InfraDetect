"""Native-pixel tiled RGB inference with explicit evaluation coverage.

Patches are cropped without scaling. Small images are padded, never enlarged.
The returned H×W probabilities use mean blending in overlaps; pixels outside
an optional ROI are unevaluated (coverage=False), not negative examples.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np


def starts(length: int, patch_size: int, stride: int) -> list[int]:
    """计算一维滑窗起点，并强制最后一块贴住图像右/下边界。

    若图像短于 patch，只返回 0；实际缺少的像素在 predict_tiled 中 pad。
    stride 不能大于 patch，否则中间可能留下未评估空隙。
    """
    if length <= 0 or patch_size <= 0 or not 0 < stride <= patch_size:
        raise ValueError("Expected positive length/patch and 0 < stride <= patch_size")
    if length <= patch_size:
        return [0]
    result = list(range(0, length - patch_size + 1, stride))
    if result[-1] != length - patch_size:
        result.append(length - patch_size)
    return result


def predict_tiled(
    image: np.ndarray,
    predict_patch: Callable[[np.ndarray], np.ndarray],
    *,
    image_id: str,
    patch_size: int | tuple[int, int] = 512,
    stride: int | tuple[int, int] = 384,
    roi: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """以原图像素裁块推理，返回完整尺寸概率、覆盖图和块坐标记录。

    predict_patch 接收固定尺寸 Hpatch×Wpatch×3 图并返回 C×Hpatch×Wpatch 概率。
    小图仅在右/下补零，预测后补齐区域不写回原图；重叠位置做算术平均。
    ROI 仅决定哪些原图像素被评估，不缩放源图，ROI 外 coverage=False，
    下游不得把这些像素当成模型判断的阴性。每块记录 image_id、x0、y0、
    w、h，供原图坐标及后续 RGB→3D 投影追溯。
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an H×W×3 image")
    height, width = image.shape[:2]
    patch_h, patch_w = ((patch_size, patch_size) if isinstance(patch_size, int) else patch_size)
    stride_h, stride_w = ((stride, stride) if isinstance(stride, int) else stride)
    ys, xs = starts(height, patch_h, stride_h), starts(width, patch_w, stride_w)
    if roi is not None:
        if roi.shape != (height, width):
            raise ValueError("ROI must match the original image shape")
        roi = roi.astype(bool, copy=False)
    totals: np.ndarray | None = None
    counts = np.zeros((height, width), dtype=np.uint16)
    records: list[dict] = []
    for y0 in ys:
        for x0 in xs:
            h, w = min(patch_h, height - y0), min(patch_w, width - x0)
            selected = None if roi is None else roi[y0:y0 + h, x0:x0 + w]
            record = {"image_id": image_id, "x0": x0, "y0": y0,
                      "w": w, "h": h, "evaluated": selected is None or bool(selected.any())}
            records.append(record)
            if not record["evaluated"]:
                continue
            crop = image[y0:y0 + h, x0:x0 + w]
            if h != patch_h or w != patch_w:
                crop = cv2.copyMakeBorder(crop, 0, patch_h - h, 0, patch_w - w,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))
            values = np.asarray(predict_patch(crop), dtype=np.float32)
            if values.ndim != 3 or values.shape[1:] != (patch_h, patch_w):
                raise ValueError("Patch predictor must return C×patch_h×patch_w")
            if not np.isfinite(values).all():
                raise ValueError("Patch predictor returned non-finite probabilities")
            if totals is None:
                totals = np.zeros((values.shape[0], height, width), dtype=np.float32)
            elif totals.shape[0] != values.shape[0]:
                raise ValueError("Patch predictor changed class count")
            block = totals[:, y0:y0 + h, x0:x0 + w]
            if selected is None:
                block += values[:, :h, :w]
                counts[y0:y0 + h, x0:x0 + w] += 1
            else:
                block += values[:, :h, :w] * selected[None]
                counts[y0:y0 + h, x0:x0 + w] += selected
    if totals is None:
        raise ValueError("ROI contains no evaluated pixels")
    coverage = counts > 0
    expected = np.ones((height, width), bool) if roi is None else roi
    if not np.array_equal(coverage, expected):
        raise RuntimeError("Tiling left requested pixels uncovered")
    np.divide(totals, np.maximum(counts, 1)[None], out=totals)
    return totals, coverage, records
