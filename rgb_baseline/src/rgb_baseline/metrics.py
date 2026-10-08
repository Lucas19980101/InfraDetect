from __future__ import annotations

import torch
import torch.nn.functional as F


class SegmentationMetrics:
    def __init__(self, classes: int, ignore_index: int = 255):
        self.classes, self.ignore_index = classes, ignore_index
        self.confusion = torch.zeros(classes, classes, dtype=torch.int64)

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction, target = prediction.cpu().reshape(-1), target.cpu().reshape(-1)
        valid = target != self.ignore_index
        indices = target[valid] * self.classes + prediction[valid]
        self.confusion += torch.bincount(indices, minlength=self.classes ** 2).reshape(self.classes, self.classes)

    def compute(self) -> dict[str, torch.Tensor | float]:
        cm = self.confusion.float()
        tp, actual, predicted = cm.diag(), cm.sum(1), cm.sum(0)
        iou = tp / (actual + predicted - tp).clamp_min(1)
        dice = 2 * tp / (actual + predicted).clamp_min(1)
        precision, recall = tp / predicted.clamp_min(1), tp / actual.clamp_min(1)
        return {"iou": iou, "mean_iou": iou.mean().item(), "dice": dice, "precision": precision, "recall": recall}


def relaxed_binary_f1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    class_id: int = 1,
    radius: int = 3,
    ignore_index: int = 255,
) -> float:
    """Return tolerance-aware F1, or NaN when prediction and truth are both empty.

    Pixels whose target equals ``ignore_index`` do not contribute. A one-sided
    empty mask scores 0; a two-sided empty mask is undefined so callers can
    exclude true-negative images when averaging localization quality.
    """
    if radius < 0:
        raise ValueError("radius must be non-negative")
    valid = target != ignore_index
    pred, true = (prediction == class_id) & valid, (target == class_id) & valid
    pred_count, true_count = pred.sum(), true.sum()
    if pred_count == 0 and true_count == 0:
        return float("nan")
    if pred_count == 0 or true_count == 0:
        return 0.0
    kernel = radius * 2 + 1
    dilated_true = F.max_pool2d(true.float()[None, None], kernel, 1, radius)[0, 0].bool()
    dilated_pred = F.max_pool2d(pred.float()[None, None], kernel, 1, radius)[0, 0].bool()
    precision = (pred & dilated_true).sum() / pred_count
    recall = (true & dilated_pred).sum() / true_count
    return float((2 * precision * recall / (precision + recall).clamp_min(1e-8)).item())
