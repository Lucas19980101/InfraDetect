from __future__ import annotations

import torch
import torch.nn.functional as F


class Metrics:
    def __init__(self, classes: int):
        self.classes = classes
        self.confusion = torch.zeros((classes, classes), dtype=torch.int64)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        prediction = logits.argmax(1).detach().cpu().reshape(-1)
        target = target.detach().cpu().reshape(-1)
        valid = target != 255
        indices = target[valid] * self.classes + prediction[valid]
        self.confusion += torch.bincount(indices, minlength=self.classes ** 2).reshape(self.classes, self.classes)

    def compute(self, names: list[str]) -> dict:
        cm = self.confusion.float()
        tp, actual, predicted = cm.diag(), cm.sum(1), cm.sum(0)
        union = actual + predicted - tp
        iou = tp / union.clamp_min(1)
        dice = 2 * tp / (actual + predicted).clamp_min(1)
        present = union > 0
        return {"mean_iou": float(iou[present].mean()) if present.any() else None,
                "iou": {name: float(iou[i]) if present[i] else None for i, name in enumerate(names)},
                "dice": {name: float(dice[i]) if present[i] else None for i, name in enumerate(names)},
                "confusion": self.confusion.tolist()}


def ce_dice(logits: torch.Tensor, target: torch.Tensor,
            class_weights: torch.Tensor | None = None) -> torch.Tensor:
    valid = target != 255
    if not valid.any():
        return logits.sum() * 0
    ce = F.cross_entropy(logits, target, weight=class_weights, ignore_index=255)
    safe = target.masked_fill(~valid, 0)
    onehot = F.one_hot(safe, logits.shape[1]).permute(0, 3, 1, 2).float()
    probs = logits.softmax(1) * valid[:, None]
    onehot *= valid[:, None]
    intersection = (probs * onehot).sum((0, 2, 3))
    denominator = (probs + onehot).sum((0, 2, 3))
    present = denominator > 0
    dice = 1 - ((2 * intersection + 1e-6) / (denominator + 1e-6))[present].mean()
    return 0.5 * ce + 0.5 * dice
