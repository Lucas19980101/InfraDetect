from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class MultilabelSegmentationMetrics:
    def __init__(self, classes: int):
        self.classes = classes
        self.tp = torch.zeros(classes, dtype=torch.int64)
        self.fp = torch.zeros(classes, dtype=torch.int64)
        self.fn = torch.zeros(classes, dtype=torch.int64)
        self.tn = torch.zeros(classes, dtype=torch.int64)

    def update(
        self, prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor
    ) -> None:
        if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != self.classes:
            raise ValueError("prediction and target must have shape [B,C,H,W]")
        if valid_mask.ndim == 3:
            valid_mask = valid_mask.unsqueeze(1)
        valid = valid_mask.bool().expand_as(target)
        prediction, target = prediction.bool(), target.bool()
        dims = (0, 2, 3)
        self.tp += (prediction & target & valid).sum(dims).cpu()
        self.fp += (prediction & ~target & valid).sum(dims).cpu()
        self.fn += (~prediction & target & valid).sum(dims).cpu()
        self.tn += (~prediction & ~target & valid).sum(dims).cpu()

    def compute(self) -> dict[str, torch.Tensor | float]:
        tp, fp, fn = self.tp.double(), self.fp.double(), self.fn.double()
        iou = tp / (tp + fp + fn).clamp_min(1)
        dice = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
        precision = tp / (tp + fp).clamp_min(1)
        recall = tp / (tp + fn).clamp_min(1)
        return {
            "iou": iou,
            "mean_iou": float(iou.mean()),
            "dice": dice,
            "precision": precision,
            "recall": recall,
        }


class PerClassThresholdTuner:
    """Dataset-level per-class Dice threshold search using probability histograms."""

    def __init__(self, classes: int, bins: int = 1000):
        if bins < 2:
            raise ValueError("bins must be at least two")
        self.classes, self.bins = classes, bins
        self.positive = torch.zeros(classes, bins + 1, dtype=torch.int64)
        self.negative = torch.zeros(classes, bins + 1, dtype=torch.int64)

    def update(self, logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> None:
        if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != self.classes:
            raise ValueError("logits and target must have shape [B,C,H,W]")
        if valid_mask.ndim == 3:
            valid_mask = valid_mask.unsqueeze(1)
        valid = valid_mask.bool().expand_as(target)
        probabilities = logits.detach().sigmoid()
        bin_indices = (probabilities * self.bins).floor().clamp(0, self.bins).long()
        for class_index in range(self.classes):
            class_valid = valid[:, class_index]
            indices = bin_indices[:, class_index][class_valid].cpu()
            truth = target[:, class_index][class_valid].bool().cpu()
            self.positive[class_index] += torch.bincount(
                indices[truth], minlength=self.bins + 1
            )
            self.negative[class_index] += torch.bincount(
                indices[~truth], minlength=self.bins + 1
            )

    def compute(self) -> dict[str, torch.Tensor]:
        thresholds = torch.arange(self.bins + 1, dtype=torch.float64) / self.bins
        selected_thresholds = torch.empty(self.classes, dtype=torch.float64)
        selected_dice = torch.empty(self.classes, dtype=torch.float64)
        selected_iou = torch.empty(self.classes, dtype=torch.float64)
        selected_precision = torch.empty(self.classes, dtype=torch.float64)
        selected_recall = torch.empty(self.classes, dtype=torch.float64)
        for class_index in range(self.classes):
            tp = self.positive[class_index].flip(0).cumsum(0).flip(0).double()
            fp = self.negative[class_index].flip(0).cumsum(0).flip(0).double()
            total_positive = self.positive[class_index].sum().double()
            fn = total_positive - tp
            dice = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
            best = torch.nonzero(dice == dice.max(), as_tuple=False).flatten()
            best_index = best[(thresholds[best] - 0.5).abs().argmin()]
            selected_thresholds[class_index] = thresholds[best_index]
            selected_dice[class_index] = dice[best_index]
            selected_iou[class_index] = tp[best_index] / (
                tp[best_index] + fp[best_index] + fn[best_index]
            ).clamp_min(1)
            selected_precision[class_index] = tp[best_index] / (
                tp[best_index] + fp[best_index]
            ).clamp_min(1)
            selected_recall[class_index] = tp[best_index] / (
                tp[best_index] + fn[best_index]
            ).clamp_min(1)
        return {
            "thresholds": selected_thresholds,
            "iou": selected_iou,
            "mean_iou": selected_iou.mean(),
            "dice": selected_dice,
            "precision": selected_precision,
            "recall": selected_recall,
        }


def relaxed_binary_f1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    radius: int = 3,
) -> float:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    valid = valid_mask.bool().squeeze()
    prediction = prediction.bool().squeeze() & valid
    target = target.bool().squeeze() & valid
    pred_count, target_count = prediction.sum(), target.sum()
    if pred_count == 0 and target_count == 0:
        return float("nan")
    if pred_count == 0 or target_count == 0:
        return 0.0
    kernel = radius * 2 + 1
    dilated_target = F.max_pool2d(target.float()[None, None], kernel, 1, radius)[0, 0].bool()
    dilated_prediction = F.max_pool2d(prediction.float()[None, None], kernel, 1, radius)[0, 0].bool()
    precision = (prediction & dilated_target).sum() / pred_count
    recall = (target & dilated_prediction).sum() / target_count
    return float((2 * precision * recall / (precision + recall).clamp_min(1e-8)).item())
