from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _expanded_valid(valid_mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if valid_mask.ndim == 3:
        valid_mask = valid_mask.unsqueeze(1)
    if valid_mask.ndim != 4 or valid_mask.shape[1] not in (1, target.shape[1]):
        raise ValueError(
            f"Expected valid_mask [B,1,H,W] or [B,C,H,W], got {tuple(valid_mask.shape)}"
        )
    if target.ndim != 4 or valid_mask.shape[0] != target.shape[0] or valid_mask.shape[2:] != target.shape[2:]:
        raise ValueError(
            f"valid_mask {tuple(valid_mask.shape)} is incompatible with target {tuple(target.shape)}"
        )
    return valid_mask.to(dtype=target.dtype).expand_as(target)


class MaskedBCEWithLogitsDiceLoss(nn.Module):
    """BCE-with-logits plus soft Dice over valid pixels only."""

    def __init__(
        self,
        classes: int,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: list[float] | None = None,
        smooth: float = 1.0,
    ):
        super().__init__()
        if classes < 1:
            raise ValueError("classes must be positive")
        if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight <= 0:
            raise ValueError("Loss weights must be non-negative and not both zero")
        self.classes = classes
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        weights = None if pos_weight is None else torch.tensor(pos_weight, dtype=torch.float32).view(1, classes, 1, 1)
        if weights is not None and weights.numel() != classes:
            raise ValueError(f"Expected {classes} pos_weight values")
        self.register_buffer("pos_weight", weights)

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != self.classes:
            raise ValueError(
                f"Expected logits and target [B,{self.classes},H,W], got "
                f"{tuple(logits.shape)} and {tuple(target.shape)}"
            )
        target = target.to(dtype=logits.dtype)
        valid = _expanded_valid(valid_mask, target)
        denominator = valid.sum()
        if denominator.item() == 0:
            raise ValueError("A batch contains no valid supervised pixels")
        bce = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none", pos_weight=self.pos_weight
        )
        bce = (bce * valid).sum() / denominator

        probabilities = logits.sigmoid() * valid
        masked_target = target * valid
        dims = (0, 2, 3)
        intersection = (probabilities * masked_target).sum(dims)
        cardinality = probabilities.sum(dims) + masked_target.sum(dims)
        dice = (2 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1 - dice.mean()
        return self.bce_weight * bce + self.dice_weight * dice_loss
