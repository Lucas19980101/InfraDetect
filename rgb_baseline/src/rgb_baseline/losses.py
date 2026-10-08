from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CrossEntropyDiceLoss(nn.Module):
    def __init__(self, classes: int, ignore_index: int = 255, ce_weight: float = 0.5,
                 dice_weight: float = 0.5, class_weights: list[float] | None = None):
        super().__init__()
        self.classes, self.ignore_index = classes, ignore_index
        self.ce_weight, self.dice_weight = ce_weight, dice_weight
        self.register_buffer("class_weights", torch.tensor(class_weights) if class_weights else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.class_weights, ignore_index=self.ignore_index)
        valid = target != self.ignore_index
        safe_target = target.masked_fill(~valid, 0)
        one_hot = F.one_hot(safe_target, self.classes).permute(0, 3, 1, 2).float()
        valid = valid.unsqueeze(1)
        probs = logits.softmax(1) * valid
        one_hot = one_hot * valid
        dims = (0, 2, 3)
        dice = (2 * (probs * one_hot).sum(dims) + 1e-6) / ((probs + one_hot).sum(dims) + 1e-6)
        return self.ce_weight * ce + self.dice_weight * (1 - dice.mean())

