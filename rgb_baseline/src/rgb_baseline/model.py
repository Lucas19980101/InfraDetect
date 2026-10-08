from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
from torch import nn


class RGBDeepLabV3Plus(nn.Module):
    """SMP wrapper exposing fusion-ready encoder and decoder tensors."""

    def __init__(self, encoder: str = "resnet50", encoder_weights: str | None = "imagenet", classes: int = 3):
        super().__init__()
        self.net = smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=3, classes=classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.net.encoder(x)
        decoder = self.net.decoder(*features)
        logits = self.net.segmentation_head(decoder)
        if return_features:
            return {"logits": logits, "encoder": features, "decoder": decoder}
        return logits


def build_model(config: dict, pretrained: bool = True) -> RGBDeepLabV3Plus:
    return RGBDeepLabV3Plus(
        encoder=config["encoder"],
        encoder_weights=config.get("encoder_weights") if pretrained else None,
        classes=config["num_classes"],
    )
