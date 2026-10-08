from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DeepLabIR(nn.Module):
    def __init__(self, classes: int, pretrained: bool):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.net = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights="imagenet" if pretrained else None,
                                     in_channels=3, classes=classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class SegFormerComponents(nn.Module):
    def __init__(self, classes: int, pretrained: bool):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerModel
        if pretrained:
            config = SegformerConfig.from_pretrained("nvidia/mit-b0")
        else:
            config = SegformerConfig(num_channels=3, num_encoder_blocks=4, depths=[2, 2, 2, 2],
                                     sr_ratios=[8, 4, 2, 1], hidden_sizes=[32, 64, 160, 256],
                                     patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2],
                                     num_attention_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
                                     decoder_hidden_size=256)
        config.num_labels = classes
        self.net = SegformerForSemanticSegmentation(config)
        if pretrained:
            self.net.segformer = SegformerModel.from_pretrained("nvidia/mit-b0", config=config)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = self.net(pixel_values=image).logits
        return F.interpolate(output, size=image.shape[-2:], mode="bilinear", align_corners=False)


def build_model(task: str, classes: int, pretrained: bool) -> nn.Module:
    if task == "ir":
        return DeepLabIR(classes, pretrained)
    if task == "components":
        return SegFormerComponents(classes, pretrained)
    raise ValueError(task)
