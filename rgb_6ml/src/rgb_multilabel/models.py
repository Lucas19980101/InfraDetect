from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DeepLabV3PlusSixLabel(nn.Module):
    def __init__(
        self,
        encoder: str = "resnet50",
        encoder_weights: str | None = "imagenet",
        classes: int = 6,
    ):
        super().__init__()
        import segmentation_models_pytorch as smp

        self.net = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=classes,
            activation=None,
        )

    def forward(self, images: torch.Tensor, return_features: bool = False):
        features = self.net.encoder(images)
        # segmentation-models-pytorch 0.5+ accepts the encoder pyramid as one
        # list, matching its public top-level forward implementation.
        decoder = self.net.decoder(features)
        logits = self.net.segmentation_head(decoder)
        if return_features:
            return {"logits": logits, "encoder": features, "decoder": decoder}
        return logits


class SegFormerB0SixLabel(nn.Module):
    def __init__(
        self,
        pretrained_name: str = "nvidia/mit-b0",
        classes: int = 6,
        pretrained: bool = True,
    ):
        super().__init__()
        try:
            from transformers import (
                SegformerConfig,
                SegformerForSemanticSegmentation,
                SegformerModel,
            )
        except ImportError as error:
            raise ImportError(
                "SegFormer requires the optional 'transformers' dependency"
            ) from error

        if pretrained:
            config = SegformerConfig.from_pretrained(pretrained_name)
        else:
            config = SegformerConfig(
                num_channels=3,
                num_encoder_blocks=4,
                depths=[2, 2, 2, 2],
                sr_ratios=[8, 4, 2, 1],
                hidden_sizes=[32, 64, 160, 256],
                patch_sizes=[7, 3, 3, 3],
                strides=[4, 2, 2, 2],
                num_attention_heads=[1, 2, 5, 8],
                mlp_ratios=[4, 4, 4, 4],
                decoder_hidden_size=256,
            )
        config.num_labels = classes
        config.id2label = {index: str(index) for index in range(classes)}
        config.label2id = {str(index): index for index in range(classes)}
        self.net = SegformerForSemanticSegmentation(config)
        if pretrained:
            # Load ImageNet MiT-B0 encoder weights only. The six-class decoder
            # and classifier stay randomly initialized for a fair task-specific comparison.
            self.net.segformer = SegformerModel.from_pretrained(
                pretrained_name, config=config
            )

    def forward(self, images: torch.Tensor, return_features: bool = False):
        input_size = images.shape[-2:]
        outputs = self.net.segformer(
            pixel_values=images,
            output_hidden_states=True,
            return_dict=True,
        )
        decoder_output = self.net.decode_head(outputs.hidden_states)
        logits = F.interpolate(
            decoder_output, size=input_size, mode="bilinear", align_corners=False
        )
        if return_features:
            return {
                "logits": logits,
                "encoder": outputs.hidden_states,
                "decoder": decoder_output,
            }
        return logits


def build_model(config: dict, pretrained: bool = True) -> nn.Module:
    architecture = str(config["architecture"]).lower()
    classes = int(config.get("num_classes", 6))
    if architecture == "deeplabv3plus":
        return DeepLabV3PlusSixLabel(
            encoder=config.get("encoder", "resnet50"),
            encoder_weights=config.get("encoder_weights", "imagenet")
            if pretrained
            else None,
            classes=classes,
        )
    if architecture == "segformer_b0":
        return SegFormerB0SixLabel(
            pretrained_name=config.get("pretrained_name", "nvidia/mit-b0"),
            classes=classes,
            pretrained=pretrained,
        )
    raise ValueError(f"Unknown model architecture: {architecture!r}")
