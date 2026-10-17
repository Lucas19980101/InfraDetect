"""只定义六类 RGB 缺陷网络结构，不负责加载权重或运行推理。

两个架构共用六个输出通道，但输出是每类 logits，而非互斥类别 ID。
阈值与原图裁块合并在推理模块处理；此处不读取数据集或运行训练。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DeepLabV3PlusSixLabel(nn.Module):
    """DeepLabV3+ network with six RGB defect output channels."""
    def __init__(
        self,
        encoder: str = "resnet50",
        encoder_weights: str | None = "imagenet",
        classes: int = 6,
    ):
        """Create the model with the architecture and settings required by its checkpoint."""
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
        """Compute segmentation logits from an input tensor."""
        features = self.net.encoder(images)
        # segmentation-models-pytorch 0.5+ accepts the encoder pyramid as one
        # list, matching its public top-level forward implementation.
        decoder = self.net.decoder(features)
        logits = self.net.segmentation_head(decoder)
        if return_features:
            return {"logits": logits, "encoder": features, "decoder": decoder}
        return logits


class SegFormerB0SixLabel(nn.Module):
    """SegFormer-B0 network with six RGB defect output channels."""
    def __init__(
        self,
        pretrained_name: str = "nvidia/mit-b0",
        classes: int = 6,
        pretrained: bool = True,
    ):
        """Create the model with the architecture and settings required by its checkpoint."""
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
        """Compute segmentation logits from an input tensor."""
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
    """根据 checkpoint 配置选择 DeepLabV3+ 或 SegFormer-B0。

    config 中的 architecture/num_classes 等必须匹配权重。交付包从
    本地 checkpoint 推理时应传 pretrained=False，避免额外下载上游权重；
    该函数只创建网络，不计算预测。
    """
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
