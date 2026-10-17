"""与交付权重严格对应的热异常和 RGB 构件/空调网络定义。

本文件只描述网络与权重加载所需的结构，不负责图像裁块、预处理或阈值。
普通推理应以 pretrained=False 建模，再从本地 checkpoint 严格加载；
pretrained=True 分支可能访问外部模型仓库，不是交付包离线推理流程。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DeepLabIR(nn.Module):
    """三通道热图输入的 DeepLabV3+ 分割分支。

    batch1_safe_aspp 将 ASPP 全局池化分支的 BatchNorm 换成 GroupNorm，
    是为了兼容批量为 1 的训练/推理设置；必须与对应 checkpoint 一致。
    """
    def __init__(self, classes: int, pretrained: bool, batch1_safe_aspp: bool = False):
        """Create the model with the architecture and settings required by its checkpoint."""
        super().__init__()
        import segmentation_models_pytorch as smp
        self.net = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights="imagenet" if pretrained else None,
                                     in_channels=3, classes=classes)
        if batch1_safe_aspp:
            # The global-pooling ASPP branch is [N, 256, 1, 1]. BatchNorm cannot
            # estimate statistics for N=1, while GroupNorm is batch-independent.
            pooled = self.net.decoder.aspp[0].convs[4]
            channels = pooled[2].num_features
            pooled[2] = nn.GroupNorm(32, channels)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Compute segmentation logits from an input tensor."""
        return self.net(image)


class SegFormerComponents(nn.Module):
    """SegFormer wrapper for RGB component and equipment evidence."""
    def __init__(self, classes: int, pretrained: bool):
        """Create the model with the architecture and settings required by its checkpoint."""
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
        """Compute segmentation logits from an input tensor."""
        output = self.net(pixel_values=image).logits
        return F.interpolate(output, size=image.shape[-2:], mode="bilinear", align_corners=False)


class SegFormerIR(nn.Module):
    """SegFormer wrapper for IR anomaly segmentation checkpoints."""
    def __init__(self, classes: int, pretrained: bool, variant: str = "b2"):
        """Create the model with the architecture and settings required by its checkpoint."""
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        model_id = f"nvidia/segformer-{variant}-finetuned-ade-512-512"
        if pretrained:
            self.net = SegformerForSemanticSegmentation.from_pretrained(
                model_id, num_labels=classes, ignore_mismatched_sizes=True)
        else:
            depths = {"b0": [2, 2, 2, 2], "b2": [3, 4, 6, 3]}[variant]
            hidden = {"b0": [32, 64, 160, 256], "b2": [64, 128, 320, 512]}[variant]
            config = SegformerConfig(num_channels=3, num_labels=classes,
                depths=depths, hidden_sizes=hidden, sr_ratios=[8, 4, 2, 1],
                patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2],
                num_attention_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
                decoder_hidden_size={"b0": 256, "b2": 768}[variant])
            self.net = SegformerForSemanticSegmentation(config)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Compute segmentation logits from an input tensor."""
        logits = self.net(pixel_values=image).logits
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)


class SegFormerRGBThermal(nn.Module):
    """把已配准的 RGB 与热图特征在四个 SegFormer 尺度上融合。

    输入是六通道张量：前三通道 RGB，后三通道是热分支预处理结果。
    RGB 门控参数控制各尺度特征贡献；输出仍是像素级热异常 logits。
    该模型的 RGB 辅助信息不能被当成独立的 RGB 缺陷判定。
    """

    def __init__(self, classes: int, variant: str = "b2"):
        """Create the model with the architecture and settings required by its checkpoint."""
        super().__init__()
        from transformers import SegformerModel, SegformerDecodeHead
        config = SegFormerIR(classes, False, variant).net.config
        self.rgb_encoder = SegformerModel(config)
        self.thermal_encoder = SegformerModel(config)
        self.decode_head = SegformerDecodeHead(config)
        self.rgb_gates = nn.Parameter(torch.zeros(4))
        self.rgb_frozen = False

    def initialize_branches(self, thermal_checkpoint: str, rgb_model_id: str) -> None:
        """Initialize network branches from their compatible checkpoints."""
        from transformers import SegformerForSemanticSegmentation
        source = torch.load(thermal_checkpoint, map_location="cpu", weights_only=True)["model"]
        thermal = {key.removeprefix("net.segformer."): value for key, value in source.items()
                   if key.startswith("net.segformer.")}
        head = {key.removeprefix("net.decode_head."): value for key, value in source.items()
                if key.startswith("net.decode_head.")}
        self.thermal_encoder.load_state_dict(thermal, strict=True)
        self.decode_head.load_state_dict(head, strict=True)
        rgb = SegformerForSemanticSegmentation.from_pretrained(rgb_model_id)
        self.rgb_encoder.load_state_dict(rgb.segformer.state_dict(), strict=True)
        del rgb

    def freeze_rgb_encoder(self) -> None:
        """Toggle gradient updates for the RGB encoder."""
        self.rgb_frozen = True
        self.rgb_encoder.requires_grad_(False)
        self.rgb_encoder.eval()

    def train(self, mode: bool = True):
        """Set model training mode while respecting frozen branch settings."""
        super().train(mode)
        if self.rgb_frozen:
            self.rgb_encoder.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """分别编码 RGB 与热输入，按门控相加，再还原到输入空间尺寸。

        输入通道数必须是 6。返回的是 logits，不是概率；调用者需按
        模型类别维执行 softmax。内部特征下采样与原图裁块推理是两回事。
        """
        if image.shape[1] != 6:
            raise ValueError(f"RGB+Thermal model expects 6 channels, got {image.shape[1]}")
        if self.rgb_frozen:
            with torch.no_grad():
                rgb_features = self.rgb_encoder(image[:, :3], output_hidden_states=True).hidden_states
        else:
            rgb_features = self.rgb_encoder(image[:, :3], output_hidden_states=True).hidden_states
        thermal_features = self.thermal_encoder(image[:, 3:], output_hidden_states=True).hidden_states
        fused = tuple(thermal + torch.tanh(gate) * rgb
                      for thermal, rgb, gate in zip(thermal_features, rgb_features, self.rgb_gates))
        logits = self.decode_head(fused)
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)


def build_model(task: str, classes: int, pretrained: bool,
                batch1_safe_aspp: bool = False, architecture: str | None = None,
                encoder: str | None = None) -> nn.Module:
    """按任务与配置构建网络，不在此处加载交付 checkpoint。

    thermal/ir 可选择 DeepLab、IR SegFormer 或 RGB+Thermal SegFormer；
    components 是构件 SegFormer。architecture、encoder、classes 必须
    与保存权重时一致，否则严格加载会报错。
    """
    if task in {"ir", "thermal"}:
        if architecture == "segformer_rgb_thermal":
            return SegFormerRGBThermal(classes, encoder or "b2")
        if architecture == "segformer":
            return SegFormerIR(classes, pretrained, encoder or "b2")
        return DeepLabIR(classes, pretrained, batch1_safe_aspp)
    if task == "components":
        return SegFormerComponents(classes, pretrained)
    raise ValueError(task)
