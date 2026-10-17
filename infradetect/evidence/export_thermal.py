"""Export one calibrated TBBR-style image's native-resolution anomaly probability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..inference.thermal_context_architectures import build_model
from ..inference.thermal import predict_full
from ..inference.config import config_from_file


@torch.no_grad()
def run(image: Path, config: Path, checkpoint: Path, output: Path) -> dict:
    """从四通道起的场景栈导出完整 H×W 的热异常概率，不做整图缩放。

    先核对配置和 checkpoint 的任务、输入模态、通道及网络结构，避免
    错把 RGB+IR 权重当 IR-only 使用。predict_full 采用配置中的原像素
    滑窗策略；输出 thermal_probability.npy 与可追溯的 summary.json。
    此概率只是该模型的热异常证据，不等于空鼓或渗漏真值。
    """
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty to preserve prior runs: {output}")
    cfg = config_from_file(config)
    if cfg["task"] != "thermal" or cfg["data"]["class_names"] != ["Background", "Thermal Anomaly"]:
        raise ValueError("Expected a binary thermal-anomaly config")
    if not cfg["data"].get("native_patches", False):
        raise ValueError("This exporter requires native-resolution sliding inference")
    array = np.load(image, mmap_mode="r", allow_pickle=False)
    if array.ndim != 3 or array.shape[2] < 4:
        raise ValueError("Input must contain at least [B,G,R,Thermal] channels")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    saved_cfg = state["config"]
    for section, key in (("data", "class_names"), ("data", "input_mode"),
                         ("data", "thermal_channel"), ("model", "architecture"),
                         ("model", "encoder")):
        if saved_cfg[section].get(key) != cfg[section].get(key):
            raise ValueError(f"Checkpoint/config mismatch: {section}.{key}")
    model = build_model("thermal", cfg["model"]["classes"], pretrained=False,
                        batch1_safe_aspp=cfg["model"].get("batch1_safe_aspp", False),
                        architecture=cfg["model"].get("architecture"),
                        encoder=cfg["model"].get("encoder")).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    logits = predict_full(model, {"image": str(image)}, cfg, device)
    probability = logits.softmax(0)[1].numpy().astype(np.float32)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "thermal_probability.npy", probability, allow_pickle=False)
    summary = {
        "source_image": str(image.resolve()),
        "config": str(config.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": state.get("epoch"),
        "device": str(device),
        "shape": list(probability.shape),
        "class_names": cfg["data"]["class_names"],
        "input_mode": cfg["data"].get("input_mode", "thermal"),
        "thermal_channel": cfg["data"].get("thermal_channel", 3),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    """Parse command-line arguments and run this module's workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.image, args.config, args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()
