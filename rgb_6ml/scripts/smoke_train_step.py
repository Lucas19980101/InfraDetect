#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from rgb_multilabel.config import load_config
from rgb_multilabel.labels import CLASS_NAMES
from rgb_multilabel.losses import MaskedBCEWithLogitsDiceLoss
from rgb_multilabel.models import build_model
from rgb_multilabel.train import make_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one real forward/backward training batch")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, _ = make_datasets(config)
    batch_size = int(config["train"]["batch_size"])
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    model = build_model(config["model"], pretrained=True).to(device).train()
    loss_config = config["train"]["loss"]
    criterion = MaskedBCEWithLogitsDiceLoss(
        classes=len(CLASS_NAMES),
        bce_weight=float(loss_config["bce_weight"]),
        dice_weight=float(loss_config["dice_weight"]),
        pos_weight=loss_config.get("pos_weight"),
    ).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    images = batch["image"].to(device)
    targets = batch["target"].to(device)
    valid = batch["valid_mask"].to(device)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        logits = model(images)
        loss = criterion(logits, targets, valid)
    loss.backward()
    result = {
        "model": config["model"]["architecture"],
        "device": str(device),
        "batch_size": batch_size,
        "input_shape": list(images.shape),
        "output_shape": list(logits.shape),
        "loss": float(loss.detach()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)
        if device.type == "cuda"
        else None,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
