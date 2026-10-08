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
from rgb_multilabel.train import make_datasets, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit one fixed real batch")
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    if args.steps < 2:
        raise ValueError("--steps must be at least two")
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, _ = make_datasets(config)
    batch_size = int(config["train"]["batch_size"])
    # Materialize one augmented batch once, then repeat exactly those tensors.
    batch = next(iter(DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0)))
    images = batch["image"].to(device)
    targets = batch["target"].to(device)
    valid = batch["valid_mask"].to(device)
    model = build_model(config["model"], pretrained=True).to(device).train()
    loss_config = config["train"]["loss"]
    criterion = MaskedBCEWithLogitsDiceLoss(
        classes=len(CLASS_NAMES),
        bce_weight=float(loss_config["bce_weight"]),
        dice_weight=float(loss_config["dice_weight"]),
        pos_weight=loss_config.get("pos_weight"),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=float(config["train"]["weight_decay"]),
    )
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    losses: list[float] = []
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            loss = criterion(model(images), targets, valid)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
        if step == 1 or step % 5 == 0 or step == args.steps:
            print(f"step={step:03d} loss={losses[-1]:.6f}", flush=True)
    print(json.dumps({
        "model": config["model"]["architecture"],
        "steps": args.steps,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "relative_change": losses[-1] / losses[0] - 1,
        "passed": losses[-1] < losses[0] * 0.9,
    }, indent=2))
    if losses[-1] >= losses[0] * 0.9:
        raise SystemExit("Loss did not decrease by at least 10%")


if __name__ == "__main__":
    main()
