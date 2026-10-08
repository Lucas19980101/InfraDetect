from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .building_data import BuildingPartialLabelDataset
from .config import load_config
from .labels import CLASS_NAMES
from .losses import MaskedBCEWithLogitsDiceLoss
from .metrics import PerClassThresholdTuner
from .models import build_model
from .train import seed_everything


def freeze_encoder(model, architecture: str):
    if architecture == "deeplabv3plus":
        encoder = model.net.encoder
    elif architecture == "segformer_b0":
        encoder = model.net.segformer
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder


@torch.no_grad()
def validate(model, loader, criterion, device) -> dict:
    model.eval()
    tuner = PerClassThresholdTuner(len(CLASS_NAMES), bins=1000)
    losses = []
    for batch in tqdm(loader, desc="building val", leave=False, unit="batch"):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True)
        logits = model(images)
        losses.append(float(criterion(logits, targets, valid)))
        tuner.update(logits, targets, valid)
    values = tuner.compute()
    return {
        "loss": float(np.mean(losses)),
        "mean_iou_crack_spalling": float(values["iou"][:2].mean()),
        **{
            key: {name: float(value) for name, value in zip(CLASS_NAMES[:2], tensor[:2])}
            for key, tensor in values.items()
            if key in {"thresholds", "iou", "dice", "precision", "recall"}
        },
    }


def run(config_path: str) -> None:
    config = load_config(config_path)
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fine = config["finetune"]
    base_path = Path(fine["base_checkpoint"])
    base = torch.load(base_path, map_location=device, weights_only=False)
    if base["config"]["model"]["architecture"] != config["model"]["architecture"]:
        raise ValueError("Base checkpoint architecture does not match fine-tune config")
    model = build_model(config["model"], pretrained=False).to(device)
    model.load_state_dict(base["model"])
    encoder = freeze_encoder(model, config["model"]["architecture"])
    root = fine["data_root"]
    size = int(config["data"]["crop_size"])
    train_set = BuildingPartialLabelDataset(root, "train", size, training=True)
    val_set = BuildingPartialLabelDataset(root, "val", size, training=False)
    batch_size = int(config["train"]["batch_size"])
    workers = int(config["data"]["num_workers"])
    options = dict(batch_size=batch_size, num_workers=workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **options)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **options)
    loss_config = config["train"]["loss"]
    criterion = MaskedBCEWithLogitsDiceLoss(
        len(CLASS_NAMES), float(loss_config["bce_weight"]), float(loss_config["dice_weight"])
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=float(fine["learning_rate"]), weight_decay=float(config["train"]["weight_decay"])
    )
    epochs = int(fine["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp = device.type == "cuda" and bool(config["train"]["mixed_precision"])
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    accumulation = int(config["train"]["gradient_accumulation_steps"])
    output = Path(fine["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    best = -1.0
    print(f"model={config['model']['architecture']}; train={len(train_set)}; val={len(val_set)}; "
          f"epochs={epochs}; frozen_encoder=True; lr={fine['learning_rate']}")
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        encoder.eval()  # keep frozen BatchNorm statistics unchanged
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        progress = tqdm(train_loader, desc=f"fine-tune {epoch}/{epochs}", unit="batch")
        for step, batch in enumerate(progress, 1):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            valid = batch["valid_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                raw_loss = criterion(model(images), targets, valid)
                loss = raw_loss / accumulation
            scaler.scale(loss).backward()
            if step % accumulation == 0 or step == len(train_loader):
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            running += float(raw_loss.detach())
            progress.set_postfix(loss=f"{float(raw_loss.detach()):.4f}", avg=f"{running/step:.4f}")
        scheduler.step()
        validation = validate(model, val_loader, criterion, device)
        score = validation["mean_iou_crack_spalling"]
        combined_thresholds = dict(base["thresholds"])
        combined_thresholds.update(validation["thresholds"])
        record = {
            "epoch": epoch,
            "train_loss": running / len(train_loader),
            "epoch_seconds": round(time.perf_counter() - started, 1),
            "validation": validation,
        }
        state = {
            "model": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "base_checkpoint": str(base_path.resolve()),
            "thresholds": combined_thresholds,
            "building_validation": validation,
        }
        torch.save(state, output / "last.pt")
        if score > best:
            best = score
            torch.save(state, output / "best.pt")
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
