from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .data import DACL10KSixLabelDataset
from .labels import CLASS_NAMES
from .losses import MaskedBCEWithLogitsDiceLoss
from .metrics import PerClassThresholdTuner
from .models import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def serializable_validation(result: dict[str, torch.Tensor | float]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in result.items():
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                output[key] = float(value)
            elif key == "thresholds":
                output[key] = {
                    name: float(item) for name, item in zip(CLASS_NAMES, value)
                }
            else:
                output[key] = {
                    name: float(item) for name, item in zip(CLASS_NAMES, value)
                }
        else:
            output[key] = value
    return output


@torch.no_grad()
def validate(model, loader, criterion, device, epoch: int) -> dict[str, Any]:
    model.eval()
    tuner = PerClassThresholdTuner(classes=len(CLASS_NAMES), bins=1000)
    losses: list[float] = []
    progress = tqdm(loader, desc=f"validate {epoch}", unit="batch", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets, valid)
        losses.append(float(loss))
        tuner.update(logits, targets, valid)
        progress.set_postfix(loss=f"{float(loss):.4f}")
    result = serializable_validation(tuner.compute())
    result["loss"] = float(np.mean(losses))
    return result


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    config: dict,
    epoch: int,
    best_mean_iou: float,
    validation: dict[str, Any],
) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config,
        "epoch": epoch,
        "best_mean_iou": best_mean_iou,
        "thresholds": validation["thresholds"],
        "validation": validation,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def make_datasets(config: dict) -> tuple[DACL10KSixLabelDataset, DACL10KSixLabelDataset]:
    data = config["data"]
    common = {
        "data_root": data["root"],
        "cache_root": data["mask_cache"],
        "manifest_path": data["split_manifest"],
        "crop_size": data["crop_size"],
    }
    return (
        DACL10KSixLabelDataset(split="train", training=True, **common),
        DACL10KSixLabelDataset(split="val", training=False, **common),
    )


def run(config_path: str, resume_path: str | None = None) -> None:
    config = load_config(config_path)
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_config, train_config = config["data"], config["train"]
    train_set, val_set = make_datasets(config)
    loader_options = {
        "batch_size": int(train_config["batch_size"]),
        "num_workers": int(data_config["num_workers"]),
        "pin_memory": device.type == "cuda",
        "persistent_workers": int(data_config["num_workers"]) > 0,
    }
    # DeepLabV3+'s ASPP pooling branch contains BatchNorm on a 1x1 feature.
    # A final batch of one sample is therefore invalid in training mode.
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_options)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_options)
    model = build_model(config["model"], pretrained=True).to(device)
    loss_config = train_config["loss"]
    criterion = MaskedBCEWithLogitsDiceLoss(
        classes=len(CLASS_NAMES),
        bce_weight=float(loss_config["bce_weight"]),
        dice_weight=float(loss_config["dice_weight"]),
        pos_weight=loss_config.get("pos_weight"),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    epochs = int(train_config["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp_enabled = bool(train_config["mixed_precision"] and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    accumulation = int(train_config["gradient_accumulation_steps"])
    if accumulation < 1:
        raise ValueError("gradient_accumulation_steps must be at least one")
    output = Path(train_config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    best, start_epoch = -1.0, 1
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint.get("best_mean_iou", -1.0))
    print(
        f"device={device}; model={config['model']['architecture']}; "
        f"train={len(train_set)}; val={len(val_set)}; batch={loader_options['batch_size']}; "
        f"accumulation={accumulation}; epochs={start_epoch}-{epochs}",
        flush=True,
    )
    for epoch in range(start_epoch, epochs + 1):
        started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"train {epoch}/{epochs}", unit="batch")
        for step, batch in enumerate(progress, 1):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            valid = batch["valid_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                raw_loss = criterion(model(images), targets, valid)
                loss = raw_loss / accumulation
            scaler.scale(loss).backward()
            if step % accumulation == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(raw_loss.detach())
            progress.set_postfix(
                loss=f"{float(raw_loss.detach()):.4f}",
                avg=f"{running_loss / step:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
        scheduler.step()
        validation = validate(model, val_loader, criterion, device, epoch)
        mean_iou = float(validation["mean_iou"])
        improved = mean_iou > best
        if improved:
            best = mean_iou
        record = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_loader),
            "epoch_seconds": round(time.perf_counter() - started, 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "best_mean_iou": best,
            "validation": validation,
        }
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        save_checkpoint(
            output / "last.pt", model, optimizer, scheduler, scaler,
            config, epoch, best, validation,
        )
        if improved:
            save_checkpoint(
                output / "best.pt", model, optimizer, scheduler, scaler,
                config, epoch, best, validation,
            )
        print(json.dumps(record), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    run(args.config, args.resume)


if __name__ == "__main__":
    main()
