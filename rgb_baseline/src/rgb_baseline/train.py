from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .data import SemanticDataset
from .losses import CrossEntropyDiceLoss
from .metrics import SegmentationMetrics
from .model import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validate(model, loader, criterion, device, classes, ignore_index, epoch):
    model.eval(); losses, metrics = [], SegmentationMetrics(classes, ignore_index)
    progress = tqdm(loader, desc=f"validate {epoch}", leave=False, unit="batch")
    for batch in progress:
        images, masks = batch["image"].to(device), batch["mask"].to(device)
        logits = model(images)
        loss = criterion(logits, masks).item(); losses.append(loss)
        metrics.update(logits.argmax(1), masks)
        progress.set_postfix(loss=f"{loss:.4f}")
    result = metrics.compute(); result["loss"] = float(np.mean(losses))
    return result


def serializable(metrics: dict) -> dict:
    return {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in metrics.items()}


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, cfg, epoch, best, metrics) -> None:
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "scaler": scaler.state_dict(), "config": cfg, "epoch": epoch, "best_mean_iou": best, "metrics": metrics}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def run(config_path: str, resume_path: str | None = None) -> None:
    cfg = load_config(config_path); seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, train_cfg, model_cfg = cfg["data"], cfg["train"], cfg["model"]
    train_set = SemanticDataset(data["root"], "train", data["tile_size"], True)
    val_set = SemanticDataset(data["root"], "val", data["tile_size"], False)
    if not train_set:
        raise ValueError(f"Training split is empty: {Path(data['root']) / 'images/train'}")
    if not val_set:
        raise ValueError(f"Validation split is empty: {Path(data['root']) / 'images/val'}")
    loader_args = dict(batch_size=train_cfg["batch_size"], num_workers=data["num_workers"], pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    model = build_model(model_cfg).to(device)
    criterion = CrossEntropyDiceLoss(model_cfg["num_classes"], data["ignore_index"], train_cfg["ce_weight"],
                                     train_cfg["dice_weight"], train_cfg.get("class_weights")).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_cfg["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg["mixed_precision"] and device.type == "cuda")
    output = Path(train_cfg["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    best, start_epoch = -1.0, 1
    if resume_path:
        resume = Path(resume_path)
        if not resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume}")
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        required = {"model", "optimizer", "scheduler", "epoch"}
        if missing := required - checkpoint.keys():
            raise ValueError(f"Checkpoint cannot resume training; missing keys: {sorted(missing)}")
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"): scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint.get("best_mean_iou", -1.0))
        print(f"Resuming {resume} at epoch {start_epoch}; best mIoU={best:.4f}")
    print(f"device={device}; train={len(train_set)}; val={len(val_set)}; batch_size={train_cfg['batch_size']}; "
          f"epochs={start_epoch}-{train_cfg['epochs']}")
    if device.type == "cpu":
        print("WARNING: CUDA is unavailable. 1024px DeepLabV3+ training on CPU can take several days.")
    last_record: dict = {}
    try:
      for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        epoch_started = time.perf_counter()
        model.train(); running = 0.0
        progress = tqdm(train_loader, desc=f"train {epoch}/{train_cfg['epochs']}", unit="batch", dynamic_ncols=True)
        for step, batch in enumerate(progress, 1):
            images, masks = batch["image"].to(device), batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=scaler.is_enabled()):
                loss = criterion(model(images), masks)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}", avg=f"{running / step:.4f}",
                                 lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        scheduler.step()
        metrics = validate(model, val_loader, criterion, device, model_cfg["num_classes"], data["ignore_index"], epoch)
        elapsed = time.perf_counter() - epoch_started
        remaining_hours = elapsed * (train_cfg["epochs"] - epoch) / 3600
        record = {"epoch": epoch, "train_loss": running / len(train_loader), "epoch_seconds": round(elapsed, 1),
                  "estimated_remaining_hours": round(remaining_hours, 2), **serializable(metrics)}
        last_record = record
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle: handle.write(json.dumps(record) + "\n")
        improved = metrics["mean_iou"] > best
        if improved: best = metrics["mean_iou"]
        save_checkpoint(output / "last.pt", model, optimizer, scheduler, scaler, cfg, epoch, best, record)
        if improved: save_checkpoint(output / "best.pt", model, optimizer, scheduler, scaler, cfg, epoch, best, record)
        checkpoint_every = int(train_cfg.get("checkpoint_every", 5))
        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            periodic = output / f"epoch_{epoch:03d}.pt"
            save_checkpoint(periodic, model, optimizer, scheduler, scaler, cfg, epoch, best, record)
            snapshots = sorted(output.glob("epoch_*.pt"))
            keep = int(train_cfg.get("keep_periodic_checkpoints", 3))
            for old in snapshots[:-keep] if keep > 0 else snapshots:
                old.unlink()
        print(json.dumps(record), flush=True)
    except KeyboardInterrupt:
        save_checkpoint(output / "interrupted.pt", model, optimizer, scheduler, scaler, cfg,
                        max(start_epoch - 1, epoch - 1), best, last_record)
        print(f"\nTraining interrupted. Recovery checkpoint saved to {output / 'interrupted.pt'}")
        raise
    except torch.OutOfMemoryError as error:
        raise RuntimeError("CUDA ran out of memory. Reduce train.batch_size (try 2 or 1), then resume from last.pt.") from error


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--resume", help="Resume optimizer, scheduler, and model state from a training checkpoint")
    args = parser.parse_args(); run(args.config, args.resume)


if __name__ == "__main__": main()
