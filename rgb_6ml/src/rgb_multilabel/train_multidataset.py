# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from .config import load_config
from .data import DACL10KSixLabelDataset
from .labels import CLASS_NAMES
from .losses import MaskedBCEWithLogitsDiceLoss
from .manifest_data import PartialLabelManifestDataset
from .models import build_model
from .train import save_checkpoint, seed_everything, validate


class ExpandedValidMaskDataset(Dataset):
    """Normalize datasets to class-specific validity masks for mixed batches."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.dataset[index]
        valid = sample["valid_mask"]
        if valid.shape[0] == 1:
            sample["valid_mask"] = valid.expand(len(CLASS_NAMES), -1, -1)
        return sample


def dacl_dataset(config: dict, split: str, training: bool) -> Dataset:
    data = config["data"]
    return DACL10KSixLabelDataset(
        data_root=data["root"],
        cache_root=data["mask_cache"],
        manifest_path=data["split_manifest"],
        split=split,
        crop_size=int(data["crop_size"]),
        training=training,
    )


def make_training_sources(config: dict) -> tuple[ConcatDataset, list[float], list[dict]]:
    datasets = []
    weights = []
    report = []
    for source in config["multidataset"]["sources"]:
        kind = source["type"]
        if kind == "dacl":
            dataset = dacl_dataset(config, source.get("split", "train"), training=True)
        elif kind == "manifest":
            dataset = PartialLabelManifestDataset(
                manifest=source["manifest"],
                splits=source.get("splits", "train"),
                image_size=int(config["data"]["crop_size"]),
                training=True,
                source=source["name"],
            )
        else:
            raise ValueError(f"Unsupported dataset type: {kind!r}")
        dataset = ExpandedValidMaskDataset(dataset)
        source_weight = float(source["weight"])
        if source_weight <= 0:
            raise ValueError("Dataset sampling weights must be positive")
        datasets.append(dataset)
        weights.extend([source_weight / len(dataset)] * len(dataset))
        report.append({"name": source["name"], "samples": len(dataset), "weight": source_weight})
    if len(datasets) < 2:
        raise ValueError("Multidataset training requires at least two sources")
    return ConcatDataset(datasets), weights, report


def run(config_path: str, resume_path: str | None = None, max_steps: int | None = None) -> None:
    config = load_config(config_path)
    seed = int(config["seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, sample_weights, sources = make_training_sources(config)
    val_set = dacl_dataset(config, "val", training=False)
    train_config = config["train"]
    samples_per_epoch = int(config["multidataset"].get("samples_per_epoch", len(train_set)))
    if samples_per_epoch < int(train_config["batch_size"]):
        raise ValueError("samples_per_epoch must be at least one batch")
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        sample_weights, samples_per_epoch, replacement=True, generator=generator
    )
    options = {
        "batch_size": int(train_config["batch_size"]),
        "num_workers": int(config["data"]["num_workers"]),
        "pin_memory": device.type == "cuda",
        "persistent_workers": int(config["data"]["num_workers"]) > 0,
    }
    train_loader = DataLoader(train_set, sampler=sampler, drop_last=True, **options)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **options)
    model = build_model(config["model"], pretrained=False).to(device)
    base_path = Path(config["multidataset"]["base_checkpoint"])
    base = torch.load(base_path, map_location=device, weights_only=False)
    if base["config"]["model"]["architecture"] != config["model"]["architecture"]:
        raise ValueError("Base checkpoint architecture does not match config")
    model.load_state_dict(base["model"])
    loss_config = train_config["loss"]
    criterion = MaskedBCEWithLogitsDiceLoss(
        len(CLASS_NAMES),
        float(loss_config["bce_weight"]),
        float(loss_config["dice_weight"]),
        loss_config.get("pos_weight"),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    epochs = int(train_config["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp = device.type == "cuda" and bool(train_config["mixed_precision"])
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    accumulation = int(train_config["gradient_accumulation_steps"])
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
    print(json.dumps({
        "device": str(device), "sources": sources,
        "samples_per_epoch": samples_per_epoch, "validation": "DACL10K val",
        "epochs": [start_epoch, epochs], "max_steps": max_steps,
    }), flush=True)
    for epoch in range(start_epoch, epochs + 1):
        started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        completed_steps = 0
        progress = tqdm(train_loader, desc=f"joint train {epoch}/{epochs}", unit="batch")
        for step, batch in enumerate(progress, 1):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            valid = batch["valid_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                raw_loss = criterion(model(images), targets, valid)
                loss = raw_loss / accumulation
            scaler.scale(loss).backward()
            if step % accumulation == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running += float(raw_loss.detach())
            completed_steps = step
            progress.set_postfix(loss=f"{float(raw_loss.detach()):.4f}")
            if max_steps and step >= max_steps:
                break
        if max_steps:
            print(json.dumps({"smoke_steps": completed_steps, "mean_loss": running / completed_steps}))
            return
        scheduler.step()
        validation = validate(model, val_loader, criterion, device, epoch)
        mean_iou = float(validation["mean_iou"])
        improved = mean_iou > best
        best = max(best, mean_iou)
        record = {
            "epoch": epoch, "train_loss": running / completed_steps,
            "epoch_seconds": round(time.perf_counter() - started, 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "best_mean_iou": best, "validation": validation,
        }
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        save_checkpoint(output / "last.pt", model, optimizer, scheduler, scaler,
                        config, epoch, best, validation)
        if improved:
            save_checkpoint(output / "best.pt", model, optimizer, scheduler, scaler,
                            config, epoch, best, validation)
        print(json.dumps(record), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int,
                        help="Run only this many training batches without validation/checkpoints")
    args = parser.parse_args()
    run(args.config, args.resume, args.max_steps)


if __name__ == "__main__":
    main()
