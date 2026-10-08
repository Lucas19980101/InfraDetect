from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .data import SegmentationDataset
from .metrics import Metrics, ce_dice
from .models import build_model


def config_from_file(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["data"]["root"] = str((path.parent / cfg["data"]["root"]).resolve())
    cfg["data"]["manifest"] = str((path.parent / cfg["data"]["manifest"]).resolve())
    cfg["train"]["output_dir"] = str((path.parent / cfg["train"]["output_dir"]).resolve())
    if len(cfg["data"]["class_names"]) != cfg["model"]["classes"]:
        raise ValueError("class_names and model.classes disagree")
    return cfg


def make_loader(cfg: dict, split: str, workers: int | None = None) -> DataLoader:
    data = cfg["data"]
    include_sources = set(data["include_sources"]) if data.get("include_sources") else None
    augmentation = data.get("augmentation", {})
    label_map = ({int(old): int(new) for old, new in data["label_map"].items()}
                 if data.get("label_map") else None)
    dataset = SegmentationDataset(Path(data["root"]), Path(data["manifest"]), split, cfg["task"],
                                  tuple(data["image_size"]), cfg["model"]["classes"], split == "train",
                                  include_sources,
                                  augmentation.get("rotation_degrees", 0.0),
                                  augmentation.get("rotation_probability", 0.5), label_map)
    sampler = None
    if split == "train" and cfg["train"].get("source_sampling_weights"):
        source_weights = cfg["train"]["source_sampling_weights"]
        weights = [float(source_weights.get(source, 1.0)) for source in dataset.sources]
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    return DataLoader(dataset, batch_size=cfg["train"]["batch_size"],
                      shuffle=split == "train" and sampler is None, sampler=sampler,
                      drop_last=split == "train",
                      num_workers=data["num_workers"] if workers is None else workers,
                      pin_memory=torch.cuda.is_available())


@torch.no_grad()
def evaluate(model, loader, device, names, class_weights=None) -> dict:
    model.eval()
    metrics = Metrics(len(names))
    losses = []
    for batch in tqdm(loader, desc="validate", leave=False, disable=not sys.stderr.isatty()):
        image, target = batch["image"].to(device), batch["mask"].to(device)
        logits = model(image)
        losses.append(float(ce_dice(logits, target, class_weights)))
        metrics.update(logits, target)
    return {"loss": float(np.mean(losses)), **metrics.compute(names)}


def atomic_save(path: Path, state: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def run(path: Path, resume: Path | None = None, init_checkpoint: Path | None = None,
        workers: int | None = None,
        stop_after_epoch: int | None = None) -> None:
    cfg = config_from_file(path)
    seed = int(cfg["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train_loader, val_loader = make_loader(cfg, "train", workers), make_loader(cfg, "val", workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_generic_pretraining = resume is None and init_checkpoint is None and cfg["model"]["pretrained"]
    model = build_model(cfg["task"], cfg["model"]["classes"], pretrained=use_generic_pretraining).to(device)
    if init_checkpoint is not None:
        if resume is not None:
            raise ValueError("--resume and --init-checkpoint cannot be combined")
        source = torch.load(init_checkpoint, map_location=device, weights_only=False)["model"]
        target = model.state_dict()
        compatible = {key: value for key, value in source.items()
                      if key in target and value.shape == target[key].shape}
        model.load_state_dict(compatible, strict=False)
        print(f"initialized {len(compatible)}/{len(target)} tensors from {init_checkpoint}")
    raw_weights = cfg["train"].get("class_weights")
    class_weights = torch.tensor(raw_weights, dtype=torch.float32, device=device) if raw_weights is not None else None
    if class_weights is not None and len(class_weights) != cfg["model"]["classes"]:
        raise ValueError("class_weights and model.classes disagree")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["learning_rate"],
                                  weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["train"]["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["mixed_precision"] and device.type == "cuda")
    output = Path(cfg["train"]["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    start, best = 1, -1.0
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        if (state["config"]["task"] != cfg["task"] or
            state["config"]["data"]["class_names"] != cfg["data"]["class_names"] or
            state["config"]["train"].get("class_weights") != raw_weights):
            raise ValueError("Resume checkpoint task, classes, or loss weights differ")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"]); scaler.load_state_dict(state["scaler"])
        start, best = state["epoch"] + 1, state["best_mean_iou"]
    print(f"task={cfg['task']} device={device} train={len(train_loader.dataset)} val={len(val_loader.dataset)}")
    final_epoch = min(cfg["train"]["epochs"], stop_after_epoch) if stop_after_epoch is not None else cfg["train"]["epochs"]
    for epoch in range(start, final_epoch + 1):
        model.train(); running = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}", disable=not sys.stderr.isatty()):
            image, target = batch["image"].to(device), batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=scaler.is_enabled()):
                loss = ce_dice(model(image), target, class_weights)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            running += float(loss.detach())
        scheduler.step()
        result = evaluate(model, val_loader, device, cfg["data"]["class_names"], class_weights)
        result.update(epoch=epoch, train_loss=running / len(train_loader))
        score = result["mean_iou"] if result["mean_iou"] is not None else -1.0
        improved = score > best
        best = max(best, score)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                 "epoch": epoch, "best_mean_iou": best, "config": cfg, "metrics": result}
        atomic_save(output / "last.pt", state)
        if improved:
            atomic_save(output / "best.pt", state)
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")
        print(json.dumps(result), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-checkpoint", type=Path,
                        help="Load shape-compatible model tensors but start a new optimizer/run")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--stop-after-epoch", type=int,
                        help="Run through this absolute epoch, save a resumable checkpoint, then stop")
    args = parser.parse_args()
    run(args.config, args.resume, args.init_checkpoint, args.num_workers, args.stop_after_epoch)


if __name__ == "__main__":
    main()
