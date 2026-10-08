"""Short real-data overfit check before full training."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .data import SegmentationDataset
from .metrics import ce_dice
from .models import build_model
from .train import config_from_file


def run(config: Path, steps: int, size: tuple[int, int], batch_size: int) -> dict:
    cfg = config_from_file(config)
    torch.manual_seed(int(cfg["seed"]))
    dataset = SegmentationDataset(Path(cfg["data"]["root"]), Path(cfg["data"]["manifest"]),
                                  "train", cfg["task"], size, cfg["model"]["classes"], augment=False,
                                  label_map={int(old): int(new)
                                             for old, new in cfg["data"].get("label_map", {}).items()}
                                  or None)
    batch = [dataset[index] for index in range(batch_size)]
    image = torch.stack([item["image"] for item in batch])
    target = torch.stack([item["mask"] for item in batch])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["task"], cfg["model"]["classes"], pretrained=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
    image, target = image.to(device), target.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    losses: list[float] = []
    for _ in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = ce_dice(model(image), target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {"task": cfg["task"], "device": str(device), "samples": [item["name"] for item in batch],
            "size": list(size), "steps": steps, "loss_first": losses[0], "loss_last": losses[-1],
            "loss_decreased": losses[-1] < losses[0] if steps > 1 else None,
            "seconds_per_step": elapsed / steps,
            "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    result = run(args.config, args.steps, (args.height, args.width), args.batch_size)
    print(json.dumps(result, indent=2))
    if result["loss_decreased"] is False:
        raise SystemExit("Real-data overfit check did not lower loss")


if __name__ == "__main__":
    main()
