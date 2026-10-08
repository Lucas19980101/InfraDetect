from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .data import BFDD_PALETTE_RGB, restore_mask
from .metrics import Metrics
from .models import build_model
from .train import config_from_file, make_loader


@torch.no_grad()
def run(config_path: Path, checkpoint_path: Path, split: str, output: Path | None, workers: int | None) -> dict:
    cfg = config_from_file(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved = state["config"]
    if saved["task"] != cfg["task"] or saved["data"]["class_names"] != cfg["data"]["class_names"]:
        raise ValueError("Checkpoint task or class names differ from config")
    model = build_model(cfg["task"], cfg["model"]["classes"], pretrained=False).to(device)
    model.load_state_dict(state["model"]); model.eval()
    loader = make_loader(cfg, split, workers)
    metrics = Metrics(cfg["model"]["classes"])
    if output:
        output.mkdir(parents=True, exist_ok=True)
        if cfg["task"] == "ir":
            (output / "color").mkdir(exist_ok=True)
    for batch in tqdm(loader, desc=f"evaluate {split}"):
        prediction = model(batch["image"].to(device))
        metrics.update(prediction, batch["mask"])
        if output:
            for name, mask, size in zip(batch["name"], prediction.argmax(1).cpu().numpy(), batch["original_size"]):
                restored = restore_mask(mask.astype(np.uint8), tuple(map(int, size)))
                cv2.imwrite(str(output / f"{name}.png"), restored)
                if cfg["task"] == "ir":
                    palette = np.asarray(cfg["data"].get("palette_rgb", BFDD_PALETTE_RGB),
                                         dtype=np.uint8)
                    preview_rgb = palette[restored]
                    cv2.imwrite(str(output / "color" / f"{name}.png"),
                                cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR))
    result = {"task": cfg["task"], "split": split, "samples": len(loader.dataset),
              "checkpoint": str(checkpoint_path.resolve()),
              **metrics.compute(cfg["data"]["class_names"])}
    if output:
        (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.checkpoint, args.split, args.output, args.num_workers), indent=2))


if __name__ == "__main__":
    main()
