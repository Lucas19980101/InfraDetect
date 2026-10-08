from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .building_data import BuildingPartialLabelDataset
from .labels import CLASS_NAMES
from .metrics import MultilabelSegmentationMetrics, PerClassThresholdTuner, relaxed_binary_f1
from .models import build_model


def load_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(checkpoint["config"]["model"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def tune(model, loader, device) -> torch.Tensor:
    tuner = PerClassThresholdTuner(len(CLASS_NAMES), bins=1000)
    for batch in tqdm(loader, desc="calibrate building val", leave=False, unit="batch"):
        images = batch["image"].to(device, non_blocking=True)
        tuner.update(
            model(images),
            batch["target"].to(device, non_blocking=True),
            batch["valid_mask"].to(device, non_blocking=True),
        )
    return tuner.compute()["thresholds"].float().to(device)


@torch.no_grad()
def evaluate(model, loader, thresholds, device) -> dict:
    metrics = MultilabelSegmentationMetrics(len(CLASS_NAMES))
    relaxed_scores: list[float] = []
    blank, blank_fp = 0, 0
    threshold_tensor = thresholds.view(1, -1, 1, 1)
    for batch in tqdm(loader, desc="evaluate building test", unit="batch"):
        targets, valid = batch["target"], batch["valid_mask"]
        predictions = (
            model(batch["image"].to(device)).sigmoid() >= threshold_tensor
        ).cpu()
        metrics.update(predictions, targets, valid)
        for prediction, target, sample_valid, source in zip(
            predictions, targets, valid, batch["source"]
        ):
            if source != "crack":
                continue
            truth = target[0].bool() & sample_valid[0].bool()
            if truth.any():
                relaxed_scores.append(
                    relaxed_binary_f1(prediction[0], target[0], sample_valid[0], radius=3)
                )
            else:
                blank += 1
                if (prediction[0].bool() & sample_valid[0].bool()).any():
                    blank_fp += 1
    values = metrics.compute()
    iou = values["iou"][:2]
    result = {
        "test_samples": len(loader.dataset),
        "thresholds": {name: float(thresholds[index]) for index, name in enumerate(CLASS_NAMES[:2])},
        "mean_iou_crack_spalling": float(iou.mean()),
        "relaxed_crack_f1": sum(relaxed_scores) / len(relaxed_scores) if relaxed_scores else None,
        "relaxed_crack_positive_samples": len(relaxed_scores),
        "blank_crack_samples": blank,
        "blank_crack_false_positive_image_rate": blank_fp / blank if blank else None,
    }
    for metric_name in ("iou", "dice", "precision", "recall"):
        result[metric_name] = {
            name: float(value)
            for name, value in zip(CLASS_NAMES[:2], values[metric_name][:2])
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate and evaluate on current-building data")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("../../DeepLabs+/dataset"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(args.checkpoint, device)
    size = int(checkpoint["config"]["data"]["crop_size"])
    val_set = BuildingPartialLabelDataset(args.data_root, "val", size, training=False)
    test_set = BuildingPartialLabelDataset(args.data_root, "test", size, training=False)
    batch_size = int(checkpoint["config"]["train"]["batch_size"])
    options = dict(batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    thresholds = tune(model, DataLoader(val_set, **options), device)
    result = {
        "model": checkpoint["config"]["model"]["architecture"],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        **evaluate(model, DataLoader(test_set, **options), thresholds, device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
