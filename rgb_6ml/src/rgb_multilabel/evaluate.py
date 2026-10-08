from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import DACL10KSixLabelDataset, MEAN, STD
from .labels import CLASS_NAMES
from .masks import encode_mask
from .metrics import MultilabelSegmentationMetrics, PerClassThresholdTuner, relaxed_binary_f1
from .models import build_model


COLORS = np.asarray(
    [
        [255, 48, 48],
        [255, 176, 32],
        [48, 220, 255],
        [184, 64, 255],
        [255, 240, 64],
        [32, 160, 255],
    ],
    dtype=np.float32,
)


def load_experiment(
    checkpoint_path: Path,
    device: torch.device,
    threshold_checkpoint_path: Path | None = None,
):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_model(config["model"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    threshold_checkpoint = checkpoint
    if threshold_checkpoint_path is not None:
        threshold_checkpoint = torch.load(
            threshold_checkpoint_path, map_location="cpu", weights_only=False
        )
    thresholds = torch.tensor(
        [threshold_checkpoint["thresholds"][name] for name in CLASS_NAMES],
        dtype=torch.float32,
        device=device,
    ).view(1, -1, 1, 1)
    return model, config, thresholds, checkpoint


def overlay(image: np.ndarray, masks: np.ndarray, alpha: float = 0.48) -> np.ndarray:
    result = image.astype(np.float32).copy()
    for index, color in enumerate(COLORS):
        selected = masks[index].astype(bool)
        result[selected] = result[selected] * (1 - alpha) + color * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def add_title(panel: np.ndarray, title: str) -> np.ndarray:
    panel = panel.copy()
    cv2.rectangle(panel, (0, 0), (290, 38), (0, 0, 0), -1)
    cv2.putText(panel, title, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return panel


def comparison_image(
    normalized_image: torch.Tensor,
    target: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    image = normalized_image.permute(1, 2, 0).numpy()
    image = image * np.asarray(STD, dtype=np.float32) + np.asarray(MEAN, dtype=np.float32)
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    ignore = image.copy()
    ignore[~valid] = (ignore[~valid].astype(np.float32) * 0.35 + 128 * 0.65).astype(np.uint8)
    panels = [
        add_title(image, "IMAGE"),
        add_title(overlay(image, target), "GROUND TRUTH"),
        add_title(overlay(image, prediction), "PREDICTION"),
        add_title(ignore, "IGNORE (GRAY)"),
    ]
    return np.concatenate(panels, axis=1)


@torch.no_grad()
def benchmark(model, images: torch.Tensor, device: torch.device, warmup: int, repeats: int) -> dict:
    images = images.to(device)
    for _ in range(warmup):
        model(images)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(repeats):
        model(images)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    image_count = len(images) * repeats
    return {
        "benchmark_batch_size": len(images),
        "warmup_batches": warmup,
        "timed_batches": repeats,
        "milliseconds_per_image": elapsed * 1000 / image_count,
        "images_per_second": image_count / elapsed,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20
        if device.type == "cuda"
        else None,
    }


@torch.no_grad()
def calibrate_thresholds(model, loader, device: torch.device) -> torch.Tensor:
    tuner = PerClassThresholdTuner(len(CLASS_NAMES), bins=1000)
    for batch in tqdm(loader, desc="calibrate DACL val", unit="batch", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        tuner.update(
            model(images),
            batch["target"].to(device, non_blocking=True),
            batch["valid_mask"].to(device, non_blocking=True),
        )
    return tuner.compute()["thresholds"].float().to(device).view(1, -1, 1, 1)


def per_sample_counts(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict:
    valid = valid.bool().expand_as(target)
    prediction, target = prediction.bool(), target.bool()
    dims = (1, 2)
    tp = (prediction & target & valid).sum(dims)
    fp = (prediction & ~target & valid).sum(dims)
    fn = (~prediction & target & valid).sum(dims)
    return {
        f"{metric}_{name}": int(values[index])
        for metric, values in (("tp", tp), ("fp", fp), ("fn", fn))
        for index, name in enumerate(CLASS_NAMES)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one frozen 6ML checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--threshold-checkpoint",
        type=Path,
        help="Use frozen thresholds from another checkpoint (e.g. pre-finetune DACL thresholds)",
    )
    parser.add_argument(
        "--calibrate-on-dacl-val",
        action="store_true",
        help="Re-tune all class thresholds on the frozen experiment val split before test",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-visualizations", type=int, default=20)
    parser.add_argument("--benchmark-warmup", type=int, default=10)
    parser.add_argument("--benchmark-repeats", type=int, default=50)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, thresholds, checkpoint = load_experiment(
        args.checkpoint, device, args.threshold_checkpoint
    )
    if args.threshold_checkpoint and args.calibrate_on_dacl_val:
        parser.error("Use either --threshold-checkpoint or --calibrate-on-dacl-val, not both")
    data = config["data"]
    dataset = DACL10KSixLabelDataset(
        data_root=data["root"],
        cache_root=data["mask_cache"],
        manifest_path=data["split_manifest"],
        split="test",
        crop_size=data["crop_size"],
        training=False,
    )
    if args.calibrate_on_dacl_val:
        calibration_set = DACL10KSixLabelDataset(
            data_root=data["root"],
            cache_root=data["mask_cache"],
            manifest_path=data["split_manifest"],
            split="val",
            crop_size=data["crop_size"],
            training=False,
        )
        calibration_loader = DataLoader(
            calibration_set,
            batch_size=int(config["train"]["batch_size"]),
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        thresholds = calibrate_thresholds(model, calibration_loader, device)
    loader = DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.output_dir / "prediction_masks"
    visualization_dir = args.output_dir / "visualizations"
    prediction_dir.mkdir(exist_ok=True)
    visualization_dir.mkdir(exist_ok=True)
    metrics = MultilabelSegmentationMetrics(len(CLASS_NAMES))
    relaxed_scores: list[float] = []
    blank_crack_samples = 0
    blank_crack_false_positive_samples = 0
    rows: list[dict] = []
    benchmark_images: torch.Tensor | None = None
    visualized = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluate test", unit="batch"):
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"]
            valid = batch["valid_mask"]
            logits = model(images)
            predictions = (logits.sigmoid() >= thresholds).cpu()
            metrics.update(predictions, targets, valid)
            if benchmark_images is None:
                benchmark_images = batch["image"].clone()
            for image, prediction, target, sample_valid, name in zip(
                batch["image"], predictions, targets, valid, batch["name"]
            ):
                crack_target = target[0].bool() & sample_valid.squeeze(0).bool()
                if crack_target.any():
                    relaxed = relaxed_binary_f1(
                        prediction[0], target[0], sample_valid,
                        radius=int(config["evaluation"]["relaxed_crack_radius"]),
                    )
                    relaxed_scores.append(relaxed)
                else:
                    relaxed = float("nan")
                    blank_crack_samples += 1
                    if (prediction[0].bool() & sample_valid.squeeze(0).bool()).any():
                        blank_crack_false_positive_samples += 1
                row = {"name": name, "relaxed_crack_f1": relaxed, **per_sample_counts(
                    prediction, target, sample_valid
                )}
                rows.append(row)
                pred_np = prediction.numpy().astype(np.bool_)
                valid_np = sample_valid.squeeze(0).numpy().astype(np.bool_)
                packed = encode_mask(pred_np, valid_np)
                cv2.imwrite(str(prediction_dir / f"{name}.png"), packed)
                if visualized < args.max_visualizations:
                    visual = comparison_image(
                        image, target.numpy().astype(bool), pred_np, valid_np
                    )
                    cv2.imwrite(
                        str(visualization_dir / f"{name}.jpg"),
                        cv2.cvtColor(visual, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92],
                    )
                    visualized += 1
    values = metrics.compute()
    valid_relaxed = [value for value in relaxed_scores if math.isfinite(value)]
    speed = benchmark(
        model, benchmark_images, device, args.benchmark_warmup, args.benchmark_repeats
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    state_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in model.state_dict().values()
    )
    result = {
        "model": config["model"]["architecture"],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "threshold_source": (
            "dacl_experiment_val_recalibrated"
            if args.calibrate_on_dacl_val
            else str((args.threshold_checkpoint or args.checkpoint).resolve())
        ),
        "test_samples": len(dataset),
        "thresholds": {name: float(thresholds[0, index, 0, 0]) for index, name in enumerate(CLASS_NAMES)},
        "mean_iou": values["mean_iou"],
        "relaxed_crack_f1": sum(valid_relaxed) / len(valid_relaxed) if valid_relaxed else None,
        "relaxed_crack_valid_samples": len(valid_relaxed),
        "blank_crack_samples": blank_crack_samples,
        "blank_crack_false_positive_samples": blank_crack_false_positive_samples,
        "blank_crack_false_positive_image_rate": (
            blank_crack_false_positive_samples / blank_crack_samples
            if blank_crack_samples
            else None
        ),
        "parameter_count": parameter_count,
        "model_state_size_mib": state_bytes / 2**20,
        **speed,
    }
    for metric_name in ("iou", "dice", "precision", "recall"):
        result[metric_name] = {
            name: float(value) for name, value in zip(CLASS_NAMES, values[metric_name])
        }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    with (args.output_dir / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
