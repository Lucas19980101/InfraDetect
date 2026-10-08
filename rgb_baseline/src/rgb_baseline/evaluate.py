from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoint import load_checkpoint
from .config import load_config
from .data import SemanticDataset
from .metrics import SegmentationMetrics, relaxed_binary_f1
from .model import build_model


PALETTE = np.asarray([[0, 0, 0], [255, 64, 64], [255, 190, 40]], dtype=np.uint8)
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def colorize(mask: np.ndarray) -> np.ndarray:
    valid = np.clip(mask, 0, len(PALETTE) - 1)
    colored = PALETTE[valid]
    colored[mask == 255] = (128, 128, 128)
    return colored


def comparison(image: torch.Tensor, target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    rgb = image.permute(1, 2, 0).numpy() * STD + MEAN
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    panels = [np.ascontiguousarray(panel) for panel in (rgb, colorize(target), colorize(prediction))]
    titles = ["IMAGE", "GROUND TRUTH", "PREDICTION"]
    for panel, title in zip(panels, titles):
        cv2.rectangle(panel, (0, 0), (250, 42), (0, 0, 0), -1)
        cv2.putText(panel, title, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return np.concatenate(panels, axis=1)


def sample_statistics(prediction: torch.Tensor, target: torch.Tensor, ignore_index: int) -> dict[str, float | int]:
    valid = target != ignore_index
    pred_fg, true_fg = (prediction > 0) & valid, (target > 0) & valid
    tp = int((pred_fg & true_fg).sum())
    fp = int((pred_fg & ~true_fg & valid).sum())
    fn = int((~pred_fg & true_fg).sum())
    union = tp + fp + fn
    return {
        "foreground_iou": tp / union if union else 1.0,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
        "false_positive_rate": fp / max(int((~true_fg & valid).sum()), 1),
        "false_negative_rate": fn / max(int(true_fg.sum()), 1),
        "true_foreground_pixels": int(true_fg.sum()),
        "predicted_foreground_pixels": int(pred_fg.sum()),
    }


def ranked_samples(rows: list[dict[str, float | int | str]], limit: int = 5) -> dict[str, list[dict]]:
    positive = [row for row in rows if int(row["true_foreground_pixels"]) > 0]
    false_positives = [row for row in rows if int(row["false_positive_pixels"]) > 0]
    false_negatives = [row for row in positive if int(row["false_negative_pixels"]) > 0]
    return {
        "best": sorted(positive, key=lambda row: float(row["foreground_iou"]), reverse=True)[:limit],
        "worst": sorted(positive, key=lambda row: float(row["foreground_iou"]))[:limit],
        "typical_false_positive": sorted(
            false_positives, key=lambda row: (float(row["false_positive_rate"]), int(row["false_positive_pixels"])), reverse=True
        )[:limit],
        "typical_false_negative": sorted(
            false_negatives, key=lambda row: (float(row["false_negative_rate"]), int(row["false_negative_pixels"])), reverse=True
        )[:limit],
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, help="Save metrics, class-index masks, and comparison images")
    parser.add_argument("--num-workers", type=int, help="Override data.num_workers for restricted environments")
    args = parser.parse_args(); cfg = load_config(args.config); data = cfg["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"], pretrained=False).to(device); load_checkpoint(model, args.checkpoint, device); model.eval()
    dataset = SemanticDataset(data["root"], args.split, data["tile_size"], False)
    workers = data["num_workers"] if args.num_workers is None else args.num_workers
    loader = DataLoader(dataset, batch_size=cfg["inference"]["batch_size"], num_workers=workers)
    output = args.output_dir
    if output:
        mask_dir, comparison_dir = output / "masks", output / "comparisons"
        mask_dir.mkdir(parents=True, exist_ok=True); comparison_dir.mkdir(parents=True, exist_ok=True)
    metrics = SegmentationMetrics(cfg["model"]["num_classes"], data["ignore_index"]); relaxed = []; samples = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"evaluate {args.split}", unit="batch"):
            masks = batch["mask"]; predictions = model(batch["image"].to(device)).argmax(1).cpu(); metrics.update(predictions, masks)
            relaxed.extend(relaxed_binary_f1(p, m, radius=cfg["evaluation"]["relaxed_crack_radius"],
                                             ignore_index=data["ignore_index"]) for p, m in zip(predictions, masks))
            for image, prediction, target, name in zip(batch["image"], predictions, masks, batch["name"]):
                valid = target != data["ignore_index"]
                row = {"name": name, **sample_statistics(prediction, target, data["ignore_index"]),
                       "true_crack_pixels": int(((target == 1) & valid).sum()),
                       "predicted_crack_pixels": int(((prediction == 1) & valid).sum())}
                samples.append(row)
                if output:
                    pred_np, target_np = prediction.numpy().astype(np.uint8), target.numpy().astype(np.uint8)
                    cv2.imwrite(str(mask_dir / f"{name}.png"), pred_np)
                    visual = comparison(image, target_np, pred_np)
                    cv2.imwrite(str(comparison_dir / f"{name}.jpg"), cv2.cvtColor(visual, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
    values = metrics.compute(); names = data["class_names"]
    valid_relaxed = [score for score in relaxed if math.isfinite(score)]
    both_empty = len(relaxed) - len(valid_relaxed)
    blank_truth = [row for row in samples if int(row["true_crack_pixels"]) == 0]
    blank_false_positives = sum(int(row["predicted_crack_pixels"]) > 0 for row in blank_truth)
    result = {
        "mean_iou": values["mean_iou"],
        "relaxed_crack_f1": sum(valid_relaxed) / len(valid_relaxed) if valid_relaxed else None,
        "relaxed_crack_valid_samples": len(valid_relaxed),
        "relaxed_crack_both_empty_samples": both_empty,
        "blank_truth_false_positive_rate": blank_false_positives / len(blank_truth) if blank_truth else None,
        "blank_truth_samples": len(blank_truth),
    }
    for metric in ("iou", "dice", "precision", "recall"):
        result[metric] = {name: float(value) for name, value in zip(names, values[metric])}
    if output:
        ranking = ranked_samples(samples)
        result.update({"checkpoint": str(Path(args.checkpoint).resolve()), "split": args.split,
                       "sample_count": len(dataset), "ranked_samples": ranking})
        with (output / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        with (output / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(samples[0])); writer.writeheader(); writer.writerows(samples)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
