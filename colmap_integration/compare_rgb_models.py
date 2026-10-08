#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Evaluate DeepLabV3+ and SegFormer with one 2D/3D metric schema.

Run from this repository's root. The models are loaded from frozen RGB/6ML
checkpoints; thresholds are read from each checkpoint and never tuned on test.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from rgb_multilabel.building_data import BuildingPartialLabelDataset
from rgb_multilabel.data import DACL10KSixLabelDataset
from rgb_multilabel.evaluate_building import load_model
from rgb_multilabel.labels import CLASS_NAMES
from rgb_multilabel.metrics import MultilabelSegmentationMetrics, relaxed_binary_f1
from rgb_multilabel.models import DeepLabV3PlusSixLabel

from segmentation_manifest import SegmentationManifestDataset


PLY_TYPES = {"char": "i1", "uchar": "u1", "short": "<i2", "ushort": "<u2",
             "int": "<i4", "uint": "<u4", "float": "<f4", "double": "<f8"}


def read_point_ply(path: Path) -> np.memmap:
    """Read scalar vertex properties from a binary little-endian point PLY."""
    fields = []
    count = None
    in_vertex = False
    with path.open("rb") as handle:
        if handle.readline() != b"ply\n" or handle.readline() != b"format binary_little_endian 1.0\n":
            raise ValueError(f"Expected binary little-endian PLY: {path}")
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError(f"Unterminated PLY header: {path}")
            line = raw.decode("ascii").strip()
            if line == "end_header":
                break
            if line.startswith("element "):
                _, element, number = line.split()
                if element != "vertex":
                    raise ValueError(f"Only point-only PLY is supported: {path}")
                count = int(number)
                in_vertex = True
            elif line.startswith("property ") and in_vertex:
                _, kind, name = line.split()
                if kind not in PLY_TYPES:
                    raise ValueError(f"Unsupported PLY property {kind}: {path}")
                fields.append((name, PLY_TYPES[kind]))
        offset = handle.tell()
    if count is None or not fields:
        raise ValueError(f"No point fields in {path}")
    return np.memmap(path, dtype=np.dtype(fields), mode="r", offset=offset, shape=(count,))


def binary_metrics(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def model_logits(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Use SMP's version-stable top-level forward for DeepLabV3+."""
    if isinstance(model, DeepLabV3PlusSixLabel):
        return model.net(images)
    return model(images)


def evaluate_point_cloud(prediction_path: Path, classes: tuple[str, ...],
                         ground_truth_path: Path | None = None,
                         chunk_size: int = 500000) -> dict:
    """Report multi-view consistency, plus point metrics only when GT exists."""
    prediction = read_point_ply(prediction_path)
    required = {"x", "y", "z", "label_bits", "valid_views"}
    if not required.issubset(prediction.dtype.names):
        raise ValueError(f"Missing point fields {required - set(prediction.dtype.names)}: {prediction_path}")
    truth = read_point_ply(ground_truth_path) if ground_truth_path else None
    if truth is not None:
        if len(truth) != len(prediction) or not {"x", "y", "z", "label_bits"}.issubset(truth.dtype.names):
            raise ValueError("Point ground truth must have same point count, XYZ, and label_bits")
    observed = 0
    labeled = 0
    overlap = 0
    consistency = {}
    point_accuracy = {} if truth is not None else None
    for class_index, name in enumerate(classes):
        positive_field = f"positive_views_{name}"
        if positive_field not in prediction.dtype.names:
            raise ValueError(f"Missing {positive_field} in {prediction_path}")
        agrees = opportunities = positive_agrees = positive_opportunities = 0
        tp = fp = fn = supervised = 0
        for start in range(0, len(prediction), chunk_size):
            stop = min(start + chunk_size, len(prediction))
            block = prediction[start:stop]
            valid = block["valid_views"].astype(np.int64)
            positive = block[positive_field].astype(np.int64)
            if np.any(positive > valid):
                raise ValueError(f"positive_views exceeds valid_views for {name}")
            selected = (block["label_bits"] & (1 << class_index)) != 0
            agrees += int(np.where(selected, positive, valid - positive).sum())
            opportunities += int(valid.sum())
            positive_agrees += int(positive[selected].sum())
            positive_opportunities += int(valid[selected].sum())
            if truth is not None:
                reference = truth[start:stop]
                for axis in ("x", "y", "z"):
                    if not np.allclose(block[axis], reference[axis], rtol=0, atol=1e-5):
                        raise ValueError("Point ground truth is not aligned with prediction")
                valid_gt = ((reference["valid_bits"] & (1 << class_index)) != 0
                            if "valid_bits" in reference.dtype.names
                            else np.ones(len(reference), dtype=bool))
                valid_gt &= valid > 0
                target = (reference["label_bits"] & (1 << class_index)) != 0
                tp += int(np.count_nonzero(selected & target & valid_gt))
                fp += int(np.count_nonzero(selected & ~target & valid_gt))
                fn += int(np.count_nonzero(~selected & target & valid_gt))
                supervised += int(np.count_nonzero(valid_gt))
        consistency[name] = {
            "point_view_agreement": agrees / opportunities if opportunities else None,
            "predicted_positive_agreement": (
                positive_agrees / positive_opportunities if positive_opportunities else None
            ),
            "visible_point_views": opportunities,
            "predicted_positive_point_views": positive_opportunities,
        }
        if truth is not None:
            point_accuracy[name] = {
                "supervised_points": supervised,
                **(binary_metrics(tp, fp, fn) if supervised else {}),
            }
    for start in range(0, len(prediction), chunk_size):
        block = prediction[start:start + chunk_size]
        observed += int(np.count_nonzero(block["valid_views"]))
        selected_count = np.zeros(len(block), dtype=np.uint8)
        for class_index in range(len(classes)):
            selected_count += ((block["label_bits"] & (1 << class_index)) != 0).astype(np.uint8)
        labeled += int(np.count_nonzero(selected_count))
        overlap += int(np.count_nonzero(selected_count > 1))
    return {
        "status": "ground_truth_and_consistency" if truth is not None else "consistency_only",
        "prediction_ply": str(prediction_path.resolve()),
        "ground_truth_ply": str(ground_truth_path.resolve()) if ground_truth_path else None,
        "point_count": len(prediction), "visible_point_count": observed,
        "labeled_point_count": labeled, "multi_label_point_count": overlap,
        "consistency": consistency, "point_accuracy": point_accuracy,
    }


@torch.inference_mode()
def benchmark_model(model: torch.nn.Module, images: torch.Tensor,
                    device: torch.device, warmup: int, repeats: int) -> dict:
    if repeats < 1 or warmup < 0:
        raise ValueError("Benchmark repeats must be positive and warmup nonnegative")
    images = images.to(device)
    for _ in range(warmup):
        model_logits(model, images)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(repeats):
        model_logits(model, images)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "batch_size": len(images), "warmup_batches": warmup, "timed_batches": repeats,
        "milliseconds_per_image": elapsed * 1000 / (len(images) * repeats),
        "images_per_second": len(images) * repeats / elapsed,
        "peak_gpu_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        ),
    }


def build_dataset(args: argparse.Namespace, crop_size: int):
    if args.dataset == "dacl":
        dataset = DACL10KSixLabelDataset(
            args.data_root, args.mask_cache, args.manifest, args.split, crop_size, training=False
        )
        classes = CLASS_NAMES
    elif args.dataset == "building":
        dataset = BuildingPartialLabelDataset(args.data_root, args.split, crop_size, training=False)
        classes = CLASS_NAMES[:2]
    else:
        dataset = SegmentationManifestDataset(args.manifest, args.split, crop_size)
        classes = CLASS_NAMES
    if args.max_samples:
        dataset = Subset(dataset, list(range(min(args.max_samples, len(dataset)))))
    return dataset, classes


@torch.inference_mode()
def evaluate_2d(model: torch.nn.Module, loader: DataLoader, thresholds: torch.Tensor,
                device: torch.device, classes: tuple[str, ...], radius: int) -> tuple[dict, torch.Tensor]:
    metrics = MultilabelSegmentationMetrics(len(CLASS_NAMES))
    supervised_pixels = torch.zeros(len(CLASS_NAMES), dtype=torch.int64)
    relaxed_scores = []
    blank_crack = blank_crack_fp = 0
    benchmark_images = None
    for batch in loader:
        images = batch["image"]
        if benchmark_images is None:
            benchmark_images = images.clone()
        predictions = (model_logits(model, images.to(device)).sigmoid() >= thresholds).cpu()
        target = batch["target"]
        valid = batch["valid_mask"]
        metrics.update(predictions, target, valid)
        supervised_pixels += valid.bool().expand_as(target).sum((0, 2, 3))
        for pred, gt, mask in zip(predictions, target, valid):
            crack_valid = mask[0].bool()
            if not crack_valid.any():
                continue
            crack_gt = gt[0].bool() & crack_valid
            if crack_gt.any():
                relaxed_scores.append(relaxed_binary_f1(pred[0], gt[0], crack_valid, radius))
            else:
                blank_crack += 1
                blank_crack_fp += int((pred[0].bool() & crack_valid).any())
    if benchmark_images is None:
        raise ValueError("Evaluation dataset is empty")
    values = metrics.compute()
    per_class = {}
    for index, name in enumerate(classes):
        per_class[name] = (
            {metric: float(values[metric][index]) for metric in ("iou", "dice", "precision", "recall")}
            if supervised_pixels[index] else None
        )
    usable = [per_class[name]["iou"] for name in classes if per_class[name] is not None]
    return {
        "samples": len(loader.dataset), "evaluated_classes": list(classes),
        "supervised_pixels": {name: int(supervised_pixels[index]) for index, name in enumerate(classes)},
        "per_class": per_class, "mean_iou": sum(usable) / len(usable) if usable else None,
        "relaxed_crack_f1": sum(relaxed_scores) / len(relaxed_scores) if relaxed_scores else None,
        "relaxed_crack_positive_samples": len(relaxed_scores),
        "blank_crack_samples": blank_crack,
        "blank_crack_false_positive_image_rate": blank_crack_fp / blank_crack if blank_crack else None,
        "relaxed_crack_radius_pixels": radius,
    }, benchmark_images


def parse_named_paths(items: list[str]) -> dict[str, Path]:
    values = {}
    for item in items:
        name, separator, path = item.partition("=")
        if not separator or not name or not path or name in values:
            raise ValueError(f"Expected unique NAME=PATH, got {item!r}")
        values[name] = Path(path)
    return values


def run(args: argparse.Namespace) -> dict:
    checkpoints = parse_named_paths(args.checkpoint)
    point_clouds = parse_named_paths(args.point_cloud)
    if not checkpoints or set(point_clouds) - set(checkpoints):
        raise ValueError("Provide checkpoints and matching point-cloud model names")
    geometry_keys = ("source_ply", "points", "images", "min_positive_views",
                     "min_positive_fraction", "depth_tolerance_relative")
    point_run_settings = None
    for name, path in point_clouds.items():
        summary_path = path.with_suffix(".summary.json")
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing point-cloud fusion summary: {summary_path}")
        summary = json.loads(summary_path.read_text())
        settings = tuple(summary.get(key) for key in geometry_keys)
        if point_run_settings is None:
            point_run_settings = settings
        elif settings != point_run_settings:
            raise ValueError("Point clouds use different geometry or fusion settings")
        if summary.get("partial_run"):
            print(f"WARNING: {name} point cloud is a partial smoke-test run", flush=True)
    if args.batch_size < 1 or args.num_workers < 0 or args.max_samples < 0:
        raise ValueError("Invalid batch, worker, or sample count")
    device = torch.device(args.device)
    crop_sizes = set()
    for name, path in checkpoints.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        crop_sizes.add(int(checkpoint["config"]["data"]["crop_size"]))
        del checkpoint
    if len(crop_sizes) != 1:
        raise ValueError("Checkpoints use different crop sizes; use matched experiments")
    crop_size = crop_sizes.pop()
    dataset, classes = build_dataset(args, crop_size)
    for name, path in point_clouds.items():
        summary = json.loads(path.with_suffix(".summary.json").read_text())
        declared = [summary["classes"][str(index + 1)] for index in range(len(classes))]
        if declared != list(classes):
            raise ValueError(f"{name} point-cloud class order differs from 2D evaluation")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    results = {}
    for name, path in checkpoints.items():
        model, checkpoint = load_model(path, device)
        threshold_values = checkpoint["thresholds"]
        thresholds = torch.tensor([threshold_values[class_name] for class_name in CLASS_NAMES],
                                  dtype=torch.float32, device=device).view(1, -1, 1, 1)
        metrics_2d, sample = evaluate_2d(model, loader, thresholds, device, classes,
                                         args.relaxed_radius)
        speed = benchmark_model(model, sample, device, args.warmup, args.repeats)
        point_metrics = (
            evaluate_point_cloud(point_clouds[name], tuple(classes), args.point_ground_truth)
            if name in point_clouds else
            {"status": "not_run", "reason": "No --point-cloud provided for this model"}
        )
        results[name] = {
            "architecture": checkpoint["config"]["model"]["architecture"],
            "checkpoint": str(path.resolve()), "checkpoint_epoch": checkpoint.get("epoch"),
            "thresholds": {class_name: float(threshold_values[class_name]) for class_name in classes},
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "model_state_size_mib": sum(t.numel() * t.element_size()
                                        for t in model.state_dict().values()) / 2**20,
            "inference": speed, "two_dimensional": metrics_2d,
            "three_dimensional": point_metrics,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output = {
        "dataset": args.dataset, "split": args.split, "crop_size": crop_size,
        "batch_size": args.batch_size, "max_samples": args.max_samples,
        "device": str(device), "models": results,
        "comparison_is_partial": bool(args.max_samples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, allow_nan=False))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="Repeat NAME=PATH, e.g. segformer=best.pt")
    parser.add_argument("--dataset", choices=("dacl", "building", "manifest"), default="dacl")
    parser.add_argument("--data-root", type=Path, default=Path("../RGB/datasets/dacl10k/raw/dacl10k_v2_devphase"))
    parser.add_argument("--mask-cache", type=Path, default=Path("../RGB/datasets/dacl10k/cache/labels_6ml"))
    parser.add_argument("--manifest", type=Path, default=Path("../RGB/6ML/data/split_manifest.csv"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path, default=Path("benchmark/reconstruction/runs/rgb-model-comparison.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0, help="Use first N samples for smoke test")
    parser.add_argument("--relaxed-radius", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--point-cloud", action="append", default=[], help="Optional NAME=labeled_dense.ply")
    parser.add_argument("--point-ground-truth", type=Path, help="Optional aligned PLY with label_bits and optional valid_bits")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
