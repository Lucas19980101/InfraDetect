"""Calibrate a binary segmentation probability threshold on one split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .models import build_model
from .train import config_from_file, make_loader


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--min-threshold", type=float, default=0.50)
    parser.add_argument("--max-threshold", type=float, default=0.99)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--max-background-fpr", type=float, default=0.02)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cfg = config_from_file(args.config)
    if cfg["model"]["classes"] != 2:
        raise ValueError("Threshold calibration requires exactly two classes")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(cfg["task"], 2, pretrained=False).to(device)
    model.load_state_dict(state["model"]); model.eval()
    loader = make_loader(cfg, args.split, args.num_workers)
    probabilities, targets = [], []
    for batch in tqdm(loader, desc=f"threshold {args.split}"):
        logits = model(batch["image"].to(device))
        probabilities.append(logits.softmax(1)[:, 1].cpu().numpy().reshape(-1))
        targets.append(batch["mask"].numpy().reshape(-1))
    probability, target = np.concatenate(probabilities), np.concatenate(targets)
    valid = target != 255
    probability, target = probability[valid], target[valid]
    positive, negative = target == 1, target == 0

    thresholds = np.arange(args.min_threshold, args.max_threshold + args.step / 2, args.step)
    results = []
    for threshold in thresholds:
        prediction = probability >= threshold
        tp = int(np.count_nonzero(prediction & positive))
        fp = int(np.count_nonzero(prediction & negative))
        fn = int(np.count_nonzero(~prediction & positive))
        tn = int(np.count_nonzero(~prediction & negative))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        iou = tp / max(tp + fp + fn, 1)
        results.append({"threshold": round(float(threshold), 4), "iou": iou,
                        "dice": 2 * tp / max(2 * tp + fp + fn, 1),
                        "precision": precision, "recall": recall,
                        "background_fpr": fpr, "confusion": [[tn, fp], [fn, tp]]})
    feasible = [row for row in results if row["background_fpr"] <= args.max_background_fpr]
    selected = max(feasible, key=lambda row: (row["iou"], row["recall"])) if feasible else None
    report = {"split": args.split, "samples": len(loader.dataset),
              "max_background_fpr": args.max_background_fpr,
              "selected": selected, "thresholds": results}
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
