#!/usr/bin/env python3
"""Evaluate RANSAC homographies from saved MFDN features."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.io import loadmat, savemat


def load_ground_truth(dataset, split, image_name, gt_format):
    if gt_format == "identity":
        return np.eye(3, dtype=np.float64)
    stem = Path(image_name).stem
    transforms = dataset / split / "transforms"
    transform_21 = transforms / f"{stem}.21.mat"
    if transform_21.exists():
        return loadmat(transform_21)["H"]
    transform_12 = transforms / f"{stem}.12.mat"
    if transform_12.exists():
        return np.linalg.inv(loadmat(transform_12)["H"])
    raise FileNotFoundError(f"No transform for {image_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--vis-dir", default="VIS")
    parser.add_argument("--feature-name", default="MFDN")
    parser.add_argument("--num-features", type=int, default=4096)
    parser.add_argument("--gt-format", choices=("identity", "redfeat"), required=True)
    parser.add_argument("--ransac-threshold", type=float, default=10.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    dataset = root / args.dataset
    image_dir = dataset / args.split / args.vis_dir
    feature_dir = root / "features" / args.dataset / args.feature_name
    names = sorted(path.name for path in image_dir.glob("*.png"))
    matcher = cv2.BFMatcher(crossCheck=True)
    records = []

    for image_name in names:
        feature_file = feature_dir / image_name.replace(".png", ".features.mat")
        features = loadmat(feature_file)
        desc1 = np.asarray(features["desc1"], dtype=np.float32)[:args.num_features]
        desc2 = np.asarray(features["desc2"], dtype=np.float32)[:args.num_features]
        kp1 = np.asarray(features["kp1"][:, :2], dtype=np.float32)[:args.num_features]
        kp2 = np.asarray(features["kp2"][:, :2], dtype=np.float32)[:args.num_features]
        matches = matcher.match(desc1, desc2)
        src = np.float32([kp1[match.queryIdx] for match in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp2[match.trainIdx] for match in matches]).reshape(-1, 1, 2)
        estimated, mask = cv2.findHomography(
            src, dst, cv2.RANSAC, args.ransac_threshold, maxIters=100000)
        if estimated is None:
            records.append((image_name, np.nan, np.nan, len(matches)))
            continue

        width, height = Image.open(image_dir / image_name).size
        corners = np.float32([
            [0, 0], [width - 1, 0],
            [width - 1, height - 1], [0, height - 1],
        ]).reshape(-1, 1, 2)
        expected = cv2.perspectiveTransform(
            corners, load_ground_truth(dataset, args.split, image_name, args.gt_format))
        predicted = cv2.perspectiveTransform(corners, estimated)
        corner_error = np.linalg.norm(predicted - expected, axis=2).mean()
        records.append((image_name, corner_error, float(mask.mean()), len(matches)))

    errors = np.array([record[1] for record in records], dtype=np.float64)
    valid = np.isfinite(errors)
    valid_errors = errors[valid]
    print(f"Dataset: {args.dataset}; evaluated: {valid.sum()}; failed: {(~valid).sum()}")
    print(f"Corner error mean/median/p90/max: {valid_errors.mean():.3f} / "
          f"{np.median(valid_errors):.3f} / {np.percentile(valid_errors, 90):.3f} / "
          f"{valid_errors.max():.3f} px")
    for threshold in (3, 5, 10, 20):
        print(f"Success <={threshold}px: {(valid_errors <= threshold).mean():.4f}")
    inlier_ratios = np.array([record[2] for record in records], dtype=np.float64)[valid]
    print(f"RANSAC inlier ratio mean/median: {inlier_ratios.mean():.4f} / "
          f"{np.median(inlier_ratios):.4f}")

    output_dir = root / "results" / args.dataset / args.feature_name
    output_dir.mkdir(parents=True, exist_ok=True)
    savemat(output_dir / "homography_evaluation.mat", {
        "image_names": np.array([record[0] for record in records], dtype=object),
        "corner_error": errors,
        "inlier_ratio": np.array([record[2] for record in records]),
        "match_count": np.array([record[3] for record in records]),
    })


if __name__ == "__main__":
    main()
