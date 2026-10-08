#!/usr/bin/env python3
"""Build the ReDFeat VIS-IR test benchmark without modifying source datasets."""

import argparse
import io
import json
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
import scipy.io as scio
from PIL import Image


def read_transform(archive, index):
    for suffix in ("21", "12"):
        member = f"test/transforms/{index}.{suffix}.mat"
        try:
            data = archive.read(member)
        except KeyError:
            continue
        return suffix, scio.loadmat(io.BytesIO(data))["H"]
    raise FileNotFoundError(f"No transform found for benchmark item {index}")


def load_pair(name, roadscene, lghd):
    if name.startswith("FLIR"):
        return (
            Image.open(roadscene / "crop_LR_visible" / name).convert("RGB"),
            Image.open(roadscene / "cropinfrared" / name).convert("RGB"),
            "RoadScene",
        )
    if name.startswith("lwir") and lghd is not None:
        vis_name = name.replace("lwir", "rgb").replace(".png", ".bmp")
        return (
            Image.open(lghd / "rgb" / vis_name).convert("RGB"),
            Image.open(lghd / "lwir" / name).convert("RGB"),
            "LGHD",
        )
    raise FileNotFoundError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True,
                        help="Path to Multimodal_Feature_Evaluation/VIS_IR")
    parser.add_argument("--roadscene", type=Path, required=True)
    parser.add_argument("--lghd", type=Path,
                        help="Optional extracted lghd_icip2015_rgb_lwir directory")
    parser.add_argument("--output", type=Path, default=Path("VIS_IR"))
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite {args.output}; move it aside or choose another output")

    test_names = [line.strip() for line in
                  (args.benchmark / "test.txt").read_text().splitlines() if line.strip()]
    vis_dir = args.output / "test" / "VIS"
    ir_dir = args.output / "test" / "IR"
    transform_dir = args.output / "test" / "transforms"
    landmark_dir = args.output / "test" / "landmarks"
    for directory in (vis_dir, ir_dir, transform_dir, landmark_dir):
        directory.mkdir(parents=True, exist_ok=True)

    built = []
    skipped = []
    with zipfile.ZipFile(args.benchmark / "transforms.zip") as transforms, \
            zipfile.ZipFile(args.benchmark / "landmarks.zip") as landmarks:
        for index, name in enumerate(test_names, start=1):
            try:
                vis, ir, source = load_pair(name, args.roadscene, args.lghd)
            except FileNotFoundError:
                skipped.append({"index": index, "source_name": name})
                continue

            suffix, homography = read_transform(transforms, index)
            if suffix == "21":
                array = np.asarray(ir)
                ir = Image.fromarray(cv2.warpPerspective(
                    array, homography, (array.shape[1], array.shape[0])))
            else:
                array = np.asarray(vis)
                vis = Image.fromarray(cv2.warpPerspective(
                    array, homography, (array.shape[1], array.shape[0])))

            filename = f"{index}.png"
            vis.save(vis_dir / filename)
            ir.save(ir_dir / filename)
            (transform_dir / f"{index}.{suffix}.mat").write_bytes(
                transforms.read(f"test/transforms/{index}.{suffix}.mat"))
            landmark_member = f"landmarks/{index}.lms.mat"
            (landmark_dir / f"{index}.lms.mat").write_bytes(
                landmarks.read(landmark_member))
            built.append({
                "index": index,
                "source_name": name,
                "source_dataset": source,
                "transform_suffix": suffix,
            })

    shutil.copy2(args.benchmark / "test.txt", args.output / "test.txt")
    manifest = {
        "benchmark": "ReDFeat Multimodal Feature Evaluation VIS-IR",
        "built_pairs": len(built),
        "skipped_pairs": len(skipped),
        "items": built,
        "skipped": skipped,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Built {len(built)} pairs in {args.output}; skipped {len(skipped)}")


if __name__ == "__main__":
    main()
