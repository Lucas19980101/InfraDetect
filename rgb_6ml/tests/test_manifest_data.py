# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from rgb_multilabel.manifest_data import PartialLabelManifestDataset


def test_partial_manifest_preserves_overlap_and_missing_classes(tmp_path: Path):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    crack = np.zeros((8, 8), dtype=np.uint8)
    spalling = np.zeros((8, 8), dtype=np.uint8)
    crack[1:4, 1:4] = 255
    spalling[2:6, 2:6] = 255
    cv2.imwrite(str(tmp_path / "sample.jpg"), image)
    cv2.imwrite(str(tmp_path / "crack.png"), crack)
    cv2.imwrite(str(tmp_path / "spalling.png"), spalling)
    fields = ["split", "image", "crack", "spalling", "exposed_rebar",
              "rust", "efflorescence", "wetspot"]
    with (tmp_path / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"split": "train", "image": "sample.jpg", "crack": "crack.png",
                         "spalling": "spalling.png", "exposed_rebar": "", "rust": "",
                         "efflorescence": "", "wetspot": ""})
    dataset = PartialLabelManifestDataset(
        tmp_path / "manifest.csv", "train", 8, False, "test"
    )
    sample = dataset[0]
    assert sample["target"][:, 2, 2].tolist()[:2] == [1.0, 1.0]
    assert sample["valid_mask"][:, 0, 0].tolist() == [True, True, False, False, False, False]
