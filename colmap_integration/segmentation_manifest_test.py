#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for external image/mask manifest loading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from segmentation_manifest import SegmentationManifestDataset


class SegmentationManifestTest(unittest.TestCase):
    def test_missing_classes_are_unsupervised(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            cv2.imwrite(str(folder / "image.png"), np.zeros((8, 8, 3), dtype=np.uint8))
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[:4] = 255
            cv2.imwrite(str(folder / "crack.png"), mask)
            (folder / "manifest.csv").write_text(
                "split,image,crack,spalling\n"
                "test,image.png,crack.png,\n", encoding="utf-8"
            )
            dataset = SegmentationManifestDataset(folder / "manifest.csv", "test", 8)
            sample = dataset[0]
            self.assertEqual(tuple(sample["target"].shape), (6, 8, 8))
            self.assertEqual(int(sample["target"][0].sum()), 32)
            self.assertEqual(int(sample["valid_mask"][0].sum()), 64)
            self.assertEqual(int(sample["valid_mask"][1].sum()), 0)


if __name__ == "__main__":
    unittest.main()
