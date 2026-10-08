#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from prepare_damcrack import prepare, stable_split


class PrepareDamCrackTest(unittest.TestCase):
    def test_stable_split_is_repeatable(self):
        self.assertEqual(stable_split("A17", "seed", 0.8, 0.1),
                         stable_split("A17", "seed", 0.8, 0.1))

    def test_prepare_preserves_two_labels_and_missing_classes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "Damage Segmentaion"
            images = root / "Images"
            masks = root / "Refined Labels" / "Mask"
            images.mkdir(parents=True)
            masks.mkdir(parents=True)
            image = np.zeros((4, 5, 3), dtype=np.uint8)
            mask = image.copy()
            mask[0, 0] = (0, 0, 255)
            mask[1, 1] = (255, 0, 0)
            cv2.imwrite(str(images / "A1.jpg"), image)
            cv2.imwrite(str(masks / "A1.png"), mask)

            output = Path(temporary) / "prepared"
            summary = prepare(root, output, "seed", 0.8, 0.1)
            self.assertEqual(summary["multi_label_images"], 1)
            with (output / "manifest.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["exposed_rebar"], "")
            crack = cv2.imread(str(output / "masks/crack/A1.png"), 0)
            spalling = cv2.imread(str(output / "masks/spalling/A1.png"), 0)
            self.assertEqual(int(crack[0, 0]), 255)
            self.assertEqual(int(spalling[1, 1]), 255)

    def test_unknown_color_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "Damage Segmentaion"
            images = root / "Images"
            masks = root / "Refined Labels" / "Mask"
            images.mkdir(parents=True)
            masks.mkdir(parents=True)
            image = np.zeros((2, 2, 3), dtype=np.uint8)
            cv2.imwrite(str(images / "A1.jpg"), image)
            image[0, 0] = (0, 255, 0)
            cv2.imwrite(str(masks / "A1.png"), image)
            with self.assertRaisesRegex(ValueError, "Unknown BGR colors"):
                prepare(root, Path(temporary) / "prepared", "seed", 0.8, 0.1)


if __name__ == "__main__":
    unittest.main()
