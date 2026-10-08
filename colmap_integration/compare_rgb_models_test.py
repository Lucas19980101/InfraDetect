#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for common 2D/3D evaluation calculations."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from compare_rgb_models import binary_metrics, evaluate_point_cloud, parse_named_paths


def write_ply(path: Path, points: np.ndarray) -> None:
    kinds = {(np.dtype("u1").kind, 1): "uchar",
             (np.dtype("u2").kind, 2): "ushort",
             (np.dtype("f4").kind, 4): "float"}
    with path.open("wb") as handle:
        handle.write(b"ply\nformat binary_little_endian 1.0\n")
        handle.write(f"element vertex {len(points)}\n".encode())
        for name in points.dtype.names:
            field_type = points.dtype[name]
            handle.write(f"property {kinds[(field_type.kind, field_type.itemsize)]} {name}\n".encode())
        handle.write(b"end_header\n")
        handle.write(points.tobytes())


class CompareRgbModelsTest(unittest.TestCase):
    def test_binary_metrics(self) -> None:
        result = binary_metrics(2, 1, 1)
        self.assertAlmostEqual(result["iou"], 0.5)
        self.assertAlmostEqual(result["dice"], 2 / 3)
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 2 / 3)

    def test_named_paths_reject_duplicates(self) -> None:
        with self.assertRaises(ValueError):
            parse_named_paths(["segformer=a.pt", "segformer=b.pt"])

    def test_point_consistency_and_accuracy(self) -> None:
        prediction_dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("label_bits", "u1"), ("valid_views", "<u2"),
            ("positive_views_crack", "<u2"), ("positive_views_spalling", "<u2"),
        ])
        truth_dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("label_bits", "u1"), ("valid_bits", "u1"),
        ])
        prediction = np.zeros(3, dtype=prediction_dtype)
        prediction["x"] = [0, 1, 2]
        prediction["label_bits"] = [1, 0, 2]
        prediction["valid_views"] = [3, 3, 2]
        prediction["positive_views_crack"] = [2, 1, 0]
        prediction["positive_views_spalling"] = [0, 0, 2]
        truth = np.zeros(3, dtype=truth_dtype)
        truth["x"] = [0, 1, 2]
        truth["label_bits"] = [1, 1, 0]
        truth["valid_bits"] = [3, 3, 1]  # Spalling at point 2 is unknown.
        with tempfile.TemporaryDirectory() as temporary:
            pred_path = Path(temporary) / "pred.ply"
            truth_path = Path(temporary) / "truth.ply"
            write_ply(pred_path, prediction)
            write_ply(truth_path, truth)
            result = evaluate_point_cloud(pred_path, ("crack", "spalling"), truth_path, 2)
        self.assertEqual(result["status"], "ground_truth_and_consistency")
        self.assertEqual(result["point_accuracy"]["crack"]["tp"], 1)
        self.assertEqual(result["point_accuracy"]["crack"]["fn"], 1)
        self.assertEqual(result["point_accuracy"]["spalling"]["supervised_points"], 2)
        self.assertEqual(result["point_accuracy"]["spalling"]["fp"], 0)
        self.assertAlmostEqual(result["consistency"]["crack"]["point_view_agreement"], 6 / 8)


if __name__ == "__main__":
    unittest.main()
