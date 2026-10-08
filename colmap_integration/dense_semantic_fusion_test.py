#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for dense semantic projection and fusion."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dense_semantic_fusion import (
    PLY_FIELDS,
    fuse_labels,
    project_original,
    read_colmap_ply,
    sample_probability,
    visible_indices,
    write_labeled_ply,
)


class DenseSemanticFusionTest(unittest.TestCase):
    def test_original_simple_radial_projection(self) -> None:
        camera = SimpleNamespace(model="SIMPLE_RADIAL", params=[100.0, 50.0, 40.0, 0.1])
        u, v = project_original(camera, np.asarray([[1.0, 0.0, 2.0]]))
        self.assertAlmostEqual(float(u[0]), 101.25)
        self.assertAlmostEqual(float(v[0]), 40.0)

    def test_depth_visibility_rejects_occluded_and_offscreen_points(self) -> None:
        camera = SimpleNamespace(model="PINHOLE", params=[1.0, 1.0, 0.0, 0.0])
        points = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [3.0, 0.0, 1.0]])
        depth = np.ones((2, 2), dtype=np.float32)
        indexes, xyz = visible_indices(points, np.eye(4), camera, depth, 0.02)
        np.testing.assert_array_equal(indexes, [0])
        np.testing.assert_array_equal(xyz, [[0.0, 0.0, 1.0]])

    def test_probability_sampling_and_multiview_label(self) -> None:
        probability = np.asarray([[0, 65535], [65535, 0]], dtype=np.uint16)
        value = sample_probability(probability, np.asarray([0.5]), np.asarray([0.5]), 2, 2)
        self.assertAlmostEqual(float(value[0]), 0.5)
        labels, accepted, means, support = fuse_labels(
            np.asarray([[1.8, 1.8], [0.9, 0.0]], dtype=np.float32),
            np.asarray([[1.7, 1.7], [0.9, 0.0]], dtype=np.float32),
            np.asarray([[2, 2], [1, 0]], dtype=np.uint16),
            np.asarray([3, 3], dtype=np.uint16),
            np.asarray([0.6, 0.6], dtype=np.float32), 2, 0.5,
        )
        np.testing.assert_array_equal(labels, [1, 0])
        np.testing.assert_array_equal(accepted, [[True, True], [False, False]])
        self.assertAlmostEqual(float(means[0, 0]), 0.6)
        self.assertAlmostEqual(float(support[0, 0]), 0.85)

    def test_colmap_ply_reader_and_output_properties(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "fused.ply"
            fields = np.dtype(list(PLY_FIELDS))
            points = np.zeros(2, dtype=fields)
            points["x"] = [1.0, 2.0]
            points["red"] = 12
            with source.open("wb") as handle:
                handle.write(b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n")
                for name, kind in PLY_FIELDS:
                    handle.write(f"property {'float' if kind == '<f4' else 'uchar'} {name}\n".encode())
                handle.write(b"end_header\n")
                handle.write(points.tobytes())
            mapped = read_colmap_ply(source)
            self.assertEqual(len(mapped), 2)
            output = folder / "labeled.ply"
            write_labeled_ply(
                output, mapped, np.asarray([1, 0], dtype=np.uint8),
                np.asarray([[True, True], [False, False]]),
                np.asarray([[0.7, 0.8], [0.1, 0.2]], dtype=np.float32),
                np.asarray([[0.9, 0.9], [0.0, 0.0]], dtype=np.float32),
                np.asarray([[2, 2], [0, 0]], dtype=np.uint16),
                np.asarray([3, 1], dtype=np.uint16), ("crack", "spalling"), 1,
            )
            with output.open("rb") as handle:
                contents = handle.read()
            header, body = contents.split(b"end_header\n", 1)
            self.assertIn(b"property uchar label", header)
            self.assertIn(b"property uchar label_bits", header)
            self.assertIn(b"property uchar is_crack", header)
            self.assertIn(b"property uchar is_spalling", header)
            self.assertIn(b"property float prob_crack", header)
            self.assertIn(b"property ushort positive_views_crack", header)
            output_dtype = np.dtype([*PLY_FIELDS, ("label", "u1"), ("label_bits", "u1"),
                                     ("valid_views", "<u2"), ("is_crack", "u1"),
                                     ("is_spalling", "u1"), ("prob_crack", "<f4"),
                                     ("prob_spalling", "<f4"), ("support_crack", "<f4"),
                                     ("support_spalling", "<f4"),
                                     ("positive_views_crack", "<u2"),
                                     ("positive_views_spalling", "<u2")])
            loaded = np.frombuffer(body, dtype=output_dtype)
            self.assertEqual(int(loaded["label_bits"][0]), 3)
            self.assertEqual(int(loaded["label_bits"][1]), 0)
            self.assertEqual(int(loaded["is_spalling"][0]), 1)
            self.assertGreater(output.stat().st_size, source.stat().st_size)


if __name__ == "__main__":
    unittest.main()
