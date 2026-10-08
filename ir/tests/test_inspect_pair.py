import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from seg_tasks.audit_components_bfdd import sample_rows
from seg_tasks.data import restore_mask
from seg_tasks.inspect_pair import (compare_defects, component_palette, fusion_overlay,
                                    filter_binary_components, linear_artifact_candidates,
                                    ir_palette_and_names, map_ir_truth, overlay, prepare)


class InspectPairTests(unittest.TestCase):
    def test_audit_sample_is_validation_only_and_deterministic(self):
        manifest = Path(__file__).resolve().parents[1] / "data/bfdd_split.csv"
        rows = sample_rows(manifest, 10)
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(row["split"] == "val" for row in rows))
        self.assertEqual(rows, sample_rows(manifest, 10))
        self.assertGreater(len({row["group"] for row in rows}), 1)

    def test_prepare_matches_training_resize_and_pad(self):
        image = np.full((20, 40, 3), (10, 80, 200), dtype=np.uint8)
        tensor = prepare(image, (32, 32))
        self.assertEqual(tuple(tensor.shape), (1, 3, 32, 32))
        self.assertTrue(np.isfinite(tensor.numpy()).all())
        mask = restore_mask(np.zeros((32, 32), np.uint8), image.shape[:2])
        self.assertEqual(mask.shape, image.shape[:2])

    def test_compare_separates_removed_true_and_false_positives(self):
        truth = np.array([[1, 0, 2, 255]], dtype=np.uint8)
        before = np.array([[1, 3, 3, 4]], dtype=np.uint8)
        after = np.array([[0, 0, 3, 0]], dtype=np.uint8)
        result = compare_defects(before, after, truth)
        self.assertEqual(result["removed_correct_class_pixels"], 1)
        self.assertEqual(result["removed_wrong_class_or_background_pixels"], 1)
        self.assertEqual(result["binary_false_positive_before"], 1)
        self.assertEqual(result["binary_false_positive_after"], 0)
        self.assertEqual(result["binary_true_positive_before"], 2)
        self.assertEqual(result["binary_true_positive_after"], 1)

    def test_overlay_preserves_unselected_pixels(self):
        image = np.full((2, 2, 3), 100, dtype=np.uint8)
        mask = np.array([[0, 1], [1, 0]], np.uint8)
        palette = np.array([(0, 0, 0), (255, 0, 0)], np.uint8)
        result = overlay(image, mask, palette, mask != 0)
        self.assertTrue(np.array_equal(result[0, 0], image[0, 0]))
        self.assertFalse(np.array_equal(result[0, 1], image[0, 1]))

    def test_three_class_palette_and_fusion_overlay(self):
        palette = component_palette(["building_surface", "opening", "other"])
        self.assertEqual(palette.tolist(), [[0, 0, 255], [0, 255, 0], [255, 255, 0]])
        image = np.zeros((1, 3, 3), np.uint8)
        confirmed = np.array([[1, 0, 0]], np.uint8)
        uncertain = np.array([[0, 1, 0]], np.uint8)
        rejected = np.array([[0, 0, 1]], np.uint8)
        result = fusion_overlay(image, confirmed, uncertain, rejected)
        self.assertFalse(np.array_equal(result[0, 0], result[0, 1]))
        self.assertTrue(np.all(result[0, 2] > 0))

    def test_linear_artifact_accepts_single_flat_hough_line(self):
        defect = np.ones((8, 8), np.uint8)
        evidence = np.ones_like(defect)
        with patch("cv2.HoughLinesP", return_value=np.array([[1, 1, 6, 1]], np.int32)):
            result = linear_artifact_candidates(defect, evidence, defect,
                                                min_length=2, max_gap=1, thickness=1)
        self.assertGreater(int(result.sum()), 0)

    def test_component_filter_drops_border_and_small_regions(self):
        mask = np.zeros((12, 12), np.uint8)
        mask[2:7, 2:7] = 1
        mask[0:5, 9:12] = 1
        mask[9:11, 2:4] = 1
        result = filter_binary_components(mask, min_area=10, drop_border=True)
        self.assertEqual(int(result.sum()), 25)

    def test_component_filter_only_drops_narrow_side_regions(self):
        mask = np.zeros((20, 30), np.uint8)
        mask[2:18, 0:3] = 1
        mask[2:18, 20:30] = 1
        result = filter_binary_components(mask, drop_narrow_side_border_width=4)
        self.assertEqual(int(result[:, :3].sum()), 0)
        self.assertEqual(int(result[:, 20:].sum()), 160)

    def test_ir_metadata_and_truth_mapping_support_specialized_classes(self):
        cfg = {"model": {"classes": 3}, "data": {
            "class_names": ["Background", "Hollow Areas", "Stains"],
            "palette_rgb": [[0, 0, 0], [174, 79, 13], [36, 179, 83]],
            "label_map": {0: 0, 3: 1, 4: 2},
        }}
        palette, names = ir_palette_and_names(cfg)
        self.assertEqual(names, ["Background", "Hollow Areas", "Stains"])
        self.assertEqual(palette.shape, (3, 3))
        truth = np.arange(6, dtype=np.uint8)[None]
        self.assertEqual(map_ir_truth(truth, cfg).tolist(), [[0, 255, 255, 1, 2, 255]])

    def test_ir_metadata_preserves_legacy_six_class_fallback(self):
        names = ["Background", "Cracks", "Peeling", "Hollow Areas", "Stains", "Erosion"]
        cfg = {"model": {"classes": 6}, "data": {"class_names": names}}
        palette, actual = ir_palette_and_names(cfg)
        self.assertEqual(actual, names)
        self.assertEqual(palette.shape, (6, 3))


if __name__ == "__main__":
    unittest.main()
