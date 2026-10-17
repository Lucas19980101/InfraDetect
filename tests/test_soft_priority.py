"""Regression checks for non-destructive wall/interference review ordering."""

from __future__ import annotations

import json

import cv2
import numpy as np

from infradetect.evidence.composite_assessment import assess
from infradetect.evidence.fusion import fuse_scene


def _png(path, pixels):
    """Write a small single-channel evidence mask in a temporary test scene."""
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.asarray(pixels, dtype=np.uint8))


def test_soft_evidence_preserves_original_thermal_candidates(tmp_path):
    """A low wall score and high AC score must never erase the IR candidate."""
    thermal = np.zeros((5, 5), np.float32)
    thermal[1, 1] = 0.8
    thermal[3, 3] = 0.9
    np.save(tmp_path / "thermal.npy", thermal)
    component = np.zeros((5, 5), np.uint8)
    _png(tmp_path / "component.png", component)
    wall = np.ones((5, 5), np.float32)
    wall[3, 3] = 0.1
    ac = np.zeros((5, 5), np.float32)
    ac[3, 3] = 1
    np.save(tmp_path / "wall.npy", wall)
    np.save(tmp_path / "ac.npy", ac)
    np.save(tmp_path / "identity.npy", np.eye(3))
    scene = {"id": "scene", "source_dataset": "synthetic", "thermal_model": "test",
             "thermal_probability": "thermal.npy", "registration_accepted": True,
             "component_ids_rgb": "component.png", "rgb_to_thermal": "identity.npy",
             "wall_probability_rgb": "wall.npy", "interference_probability_rgb": "ac.npy"}
    maps, _ = fuse_scene(scene, tmp_path, 0.5, 0, 0.78)
    assert int(maps["thermal_anomaly_mask.png"].sum()) == 2
    np.testing.assert_array_equal(maps["thermal_anomaly_probability.npy"], thermal)
    assert maps["wall_probability_in_ir.npy"][3, 3] == 0.1
    assert maps["interference_probability_in_ir.npy"][3, 3] == 1


def test_review_priority_changes_order_without_changing_regions(tmp_path):
    """A hotter interfering region may move down, but stays in the report."""
    source = tmp_path / "source"
    primary = source / "fusion" / "scene"
    diagnostic = source / "fusion_diagnostic" / "scene"
    for directory in (primary, diagnostic):
        _png(directory / "thermal_anomaly_mask.png", np.array(
            [[1, 0, 0], [0, 0, 0], [0, 0, 1]], np.uint8))
        np.save(directory / "thermal_anomaly_probability.npy",
                np.array([[0.8, 0, 0], [0, 0, 0], [0, 0, 0.9]], np.float32))
    for name, array in {
        "valid_thermal_mask.png": np.ones((3, 3)),
        "wall_core_in_ir.png": np.ones((3, 3)),
        "registration_coverage_in_ir.png": np.ones((3, 3)),
        "equipment_mask_in_ir.png": np.zeros((3, 3)),
    }.items():
        _png(primary / name, array)
    np.save(primary / "wall_probability_in_ir.npy",
            np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], np.float32))
    np.save(primary / "interference_probability_in_ir.npy",
            np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], np.float32))
    _png(source / "cross_modal" / "rgb_defect_bits_in_ir.png", np.zeros((3, 3)))
    _png(source / "cross_modal" / "rgb_defect_near_bits_in_ir.png", np.zeros((3, 3)))
    np.save(tmp_path / "stack.npy", np.zeros((3, 3, 5), np.uint8))
    (source / "report.json").write_text(json.dumps({
        "scene": {"scene_id": "scene", "shape": [3, 3], "image": str(tmp_path / "stack.npy"),
                  "source_dataset": "synthetic"},
        "primary_thermal": {"input_mode": "rgb_thermal"},
        "diagnostic_thermal": {"input_mode": "thermal"},
        "three_dimensional_projection": "not_run",
    }), encoding="utf-8")
    result = assess(source, tmp_path / "result")
    assert result["candidate_regions"] == result["ranked_regions"] == 2
    first, second = result["region_assessments"]
    assert first["raw_ir_score"] < second["raw_ir_score"]
    assert first["review_priority"] > second["review_priority"] > 0
    assert [first["review_rank"], second["review_rank"]] == [1, 2]
    assert first["diagnosis"] == second["diagnosis"] == "unverified_thermal_anomaly_candidate"
