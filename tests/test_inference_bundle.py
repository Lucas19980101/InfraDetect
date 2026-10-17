"""CPU-only checks that the delivered inference code preserves native pixels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]

from infradetect.inference.rgb_predict import (predict_native_class_probabilities,
                                       predict_rgb_defects)
from infradetect.inference.thermal import predict_full


class EchoThermal(torch.nn.Module):
    """Tiny deterministic model for checking overlap reconstruction."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Expose channel 0 as class 1 without changing spatial dimensions."""

        return torch.cat((torch.zeros_like(images[:, :1]), images[:, :1]), dim=1)


class FlatRgb(torch.nn.Module):
    """Tiny deterministic RGB model with one constant output channel."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return full-size logits so patch coverage can be checked."""

        return torch.zeros((images.shape[0], 1, *images.shape[-2:]), device=images.device)


def test_thermal_overlap_preserves_native_values(tmp_path: Path):
    """Four-channel sliding-window blending reconstructs every thermal pixel."""

    source = np.zeros((7, 9, 4), dtype=np.uint8)
    source[..., 3] = np.arange(63, dtype=np.uint8).reshape(7, 9)
    image = tmp_path / "stack.npy"
    np.save(image, source)
    cfg = {"data": {"thermal_channel": 3, "image_size": [4, 5],
                    "normalization": {"mean": 0, "std": 1}, "tile_overlap": 0.25},
           "model": {"classes": 2}, "train": {"mixed_precision": False}}
    logits = predict_full(EchoThermal(), {"image": str(image)}, cfg, torch.device("cpu"))
    assert logits.shape == (2, 7, 9)
    np.testing.assert_allclose(logits[1].numpy(), source[..., 3], atol=1e-6)


def test_rgb_component_output_stays_full_size():
    """Component patches retain the H×W grid and original-pixel coordinates."""

    image = np.zeros((9, 11, 3), dtype=np.uint8)
    probabilities, patches = predict_native_class_probabilities(
        FlatRgb(), image, (4, 5), torch.device("cpu"), image_id="test")
    assert probabilities.shape == (1, 9, 11)
    assert all(item["image_id"] == "test" for item in patches)
    assert max(item["x0"] + item["w"] for item in patches) == 11
    assert max(item["y0"] + item["h"] for item in patches) == 9


def test_rgb_roi_outside_is_unobserved():
    """Pixels outside the wall ROI are coverage=False, not model negatives."""

    image = np.zeros((9, 11, 3), dtype=np.uint8)
    roi = np.zeros((9, 11), dtype=bool)
    roi[6:8, 8:10] = True
    state = {"config": {"data": {"class_names": ["crack"]}},
             "thresholds": {"crack": 0.5}}
    probabilities, masks, names, coverage, patches = predict_rgb_defects(
        FlatRgb(), state, image, torch.device("cpu"), image_id="test",
        patch_size=4, stride=3, roi=roi, return_coverage=True)
    assert probabilities.shape == (1, 9, 11)
    assert names == ["crack"]
    assert np.array_equal(coverage, roi)
    assert np.array_equal(masks[0], roi)
    assert any(not patch["evaluated"] for patch in patches)


def test_no_training_files_in_delivery():
    """The delivered layout groups algorithms by role and omits training code."""

    assert not (ROOT / "IR").exists()
    assert not (ROOT / "RGB").exists()
    assert not (ROOT / "infradetect" / "inference" / "train.py").exists()
    assert (ROOT / "infradetect" / "inference" / "rgb_defect_architectures.py").is_file()
    assert (ROOT / "infradetect" / "inference" / "config.py").is_file()
    assert (ROOT / "infradetect" / "evidence" / "export_rgb.py").is_file()
    assert (ROOT / "infradetect" / "evidence" / "fusion.py").is_file()
    assert (ROOT / "infradetect" / "geometry" / "project.py").is_file()
