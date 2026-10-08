import numpy as np
import pytest
import torch

from rgb_baseline.inference import positions, sliding_window_predict
from rgb_baseline.losses import CrossEntropyDiceLoss
from rgb_baseline.metrics import SegmentationMetrics, relaxed_binary_f1


class ToyModel(torch.nn.Module):
    def forward(self, x, return_features=False):
        logits = torch.stack((x[:, 0], -x[:, 0], torch.zeros_like(x[:, 0])), 1)
        return {"logits": logits, "decoder": x} if return_features else logits


def test_positions_cover_last_pixel():
    assert positions(2500, 1024, 768) == [0, 768, 1476]


def test_loss_ignores_255_and_backpropagates():
    logits = torch.randn(1, 3, 4, 4, requires_grad=True); target = torch.zeros(1, 4, 4, dtype=torch.long); target[0, 0, 0] = 255
    loss = CrossEntropyDiceLoss(3)(logits, target); loss.backward()
    assert torch.isfinite(loss) and logits.grad is not None


def test_metrics_perfect():
    target = torch.tensor([[0, 1], [2, 255]]); metric = SegmentationMetrics(3); metric.update(target, target)
    assert metric.compute()["mean_iou"] == 1.0
    assert relaxed_binary_f1(target, target) == 1.0


def test_relaxed_f1_tolerance_and_empty_masks():
    target = torch.zeros(24, 24, dtype=torch.long); target[4:20, 8] = 1
    for shift in range(4):
        prediction = torch.zeros_like(target); prediction[4:20, 8 + shift] = 1
        assert relaxed_binary_f1(prediction, target, radius=3) == pytest.approx(1.0)
    prediction = torch.zeros_like(target); prediction[4:20, 12] = 1
    assert relaxed_binary_f1(prediction, target, radius=3) < 1.0

    empty = torch.zeros_like(target)
    assert np.isnan(relaxed_binary_f1(empty, empty))
    assert relaxed_binary_f1(target, empty) == 0.0
    assert relaxed_binary_f1(empty, target) == 0.0


def test_relaxed_f1_ignores_predictions_in_ignored_regions():
    target = torch.zeros(16, 16, dtype=torch.long); target[:, 8:] = 255
    prediction = torch.zeros_like(target); prediction[:, 8:] = 1
    assert np.isnan(relaxed_binary_f1(prediction, target, ignore_index=255))


def test_sliding_output_keeps_original_shape():
    image = np.full((19, 23, 3), 128, np.uint8)
    logits = sliding_window_predict(ToyModel(), image, 16, 12, 3, torch.device("cpu"), 2)
    assert logits.shape == (3, 19, 23) and torch.isfinite(logits).all()
