import pytest
import torch

from rgb_multilabel.losses import MaskedBCEWithLogitsDiceLoss
from rgb_multilabel.metrics import MultilabelSegmentationMetrics, PerClassThresholdTuner


def test_masked_loss_ignores_invalid_pixels_and_backpropagates():
    criterion = MaskedBCEWithLogitsDiceLoss(classes=2)
    target = torch.zeros(1, 2, 3, 3)
    valid = torch.ones(1, 1, 3, 3, dtype=torch.bool)
    valid[:, :, 0, 0] = False
    logits_a = torch.zeros(1, 2, 3, 3, requires_grad=True)
    logits_b = logits_a.detach().clone()
    logits_b[:, :, 0, 0] = 100
    assert criterion(logits_a, target, valid).item() == pytest.approx(
        criterion(logits_b, target, valid).item()
    )
    criterion(logits_a, target, valid).backward()
    assert logits_a.grad is not None
    assert torch.all(logits_a.grad[:, :, 0, 0] == 0)


def test_multilabel_metrics_are_perfect_and_ignore_invalid_pixels():
    target = torch.tensor([[[[1, 0]], [[1, 0]]]], dtype=torch.bool)
    prediction = target.clone()
    prediction[:, :, :, 1] = ~prediction[:, :, :, 1]
    valid = torch.tensor([[[[1, 0]]]], dtype=torch.bool)
    metrics = MultilabelSegmentationMetrics(classes=2)
    metrics.update(prediction, target, valid)
    result = metrics.compute()
    assert result["mean_iou"] == 1.0


def test_threshold_tuner_selects_per_class_thresholds():
    probabilities = torch.tensor(
        [[[[0.9, 0.8, 0.4, 0.1]], [[0.8, 0.6, 0.3, 0.2]]]]
    )
    logits = torch.logit(probabilities)
    target = torch.tensor([[[[1, 1, 0, 0]], [[1, 0, 0, 0]]]], dtype=torch.float32)
    valid = torch.ones(1, 1, 1, 4, dtype=torch.bool)
    tuner = PerClassThresholdTuner(classes=2, bins=10)
    tuner.update(logits, target, valid)
    result = tuner.compute()
    assert result["dice"].tolist() == pytest.approx([1.0, 1.0])
    assert result["thresholds"][0] != result["thresholds"][1]
