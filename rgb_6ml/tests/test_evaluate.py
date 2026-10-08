import numpy as np
import torch

from rgb_multilabel.evaluate import comparison_image, per_sample_counts


def test_per_sample_counts_respect_valid_mask():
    target = torch.zeros(6, 1, 2, dtype=torch.bool)
    prediction = torch.zeros_like(target)
    target[0, 0, 0] = True
    prediction[0, 0] = True
    target[1, 0, 1] = True
    valid = torch.tensor([[[1, 0]]], dtype=torch.bool)
    counts = per_sample_counts(prediction, target, valid)
    assert counts["tp_crack"] == 1
    assert counts["fp_crack"] == 0
    assert counts["fn_spalling"] == 0


def test_comparison_has_four_panels():
    image = torch.zeros(3, 8, 9)
    masks = np.zeros((6, 8, 9), dtype=bool)
    valid = np.ones((8, 9), dtype=bool)
    result = comparison_image(image, masks, masks, valid)
    assert result.shape == (8, 9 * 4, 3)
