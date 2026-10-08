import torch

from rgb_multilabel.losses import MaskedBCEWithLogitsDiceLoss


def test_class_level_partial_mask_has_no_gradient_for_unlabeled_classes():
    logits = torch.zeros(1, 6, 2, 2, requires_grad=True)
    target = torch.zeros_like(logits)
    valid = torch.zeros_like(logits, dtype=torch.bool)
    valid[:, 0] = True
    loss = MaskedBCEWithLogitsDiceLoss(6)(logits, target, valid)
    loss.backward()
    assert logits.grad[:, 0].abs().sum() > 0
    assert logits.grad[:, 1:].abs().sum() == 0
