import pytest
import torch

from rgb_multilabel.models import build_model


@pytest.mark.parametrize(
    "config",
    [
        {"architecture": "deeplabv3plus", "encoder": "resnet50", "num_classes": 6},
        {"architecture": "segformer_b0", "num_classes": 6},
    ],
)
def test_model_output_and_backward(config):
    if config["architecture"] == "segformer_b0":
        pytest.importorskip("transformers")
    model = build_model(config, pretrained=False)
    model.train()
    images = torch.randn(2, 3, 128, 128)
    logits = model(images)
    assert logits.shape == (2, 6, 128, 128)
    logits.mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_model_feature_contract():
    model = build_model(
        {"architecture": "deeplabv3plus", "encoder": "resnet50", "num_classes": 6},
        pretrained=False,
    )
    output = model(torch.randn(2, 3, 128, 128), return_features=True)
    assert output["logits"].shape == (2, 6, 128, 128)
    assert "encoder" in output and "decoder" in output
