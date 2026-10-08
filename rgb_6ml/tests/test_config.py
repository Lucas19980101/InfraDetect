from pathlib import Path

from rgb_multilabel.config import load_config


def test_model_configs_share_training_protocol():
    root = Path(__file__).parents[1]
    deeplab = load_config(root / "configs/deeplabv3plus_6ml.yaml")
    segformer = load_config(root / "configs/segformer_b0_6ml.yaml")
    assert deeplab["data"] == segformer["data"]
    deeplab_train = {k: v for k, v in deeplab["train"].items() if k != "output_dir"}
    segformer_train = {k: v for k, v in segformer["train"].items() if k != "output_dir"}
    assert deeplab_train == segformer_train
    assert deeplab["model"]["num_classes"] == segformer["model"]["num_classes"] == 6
