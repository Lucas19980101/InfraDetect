from pathlib import Path
from collections import Counter

import numpy as np
import torch

from seg_tasks.data import (SegmentationDataset, component_pairs, load_component_mask,
                            load_ir_mask, resize_and_pad, restore_mask)
from seg_tasks.filter import filter_defects, partition_defects
from seg_tasks.metrics import Metrics, ce_dice
from seg_tasks.splits import bfdd_rows, bfdd_specialized_rows


ROOT = Path(__file__).resolve().parents[2] / "collected_datasets"


def test_real_data_pairing_and_split():
    pairs = component_pairs(ROOT / "3DOM")
    assert len(pairs) == 227
    assert load_component_mask(pairs[0][1]).ndim == 2
    rows = bfdd_rows(ROOT / "Dataset_1x")
    assert len(rows) == 838
    assert {row["split"] for row in rows} == {"train", "val", "test"}
    dataset = SegmentationDataset(
        ROOT / "3DOM", Path(__file__).resolve().parents[1] / "data/3dom_split.csv",
        "train", "components", (128, 192), 5)
    rotated_index = next(i for i, (image, _) in enumerate(dataset.pairs) if image.stem == "IMG_1233")
    assert tuple(dataset[rotated_index]["mask"].shape) == (128, 192)


def test_padding_round_trip_and_filter():
    image = np.zeros((8, 12, 3), np.uint8)
    mask = np.zeros((8, 12), np.uint8)
    mask[:, 6:] = 1
    _, padded = resize_and_pad(image, mask, 16, 32)
    assert np.array_equal(restore_mask(padded, (8, 12)), mask)
    defects = np.ones((8, 12), np.uint8)
    filtered, wall = filter_defects(defects, mask, np.eye(3), defects.shape)
    assert np.array_equal(filtered, (mask == 0).astype(np.uint8))
    assert np.array_equal(wall, filtered)


def test_three_zone_filter_preserves_edge_candidates():
    defects = np.ones((9, 9), np.uint8)
    components = np.full((9, 9), 2, np.uint8)
    components[2:7, 2:7] = 0
    zones = partition_defects(defects, components, np.eye(3), defects.shape, edge_radius=1)
    assert zones["confirmed"][4, 4] == 1
    assert zones["uncertain"][2, 2] == 1
    assert zones["uncertain"][1, 2] == 1
    assert zones["rejected"][0, 0] == 1
    assert np.array_equal((zones["confirmed"] != 0) | (zones["uncertain"] != 0),
                          zones["lenient"] != 0)


def test_loss_and_metrics_ignore_boundary():
    logits = torch.randn(1, 3, 2, 2, requires_grad=True)
    target = torch.tensor([[[0, 1], [2, 255]]])
    loss = ce_dice(logits, target)
    loss.backward()
    assert torch.isfinite(loss) and logits.grad is not None
    perfect = torch.nn.functional.one_hot(target.masked_fill(target == 255, 0), 3).permute(0, 3, 1, 2).float()
    metrics = Metrics(3)
    metrics.update(perfect, target)
    assert metrics.compute(["a", "b", "c"])["mean_iou"] == 1.0


def test_rotation_augmentation_preserves_label_ids():
    dataset = SegmentationDataset(
        ROOT / "3DOM", Path(__file__).resolve().parents[1] / "data/3dom_split.csv",
        "train", "components", (128, 192), 5, augment=True,
        rotation_degrees=10, rotation_probability=1.0)
    sample = dataset[0]
    assert tuple(sample["mask"].shape) == (128, 192)
    assert set(sample["mask"].unique().tolist()) <= {0, 1, 2, 3, 4, 255}


def test_specialized_ir_split_keeps_sessions_disjoint():
    rows = bfdd_specialized_rows(ROOT / "Dataset_1x")
    assert Counter(row["split"] for row in rows) == Counter(train=570, val=44, test=224)
    groups = {split: {row["group"] for row in rows if row["split"] == split}
              for split in ("train", "val", "test")}
    assert groups["train"].isdisjoint(groups["val"] | groups["test"])
    assert groups["val"].isdisjoint(groups["test"])


def test_specialized_ir_label_map_ignores_other_defects(tmp_path):
    path = tmp_path / "mask.png"
    import cv2
    cv2.imwrite(str(path), np.arange(6, dtype=np.uint8)[None])
    mapped = load_ir_mask(path, 3, {0: 0, 3: 1, 4: 2})
    assert mapped.tolist() == [[0, 255, 255, 1, 2, 255]]
