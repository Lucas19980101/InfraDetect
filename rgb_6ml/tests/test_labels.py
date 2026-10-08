import numpy as np

from rgb_multilabel.labels import CLASS_NAMES, rasterize_annotation


def polygon(label, x0, y0, x1, y1):
    return {"label": label, "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}


def test_six_channels_and_crack_union():
    annotation = {
        "imageWidth": 8,
        "imageHeight": 8,
        "shapes": [polygon("Crack", 1, 1, 2, 2), polygon("ACrack", 5, 5, 6, 6)],
    }
    targets, valid, _ = rasterize_annotation(annotation)
    assert targets.shape == (len(CLASS_NAMES), 8, 8)
    assert targets[0, 1, 1] and targets[0, 5, 5]
    assert valid.all()


def test_selected_positive_wins_over_other_label_ignore():
    annotation = {
        "imageWidth": 8,
        "imageHeight": 8,
        "shapes": [
            polygon("Rust", 2, 2, 5, 5),
            polygon("Graffiti", 0, 0, 3, 3),
        ],
    }
    targets, valid, _ = rasterize_annotation(annotation)
    rust = CLASS_NAMES.index("rust")
    assert targets[rust, 2, 2]
    assert valid[2, 2]
    assert not valid[0, 0]
    assert valid[7, 7]
    assert np.all(~targets[:, 0, 0])
