import numpy as np

from rgb_multilabel.labels import CLASS_NAMES
from rgb_multilabel.masks import decode_mask, encode_mask


def test_mask_encoding_round_trip():
    rng = np.random.default_rng(42)
    targets = rng.integers(0, 2, size=(len(CLASS_NAMES), 11, 13), dtype=np.uint8).astype(bool)
    valid = rng.integers(0, 2, size=(11, 13), dtype=np.uint8).astype(bool)
    encoded = encode_mask(targets, valid)
    decoded_targets, decoded_valid = decode_mask(encoded)
    assert encoded.dtype == np.uint8
    assert np.array_equal(decoded_targets, targets)
    assert np.array_equal(decoded_valid, valid)
