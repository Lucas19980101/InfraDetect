import numpy as np

from scripts.create_split_manifest import stratified_selection


def test_stratified_selection_is_exact_and_deterministic():
    features = np.asarray(
        [[1, 0], [1, 1], [0, 1], [0, 0]] * 10,
        dtype=np.int16,
    )
    first = stratified_selection(features, validation_size=8, seed=42, iterations=2_000)
    second = stratified_selection(features, validation_size=8, seed=42, iterations=2_000)
    assert first.sum() == 8
    assert np.array_equal(first, second)
    assert np.array_equal(features[first].sum(axis=0), np.asarray([4, 4]))
