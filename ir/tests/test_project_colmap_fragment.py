import tempfile
import unittest
from pathlib import Path

import numpy as np

from seg_tasks.project_colmap_fragment import accepted_votes, apply_homography, defect_metadata


class ProjectColmapFragmentTest(unittest.TestCase):
    def test_homography(self):
        u = np.asarray([1.0, 3.0])
        v = np.asarray([2.0, 4.0])
        matrix = np.asarray([[2, 0, 5], [0, 3, 7], [0, 0, 1]], dtype=float)
        x, y, valid = apply_homography(u, v, matrix)
        np.testing.assert_allclose(x, [7, 11])
        np.testing.assert_allclose(y, [13, 19])
        self.assertTrue(valid.all())

    def test_vote_thresholds(self):
        counts = np.asarray([[2, 1], [1, 0], [0, 0]], dtype=np.uint16)
        valid = np.asarray([3, 4, 0], dtype=np.uint16)
        accepted, primary = accepted_votes(counts, valid, min_views=2, min_fraction=0.3)
        np.testing.assert_array_equal(accepted, [[True, False], [False, False], [False, False]])
        np.testing.assert_array_equal(primary, [1, 0, 0])

    def test_specialized_defect_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("""data:\n  class_names: [Background, Hollow Areas, Stains]\n  palette_rgb: [[0, 0, 0], [174, 79, 13], [36, 179, 83]]\n""")
            names, colors = defect_metadata(path)
        self.assertEqual(names, ("hollow_areas", "stains"))
        self.assertEqual(colors.tolist(), [[174, 79, 13], [36, 179, 83]])


if __name__ == "__main__":
    unittest.main()
