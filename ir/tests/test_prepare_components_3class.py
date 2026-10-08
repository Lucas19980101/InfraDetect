import unittest

import numpy as np

from seg_tasks.prepare_components_3class import CMP_MAP, ETRIMS_MAP, DOM_MAP, hznu_mask, remap


class ThreeClassPreparationTests(unittest.TestCase):
    def test_3dom_and_cmp_mapping(self):
        self.assertEqual(remap(np.array([[0, 1, 2, 3, 4, 5]], np.uint8), DOM_MAP, "3DOM").tolist(),
                         [[255, 0, 1, 2, 2, 1]])
        self.assertEqual(remap(np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]], np.uint8), CMP_MAP, "CMP").tolist(),
                         [[255, 255, 0, 1, 1, 0, 0, 255, 1, 0, 0, 0, 255]])

    def test_etrims_mapping(self):
        self.assertEqual(remap(np.arange(9, dtype=np.uint8)[None], ETRIMS_MAP, "eTRIMS").tolist(),
                         [[255, 0, 2, 1, 2, 2, 2, 2, 1]])

    def test_hznu_unlabeled_and_opening_priority(self):
        annotation = {"imageWidth": 4, "imageHeight": 4, "shapes": [
            {"label": "window", "points": [[1, 1], [2, 1], [2, 2], [1, 2]]},
            {"label": "building", "points": [[0, 0], [3, 0], [3, 3], [0, 3]]},
            {"label": "56", "points": [[0, 0], [1, 0], [1, 1]]},
        ]}
        mask, counts = hznu_mask(annotation, (4, 4))
        self.assertEqual(mask[1, 1], 1)
        self.assertEqual(mask[0, 0], 0)
        self.assertEqual(counts["56"], 1)


if __name__ == "__main__":
    unittest.main()
