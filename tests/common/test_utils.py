from unittest import TestCase

import numpy as np

from evo.data_converters.common.utils import vertices_bounding_box


class TestUtils(TestCase):
    """
    Tests for utils
    """

    def test_vertices_bounding_box(self) -> None:
        # given a points array
        points = np.array(
            [
                [-100.0, 1.0, 2.0],
                [3.0, -200.0, 4.0],
                [5.0, 6.0, -300.0],
                [400.0, 7.0, 8.0],
                [9.0, 500.0, 10.0],
                [11.0, 12.0, 600.0],
            ]
        )

        # Then vertices_bounding_box returns the expected values
        bb = vertices_bounding_box(points)
        self.assertEqual(-100, bb.min_x)
        self.assertEqual(-200, bb.min_y)
        self.assertEqual(-300, bb.min_z)
        self.assertEqual(400, bb.max_x)
        self.assertEqual(500, bb.max_y)
        self.assertEqual(600, bb.max_z)
