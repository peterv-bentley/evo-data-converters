""" """

from unittest import TestCase

import numpy as np
import numpy.typing as npt

from evo.data_converters.resqml.importer._grid_converter import _unique_points


class TestConverterUniquePoints(TestCase):
    r"""
    Tests for grid_converter::_unique_points
    """

    def test_unique_points_empty_array(self) -> None:
        (points, indices) = _unique_points(np.array([]))
        self.assertEqual(0, len(points))
        self.assertEqual(0, len(indices))

    def test_unique_points_one_element(self) -> None:
        points = np.array([[1, 2, 3]])
        (corners, indices) = _unique_points(points)
        self.assertEqual(1, len(corners))

        self._unique_points_post_conditions(points, indices, corners)

    def test_unique_points_all_points_identical(self) -> None:
        points = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        (corners, indices) = _unique_points(points)
        self.assertEqual(1, len(corners))

        self._unique_points_post_conditions(points, indices, corners)

    def test_unique_points_all_points_unique(self) -> None:
        points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        (corners, indices) = _unique_points(points)
        self.assertEqual(3, len(corners))

        self._unique_points_post_conditions(points, indices, corners)

    def test_unique_points_first_points_same(self) -> None:
        points = np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6], [7, 8, 9]])
        (corners, indices) = _unique_points(points)
        self.assertEqual(3, len(corners))

        self._unique_points_post_conditions(points, indices, corners)

    def test_unique_points_last_points_same(self) -> None:
        points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [7, 8, 9]])
        (corners, indices) = _unique_points(points)
        self.assertEqual(3, len(corners))

        self._unique_points_post_conditions(points, indices, corners)

    def test_unique_points_middle_points_same(self) -> None:
        points = np.array([[1, 2, 3], [4, 5, 6], [4, 5, 6], [7, 8, 9]])
        (corners, indices) = _unique_points(points)
        self.assertEqual(3, len(corners))

        self._unique_points_post_conditions(points, indices, corners)

    def _unique_points_post_conditions(self, points: npt.NDArray, indices: npt.NDArray, corners: npt.NDArray) -> None:
        r"""
        Check the post conditions for _unique_points
        """
        # Length of indices should equal the length of points
        self.assertEqual(len(points), len(indices))

        # The corners list should be a subset of the original points list
        pl = points.tolist()
        cl = corners.tolist()
        self.assertTrue(all([c in pl for c in cl]))

        # The indices should be valid indices for corners
        # i.e. the should be be >= 0 and < len(corners)
        self.assertTrue(all([i >= 0 for i in indices]))
        self.assertTrue(all([i < len(corners) for i in indices]))

        # all elements in corners should be unique.
        self.assertTrue(len(corners), len(np.unique(corners)))

        # Test that the two representations are equivalent
        same = [points[i] == corners[indices[i]] for i in range(len(points))]
        self.assertTrue(np.all(same))
