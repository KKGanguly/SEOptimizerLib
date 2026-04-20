
import unittest
import pandas as pd

from models.Data import Data                    # adjust if needed

class TestXDist(unittest.TestCase):

    def _make_data(self, rows, col_types):
        return Data(
            rows=rows,
            column_types=col_types,
            use_kdtree=False
        )

    def test_xdist_numeric_zero(self):
        rows = [
            [1.0, 2.0],
            [1.0, 2.0],
        ]
        col_types = {"a": "numeric", "b": "numeric"}
        d = self._make_data(rows, col_types)

        self.assertEqual(d.xdist(rows[0], rows[1]), 0.0)

    def test_xdist_numeric_nonzero(self):
        rows = [
            [0.0],
            [10.0],
        ]
        col_types = {"a": "numeric"}
        d = self._make_data(rows, col_types)

        # normalized difference = 1
        self.assertAlmostEqual(d.xdist(rows[0], rows[1]), 1.0)

    def test_xdist_categorical_equal(self):
        rows = [
            ["dog"],
            ["dog"],
        ]
        col_types = {"a": "categorical"}
        d = self._make_data(rows, col_types)

        self.assertEqual(d.xdist(rows[0], rows[1]), 0.0)

    def test_xdist_categorical_mismatch(self):
        rows = [
            ["dog"],
            ["cat"],
        ]
        col_types = {"a": "categorical"}
        d = self._make_data(rows, col_types)

        self.assertEqual(d.xdist(rows[0], rows[1]), 1.0)

    def test_xdist_mixed_numeric_categorical(self):
        rows = [
            [0.0, "dog"],
            [10.0, "cat"],
        ]
        col_types = {"a": "numeric", "b": "categorical"}
        d = self._make_data(rows, col_types)

        # numeric = 1, categorical = 1
        # sqrt((1^2 + 1^2) / 2) = 1
        self.assertAlmostEqual(d.xdist(rows[0], rows[1]), 1.0)

    def test_xdist_missing_both(self):
        rows = [
            ["?", "?"],
            ["?", "?"],
        ]
        col_types = {"a": "numeric", "b": "categorical"}
        d = self._make_data(rows, col_types)

        self.assertEqual(d.xdist(rows[0], rows[1]), 1.0)

    def test_xdist_missing_one_side(self):
        rows = [
            ["?", 5.0],
            [10.0, 5.0],
        ]
        col_types = {"a": "numeric", "b": "numeric"}
        d = self._make_data(rows, col_types)

        dist = d.xdist(rows[0], rows[1])
        self.assertTrue(0.0 <= dist <= 1.0)

    def test_xdist_symmetry(self):
        rows = [
            [1.0, "dog"],
            [2.0, "cat"],
        ]
        col_types = {"a": "numeric", "b": "categorical"}
        d = self._make_data(rows, col_types)

        self.assertEqual(
            d.xdist(rows[0], rows[1]),
            d.xdist(rows[1], rows[0])
        )


if __name__ == "__main__":
    unittest.main()