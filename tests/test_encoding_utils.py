import unittest
import pandas as pd
from utils.EncodingUtils import EncodingUtils   # <-- adjust import if needed
from models.Data import Data
import time
class TestEncodingUtils(unittest.TestCase):

    def test_infer_numeric(self):
        vals = [1, 2, 3.5, 4]
        col_type, processed = EncodingUtils.infer_column_type(vals)
        self.assertEqual(col_type, "numeric")
        self.assertEqual(sorted(processed), [1.0, 2.0, 3.5, 4.0])

    def test_infer_categorical_dates(self):
        vals = ["12/25/2021", "2021-12-25"]
        col_type, processed = EncodingUtils.infer_column_type(vals)
        self.assertEqual(col_type, "date")
        self.assertEqual(sorted(processed), ["20211225","20211225"])

    def test_encode_numeric(self):
        encoded = EncodingUtils.encode_value("3.5", "numeric")
        self.assertEqual(encoded, 3.5)

    def test_encode_categorical(self):
        encoded = EncodingUtils.encode_value("dog", "categorical")
        self.assertEqual(encoded, "dog")

    def test_encode_dataframe(self):
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["2021-12-25", "12/25/2021", "random"]
        })

        col_types = {
            "a": "numeric",
            "b": "date",
        }

        df_encoded = EncodingUtils.encode_dataframe(df, col_types)

        self.assertEqual(df_encoded["a"].iloc[0], 1.0)
        self.assertEqual(df_encoded["b"].iloc[0], "20211225")
        self.assertEqual(df_encoded["b"].iloc[2], "random")

    def test_xdist_speed_large_scale(self):
        """
        Sanity performance test for xdist.

        This does NOT aim for exact timing.
        It only checks that xdist scales reasonably
        and does not regress to O(N^2) per call.
        """

        n_rows = 50000
        n_cols = 20

        # build synthetic dataset
        rows = [
            [float(i % 10) if j % 2 == 0 else f"cat{j % 3}"
             for j in range(n_cols)]
            for i in range(n_rows)
        ]

        col_types = {
            f"c{j}": ("numeric" if j % 2 == 0 else "categorical")
            for j in range(n_cols)
        }

        d = Data(rows=rows, column_types=col_types, use_kdtree=False)

        # warm up (fills cache, JIT paths, etc.)
        for i in range(10):
            d.xdist(rows[i], rows[i + 1])

        # timed section
        start = time.perf_counter()
        for i in range(200):
            d.xdist(rows[i], rows[i + 1])
        elapsed = time.perf_counter() - start

        # This threshold is intentionally loose.
        # On a typical laptop this is ~0.02–0.05s.
        self.assertLess(
            elapsed,
            2,
            f"xdist too slow: {elapsed:.3f}s for 200 calls"
        )
if __name__ == "__main__":
    unittest.main()
