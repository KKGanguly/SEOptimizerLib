import unittest
import pandas as pd
from unittest.mock import patch

from models.model_wrapper_static import ModelWrapperStatic


# Dummy encoder
class DummyEncodingUtils:
    @staticmethod
    def encode_value(v, col_type):
        if col_type == "numeric":
            return float(v)
        return str(v)


# Dummy d2h implementation
class DummyDistanceUtil:
    @staticmethod
    def d2h(a, b):
        return 0.2  # deterministic


class TestModelWrapperStatic(unittest.TestCase):

    @patch("utils.DistanceUtil.d2h", DummyDistanceUtil.d2h)
    @patch("utils.EncodingUtils.EncodingUtils.encode_value", DummyEncodingUtils.encode_value)
    def test_initialization_and_normalization(self):
        X = pd.DataFrame({
            "hp1": ["A", "B"],
            "hp2": [1, 2]
        })

        y = pd.DataFrame({
            "acc": [True, False],
            "bin": [0, 1],
            "loss-": [5.0, 10.0],
        })

        model_config = type("cfg", (), {})()
        model_config.column_types = {"hp1": "categorical", "hp2": "numeric"}

        wrapper = ModelWrapperStatic(X, y, model_config)

        # Check normalization
        self.assertEqual(wrapper.y["acc"].tolist(), [1.0, 0.0])   # normalized acc
        self.assertEqual(wrapper.y["bin"].tolist(), [0.0, 1.0])    # normalized bin
        self.assertEqual(wrapper.y["loss-"].tolist(), [0.0, 1.0])  # normalized loss-

        # Lookup correctness
        key0 = ("A", 1.0)
        key1 = ("B", 2.0)
        self.assertEqual(wrapper.lookup[key0], 0)
        self.assertEqual(wrapper.lookup[key1], 1)

    @patch("utils.DistanceUtil.d2h", DummyDistanceUtil.d2h)
    @patch("utils.EncodingUtils.EncodingUtils.encode_value", DummyEncodingUtils.encode_value)
    def test_find_row_fast(self):
        X = pd.DataFrame({"hp1": ["A", "B"], "hp2": [1, 2]})
        y = pd.DataFrame({"acc": [0.5, 1.0]})

        model_config = type("cfg", (), {})()
        model_config.column_types = {"hp1": "categorical", "hp2": "numeric"}

        wrapper = ModelWrapperStatic(X, y, model_config)

        self.assertEqual(wrapper._find_row_fast({"hp1": "A", "hp2": 1}), 0)
        self.assertEqual(wrapper._find_row_fast({"hp1": "B", "hp2": 2}), 1)

    @patch("utils.DistanceUtil.d2h", DummyDistanceUtil.d2h)
    @patch("utils.EncodingUtils.EncodingUtils.encode_value", DummyEncodingUtils.encode_value)
    def test_score_tuple_flip(self):
        X = pd.DataFrame({"hp": ["X", "Y"]})
        y = pd.DataFrame({
            "acc": [0.2, 0.8],         # Example column (does not end with '-')
            "loss-": [0.7, 0.3],       # Example column (ends with '-')
        })

        model_config = type("cfg", (), {})()
        model_config.column_types = {"hp": "categorical"}

        # Normalize y
        normalized_y = (y - y.min()) / (y.max() - y.min())

        # Initialize the ModelWrapperStatic object with X, normalized_y, and model_config
        wrapper = ModelWrapperStatic(X, y, model_config)

        # Check if normalization was done correctly (expected output after normalization)
        self.assertEqual(wrapper.y["acc"].tolist(), [0.0, 1.0])   # normalized acc
        self.assertEqual(wrapper.y["loss-"].tolist(), [1.0, 0.0])  # normalized loss-

        # Now let's test the flipping logic for the 'loss-' column
        scores = wrapper._score_tuple(0)  # index 0 (first row)
        # acc should be the same (0.0), loss- should be flipped (1 - 0.7 = 0.3)
        self.assertAlmostEqual(scores[1], 1 - normalized_y["loss-"].tolist()[0], places=7)  # Allow some floating-point tolerance
        
        scores = wrapper._score_tuple(1)  # index 1 (second row)
        # acc should be the same (1.0), loss- should be flipped (1 - 0.3 = 0.7)
        self.assertAlmostEqual(scores[1], 1 - normalized_y["loss-"].tolist()[1], places=7)  # Allow some floating-point tolerance

    @patch("utils.DistanceUtil.d2h", DummyDistanceUtil.d2h)
    @patch("utils.EncodingUtils.EncodingUtils.encode_value", DummyEncodingUtils.encode_value)
    def test_missing_configuration_raises(self):
        X = pd.DataFrame({"hp": ["X"]})
        y = pd.DataFrame({"acc": [1.0]})

        model_config = type("cfg", (), {})()
        model_config.column_types = {"hp": "categorical"}

        wrapper = ModelWrapperStatic(X, y, model_config)

        with self.assertRaises(ValueError):
            wrapper.get_score({"hp": "Z"})  # Not in dataset


if __name__ == "__main__":
    unittest.main()