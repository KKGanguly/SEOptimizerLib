import unittest
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    OrdinalHyperparameter,
)
from models.configurations.model_config_static import ModelConfigurationStatic


class TestModelConfigurationStatic(unittest.TestCase):

    def test_initialization(self):
        config = {
            "num_param": [1, 2, 3],
            "cat_param": ["A", "B"],
        }
        column_types = {"num_param": "numeric", "cat_param": "categorical"}

        mcs = ModelConfigurationStatic(config, dataset_file="data.csv", seed=42, column_types=column_types)

        self.assertEqual(mcs.dataset_file, "data.csv")
        self.assertEqual(mcs.seed, 42)
        self.assertEqual(mcs.column_types, column_types)
        self.assertIsInstance(mcs.configspace, ConfigurationSpace)

    def test_hyperparameter_construction(self):
        config = {
            "num_param": [3, 1, 2],      # numeric → ordinal
            "cat_param": ["B", "A"],     # categorical → categorical HP
            "constant_param": ["X"],     # should become Constant HP
        }
        column_types = {"num_param": "numeric", "cat_param": "categorical", "constant_param": "categorical"}

        mcs = ModelConfigurationStatic(config, dataset_file="dummy.csv", seed=0, column_types=column_types)
        cs, names, space = mcs.get_configspace()

        # Access hyperparameters using the configuration space (direct access)
        hp_num = cs["num_param"]  # Access from ConfigurationSpace directly
        hp_cat = cs["cat_param"]
        hp_const = cs["constant_param"]

        # Numeric should be ordinal and sorted (handling tuples)
        self.assertIsInstance(hp_num, OrdinalHyperparameter)
        self.assertEqual(tuple(hp_num.sequence), (1, 2, 3))  # Ensure it's a tuple and matches the expected order

        # Categorical should remain categorical
        self.assertIsInstance(hp_cat, CategoricalHyperparameter)
        self.assertEqual(sorted(hp_cat.choices), ["A", "B"])

        # Single-value list should become Constant
        self.assertIsInstance(hp_const, Constant)
        self.assertEqual(hp_const.value, "X")

    def test_seed_reproducibility(self):
        config = {"p": [1, 2, 3]}
        mcs1 = ModelConfigurationStatic(config, dataset_file="d.csv", seed=123)
        mcs2 = ModelConfigurationStatic(config, dataset_file="d.csv", seed=123)

        # Draw sample configurations from both spaces
        c1 = mcs1.configspace.sample_configuration()
        c2 = mcs2.configspace.sample_configuration()

        # With same seed, they must match
        self.assertEqual(c1.get_dictionary(), c2.get_dictionary())

    def test_hyperparam_dict_output(self):
        config = {
            "a": [1, 1, 2],
            "b": ["X", "Y", "Y"]
        }
        mcs = ModelConfigurationStatic(config, dataset_file="file.csv")

        expected = {
            "a": [1, 2],
            "b": ["X", "Y"]
        }

        self.assertEqual({k: sorted(v) for k, v in mcs.get_hyperparam_dict().items()}, expected)

    def test_cs_to_dict(self):
        config = {"p": [1, 2]}
        mcs = ModelConfigurationStatic(config, dataset_file="d.csv", seed=0)

        cs, _, _ = mcs.get_configspace()
        sample = cs.sample_configuration()

        result = mcs.cs_to_dict(sample)
        self.assertIsInstance(result, dict)
        self.assertIn("p", result)
        self.assertIn(result["p"], [1, 2])


if __name__ == '__main__':
    unittest.main()