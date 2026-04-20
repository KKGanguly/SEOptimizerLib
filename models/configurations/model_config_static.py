import numpy as np
from datetime import datetime
from ConfigSpace.hyperparameters import (
    OrdinalHyperparameter,
    CategoricalHyperparameter,
    Constant
)
from ConfigSpace import ConfigurationSpace
from utils.EncodingUtils import EncodingUtils
class ModelConfigurationStatic:
    def __init__(self, config, dataset_file, seed=42, column_types=None):
        self.config = config
        self.dataset_file = dataset_file
        self.seed = seed
        self.configspace = None
        self.param_names = None
        self.hyperparam_space = None
        self.hyperparam_dict = None
        # Store column type info for consistent encoding
        self.column_types = column_types or {}
        self.get_configspace()
    
    def set_seed(self, seed):
        self.seed = seed
        if self.configspace:
            self.configspace.seed(seed)
    
    def get_dataset_file(self):
        return self.dataset_file
    
    def get_hyperparam_dict(self):
        return self.hyperparam_dict
    
    def get_hyperconfig_distribution(self):
        """
        Build a purely discrete ConfigSpace:
        - numeric -> ordinal via CategoricalHyperparameter with sorted numeric choices
        - date & categorical -> CategoricalHyperparameter with string choices
        """
        cs = ConfigurationSpace(seed=self.seed)

        for param_name, values in self.config.items():
            # values are already ENCODED
            col_type = self.column_types.get(param_name, "categorical")
            unique_vals = sorted(set(values))

            if len(unique_vals) == 1:
                # Constant hyperparameter
                cs.add_hyperparameter(Constant(param_name, unique_vals[0]))
                continue

            # Numeric -> ordinal (discrete)
            if col_type == "numeric":
                hp = OrdinalHyperparameter(param_name, sequence=unique_vals)
            else:
                # date or categorical: treat as categorical
                hp = CategoricalHyperparameter(param_name, choices=unique_vals)

            cs.add_hyperparameter(hp)

        return cs

    def get_configspace(self, recompute=False):
        if recompute or self.configspace is None:
            self.configspace = self.get_hyperconfig_distribution()
            self.param_names = list(self.config.keys())
            self.hyperparam_space = [[value for value in config_values] for config_values in self.config.values()]
            param_values = {param: [] for param in  self.param_names}
            for index, param_set in enumerate(self.hyperparam_space):
                param_values[self.param_names[index]].extend(param_set)
            self.hyperparam_dict = param_values
        return self.configspace, self.param_names, self.hyperparam_space
    
    def cs_to_dict(self, config):
        return dict(config)