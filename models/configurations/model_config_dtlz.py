# models/configurations/model_config_dtlz.py
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

class ModelConfigurationDTLZ:
    def __init__(self, problem: str, n_vars: int, n_objs: int, seed: int = 0):
        self.problem = problem
        self.n_vars = n_vars
        self.n_objs = n_objs
        self.seed = seed

        # stable ordering for sorted(hyperparams)
        self.param_names = [f"x{i:03d}" for i in range(n_vars)]

    def set_seed(self, seed: int):
        self.seed = seed

    def get_dataset_file(self):
        return None  # keep your framework happy if it calls this

    def get_configspace(self):
        cs = ConfigurationSpace(seed=self.seed)
        for name in self.param_names:
            cs.add_hyperparameter(UniformFloatHyperparameter(name, lower=0.0, upper=1.0))
        return cs, None, None
