# optimizers/GPOptimizer.py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from optimizers.base_optimizer import BaseOptimizer
from ConfigSpace.hyperparameters import (
    OrdinalHyperparameter,
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    Constant,
)
from utils import DistanceUtil
import time
import copy
import numpy as np
from skopt import Optimizer
from skopt.space import Real, Integer, Categorical

class GPOptimizer(BaseOptimizer):
    """
    Plain Gaussian Process (GP-EI) Optimizer.
    - Uses skopt.Optimizer with 'GP' base estimator.
    - Native support for mixed-type spaces (Real, Integer, Categorical).
    - No snapping: direct evaluation on the continuous RF surrogate.
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)
        self.config_space, _, _ = self.model_config.get_configspace()
        self.cache = {}

        # Reset state
        self.iteration = 0
        self.best_config = None
        self.best_value = float("inf")

        # Define skopt dimensions
        self.skopt_space = []
        self.skopt_names = []
        self.constants = {}

        for hp in self.config_space.get_hyperparameters():
            hp_type = type(hp).__name__
            if isinstance(hp, Constant):
                self.constants[hp.name] = hp.value
            elif hp_type == "UniformFloatHyperparameter":
                self.skopt_space.append(Real(hp.lower, hp.upper, name=hp.name))
                self.skopt_names.append(hp.name)
            elif hp_type == "UniformIntegerHyperparameter":
                self.skopt_space.append(Integer(int(hp.lower), int(hp.upper), name=hp.name))
                self.skopt_names.append(hp.name)
            elif isinstance(hp, (CategoricalHyperparameter, OrdinalHyperparameter)):
                choices = list(hp.choices) if hasattr(hp, 'choices') else list(hp.sequence)
                self.skopt_space.append(Categorical(choices, name=hp.name))
                self.skopt_names.append(hp.name)

    def _safe_clean(self, v):
        val = v.item() if hasattr(v, "item") else v
        return round(val, 6) if isinstance(val, float) else val

    def _row_tuple(self, hp_dict):
        return tuple(self._safe_clean(hp_dict[c]) for c in self.columns)

    def optimize(self):
        n_trials = self.config["n_trials"]
        self.start_time = time.time()
        
        # Reset trackers
        self.iteration = 0
        self.best_value = float("inf")
        self.best_config = None

        # Base estimator 'GP' automatically uses Matérn/Hamming kernels
        opt = Optimizer(
            dimensions=self.skopt_space,
            base_estimator="GP",
            acq_func="EI",
            random_state=self.seed
        )

        for _ in range(n_trials):
            # 1. Ask GP for next point
            suggested = opt.ask()
            config = {name: val for name, val in zip(self.skopt_names, suggested)}
            config.update(self.constants)

            # 2. Evaluate via surrogate
            key = self._row_tuple(config)
            if key in self.cache:
                scores, d2h_val = self.cache[key]
            else:
                try:
                    scores = tuple(self.model_wrapper.get_score(config))
                except Exception:
                    scores = tuple(1.0 for _ in range(len(self.columns)))
                ideal = [0] * len(scores)
                d2h_val = DistanceUtil.d2h(ideal, list(scores))
                self.cache[key] = (scores, d2h_val)

            # 3. Update trackers
            self.iteration += 1
            self.track_evaluation(config, list(scores), self.iteration)
            
            if d2h_val < self.best_value:
                self.best_value = d2h_val
                self.best_config = copy.deepcopy(config)

            # 4. Update GP posterior
            opt.tell(suggested, d2h_val)

        self.end_time = time.time()
        return self.best_config, self.best_value