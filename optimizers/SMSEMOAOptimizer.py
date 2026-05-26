# optimizers/SMSEMOAOptimizer.py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
import copy
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from optimizers.base_optimizer import BaseOptimizer
from utils import DistanceUtil

class PymooProblemWrapper(ElementwiseProblem):
    """Wraps your model_wrapper into a pymoo-compatible Problem definition."""
    def __init__(self, model_wrapper, config_space, n_obj):
        self.wrapper = model_wrapper
        self.cs = config_space
        # Infer bounds from ConfigSpace
        lower = [hp.lower if hasattr(hp, 'lower') else 0 for hp in config_space.get_hyperparameters()]
        upper = [hp.upper if hasattr(hp, 'upper') else 1 for hp in config_space.get_hyperparameters()]
        super().__init__(n_var=len(config_space.get_hyperparameters()), n_obj=n_obj, xl=lower, xu=upper)

    def _evaluate(self, x, out, *args, **kwargs):
        # Map flat numpy array 'x' back to config dict
        hp_dict = {hp.name: val for hp, val in zip(self.cs.get_hyperparameters(), x)}
        scores = list(self.wrapper.get_score(hp_dict))
        
        # Jitter: Microscopic noise to prevent division-by-zero in Hypervolume survival
        jitter = np.random.normal(0, 1e-9, len(scores))
        out["F"] = [s + j for s, j in zip(scores, jitter)]

class SMSEMOAOptimizer(BaseOptimizer):
    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)
        self.config_space, _, _ = self.model_config.get_configspace()
        
        # Determine number of objectives
        test_config = dict(self.config_space.sample_configuration())
        self.num_objectives = len(model_wrapper.get_score(test_config))
        
        self.iteration = 0
        self.best_config = None
        self.best_value = float("inf")
        self.pop_size = int(self.config.get("pop_size", 20))

    def optimize(self):
        self.start_time = time.time()
        
        # Setup Problem and Algorithm
        problem = PymooProblemWrapper(self.model_wrapper, self.config_space, self.num_objectives)
        algorithm = SMSEMOA(pop_size=self.pop_size)
        
        # Optimization Loop
        # We run for n_trials // pop_size generations
        res = minimize(
            problem,
            algorithm,
            ('n_gen', self.config["n_trials"] // self.pop_size),
            seed=self.seed,
            verbose=False
        )

        # Track results manually using the results object
        ideal = [0] * self.num_objectives
        for i in range(len(res.F)):
            self.iteration += 1
            config = {hp.name: val for hp, val in zip(self.config_space.get_hyperparameters(), res.X[i])}
            scores = list(res.F[i])
            self.track_evaluation(config, scores, self.iteration)
            
            # Update best seen
            d2h = DistanceUtil.d2h(ideal, scores)
            if d2h < self.best_value:
                self.best_value = d2h
                self.best_config = copy.deepcopy(config)

        self.end_time = time.time()
        return self.best_config, self.best_value