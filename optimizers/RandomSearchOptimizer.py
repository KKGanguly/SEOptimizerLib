# optimizers/RandomSearchOptimizer.py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import random
import time
import copy
import numpy as np

from optimizers.base_optimizer import BaseOptimizer
from ConfigSpace.hyperparameters import (
    OrdinalHyperparameter,
    CategoricalHyperparameter,
    Constant,
)
from utils import DistanceUtil


class RandomSearchOptimizer(BaseOptimizer):
    """
    Pure Random Search optimizer.
    Archetype: The Ultimate Sanity Check.
    
    Fixed to enforce the exact same empirical 'Initialization Tax' as the 
    Bayesian and Causal algorithms, ensuring apples-to-apples learning curves.
    """
    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        random.seed(seed)
        np.random.seed(seed)

        self.X_df = self.model_wrapper.X
        self.n_rows = len(self.X_df)
        self.columns = list(self.X_df.columns)
        self.config_space, _, _ = self.model_config.get_configspace()
        self.cache = {}

        test_config = {c: self._safe_clean(self.X_df.iloc[0][c]) for c in self.columns}
        self.num_objectives = len(self.model_wrapper.get_score(test_config))

        self.iteration = 0
        self.best_config = None
        self.best_value = float("inf")

        # The Fairness Tax
        self.initial_budget = int(self.config.get("initial_size", 10))

    # ------------------------------------------------------------
    # Helpers & Sampling
    # ------------------------------------------------------------
    def _safe_clean(self, v):
        """Extracts native python types and truncates floats for cache stability."""
        val = v.item() if hasattr(v, "item") else v
        return round(val, 6) if isinstance(val, float) else val

    def _row_tuple(self, hp_dict):
        return tuple(self._safe_clean(hp_dict[c]) for c in self.columns)

    def _idx_to_config(self, idx):
        """Extracts a configuration directly from the empirical dataset."""
        row = self.X_df.iloc[idx]
        return {c: self._safe_clean(row[c]) for c in self.columns}

    def _sample_config(self):
        """Generates a random configuration across the continuous ConfigSpace boundaries."""
        hp_dict = {}
        for hp in self.config_space.get_hyperparameters():
            hp_type = type(hp).__name__
            if isinstance(hp, Constant):
                hp_dict[hp.name] = hp.value
            elif isinstance(hp, OrdinalHyperparameter):
                hp_dict[hp.name] = random.choice(list(hp.sequence))
            elif isinstance(hp, CategoricalHyperparameter):
                hp_dict[hp.name] = random.choice(list(hp.choices))
            elif hp_type == "UniformFloatHyperparameter":
                hp_dict[hp.name] = random.uniform(hp.lower, hp.upper)
            elif hp_type == "UniformIntegerHyperparameter":
                hp_dict[hp.name] = random.randint(int(hp.lower), int(hp.upper))
            else:
                raise ValueError(f"Unsupported hyperparameter type: {hp_type}")
        return hp_dict

    # ------------------------------------------------------------
    # Evaluation & Tracking
    # ------------------------------------------------------------
    def _eval_safe(self, hp_dict):
        """Evaluates, tracks budget, and caches configuration scores."""
        key = self._row_tuple(hp_dict)
        if key in self.cache:
            # If random search accidentally guesses a duplicate, it still burns an evaluation
            self.iteration += 1
            scores, d2h = self.cache[key]
            self.track_evaluation(hp_dict, list(scores), self.iteration)
            return d2h

        try:
            scores, d2h = self.model_wrapper.evaluate(hp_dict)
        except Exception:
            scores = tuple(float('inf') for _ in range(self.num_objectives))
            d2h_val = float('inf')

        self.cache[key] = (scores, d2h)
        self.iteration += 1
        self.track_evaluation(hp_dict, list(scores), self.iteration)
        
        if d2h < self.best_value:
            self.best_value = d2h
            self.best_config = copy.deepcopy(hp_dict)
            
        return d2h

    # ------------------------------------------------------------
    # Main Optimization Loop
    # ------------------------------------------------------------
    def optimize(self):
        n_trials = self.config["n_trials"]
        self.start_time = time.time()

        
        while self.iteration < n_trials:
            config = self._sample_config()
            self._eval_safe(config)

        self.end_time = time.time()
        return self.best_config, self.best_value