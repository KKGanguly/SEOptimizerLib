# optimizers/RandomSearchOptimizer.py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import random
import time
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
    Pure Random Search optimizer operating over the ConfigSpace.
    
    Correctly handles both discrete choices and continuous bounds,
    scoring against the RF surrogate without any KD-tree restrictions.
    """
    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        self.config_space, _, _ = self.model_config.get_configspace()
        self.cache = {}
        
        # Extract column names for consistent tuple generation
        self.columns = list(self.model_wrapper.X.columns)

        self.num_objectives = len(
            self.model_wrapper.get_score(
                {c: self.model_wrapper.X.iloc[0][c] for c in self.columns}
            )
        )

        self.iteration = 0
        self.start_time = None
        self.end_time = None

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _safe_clean(self, v):
        """Clean items and round floats slightly to prevent cache bloat."""
        val = v.item() if hasattr(v, "item") else v
        return round(val, 6) if isinstance(val, float) else val

    def _row_tuple(self, hp_dict):
        """Consistent caching key generation."""
        return tuple(self._safe_clean(hp_dict[c]) for c in self.columns)

    # ------------------------------------------------------------
    # Sample ONLY from ConfigSpace (no dataset projection)
    # ------------------------------------------------------------
    def _sample_config(self, rng):
        hp_dict = {}

        for hp in self.config_space.get_hyperparameters():
            hp_type = type(hp).__name__

            if isinstance(hp, Constant):
                hp_dict[hp.name] = hp.value

            elif isinstance(hp, OrdinalHyperparameter):
                hp_dict[hp.name] = rng.choice(list(hp.sequence))

            elif isinstance(hp, CategoricalHyperparameter):
                hp_dict[hp.name] = rng.choice(list(hp.choices))

            elif hp_type == "UniformFloatHyperparameter":
                hp_dict[hp.name] = rng.uniform(hp.lower, hp.upper)

            elif hp_type == "UniformIntegerHyperparameter":
                hp_dict[hp.name] = rng.randint(int(hp.lower), int(hp.upper))

            else:
                raise ValueError(f"Unsupported hyperparameter type: {hp_type}")

        return hp_dict

    # ------------------------------------------------------------
    # Main optimize loop
    # ------------------------------------------------------------
    def optimize(self):
        n_trials = self.config["n_trials"]
        rng = random.Random(self.seed)

        self.start_time = time.time()

        all_evals = []

        for _ in range(n_trials):

            config = self._sample_config(rng)
            key = self._row_tuple(config)

            # caching
            if key in self.cache:
                scores, d2h_val = self.cache[key]
            else:
                try:
                    scores = tuple(self.model_wrapper.get_score(config))
                except Exception as e:
                    print("[RandomSearch ERROR]", e, config)
                    # Use 1.0 (worst case) for failures to match other optimizers
                    scores = tuple(1.0 for _ in range(self.num_objectives))
                
                ideal = [0] * self.num_objectives
                d2h_val = DistanceUtil.d2h(ideal, list(scores))
                self.cache[key] = (scores, d2h_val)

            self.iteration += 1
            self.track_evaluation(config, list(scores), self.iteration)

            all_evals.append((config, d2h_val))

        # ------------------------------------------------------------
        # Best selection
        # ------------------------------------------------------------
        best_config = None
        best_d2h = float("inf")

        for config, d2h in all_evals:
            if d2h < best_d2h:
                best_d2h = d2h
                best_config = config

        self.best_config = best_config
        self.best_value = best_d2h
        self.end_time = time.time()

        return self.best_config, self.best_value