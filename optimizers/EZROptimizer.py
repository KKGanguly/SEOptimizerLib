# optimizers/EZROptimizer.py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from optimizers.base_optimizer import BaseOptimizer
from ConfigSpace.hyperparameters import (
    OrdinalHyperparameter,
    CategoricalHyperparameter,
    Constant,
)
from utils import DistanceUtil

import numpy as np
import time
import random
import copy

BIG = 1e32

class EZROptimizer(BaseOptimizer):
    """
    BL-style active learner over the continuous RF surrogate.
    Fixed for initialization fairness, strict budget enforcement, 
    and robust centroid calculations.
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        random.seed(seed)
        np.random.seed(seed)

        assert hasattr(model_wrapper, "rf_model")

        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)
        self.n_rows = len(self.X_df)

        self.config_space, _, _ = self.model_config.get_configspace()
        self.cache = {}

        self.num_objectives = len(
            self.model_wrapper.get_score(
                {c: self.X_df.iloc[0][c] for c in self.columns}
            )
        )

        self.iteration = 0
        self.best_config = None
        self.best_value = float("inf")

        # EZR Hyperparameters
        self.p = 2
        self.start_evals = int(self.config.get("initial_size", 20)) # FIX: Fair initialization
        self.guess = 0.5
        self.Few = 128

        # Extract bounds for distance normalization
        self.bounds = {}
        for hp in self.config_space.get_hyperparameters():
            if type(hp).__name__ in ["UniformFloatHyperparameter", "UniformIntegerHyperparameter"]:
                self.bounds[hp.name] = (hp.lower, hp.upper)

    def _safe_clean(self, v):
        val = v.item() if hasattr(v, "item") else v
        return round(val, 6) if isinstance(val, float) else val

    def _row_tuple(self, hp_dict):
        return tuple(self._safe_clean(hp_dict[c]) for c in self.columns)

    def _idx_to_config(self, idx):
        row = self.X_df.iloc[idx]
        return {c: self._safe_clean(row[c]) for c in self.columns}

    def _sample_config(self):
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
        return hp_dict

    def _rf(self, hp_dict):
        key = self._row_tuple(hp_dict)
        hit = self.cache.get(key)
        if hit is not None:
            return hit

        try:
            scores = tuple(self.model_wrapper.get_score(hp_dict))
        except Exception:
            scores = tuple(1.0 for _ in range(self.num_objectives))

        ideal = [0] * self.num_objectives
        d2h = DistanceUtil.d2h(ideal, list(scores))

        self.cache[key] = (scores, d2h)
        return scores, d2h

    def _eval(self, hp_dict):
        # FIX: Ensure cache hits don't artificially exhaust the iteration budget
        key = self._row_tuple(hp_dict)
        if key in self.cache:
            return self.cache[key][1]

        scores, d2h = self._rf(hp_dict)
        self.iteration += 1
        self.track_evaluation(hp_dict, list(scores), self.iteration)
        
        if d2h < self.best_value:
            self.best_value = d2h
            self.best_config = copy.deepcopy(hp_dict)
            
        return d2h

    def _distx(self, r1, r2):
        d = 0.0
        n = len(self.columns)

        for col in self.columns:
            a, b = r1[col], r2[col]
            if col in self.bounds:
                lower, upper = self.bounds[col]
                span = upper - lower if upper > lower else 1.0
                na = (float(a) - lower) / span
                nb = (float(b) - lower) / span
                inc = abs(na - nb)
            else:
                inc = 0.0 if a == b else 1.0
            d += inc ** self.p

        return (d / (n + 1 / BIG)) ** (1 / self.p)

    def _mid(self, configs):
        mid = {}
        # FIX: Protect against empty list crash
        if not configs:
            return self._sample_config()

        for col in self.columns:
            vals = [c[col] for c in configs]
            if col in self.bounds:
                mid[col] = sum(vals) / len(vals)
            else:
                mid[col] = max(set(vals), key=vals.count)
        return mid

    def _nearer(self, best_configs, rest_configs):
        bmid = self._mid(best_configs)
        rmid = self._mid(rest_configs)

        fallback_candidate = None
        for _ in range(self.Few):
            candidate = self._sample_config()
            fallback_candidate = candidate
            
            if self._distx(bmid, candidate) < self._distx(rmid, candidate):
                return candidate

        return fallback_candidate

    def optimize(self):
        n_trials = self.config["n_trials"]
        self.start_time = time.time()
        
        n = min(self.start_evals, n_trials)

        # 1. Ground the algorithm with real dataset rows
        initial_configs = [self._idx_to_config(i) for i in random.sample(range(self.n_rows), min(n, self.n_rows))]
        while len(initial_configs) < n:
            initial_configs.append(self._sample_config())

        # 2. Evaluate initials (with strict budget check)
        for config in initial_configs:
            if self.iteration >= n_trials:
                self.end_time = time.time()
                return self.best_config, self.best_value
            self._eval(config)

        # 3. Sort initials to split into best/rest
        done = sorted(initial_configs, key=lambda c: self._rf(c)[1])

        # FIX: Safeguard against empty splits
        cut = max(1, round(n ** self.guess))
        best = done[:cut]
        rest = done[cut:] if len(done) > cut else [done[-1]]

        # 4. Active learning loop over continuous space
        while self.iteration < n_trials:
            hi = self._nearer(best, rest)
            self._eval(hi)

            best.append(hi)
            best = sorted(best, key=lambda c: self._rf(c)[1])
            n += 1

            if len(best) >= round(n ** self.guess):
                if len(best) > 1: # FIX: Protect against popping the only best item
                    rest.append(best.pop(-1))

        self.end_time = time.time()
        return self.best_config, self.best_value