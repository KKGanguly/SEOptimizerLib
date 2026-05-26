# optimizers/SWAYOptimizer.py
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
import math

class SWAYOptimizer(BaseOptimizer):
    """
    Exact SWAY (Sampling Way) Optimizer by Menzies et al.
    Strictly follows the authors' published implementation, including 
    their specific FastMap mathematics, exponential loss function, 
    and recursive pole-pruning architecture.
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        random.seed(seed)
        np.random.seed(seed)

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

        # Exact SWAY Parameters
        self.leaf_size = 100 
        self.pool_size = 10000 

        # Extract bounds for normalized distance calculation
        self.bounds = {}
        for hp in self.config_space.get_hyperparameters():
            if type(hp).__name__ in ["UniformFloatHyperparameter", "UniformIntegerHyperparameter"]:
                self.bounds[hp.name] = (hp.lower, hp.upper)

    # ------------------------------------------------------------
    # Helpers & Sampling
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Evaluation & Authors' Exact Exponential Loss
    # ------------------------------------------------------------

    def _eval_safe(self, hp_dict):
        """Evaluates and caches. Returns raw scores for SWAY domination."""
        key = self._row_tuple(hp_dict)
        if key in self.cache:
            return self.cache[key][0] 

        try:
            scores = tuple(self.model_wrapper.get_score(hp_dict))
        except Exception:
            scores = tuple(float('inf') for _ in range(self.num_objectives))

        ideal = [0] * self.num_objectives
        d2h = DistanceUtil.d2h(ideal, list(scores))

        self.cache[key] = (scores, d2h)
        self.iteration += 1
        self.track_evaluation(hp_dict, list(scores), self.iteration)
        
        if d2h < self.best_value:
            self.best_value = d2h
            self.best_config = copy.deepcopy(hp_dict)
            
        return scores

    def _loss(self, f1, f2):
        """Exact authors' implementation: sum(exp(i - j)) / len(f1)"""
        l = 0.0
        for i, j in zip(f1, f2):
            # Clip difference to prevent math.exp overflow on unscaled surrogate objectives
            diff = max(min(i - j, 100), -100)
            l += math.exp(diff)
        return l / len(f1)

    def _cont_dominate(self, config1, config2):
        """Exact authors' implementation of continuous domination."""
        f1 = self._eval_safe(config1)
        f2 = self._eval_safe(config2)
        return self._loss(f1, f2) < self._loss(f2, f1)

    # ------------------------------------------------------------
    # Authors' Exact FastMap Distance & Splitting
    # ------------------------------------------------------------

    def _dist(self, r1, r2):
        """
        Exact authors' distance implementation. 
        Note: This computes SQUARED distance (d += (i-j)**2).
        """
        d = 0.0
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
            d += inc ** 2 
        return d

    def _where(self, pop):
        """Exact authors' FastMap implementation."""
        rand_point = random.choice(pop)
        
        ds = [self._dist(i, rand_point) for i in pop]
        east = pop[ds.index(max(ds))]
        
        ds = [self._dist(i, east) for i in pop]
        west = pop[ds.index(max(ds))]
        
        c = self._dist(east, west)
        
        # Safety for identical populations to prevent ZeroDivisionError
        if c < 1e-9:
            n = len(pop)
            return west, east, pop[:n//2], pop[n//2:]

        cc = 2 * (c ** 0.5)

        mappings = []
        for x in pop:
            a = self._dist(x, west)
            b = self._dist(x, east)
            
            # Exact authors' math for FastMap projection using squared distances
            d = (a + c - b) / cc
            mappings.append((x, d))
            
        mappings = sorted(mappings, key=lambda item: item[1])
        sorted_pop = [item[0] for item in mappings]
        
        n = len(sorted_pop)
        east_items = sorted_pop[:n//2]
        west_items = sorted_pop[n//2:]
        
        return west, east, west_items, east_items

    # ------------------------------------------------------------
    # Exact Recursive SWAY Algorithm
    # ------------------------------------------------------------

    def _sway_cluster(self, items, n_trials):
        """Exact authors' cluster implementation."""
        if len(items) < self.leaf_size:
            return items

        west, east, west_items, east_items = self._where(items)

        # Budget Check to prevent infinite loops in the benchmark runner
        if self.iteration + 1 >= n_trials:
            return items

        east_better = self._cont_dominate(east, west)
        west_better = self._cont_dominate(west, east)

        if east_better and not west_better:
            selected = east_items
        elif west_better and not east_better:
            selected = west_items
        else:
            selected = random.sample(west_items + east_items, len(items) // 2)

        return self._sway_cluster(selected, n_trials)

    # ------------------------------------------------------------
    # Main Optimization Loop
    # ------------------------------------------------------------

    def optimize(self):
        n_trials = self.config["n_trials"]
        self.start_time = time.time()

        # ---------------------------------------------------------
        # GENERATE UNLABELLED POOL
        # ---------------------------------------------------------
        pool = []
        # Dump the entire empirical dataset in
        for i in range(self.n_rows):
            pool.append(self._idx_to_config(i))
            
        # Fill the rest with continuous random samples
        while len(pool) < self.pool_size:
            pool.append(self._sample_config())

        # ---------------------------------------------------------
        # RECURSIVE SWAY PRUNING
        # ---------------------------------------------------------
        leaf_cluster = self._sway_cluster(pool, n_trials)

        # ---------------------------------------------------------
        # LEAF EXPLOITATION
        # ---------------------------------------------------------
        for config in leaf_cluster:
            # Stop condition for the benchmark runner
            if self.iteration >= n_trials:
                break
            self._eval_safe(config)

        self.end_time = time.time()
        return self.best_config, self.best_value