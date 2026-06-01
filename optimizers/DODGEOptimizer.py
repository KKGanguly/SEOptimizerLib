# optimizers/DODGEOptimizer.py
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

class DODGEOptimizer(BaseOptimizer):
    """
    DODGE Optimizer by Agrawal et al.
    Archetype: Model-Free Tabu Search with Epsilon-Redundancy.

    Mechanism:
      - Evaluates an initial empirical set to ensure fairness.
      - Generates random candidate configurations in the continuous space.
      - Discretizes the objective scores into epsilon-bins (e-Tabu list).
      - If a configuration falls into a previously seen bin, it counts as a "strike" (redundant).
      - If consecutive strikes exceed the patience threshold, it terminates early.
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

        # Objective counting
        test_config = {c: self._safe_clean(self.X_df.iloc[0][c]) for c in self.columns}
        self.num_objectives = len(self.model_wrapper.get_score(test_config))

        self.iteration = 0
        self.best_config = None
        self.best_value = float("inf")

        # DODGE Specific Hyperparameters
        self.initial_budget = int(self.config.get("initial_size", 10))
        self.epsilon = float(self.config.get("epsilon", 0.05)) # Standard 5% epsilon grid
        self.patience = int(self.config.get("patience", 30))   # Strikes before early stopping
        
        self.seen_bins = set()
        self.strikes = 0

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
        """Blind random sampling (DODGE relies on sampling over modeling)."""
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
    # Evaluation & Epsilon Discretization
    # ------------------------------------------------------------

    def _discretize(self, scores):
        """Maps continuous performance scores into an epsilon-grid bin."""
        return tuple(round(s / self.epsilon) for s in scores)

    def _eval_safe(self, hp_dict):
        """Evaluates and caches the configuration."""
        key = self._row_tuple(hp_dict)
        if key in self.cache:
            self.iteration += 1
            scores, d2h = self.cache[key]
            try:
                self.track_evaluation(hp_dict, list(scores), self.iteration)
            except Exception:
                self.logging_util.log("iteration", self.iteration)
            return scores, d2h 

        try:
            scores, d2h = self.model_wrapper.evaluate(hp_dict)
        except Exception:
            scores = tuple(float('inf') for _ in range(self.num_objectives))
            d2h = float('inf')


        self.cache[key] = (scores, d2h)
        self.iteration += 1
        self.track_evaluation(hp_dict, list(scores), self.iteration)
        
        if d2h < self.best_value:
            self.best_value = d2h
            self.best_config = copy.deepcopy(hp_dict)
            
        return scores, d2h

    # ------------------------------------------------------------
    # Main Optimization Loop
    # ------------------------------------------------------------

    def optimize(self):
        n_trials = self.config["n_trials"]
        self.start_time = time.time()

        # ---------------------------------------------------------
        # PHASE 1: The Fairness Tax (Blind Empirical Start)
        # ---------------------------------------------------------
        obs_budget = min(self.initial_budget, n_trials)
        for _ in range(obs_budget):
            config = self._sample_config()
            scores, _ = self._eval_safe(config)
            
            # Map to epsilon grid
            perf_bin = self._discretize(scores)
            self.seen_bins.add(perf_bin)

        early_stop_triggered = False
        # ---------------------------------------------------------
        # PHASE 2: DODGE Tabu Search
        # ---------------------------------------------------------
        while self.iteration < n_trials:
            # 1. Generate random candidate
            config = self._sample_config()
            
            # 2. Evaluate
            scores, _ = self._eval_safe(config)
            
            # 3. Discretize into Epsilon-Bin
            perf_bin = self._discretize(scores)
            
            # 4. Tabu / Redundancy Check
            if perf_bin in self.seen_bins:
                self.strikes += 1
            else:
                self.strikes = 0  # Reset strikes on a novel discovery
                self.seen_bins.add(perf_bin)
                
            # 5. Early Stopping (The DODGE philosophy)
            if self.strikes >= self.patience:
                # DODGE assumes the space is fully mapped / flat and terminates to save budget.
                early_stop_triggered = True
                self.logging_util.log("info", f"DODGE early stopping triggered at iteration {self.iteration} due to {self.patience} redundant strikes.")
                break

        # Added AFTER the main while loop:
        if early_stop_triggered:
            while self.iteration < n_trials:
                self.iteration += 1
                try:
                    # Pad the remaining budget with the best known configuration
                    self.track_evaluation(self.best_config, [0]*self.num_objectives, self.iteration)
                except Exception:
                    self.logging_util.log("iteration", self.iteration)

        self.end_time = time.time()
        return self.best_config, self.best_value