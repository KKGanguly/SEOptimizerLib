# optimizers/OnePlusOneESOptimizer.py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from optimizers.base_optimizer import BaseOptimizer
from utils import DistanceUtil
import time
import random
import copy
import numpy as np
from ConfigSpace.hyperparameters import (
    OrdinalHyperparameter,
    CategoricalHyperparameter,
    Constant,
)

class OnePlusOneESOptimizer(BaseOptimizer):
    """
    (1+1) Evolution Strategy optimizer over the continuous RF surrogate.

    Classic (1+1)-ES:
      - Single parent config (initialized from dataset)
      - Each generation: mutate parent → offspring
      - Keep offspring if it is at least as good (d2h ≤ parent)
      - Mutation:
          Numeric: Gaussian perturbation scaled by sigma * bounds_span, then clipped.
          Categorical: uniform resample from ConfigSpace choices.
      - 1/5 success rule: adapt sigma every `adapt_every` steps
          more than 1/5 successes → sigma *= 1.22  (increase step size)
          fewer than 1/5 successes → sigma *= 0.82  (decrease step size)
      - Restart from random dataset row when stuck for `patience` steps

    No KD-tree snapping. Operates on the continuous space but rigorously 
    enforces hyperparameter bounds.
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        random.seed(seed)
        np.random.seed(seed)

        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)
        self.n_rows = len(self.X_df)

        self.cache = {}
        self.num_objectives = len(
            self.model_wrapper.get_score(
                {c: self.X_df.iloc[0][c] for c in self.columns}
            )
        )

        self.iteration = 0
        self.best_config = None
        self.best_value = float("inf")

        # ES parameters
        self.sigma       = float(self.config.get("sigma", 0.1))       # initial step size
        self.patience    = int(self.config.get("patience", 20))       # steps before restart
        self.adapt_every = int(self.config.get("adapt_every", 10))    # 1/5 rule window

        # Column type split
        self.num_cols = [c for c in self.columns
                         if self.model_config.column_types.get(c) == 'numeric']
        self.cat_cols = [c for c in self.columns
                         if self.model_config.column_types.get(c) != 'numeric']

        # Extract bounds and choices from ConfigSpace for safe continuous mutation
        self.config_space, _, _ = self.model_config.get_configspace()
        self.bounds = {}
        self.is_int = {}
        self.cat_choices = {}
        
        for hp in self.config_space.get_hyperparameters():
            name = hp.name
            hp_type = type(hp).__name__
            
            if hp_type in ["UniformFloatHyperparameter", "UniformIntegerHyperparameter"]:
                self.bounds[name] = (hp.lower, hp.upper)
                self.is_int[name] = (hp_type == "UniformIntegerHyperparameter")
            elif hp_type in ["CategoricalHyperparameter", "OrdinalHyperparameter"]:
                self.cat_choices[name] = list(hp.choices) if hasattr(hp, 'choices') else list(hp.sequence)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _safe_clean(self, v):
        """Clean items and round floats slightly to prevent cache bloat."""
        val = v.item() if hasattr(v, "item") else v
        return round(val, 6) if isinstance(val, float) else val

    def _row_tuple(self, hp_dict):
        return tuple(self._safe_clean(hp_dict[c]) for c in self.columns)

    def _idx_to_config(self, idx):
        row = self.X_df.iloc[idx]
        return {c: self._safe_clean(row[c]) for c in self.columns}

    def _random_config(self):
        return self._idx_to_config(random.randrange(self.n_rows))

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

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, hp_dict):
        key = self._row_tuple(hp_dict)
        if key in self.cache:
            scores, d2h_val = self.cache[key]
        else:
            try:
                scores, d2h_val = self.model_wrapper.evaluate(hp_dict)
            except Exception as e:
                # Mathematical infinity ensures this configuration is never selected
                scores = tuple(float('inf') for _ in range(self.num_objectives))
                d2h_val = float('inf')
            self.cache[key] = (scores, d2h_val)

        self.iteration += 1
        try:
            self.track_evaluation(hp_dict, list(scores), self.iteration)
        except Exception:
            self.logging_util.log("iteration", self.iteration)
        return scores, d2h_val

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def _mutate(self, parent):
        """
        Gaussian mutation on numeric dims (scaled by sigma * bounds_span), then clipped.
        Uniform resample on categorical dims with prob 1/n_cat_cols.
        """
        offspring = dict(parent)

        # Numeric: Gaussian perturbation safely bounded
        for c in self.num_cols:
            if c in self.bounds:
                lower, upper = self.bounds[c]
                span = upper - lower
                
                # FIX: Catch all small integers (span <= 3), not just binary (span <= 1.0)
                if self.is_int.get(c, False) and span <= 3:
                    current_val = offspring[c]
                    valid_choices = [x for x in range(int(lower), int(upper) + 1) if x != current_val]
                    val = random.choice(valid_choices) if valid_choices else current_val
                else:
                    # Perturb floats and large integers
                    val = float(parent[c]) + np.random.normal(0.0, self.sigma * span)
                    
                    # Clip to bounds
                    val = max(lower, min(upper, val))
                    
                    # Enforce large integers
                    if self.is_int.get(c, False):
                        val = int(round(val))
                    
                offspring[c] = val

        # Categorical: each dim flips independently with low prob
        n_cat = max(len(self.cat_cols), 1)
        for c in self.cat_cols:
            if random.random() < 1.0 / n_cat:
                if c in self.cat_choices:
                    choices = list(self.cat_choices[c])
                    current_val = offspring[c]
                    if len(choices) > 1:
                        if current_val in choices:
                            choices.remove(current_val)
                        offspring[c] = random.choice(choices)

        return offspring

    # ------------------------------------------------------------------
    # Main optimise loop
    # ------------------------------------------------------------------

    def optimize(self):
        n_trials = self.config["n_trials"]
        self.start_time = time.time()

        # ── Initial parent ──────────────────────────────────────────────
        parent = self._sample_config()
        _, parent_d2h = self._evaluate(parent)

        if parent_d2h < self.best_value:
            self.best_value = parent_d2h
            self.best_config = copy.deepcopy(parent)

        # 1/5 rule tracking
        window_successes = 0
        window_total     = 0
        stagnation       = 0

        # ── (1+1)-ES loop ───────────────────────────────────────────────
        while self.iteration < n_trials:
            offspring = self._mutate(parent)
            _, offspring_d2h = self._evaluate(offspring)

            window_total += 1
            success = (offspring_d2h <= parent_d2h) and (offspring_d2h != float('inf'))

            if success:
                parent     = offspring
                parent_d2h = offspring_d2h
                window_successes += 1
                stagnation = 0
                if offspring_d2h < self.best_value:
                    self.best_value  = offspring_d2h
                    self.best_config = copy.deepcopy(offspring)
            else:
                stagnation += 1

            # 1/5 success rule — adapt sigma every adapt_every steps
            if window_total >= self.adapt_every:
                rate = window_successes / window_total
                # Hans-Paul Schwefel's suggestion
                if rate > 0.2:
                    self.sigma *= 1.176
                else:
                    self.sigma *= 0.85
                # Keep sigma within a sane range (e.g. 0.01% to 100% of domain)
                self.sigma = max(1e-4, min(self.sigma, 1.0))
                window_successes = 0
                window_total     = 0

            # Restart if stuck
            if stagnation >= self.patience:
                parent = self._sample_config()
                _, parent_d2h = self._evaluate(parent)
                stagnation = 0
                if parent_d2h < self.best_value:
                    self.best_value  = parent_d2h
                    self.best_config = copy.deepcopy(parent)

        self.end_time = time.time()
        return self.best_config, self.best_value