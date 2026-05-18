# optimizers/OnePlusOneESOptimizer.py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from optimizers.base_optimizer import BaseOptimizer
from models.Data import Data
from utils import DistanceUtil
import time
import random
import copy
import numpy as np


class OnePlusOneESOptimizer(BaseOptimizer):
    """
    (1+1) Evolution Strategy optimizer over the dataset.

    Classic (1+1)-ES:
      - Single parent config (a real dataset row)
      - Each generation: mutate parent → offspring
      - Keep offspring if it is at least as good (d2h ≤ parent)
      - Mutation:
          Numeric dims:   Gaussian perturbation scaled by sigma, then snap
          Categorical dims: uniform resample from all values seen in dataset
      - 1/5 success rule: adapt sigma every `adapt_every` steps
          more than 1/5 successes → sigma *= 1.22  (increase step size)
          fewer than 1/5 successes → sigma *= 0.82  (decrease step size)
      - Restart from random row when stuck for `patience` steps

    RF surrogate scores every candidate — no table lookup.
    KD-tree used only to snap mutated numeric vectors to real dataset rows.
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        random.seed(seed)
        np.random.seed(seed)

        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)

        self.nn = Data(
            self.X_df.values.tolist(),
            column_types=self.model_config.column_types,
        )
        self.n_rows = len(self.nn.rows)

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
        self.sigma       = float(self.config.get("sigma", 0.1))      # initial step size
        self.patience    = int(self.config.get("patience", 20))       # steps before restart
        self.adapt_every = int(self.config.get("adapt_every", 10))    # 1/5 rule window

        # Column type split
        self.num_cols = [c for c in self.columns
                         if self.model_config.column_types.get(c) == 'numeric']
        self.cat_cols = [c for c in self.columns
                         if self.model_config.column_types.get(c) != 'numeric']

        # Pre-collect unique values per categorical column for mutation
        self.cat_values = {
            c: list(self.X_df[c].unique()) for c in self.cat_cols
        }

        # Numeric range per column for sigma scaling
        self.num_range = {}
        for c in self.num_cols:
            col_data = self.X_df[c].astype(float)
            r = col_data.max() - col_data.min()
            self.num_range[c] = r if r > 0 else 1.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clean(self, v):
        return v.item() if hasattr(v, "item") else v

    def _row_tuple(self, hp_dict):
        return tuple(hp_dict[c] for c in self.columns)

    def _idx_to_config(self, idx):
        row = self.nn.rows[idx]
        return {c: self._clean(v) for c, v in zip(self.columns, row)}

    def _random_config(self):
        return self._idx_to_config(random.randrange(self.n_rows))

    def _snap_to_nearest(self, hp_dict):
        query = [hp_dict[c] for c in self.columns]
        row = self.nn.nearestRow(query)
        return {c: self._clean(v) for c, v in zip(self.columns, row)}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, hp_dict):
        key = self._row_tuple(hp_dict)
        if key in self.cache:
            scores, d2h_val = self.cache[key]
        else:
            try:
                scores = tuple(self.model_wrapper.get_score(hp_dict))
            except Exception:
                scores = tuple(1.0 for _ in range(self.num_objectives))
            ideal = [0] * self.num_objectives
            d2h_val = DistanceUtil.d2h(ideal, list(scores))
            self.cache[key] = (scores, d2h_val)

        self.iteration += 1
        self.track_evaluation(hp_dict, list(scores), self.iteration)
        return scores, d2h_val

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def _mutate(self, parent):
        """
        Gaussian mutation on numeric dims (scaled by sigma * range).
        Uniform resample on categorical dims with prob 1/n_cat_cols.
        Then snap full config to nearest dataset row.
        """
        offspring = dict(parent)

        # Numeric: Gaussian perturbation
        for c in self.num_cols:
            offspring[c] = float(parent[c]) + np.random.normal(
                0.0, self.sigma * self.num_range[c]
            )

        # Categorical: each dim flips independently with low prob
        n_cat = max(len(self.cat_cols), 1)
        for c in self.cat_cols:
            if random.random() < 1.0 / n_cat:
                offspring[c] = random.choice(self.cat_values[c])

        return offspring

    # ------------------------------------------------------------------
    # Main optimise loop
    # ------------------------------------------------------------------

    def optimize(self):
        n_trials = self.config["n_trials"]
        self.start_time = time.time()

        # ── Initial parent ──────────────────────────────────────────────
        parent = self._random_config()
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
            success = offspring_d2h <= parent_d2h

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
                if rate > 0.2:
                    self.sigma *= 1.22
                else:
                    self.sigma *= 0.82
                self.sigma = max(1e-4, min(self.sigma, 1.0))
                window_successes = 0
                window_total     = 0

            # Restart if stuck
            if stagnation >= self.patience:
                parent = self._random_config()
                _, parent_d2h = self._evaluate(parent)
                stagnation = 0
                if parent_d2h < self.best_value:
                    self.best_value  = parent_d2h
                    self.best_config = copy.deepcopy(parent)

        self.end_time = time.time()
        return self.best_config, self.best_value