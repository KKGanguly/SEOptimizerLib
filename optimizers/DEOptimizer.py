# optimizers/DEOptimizer.py
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


class DEOptimizer(BaseOptimizer):
    """
    Differential Evolution optimizer over the dataset.

    Classic DE/rand/1/bin:
      - Population of `pop_size` real dataset rows (by index)
      - For each target x_i, pick 3 distinct random individuals a, b, c
      - Mutant v = snap(a + F * (b - c))  →  nearest dataset row to the
        perturbed numeric vector
      - Trial u = crossover(x_i, v) per dimension with probability CR
      - Replace x_i with u if u is better (d2h)

    Categorical dimensions: crossover only (no arithmetic mutation).
    Mutation is numeric-only; categorical dims are inherited from 'a'.
    RF surrogate scores every candidate — no table lookup.
    KD-tree used only to snap mutant vectors back to real dataset rows.
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        random.seed(seed)
        np.random.seed(seed)

        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)
        self.n_cols = len(self.columns)

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

        # DE parameters
        self.pop_size = int(self.config.get("pop_size", 20))
        self.F        = float(self.config.get("F", 0.8))   # mutation factor
        self.CR       = float(self.config.get("CR", 0.9))  # crossover rate

        # Which columns are numeric vs categorical
        self.num_cols  = [c for c in self.columns
                          if self.model_config.column_types.get(c) == 'numeric']
        self.cat_cols  = [c for c in self.columns
                          if self.model_config.column_types.get(c) != 'numeric']

        # Build col → position index for fast access
        self.col_idx = {c: i for i, c in enumerate(self.columns)}

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

    def _snap_to_nearest(self, hp_dict):
        """Snap an arbitrary (possibly synthetic) config to nearest dataset row."""
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
    # DE operators
    # ------------------------------------------------------------------

    def _mutate(self, a, b, c):
        """
        DE/rand/1 mutation.
        Numeric dims:     v[i] = a[i] + F * (b[i] - c[i])
        Categorical dims: v[i] = a[i]  (no arithmetic meaning)
        Then snap the whole vector to the nearest real dataset row.
        """
        mutant = {}
        for col in self.columns:
            if col in self.num_cols:
                mutant[col] = self._mutate_numeric(a, b, c, col)
            else:
                mutant[col] = self._mutate_categorical(a, b, c, col)
        return mutant

    def _mutate_numeric(self, a, b, c, col):
        return a[col] + self.F * (b[col] - c[col])

    def _mutate_categorical(self, a, b, c, col):

        if random.random() >= self.CR:
            return a[col]

        return b[col] if random.random() < self.F else c[col]

    def _crossover(self, target, mutant):
        """
        Binomial crossover.
        Each dimension copied from mutant with prob CR.
        At least one dimension guaranteed from mutant (j_rand).
        """
        j_rand = random.randrange(self.n_cols)
        trial = {}
        for i, col in enumerate(self.columns):
            if i == j_rand or random.random() < self.CR:
                trial[col] = mutant[col]
            else:
                trial[col] = target[col]
        return trial

    # ------------------------------------------------------------------
    # Main optimise loop
    # ------------------------------------------------------------------

    def optimize(self):
        n_trials = self.config["n_trials"]
        self.start_time = time.time()

        # ── Initialise population with random dataset rows ──────────────
        indices = random.sample(range(self.n_rows), min(self.pop_size, self.n_rows))
        population = [self._idx_to_config(i) for i in indices]
        fitness = []

        for ind in population:
            if self.iteration >= n_trials:
                break
            scores, d2h_val = self._evaluate(ind)
            fitness.append(d2h_val)
            if d2h_val < self.best_value:
                self.best_value = d2h_val
                self.best_config = copy.deepcopy(ind)

        # ── DE main loop ────────────────────────────────────────────────
        while self.iteration < n_trials:
            new_population = []
            new_fitness    = []

            for i, target in enumerate(population):
                if self.iteration >= n_trials:
                    new_population.append(target)
                    new_fitness.append(fitness[i])
                    continue

                # Pick 3 distinct individuals ≠ i
                candidates = [j for j in range(len(population)) if j != i]
                a_idx, b_idx, c_idx = random.sample(candidates, 3)
                a, b, c = population[a_idx], population[b_idx], population[c_idx]

                # Mutate + crossover
                mutant = self._mutate(a, b, c)
                trial  = self._crossover(target, mutant)

                # Score trial
                scores, trial_d2h = self._evaluate(trial)

                # Selection: greedy replacement
                if trial_d2h <= fitness[i]:
                    new_population.append(trial)
                    new_fitness.append(trial_d2h)
                    if trial_d2h < self.best_value:
                        self.best_value = trial_d2h
                        self.best_config = copy.deepcopy(trial)
                else:
                    new_population.append(target)
                    new_fitness.append(fitness[i])

            population = new_population
            fitness    = new_fitness

        self.end_time = time.time()
        return self.best_config, self.best_value