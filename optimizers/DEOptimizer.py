# optimizers/DEOptimizer.py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from optimizers.base_optimizer import BaseOptimizer
from utils import DistanceUtil
import time
import random
import copy
import numpy as np


class DEOptimizer(BaseOptimizer):
    """
    Differential Evolution optimizer over the continuous RF surrogate.

    Classic DE/rand/1/bin:
      - Population of `pop_size` individuals initialized from dataset.
      - For each target x_i, pick 3 distinct random individuals a, b, c.
      - Mutant v = a + F * (b - c) for numeric, bounds-clipped.
      - Trial u = crossover(x_i, v) per dimension with probability CR.
      - Replace x_i with u if u is strictly better (d2h).

    No KD-tree snapping. The optimizer explores the continuous gaps 
    between dataset rows but respects ConfigSpace bounds.
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

        # DE parameters
        self.pop_size = int(self.config.get("pop_size", 20))
        self.F        = float(self.config.get("F", 0.8))   # mutation factor
        self.CR       = float(self.config.get("CR", 0.9))  # crossover rate

        # Which columns are numeric vs categorical
        self.num_cols  = [c for c in self.columns
                          if self.model_config.column_types.get(c) == 'numeric']
        self.cat_cols  = [c for c in self.columns
                          if self.model_config.column_types.get(c) != 'numeric']

        # Get bounds for clipping continuous variables
        self.config_space, _, _ = self.model_config.get_configspace()
        self.bounds = {}
        self.is_int = {}
        for hp in self.config_space.get_hyperparameters():
            if type(hp).__name__ in ["UniformFloatHyperparameter", "UniformIntegerHyperparameter"]:
                self.bounds[hp.name] = (hp.lower, hp.upper)
                self.is_int[hp.name] = (type(hp).__name__ == "UniformIntegerHyperparameter")

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
        """Initialize from a real dataset row."""
        row = self.X_df.iloc[idx]
        return {c: self._safe_clean(row[c]) for c in self.columns}

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
                ideal = [0] * self.num_objectives
                d2h_val = DistanceUtil.d2h(ideal, list(scores))
            except Exception as e:
                # Mathematical infinity ensures this configuration is never selected
                scores = tuple(float('inf') for _ in range(self.num_objectives))
                d2h_val = float('inf')
            self.cache[key] = (scores, d2h_val)

        self.iteration += 1
        self.track_evaluation(hp_dict, list(scores), self.iteration)
        return scores, d2h_val

    # ------------------------------------------------------------------
    # DE operators
    # ------------------------------------------------------------------

    def _mutate(self, a, b, c):
        """
        DE/rand/1 mutation with bounds clipping.
        """
        mutant = {}
        for col in self.columns:
            if col in self.num_cols:
                mutant[col] = self._mutate_numeric(a, b, c, col)
            else:
                mutant[col] = self._mutate_categorical(b, c, col)
        return mutant

    def _clamp(self, hp_dict):
            """Helper to enforce bounds after any operation."""
            for col, (lower, upper) in self.bounds.items():
                if col in hp_dict:
                    val = max(lower, min(upper, hp_dict[col]))
                    if self.is_int.get(col, False):
                        val = int(round(val))
                    hp_dict[col] = val
            return hp_dict   
    def _mutate_numeric(self, a, b, c, col):
        # Arithmetic mutation
        val = a[col] + self.F * (b[col] - c[col])
        
        # Clip to valid bounds so surrogate doesn't break
        if col in self.bounds:
            lower, upper = self.bounds[col]
            val = max(lower, min(upper, val))
            
        # Enforce integer types
        if self.is_int.get(col, False):
            val = int(round(val))
            
        return val

    def _mutate_categorical(self, a, b, c, col):
        """DE logic for categorical variables."""
        if b[col] == c[col]:
            return a[col]
        return b[col] if random.random() < self.F else c[col]

    def _mutate(self, a, b, c):
        """
        DE/rand/1 mutation: Categorical fix (added missing 'a' arg) 
        and added clamping.
        """
        mutant = {}
        for col in self.columns:
            if col in self.num_cols:
                mutant[col] = self._mutate_numeric(a, b, c, col)
            else:
                # FIX: Passed 'a' to match your function signature
                mutant[col] = self._mutate_categorical(a, b, c, col)
        return self._clamp(mutant)

    def _crossover(self, target, mutant):
        """Binomial crossover followed by mandatory clamping."""
        j_rand = random.randrange(len(self.columns))
        trial = {}
        for i, col in enumerate(self.columns):
            if i == j_rand or random.random() < self.CR:
                trial[col] = mutant[col]
            else:
                trial[col] = target[col]
        # Ensure the blended trial is valid before it hits the surrogate
        return self._clamp(trial)

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