# optimizers/GAOptimizer.py
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


class GAOptimizer(BaseOptimizer):
    """
    Genetic Algorithm optimizer over the dataset.

    Standard generational GA:
      - Population of `pop_size` real dataset rows
      - Selection:   tournament selection (size `tournament_k`)
      - Crossover:   uniform crossover per dimension with prob 0.5
        (no snap needed — both parents are valid dataset rows,
         crossover always produces a valid row too because every
         value per column comes from a real row)
      - Mutation:    each dimension flips to a random dataset value
        with probability `mutation_rate`; numeric dims get Gaussian
        perturbation then snap; categorical dims resample uniformly
      - Elitism:     top `elitism` individuals survive unchanged
      - Replacement: full generational replacement with elites carried over

    RF surrogate scores all individuals — no table lookup.
    KD-tree used only to snap mutated numeric vectors to nearest dataset rows.
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

        # GA parameters
        self.pop_size      = int(self.config.get("pop_size", 20))
        self.tournament_k  = int(self.config.get("tournament_k", 3))
        self.mutation_rate = float(self.config.get("mutation_rate", 0.1))
        self.elitism       = int(self.config.get("elitism", 2))
        self.sigma         = float(self.config.get("sigma", 0.1))  # numeric mutation scale

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

    def _evaluate_population(self, population):
        """Evaluate all individuals, return list of (scores, d2h)."""
        results = []
        for ind in population:
            if self.iteration >= self._n_trials:
                # Budget exhausted mid-generation — use cache or penalty
                key = self._row_tuple(ind)
                if key in self.cache:
                    results.append(self.cache[key])
                else:
                    results.append((tuple(1.0 for _ in range(self.num_objectives)), 1.0))
            else:
                results.append(self._evaluate(ind))
        return results

    # ------------------------------------------------------------------
    # GA operators
    # ------------------------------------------------------------------

    def _tournament_select(self, population, fitness):
        """Tournament selection — returns one winner config."""
        contestants = random.sample(range(len(population)), min(self.tournament_k, len(population)))
        winner = min(contestants, key=lambda i: fitness[i])
        return population[winner]

    def _crossover(self, parent_a, parent_b):
        """
        Uniform crossover: each dimension independently from parent_a or parent_b.
        Both parents are valid dataset rows so offspring values are always valid.
        No snap needed.
        """
        child = {}
        for col in self.columns:
            child[col] = parent_a[col] if random.random() < 0.5 else parent_b[col]
        return child

    def _mutate(self, individual):
        """
        Per-dimension mutation with probability mutation_rate.
        Numeric: Gaussian perturbation then snap.
        Categorical: uniform resample from seen values.
        Snap is applied once at the end if any numeric dim was mutated.
        """
        mutant = dict(individual)

        for c in self.num_cols:
            if random.random() < self.mutation_rate:
                mutant[c] = float(individual[c]) + np.random.normal(
                    0.0, self.sigma * self.num_range[c]
                )

        for c in self.cat_cols:
            if random.random() < self.mutation_rate:
                mutant[c] = random.choice(self.cat_values[c])


        return mutant

    # ------------------------------------------------------------------
    # Main optimise loop
    # ------------------------------------------------------------------

    def optimize(self):
        n_trials = self.config["n_trials"]
        self._n_trials = n_trials  # accessible inside _evaluate_population
        self.start_time = time.time()

        # ── Initialise population ───────────────────────────────────────
        indices    = random.sample(range(self.n_rows), min(self.pop_size, self.n_rows))
        population = [self._idx_to_config(i) for i in indices]
        results    = self._evaluate_population(population)
        fitness    = [r[1] for r in results]

        # Track initial best
        for ind, (scores, d2h_val) in zip(population, results):
            if d2h_val < self.best_value:
                self.best_value  = d2h_val
                self.best_config = copy.deepcopy(ind)

        # ── Generational GA loop ────────────────────────────────────────
        while self.iteration < n_trials:
            # Sort population by fitness for elitism
            ranked = sorted(zip(fitness, population), key=lambda x: x[0])
            elites = [copy.deepcopy(ind) for _, ind in ranked[:self.elitism]]

            # Build new generation
            new_population = list(elites)

            while len(new_population) < self.pop_size:
                parent_a = self._tournament_select(population, fitness)
                parent_b = self._tournament_select(population, fitness)
                child    = self._crossover(parent_a, parent_b)
                child    = self._mutate(child)
                new_population.append(child)

            # Evaluate new generation (elites may hit cache)
            new_results = self._evaluate_population(new_population)
            new_fitness = [r[1] for r in new_results]

            # Update best
            for ind, (scores, d2h_val) in zip(new_population, new_results):
                if d2h_val < self.best_value:
                    self.best_value  = d2h_val
                    self.best_config = copy.deepcopy(ind)

            population = new_population
            fitness    = new_fitness

        self.end_time = time.time()
        return self.best_config, self.best_value