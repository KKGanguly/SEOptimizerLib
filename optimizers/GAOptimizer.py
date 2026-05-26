# optimizers/GAOptimizer.py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from optimizers.base_optimizer import BaseOptimizer
from utils import DistanceUtil
import time
import random
import copy
import numpy as np


class GAOptimizer(BaseOptimizer):
    """
    Genetic Algorithm optimizer over the continuous RF surrogate.

    Standard generational GA:
      - Population of `pop_size` initialized from real dataset rows.
      - Selection:   tournament selection (size `tournament_k`).
      - Crossover:   uniform crossover per dimension with prob 0.5.
      - Mutation:    each dimension mutates with probability `mutation_rate`.
                     Numeric: Gaussian perturbation clipped to ConfigSpace bounds.
                     Categorical: uniform resample from ConfigSpace choices.
      - Elitism:     top `elitism` individuals survive unchanged.
      - Replacement: full generational replacement.

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
                    results.append((tuple(float('inf') for _ in range(self.num_objectives)), float('inf')))
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
        """
        child = {}
        for col in self.columns:
            child[col] = parent_a[col] if random.random() < 0.5 else parent_b[col]
        return child

    def _mutate(self, individual):
        """
        Per-dimension mutation with probability mutation_rate.
        Numeric: Gaussian perturbation constrained by ConfigSpace bounds.
        Categorical: Uniform resample from ConfigSpace choices.
        """
        mutant = dict(individual)

        for col in self.columns:
            if random.random() < self.mutation_rate:
                
                if col in self.num_cols and col in self.bounds:
                    lower, upper = self.bounds[col]
                    span = upper - lower
                    if span <= 1.0:
                        # Uniformly resample across the entire bound to guarantee a chance to flip
                        new_val = random.uniform(hp.lower, hp.upper)
                    else:
                        # Add Gaussian noise scaled to the parameter's range
                        new_val = float(individual[col]) + np.random.normal(0.0, self.sigma * span)
                    
                    # Clip to bounds
                    new_val = max(lower, min(upper, new_val))
                    
                    # Enforce integer constraints
                    if self.is_int.get(col, False):
                        new_val = int(round(new_val))
                        
                    mutant[col] = new_val

                elif col in self.cat_cols and col in self.cat_choices:
                    choices = list(self.cat_choices[col])
                    current_val = mutant[col]
                    
                    if len(choices) > 1:
                        if current_val in choices:
                            choices.remove(current_val)
                        mutant[col] = random.choice(choices)

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