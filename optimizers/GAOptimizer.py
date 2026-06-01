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
from ConfigSpace.hyperparameters import (
    OrdinalHyperparameter,
    CategoricalHyperparameter,
    Constant,
)

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
        self.pop_size      = int(self.config.get("pop_size", 10))
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
                    current_val = mutant[col]
                    
                    # 1. Zero-Span Protection
                    if span == 0:
                        continue 
                        
                    # 2. THE BOOLEAN / SMALL INTEGER CHECK
                    elif self.is_int.get(col, False) and span <= 3:
                        # Guarantees a mutation for binary [0, 1] and small ordinal ranges
                        valid_choices = [x for x in range(int(lower), int(upper) + 1) if x != current_val]
                        new_val = random.choice(valid_choices) if valid_choices else current_val
                        mutant[col] = new_val
                        
                    # 3. Large Integer Mutation
                    elif self.is_int.get(col, False):
                        # Random integer step scaled by your GA's sigma parameter
                        step_size = max(1, int(span * self.sigma))
                        new_val = current_val + random.randint(-step_size, step_size)
                        
                        # Clip and enforce
                        new_val = max(lower, min(upper, new_val))
                        mutant[col] = int(round(new_val))
                        
                    # 4. Continuous Float Mutation
                    else:
                        # True Gaussian perturbation for floats, regardless of span size
                        new_val = float(current_val) + np.random.normal(0.0, self.sigma * span)
                        
                        # Clip
                        mutant[col] = max(lower, min(upper, new_val))

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
        initial_count = min(self.pop_size, self.config["n_trials"])
        
        # Fill the population up to initial_count
        if hasattr(self, "_sample_config"):
            population = [self._sample_config() for _ in range(initial_count)]
        else:
            population = [self.config_space.sample_configuration().get_dictionary() for _ in range(initial_count)]
            
        # Evaluate
        results = self._evaluate_population(population)
        fitness = [r[1] for r in results]

        # Track initial best
        for ind, (scores, d2h_val) in zip(population, results):
            if d2h_val < self.best_value:
                self.best_value  = d2h_val
                self.best_config = copy.deepcopy(ind)

        # ── Generational GA loop ────────────────────────────────────────
        while self.iteration < n_trials:
            # Sort population by fitness for elitism
            ranked = sorted(zip(fitness, population), key=lambda x: x[0])
            
            # 1. Store both the config AND the fitness for elites
            new_population = [copy.deepcopy(ind) for _, ind in ranked[:self.elitism]]
            new_fitness = [fit for fit, _ in ranked[:self.elitism]]

            # 2. Build ONLY children
            children = []
            while (len(new_population) + len(children)) < self.pop_size:
                parent_a = self._tournament_select(population, fitness)
                parent_b = self._tournament_select(population, fitness)
                child    = self._crossover(parent_a, parent_b)
                child    = self._mutate(child)
                children.append(child)

            # 3. Evaluate ONLY the new children
            child_results = self._evaluate_population(children)
            child_fitness = [r[1] for r in child_results]

            # 4. Update best (checking only the newly evaluated children)
            for ind, (scores, d2h_val) in zip(children, child_results):
                if d2h_val < self.best_value:
                    self.best_value  = d2h_val
                    self.best_config = copy.deepcopy(ind)

            # 5. Merge elites and evaluated children for the next generation
            new_population.extend(children)
            new_fitness.extend(child_fitness)

            population = new_population
            fitness    = new_fitness

        self.end_time = time.time()
        return self.best_config, self.best_value