# optimizers/HillClimbingOptimizer.py
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
import time
import random
import copy
import numpy as np


class HillClimbingOptimizer(BaseOptimizer):
    """
    Stochastic Hill Climbing optimizer over the continuous Random Forest surrogate.

    At each step:
      1. Generate `neighbor_size` synthetic neighbors by mutating the current config.
      2. Evaluate all, move to the best if it strictly improves d2h.
      3. If stuck (no neighbors improve), generate a new batch of neighbors 
         around the same point until the budget is exhausted.
      
    Numeric dimensions are perturbed with Gaussian noise and clipped to valid bounds.
    Categorical dimensions are uniformly resampled.
    No random restarts.
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
        
        assert hasattr(self.model_wrapper, 'rf_model'), \
            "ModelWrapper must have RF trained before optimizer init"
            
        self.num_objectives = len(
            self.model_wrapper.get_score(
                {c: self.X_df.iloc[0][c] for c in self.columns}
            )
        )

        self.iteration = 0
        self.best_config = None
        self.best_value = float("inf")

        # Hill climbing parameters
        self.neighbor_size = int(self.config.get("neighbor_size", 5))
        self.mutation_rate = float(self.config.get("mutation_rate", 0.1))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clean(self, v):
        return v.item() if hasattr(v, "item") else v

    def _row_tuple(self, hp_dict):
        """Round floats slightly to prevent hash misses in cache."""
        return tuple(
            round(hp_dict[c], 6) if isinstance(hp_dict[c], float) else hp_dict[c]
            for c in self.columns
        )

    def _idx_to_config(self, idx):
        """Extract a real dataset row to use as a starting point."""
        row = self.X_df.iloc[idx]
        return {c: self._clean(row[c]) for c in self.columns}

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
            except Exception:
                #scores = tuple(1.0 for _ in range(self.num_objectives))
                scores = tuple(float('inf') for _ in range(self.num_objectives))
                d2h_val = float('inf')
            
            self.cache[key] = (scores, d2h_val)

        self.iteration += 1
        try:
        # Use the appropriate config variable name depending on where you are pasting this
            self.track_evaluation(hp_dict, list(scores), self.iteration) 
        except Exception:
            self.logging_util.log("iteration", self.iteration)
        return scores, d2h_val

    # ------------------------------------------------------------------
    # Neighbor Generation (Mutation)
    # ------------------------------------------------------------------
    def _mutate(self, config):
        """Create a neighbor by mutating some hyperparameters."""
        mutant = copy.deepcopy(config)
        
        for hp in self.config_space.get_hyperparameters():
            if isinstance(hp, Constant):
                continue
                
            # Only mutate a subset of dimensions to keep it a "neighbor"
            if random.random() > self.mutation_rate:
                continue

            name = hp.name
            current_val = mutant[name]

            if type(hp).__name__ in ["UniformFloatHyperparameter", "UniformIntegerHyperparameter"]:
                span = hp.upper - hp.lower
                
                # FIX: Catch all small integers, not just binary
                if type(hp).__name__ == "UniformIntegerHyperparameter" and span <= 3:
                    valid_choices = [x for x in range(int(hp.lower), int(hp.upper) + 1) if x != current_val]
                    new_val = random.choice(valid_choices) if valid_choices else current_val
                    mutant[name] = new_val
                else:
                    # Standard Gaussian noise for floats and large integers
                    std = span * 0.1
                    new_val = current_val + random.gauss(0, std)
                    new_val = max(hp.lower, min(hp.upper, new_val))
                    if type(hp).__name__ == "UniformIntegerHyperparameter":
                        new_val = int(round(new_val))
                    mutant[name] = new_val

            elif isinstance(hp, CategoricalHyperparameter):
                choices = list(hp.choices)
                if len(choices) > 1:
                    choices.remove(current_val)
                    mutant[name] = random.choice(choices)

            elif isinstance(hp, OrdinalHyperparameter):
                seq = list(hp.sequence)
                idx = seq.index(current_val)
                moves = []
                if idx > 0: moves.append(idx - 1)
                if idx < len(seq) - 1: moves.append(idx + 1)
                if moves:
                    mutant[name] = seq[random.choice(moves)]

        return mutant

    def _get_neighbours(self, current_config):
        """Generate `neighbor_size` synthetic neighbors."""
        neighbours = []
        for _ in range(self.neighbor_size):
            neighbours.append(self._mutate(current_config))
        return neighbours

    # ------------------------------------------------------------------
    # Main optimize loop
    # ------------------------------------------------------------------

    def optimize(self):
        n_trials = self.config["n_trials"]
        self.start_time = time.time()

        current_config = self._sample_config()
        _, current_d2h = self._evaluate(current_config)

        if current_d2h < self.best_value:
            self.best_value = current_d2h
            self.best_config = copy.deepcopy(current_config)

        # ── Hill climbing loop ──────────────────────────────────────────
        while self.iteration < n_trials:
            neighbours = self._get_neighbours(current_config)

            best_neighbour = None
            best_neighbour_d2h = current_d2h

            for neighbour in neighbours:
                if self.iteration >= n_trials:
                    break

                key = self._row_tuple(neighbour)
                if key in self.cache:
                    scores, new_d2h = self.cache[key]
                    self.iteration += 1
                    self.track_evaluation(neighbour, list(scores), self.iteration)
                else:
                    scores, new_d2h = self._evaluate(neighbour)

                # Strict improvement needed to climb
                if new_d2h < best_neighbour_d2h:
                    best_neighbour_d2h = new_d2h
                    best_neighbour = neighbour

            if best_neighbour is not None:
                # Improvement found — step to best neighbor
                current_config = best_neighbour
                current_d2h = best_neighbour_d2h

                if current_d2h < self.best_value:
                    self.best_value = current_d2h
                    self.best_config = copy.deepcopy(current_config)
            
            # If best_neighbour is None, we are stuck at a local optimum.
            # Without restarts, the loop simply continues and generates a 
            # *new* batch of stochastic neighbors around the current_config.

        self.end_time = time.time()
        return self.best_config, self.best_value