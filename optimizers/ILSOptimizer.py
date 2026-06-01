# optimizers/IteratedLocalSearchOptimizer.py
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


class IteratedLocalSearchOptimizer(BaseOptimizer):
    """
    Iterated Local Search (ILS) optimizer operating over the continuous Random Forest surrogate.

    Classic ILS:
      1. Start at a random solution (drawn from the real dataset to ground it).
      2. Apply hill climbing until local optimum.
      3. Perturb the best solution found so far (Medium random walk).
      4. Repeat from step 2 using the perturbed point as the new start.

    Hill climbing inner loop: evaluates `neighbor_size` synthetic continuous 
    neighbors, moves to best if it strictly improves d2h.

    Perturbation: walks `perturbation_hops` steps away using forced continuous 
    mutation to guarantee it escapes the basin while respecting bounds.
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

        # ILS parameters
        self.neighbor_size      = int(self.config.get("neighbor_size", 5))
        self.perturbation_hops  = int(self.config.get("perturbation_hops", 3))
        self.mutation_rate      = float(self.config.get("mutation_rate", 0.1))

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

    def _idx_to_config(self, idx):
        """Extract a real dataset row to use as an initial starting point."""
        row = self.X_df.iloc[idx]
        return {c: self._clean(row[c]) for c in self.columns}

    def _random_config(self):
        idx = random.randrange(self.n_rows)
        return self._idx_to_config(idx)

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
            # Use the appropriate variable (hp_dict, neighbour, or perturbed)
            self.track_evaluation(hp_dict, list(scores), self.iteration)
        except Exception:
            self.logging_util.log("iteration", self.iteration)
        return scores, d2h_val

    # ------------------------------------------------------------------
    # Continuous Mutation (Shared logic with Hill Climber)
    # ------------------------------------------------------------------
    def _mutate(self, config, force_one=False):
        """
        Create a neighbor by perturbing hyperparameters. 
        If force_one=True, guarantees exactly one random parameter mutates 
        regardless of mutation_rate, while the rest follow probability.
        """
        mutant = copy.deepcopy(config)
        mutated_any = False
        
        hps = [hp for hp in self.config_space.get_hyperparameters() if not isinstance(hp, Constant)]
        if not hps:
            return mutant
            
        # If we must guarantee a change, pick exactly one HP to force
        forced_hp = random.choice(hps).name if force_one else None
        
        for hp in hps:
            name = hp.name
            current_val = mutant[name]

            # Skip mutation if it's not the forced HP AND it fails the probability check
            if name != forced_hp and random.random() > self.mutation_rate:
                continue

            if type(hp).__name__ in ["UniformFloatHyperparameter", "UniformIntegerHyperparameter"]:
                span = hp.upper - hp.lower
                
                # FIX: Catch all small integers, not just binary
                if type(hp).__name__ == "UniformIntegerHyperparameter" and span <= 3:
                    valid_choices = [x for x in range(int(hp.lower), int(hp.upper) + 1) if x != current_val]
                    new_val = random.choice(valid_choices) if valid_choices else current_val
                    mutant[name] = new_val
                else:
                    # Standard Gaussian noise
                    std = span * 0.1
                    new_val = current_val + random.gauss(0, std)
                    
                    # Strict bounds clipping
                    new_val = max(hp.lower, min(hp.upper, new_val))
                    if type(hp).__name__ == "UniformIntegerHyperparameter":
                        new_val = int(round(new_val))
                        
                    mutant[name] = new_val
                mutated_any = True

            elif isinstance(hp, CategoricalHyperparameter):
                choices = list(hp.choices)
                if len(choices) > 1:
                    if current_val in choices:
                        choices.remove(current_val)
                    mutant[name] = random.choice(choices)
                    mutated_any = True

            elif isinstance(hp, OrdinalHyperparameter):
                seq = list(hp.sequence)
                if current_val in seq:
                    idx = seq.index(current_val)
                    moves = []
                    if idx > 0: moves.append(idx - 1)
                    if idx < len(seq) - 1: moves.append(idx + 1)
                    if moves:
                        mutant[name] = seq[random.choice(moves)]
                        mutated_any = True
                    
        # If pure probability missed everything, recursively force exactly one to change
        if not mutated_any and not force_one:
            return self._mutate(config, force_one=True)

        return mutant

    def _get_neighbours(self, current_config):
        """Generate `neighbor_size` synthetic continuous neighbors."""
        return [self._mutate(current_config, force_one=False) for _ in range(self.neighbor_size)]

    # ------------------------------------------------------------------
    # Perturbation — the ILS-specific step
    # ------------------------------------------------------------------

    def _perturb(self, best_config):
        """
        Take `perturbation_hops` guaranteed steps away from the global best.
        force_one=True guarantees it doesn't get stuck generating duplicates.
        """
        current = copy.deepcopy(best_config)
        for _ in range(self.perturbation_hops):
            current = self._mutate(current, force_one=True)
        return current

    # ------------------------------------------------------------------
    # Inner hill climbing
    # ------------------------------------------------------------------

    def _hill_climb(self, start_config, start_d2h):
        """
        Best-improvement hill climbing from start_config.
        """
        current_config = start_config
        current_d2h = start_d2h

        while self.iteration < self.config["n_trials"]:
            neighbours = self._get_neighbours(current_config)

            best_neighbour = None
            best_neighbour_d2h = current_d2h

            for neighbour in neighbours:
                if self.iteration >= self.config["n_trials"]:
                    break

                key = self._row_tuple(neighbour)
                if key in self.cache:
                    scores, new_d2h = self.cache[key]
                    self.iteration += 1
                    try:
                        # Use the appropriate variable (hp_dict, neighbour, or perturbed)
                        self.track_evaluation(neighbour, list(scores), self.iteration)
                    except Exception:
                        self.logging_util.log("iteration", self.iteration)
                else:
                    scores, new_d2h = self._evaluate(neighbour)

                if new_d2h < best_neighbour_d2h:
                    best_neighbour_d2h = new_d2h
                    best_neighbour = neighbour

            if best_neighbour is None:
                # Local optimum reached
                break

            current_config = best_neighbour
            current_d2h = best_neighbour_d2h

        return current_config, current_d2h

    # ------------------------------------------------------------------
    # Main optimise loop
    # ------------------------------------------------------------------

    def optimize(self):
        n_trials = self.config["n_trials"]
        self.start_time = time.time()

        # ── Step 1: Initial solution from dataset ────────────────────
        current_config = self._sample_config()
        _, current_d2h = self._evaluate(current_config)
        
        # ── Step 2: First hill climb ─────────────────────────────────
        current_config, current_d2h = self._hill_climb(current_config, current_d2h)

        self.best_value = current_d2h
        self.best_config = copy.deepcopy(current_config)

        # ── Step 3: ILS loop (Perturb -> Climb) ──────────────────────
        while self.iteration < n_trials:
            
            # Perturb the BEST solution found so far
            perturbed = self._perturb(self.best_config)
            
            # Evaluate the perturbed start point
            key = self._row_tuple(perturbed)
            if key in self.cache:
                scores, perturbed_d2h = self.cache[key]
                self.iteration += 1
                try:
                    # Use the appropriate variable (hp_dict, neighbour, or perturbed)
                    self.track_evaluation(perturbed, list(scores), self.iteration)
                except Exception:
                    self.logging_util.log("iteration", self.iteration)
            else:
                _, perturbed_d2h = self._evaluate(perturbed)

            # Hill climb from the perturbed point
            candidate_config, candidate_d2h = self._hill_climb(perturbed, perturbed_d2h)

            # Update global best
            if candidate_d2h < self.best_value:
                self.best_value = candidate_d2h
                self.best_config = copy.deepcopy(candidate_config)

        self.end_time = time.time()
        return self.best_config, self.best_value