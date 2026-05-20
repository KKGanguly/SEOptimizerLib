# optimizers/TabuSearchOptimizer.py
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
from collections import deque
import numpy as np


class TabuSearchOptimizer(BaseOptimizer):
    """
    Tabu Search optimizer over the continuous RF surrogate.

    Classic TS:
      - Start at a random solution (initialized from dataset).
      - At each step, generate `neighbor_size` synthetic continuous neighbours.
      - Move to the best neighbour even if it worsens d2h (unlike hill climbing).
      - But never move to a solution in the tabu list.
      - Add the move to the tabu list, evict oldest if list exceeds tabu_tenure.
      - Aspiration criterion: override tabu if candidate beats best known d2h.
      - Stop when budget exhausted.

    No KD-tree snapping. Operates on the continuous space but rigorously 
    enforces hyperparameter bounds via ConfigSpace.
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

        # Tabu Search parameters
        self.neighbor_size = int(self.config.get("neighbor_size", 10))
        self.tabu_tenure   = int(self.config.get("tabu_tenure", 10))
        self.mutation_rate = float(self.config.get("mutation_rate", 0.2))

        # Tabu list — fixed-size queue of row tuples
        self.tabu_list = deque(maxlen=self.tabu_tenure)

        # Extract bounds and choices from ConfigSpace for safe continuous mutation
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
        """Clean items and round floats slightly to prevent cache bloat and allow Tabu matching."""
        val = v.item() if hasattr(v, "item") else v
        return round(val, 6) if isinstance(val, float) else val

    def _row_tuple(self, hp_dict):
        return tuple(self._safe_clean(hp_dict[c]) for c in self.columns)

    def _idx_to_config(self, idx):
        row = self.X_df.iloc[idx]
        return {c: self._safe_clean(row[c]) for c in self.columns}

    def _random_config(self):
        """Pick a uniformly random row from the dataset as a starting point."""
        return self._idx_to_config(random.randrange(self.n_rows))

    def _is_tabu(self, hp_dict):
        # Because we use _safe_clean, floats are rounded, effectively 
        # making a tiny "tabu region" rather than an infinitely strict point.
        return self._row_tuple(hp_dict) in self.tabu_list

    def _add_tabu(self, hp_dict):
        self.tabu_list.append(self._row_tuple(hp_dict))

    # ------------------------------------------------------------------
    # Evaluation with caching and tracking
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
        return hp_dict, scores, d2h_val

    # ------------------------------------------------------------------
    # Continuous Mutation
    # ------------------------------------------------------------------

    def _mutate(self, config, force_one=False):
        """
        Create a neighbor by perturbing hyperparameters. 
        Matches the mutation logic from Hill Climbing, SA, and ILS exactly.
        """
        mutant = copy.deepcopy(config)
        mutated_any = False
        
        hps = [hp for hp in self.config_space.get_hyperparameters() if not isinstance(hp, Constant)]
        if not hps:
            return mutant
            
        forced_hp = random.choice(hps).name if force_one else None
        
        for hp in hps:
            name = hp.name
            current_val = mutant[name]

            if name != forced_hp and random.random() > self.mutation_rate:
                continue

            if name in self.bounds:
                lower, upper = self.bounds[name]
                span = upper - lower
                std = span * 0.1 # 10% Gaussian noise
                new_val = current_val + random.gauss(0, std)
                
                # Bounds clipping
                new_val = max(lower, min(upper, new_val))
                
                # Integer enforcing
                if self.is_int.get(name, False):
                    new_val = int(round(new_val))
                    
                mutant[name] = new_val
                mutated_any = True

            elif name in self.cat_choices:
                choices = list(self.cat_choices[name])
                if len(choices) > 1:
                    if current_val in choices:
                        choices.remove(current_val)
                    mutant[name] = random.choice(choices)
                    mutated_any = True
                    
        # Recursively force a mutation if pure probability missed everything
        if not mutated_any and not force_one:
            return self._mutate(config, force_one=True)

        return mutant

    def _get_neighbours(self, current_config):
        """Generate `neighbor_size` synthetic continuous neighbors."""
        return [self._mutate(current_config, force_one=False) for _ in range(self.neighbor_size)]

    # ------------------------------------------------------------------
    # Main optimise loop
    # ------------------------------------------------------------------

    def optimize(self):
        n_trials = self.config["n_trials"]
        self.start_time = time.time()

        # ── Random initial solution ─────────────────────────────────────
        current_config = self._random_config()
        current_config, _, current_d2h = self._evaluate(current_config)
        self._add_tabu(current_config)

        self.best_value = current_d2h
        self.best_config = copy.deepcopy(current_config)

        # ── Tabu Search loop ────────────────────────────────────────────
        while self.iteration < n_trials:
            neighbours = self._get_neighbours(current_config)

            best_candidate        = None
            best_candidate_d2h    = float("inf")

            for neighbour in neighbours:
                if self.iteration >= n_trials:
                    break

                is_tabu = self._is_tabu(neighbour)

                key = self._row_tuple(neighbour)
                if key in self.cache:
                    scores, candidate_d2h = self.cache[key]
                    self.iteration += 1
                    self.track_evaluation(neighbour, list(scores), self.iteration)
                else:
                    neighbour, scores, candidate_d2h = self._evaluate(neighbour)

                # Aspiration criterion: override tabu if beats global best
                aspiration = candidate_d2h < self.best_value

                # Accept if (not Tabu OR Aspiration) AND it's the best of this neighborhood batch
                if (not is_tabu or aspiration) and candidate_d2h < best_candidate_d2h:
                    best_candidate     = neighbour
                    best_candidate_d2h = candidate_d2h

            if best_candidate is None:
                # All neighbours were tabu and none triggered aspiration — random restart
                current_config = self._random_config()
                current_config, _, current_d2h = self._evaluate(current_config)
            else:
                # Move to best admissible neighbour (even if it is worse than current_d2h)
                current_config = best_candidate
                current_d2h    = best_candidate_d2h

            # Add current position to tabu list
            self._add_tabu(current_config)

            # Update global best
            if current_d2h < self.best_value:
                self.best_value = current_d2h
                self.best_config = copy.deepcopy(current_config)

        self.end_time = time.time()
        return self.best_config, self.best_value