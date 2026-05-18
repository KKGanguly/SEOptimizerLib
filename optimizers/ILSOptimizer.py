# optimizers/IteratedLocalSearchOptimizer.py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from optimizers.base_optimizer import BaseOptimizer
from models.Data import Data
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
    Iterated Local Search (ILS) optimizer.

    Classic ILS:
      1. Start at a random solution
      2. Apply hill climbing until local optimum
      3. Perturb the best solution found so far (not the local optimum)
      4. Repeat from step 2 using the perturbed point as the new start
      5. Accept new solution if it improves best (or use acceptance criterion)

    Perturbation for discrete data: walk `perturbation_hops` steps away
    from the best solution using the KD-tree neighbourhood, giving a
    starting point that is near-but-not-identical to the best basin.

    Hill climbing inner loop: evaluate all `neighbor_size` KD-tree
    neighbours, move to best if it improves d2h, stop when no improvement.

    Single-objectivisation via d2h.
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

        self.config_space, _, _ = self.model_config.get_configspace()
        self.cache = {}

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
        self.max_restarts       = int(self.config.get("max_restarts", 10))

        # Row index: tuple(row) → dataset index for KD-tree lookup
        self.row_to_idx = {
            tuple(v.item() if hasattr(v, "item") else v for v in row): i
            for i, row in enumerate(self.nn.rows)
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clean(self, v):
        return v.item() if hasattr(v, "item") else v

    def _nearest_row(self, hp_dict):
        query = [hp_dict[c] for c in self.columns]
        row = self.nn.nearestRow(query)
        return {c: self._clean(v) for c, v in zip(self.columns, row)}

    def _row_tuple(self, hp_dict):
        return tuple(hp_dict[c] for c in self.columns)

    def _config_to_idx(self, hp_dict):
        return self.row_to_idx.get(self._row_tuple(hp_dict), None)

    def _idx_to_config(self, idx):
        row = self.nn.rows[idx]
        return {c: self._clean(v) for c, v in zip(self.columns, row)}

    def _random_config(self):
        idx = random.randrange(len(self.nn.rows))
        return self._idx_to_config(idx)

    # ------------------------------------------------------------------
    # Evaluation with caching and tracking
    # ------------------------------------------------------------------
    def _evaluate(self, hp_dict):
        """RF surrogate scoring — no table lookup."""
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
    # Neighbour generation via KD-tree
    # ------------------------------------------------------------------

    def _get_neighbours(self, current_config):
        idx = self._config_to_idx(current_config)

        if idx is None:
            all_indices = list(range(len(self.nn.rows)))
            sampled = random.sample(all_indices, min(self.neighbor_size, len(all_indices)))
            return [self._idx_to_config(i) for i in sampled]

        neighbour_indices = self.nn.k_nearest_indices(idx, k=self.neighbor_size)
        return [self._idx_to_config(i) for i in neighbour_indices]

    # ------------------------------------------------------------------
    # Perturbation — the ILS-specific step
    # ------------------------------------------------------------------

    def _perturb(self, best_config):
        """
        Walk `perturbation_hops` random KD-tree steps away from best_config.
        Each hop picks one random neighbour from the current position,
        producing a starting point that is near-but-outside the best basin.

        This is the discrete analogue of:
            start_pt = best + randn(len(bounds)) * p_size
        from the reference implementation, but respects the actual data
        geometry instead of assuming a continuous space.
        """
        current = best_config
        visited = {self._row_tuple(current)}

        for _ in range(self.perturbation_hops):
            neighbours = self._get_neighbours(current)
            # Prefer unvisited neighbours to avoid immediately looping back
            unvisited = [n for n in neighbours if self._row_tuple(n) not in visited]
            candidates = unvisited if unvisited else neighbours
            if not candidates:
                break
            current = random.choice(candidates)
            visited.add(self._row_tuple(current))

        return current

    # ------------------------------------------------------------------
    # Inner hill climbing — runs until local optimum or budget exhausted
    # ------------------------------------------------------------------

    def _hill_climb(self, start_config, start_d2h):
        """
        Best-improvement hill climbing from start_config.
        Evaluates all neighbours, moves to best if it improves d2h.
        Stops when no neighbour improves (local optimum).
        Returns the local optimum config and its d2h.
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
                    self.track_evaluation(neighbour, list(scores), self.iteration)
                else:
                    neighbour, scores, new_d2h = self._evaluate(neighbour)

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

        # ── Step 1: random initial solution ────────────────────────────
        current_config = self._random_config()
        current_config, _, current_d2h = self._evaluate(current_config)

        # ── Step 2: first hill climb ────────────────────────────────────
        current_config, current_d2h = self._hill_climb(current_config, current_d2h)

        self.best_value = current_d2h
        self.best_config = copy.deepcopy(current_config)

        # ── Step 3: ILS restarts ────────────────────────────────────────
        for _ in range(self.max_restarts):
            if self.iteration >= n_trials:
                break

            # Perturb the BEST solution found so far (not the local optimum)
            # This is the defining characteristic of ILS vs random restarts
            perturbed = self._perturb(self.best_config)
            perturbed, _, perturbed_d2h = self._evaluate(perturbed)

            # Hill climb from the perturbed point
            candidate_config, candidate_d2h = self._hill_climb(perturbed, perturbed_d2h)

            # Update global best
            if candidate_d2h < self.best_value:
                self.best_value = candidate_d2h
                self.best_config = copy.deepcopy(candidate_config)

        self.end_time = time.time()
        return self.best_config, self.best_value