# optimizers/SimulatedAnnealingOptimizer.py
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
import math
import numpy as np


class SimulatedAnnealingOptimizer(BaseOptimizer):
    """
    Simulated Annealing optimizer.

    Classic SA:
      - Start at a random solution
      - At each step, pick ONE random neighbour
      - If it improves d2h, always accept
      - If it worsens d2h, accept with probability exp(-delta / T)
      - Cool temperature by factor `cooling_rate` each step
      - Stop when budget exhausted or T < min_temp

    Neighbours are real dataset rows via KD-tree (Data.k_nearest_indices).
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

        # SA parameters
        self.neighbor_size = int(self.config.get("neighbor_size", 5))
        self.initial_temp  = float(self.config.get("initial_temp", 1.0))
        self.cooling_rate  = float(self.config.get("cooling_rate", 0.95))
        self.min_temp      = float(self.config.get("min_temp", 1e-5))

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
        """Pick a uniformly random row from the dataset as starting point."""
        idx = random.randrange(len(self.nn.rows))
        return self._idx_to_config(idx)

    # ------------------------------------------------------------------
    # Evaluation with caching and tracking
    # ------------------------------------------------------------------

    def _evaluate(self, hp_dict):
        valid_hp = self._nearest_row(hp_dict)
        key = self._row_tuple(valid_hp)

        if key in self.cache:
            scores, d2h_val = self.cache[key]
        else:
            try:
                scores = tuple(self.model_wrapper.get_score(valid_hp))
            except Exception:
                scores = tuple(1.0 for _ in range(self.num_objectives))
            ideal = [0] * self.num_objectives
            d2h_val = DistanceUtil.d2h(ideal, list(scores))
            self.cache[key] = (scores, d2h_val)

        self.iteration += 1
        self.track_evaluation(valid_hp, list(scores), self.iteration)

        return valid_hp, scores, d2h_val

    # ------------------------------------------------------------------
    # Neighbour generation via KD-tree
    # ------------------------------------------------------------------

    def _random_neighbour(self, current_config):
        """
        Pick ONE random neighbour from the k nearest dataset rows.
        SA moves one step at a time — we do not evaluate all neighbours.
        """
        idx = self._config_to_idx(current_config)

        if idx is None:
            return self._random_config()

        neighbour_indices = self.nn.k_nearest_indices(idx, k=self.neighbor_size)
        chosen_idx = random.choice(neighbour_indices)
        return self._idx_to_config(chosen_idx)

    # ------------------------------------------------------------------
    # Metropolis acceptance criterion
    # ------------------------------------------------------------------

    def _accept(self, current_d2h, candidate_d2h, temperature):
        delta = candidate_d2h - current_d2h
        if delta < 0:
            # Always accept improvements
            return True
        # Accept worse solution with probability exp(-delta / T)
        return random.random() < math.exp(-delta / temperature)

    # ------------------------------------------------------------------
    # Main optimise loop
    # ------------------------------------------------------------------

    def optimize(self):
        n_trials = self.config["n_trials"]
        self.start_time = time.time()

        temperature = self.initial_temp

        # ── Random initial solution ─────────────────────────────────────
        current_config = self._random_config()
        current_config, _, current_d2h = self._evaluate(current_config)

        self.best_value = current_d2h
        self.best_config = copy.deepcopy(current_config)

        # ── SA loop ─────────────────────────────────────────────────────
        while self.iteration < n_trials and temperature > self.min_temp:

            # 1. Pick ONE random neighbour
            candidate = self._random_neighbour(current_config)

            # 2. Evaluate it
            candidate, scores, candidate_d2h = self._evaluate(candidate)

            # 3. Metropolis acceptance
            if self._accept(current_d2h, candidate_d2h, temperature):
                current_config = candidate
                current_d2h = candidate_d2h

            # 4. Update global best (best seen, not current position)
            if current_d2h < self.best_value:
                self.best_value = current_d2h
                self.best_config = copy.deepcopy(current_config)

            # 5. Cool down
            temperature *= self.cooling_rate


        self.end_time = time.time()
        return self.best_config, self.best_value