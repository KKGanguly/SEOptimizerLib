# optimizers/TabuSearchOptimizer.py
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
from collections import deque
import numpy as np


class TabuSearchOptimizer(BaseOptimizer):
    """
    Tabu Search optimizer.

    Classic TS:
      - Start at a random solution
      - At each step, evaluate ALL neighbours of current solution
      - Move to the best neighbour even if it worsens d2h (unlike hill climbing)
      - But never move to a solution in the tabu list
      - Add the move to the tabu list, evict oldest if list exceeds tabu_tenure
      - Aspiration criterion: override tabu if candidate beats best known d2h
      - Stop when budget exhausted

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

        # Tabu Search parameters
        self.neighbor_size = int(self.config.get("neighbor_size", 10))
        self.tabu_tenure   = int(self.config.get("tabu_tenure", 10))

        # Tabu list — fixed-size queue of row tuples
        self.tabu_list = deque(maxlen=self.tabu_tenure)

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

    def _is_tabu(self, hp_dict):
        return self._row_tuple(hp_dict) in self.tabu_list

    def _add_tabu(self, hp_dict):
        self.tabu_list.append(self._row_tuple(hp_dict))

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

    def _get_neighbours(self, current_config):
        """
        Retrieve `neighbor_size` nearest dataset rows via KD-tree.
        TS evaluates ALL neighbours to find the best non-tabu move.
        """
        idx = self._config_to_idx(current_config)

        if idx is None:
            all_indices = list(range(len(self.nn.rows)))
            sampled = random.sample(all_indices, min(self.neighbor_size, len(all_indices)))
            return [self._idx_to_config(i) for i in sampled]

        neighbour_indices = self.nn.k_nearest_indices(idx, k=self.neighbor_size)
        return [self._idx_to_config(i) for i in neighbour_indices]

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

                if (not is_tabu or aspiration) and candidate_d2h < best_candidate_d2h:
                    best_candidate     = neighbour
                    best_candidate_d2h = candidate_d2h

            if best_candidate is None:
                # All neighbours tabu and none triggered aspiration — random restart
                current_config = self._random_config()
                current_config, _, current_d2h = self._evaluate(current_config)
            else:
                # Move to best admissible neighbour (may be worse than current)
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