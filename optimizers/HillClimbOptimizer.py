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
import optuna
import time
import random
import copy
import numpy as np

optuna.logging.set_verbosity(optuna.logging.WARNING)


class HillClimbingOptimizer(BaseOptimizer):
    """
    Hill Climbing optimizer using Optuna for random restarts and
    the dataset's KD-tree (via Data.k_nearest_indices) for neighbour
    generation.

    Neighbours are real rows from the dataset — no synthetic configs,
    no snapping artefacts. The KD-tree makes neighbourhood lookup O(log N).

    At each step:
      1. Find the `neighbor_size` nearest dataset rows to the current config.
      2. Evaluate all, move to the best if it improves d2h.
      3. If stuck, restart from a random dataset row (up to `max_restarts`).

    Single-objectivisation via d2h, consistent with the rest of the framework.
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

        # Hill climbing parameters (this is the default in optuna)
        self.neighbor_size = int(self.config.get("neighbor_size", 5))
        self.max_restarts = int(self.config.get("max_restarts", 10))

        # Pre-build row index mapping: tuple(row) → dataset row index
        # so we can call k_nearest_indices by index
        self.row_to_idx = {
            tuple(self.nn._clean(v) if hasattr(v, "item") else v for v in row): i
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
        """Look up the dataset index for a snapped config dict."""
        key = self._row_tuple(hp_dict)
        return self.row_to_idx.get(key, None)

    def _idx_to_config(self, idx):
        """Convert a dataset row index back to a config dict."""
        row = self.nn.rows[idx]
        return {c: self._clean(v) for c, v in zip(self.columns, row)}

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
        Use Data.k_nearest_indices to retrieve the `neighbor_size` closest
        real dataset rows to the current config. These are guaranteed to be
        valid, distinct rows — no synthetic configs, no snapping artefacts.
        """
        idx = self._config_to_idx(current_config)

        if idx is None:
            # Current config not found in index (shouldn't happen after snap)
            # Fall back to random dataset rows
            all_indices = list(range(len(self.nn.rows)))
            sampled = random.sample(all_indices, min(self.neighbor_size, len(all_indices)))
            return [self._idx_to_config(i) for i in sampled]

        neighbour_indices = self.nn.k_nearest_indices(idx, k=self.neighbor_size)
        return [self._idx_to_config(i) for i in neighbour_indices]

    # ------------------------------------------------------------------
    # Random start via Optuna
    # ------------------------------------------------------------------

    def _sample_random_config(self, trial):
        """Sample a fully random config using Optuna trial suggestions."""
        config = {}
        for hp in self.config_space.get_hyperparameters():
            if isinstance(hp, Constant):
                config[hp.name] = hp.value
            elif isinstance(hp, OrdinalHyperparameter):
                config[hp.name] = trial.suggest_categorical(hp.name, list(hp.sequence))
            elif isinstance(hp, CategoricalHyperparameter):
                config[hp.name] = trial.suggest_categorical(hp.name, list(hp.choices))
            else:
                raise ValueError(f"Unsupported hyperparameter type: {type(hp)}")
        return config

    # ------------------------------------------------------------------
    # Main optimise loop
    # ------------------------------------------------------------------

    def optimize(self):
        n_trials = self.config["n_trials"]
        self.start_time = time.time()

        restarts = 0

        sampler = optuna.samplers.RandomSampler(seed=self.iteration)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        # ── Initial random start ────────────────────────────────────────
        trial = study.ask()
        current_config = self._sample_random_config(trial)
        current_config, _, current_d2h = self._evaluate(current_config)
        study.tell(trial, current_d2h)

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
                    neighbour, scores, new_d2h = self._evaluate(neighbour)

                trial = study.ask()
                study.tell(trial, new_d2h)

                if new_d2h < best_neighbour_d2h:
                    best_neighbour_d2h = new_d2h
                    best_neighbour = neighbour

            if best_neighbour is not None:
                # Improvement found — move to best neighbour
                current_config = best_neighbour
                current_d2h = best_neighbour_d2h

                if current_d2h < self.best_value:
                    self.best_value = current_d2h
                    self.best_config = copy.deepcopy(current_config)
            else:
                # Local optimum — restart if budget allows
                if restarts >= self.max_restarts or self.iteration >= n_trials:
                    break

                restarts += 1
                trial = study.ask()
                current_config = self._sample_random_config(trial)
                current_config, _, current_d2h = self._evaluate(current_config)
                study.tell(trial, current_d2h)

                if current_d2h < self.best_value:
                    self.best_value = current_d2h
                    self.best_config = copy.deepcopy(current_config)

          
        self.end_time = time.time()
        return self.best_config, self.best_value