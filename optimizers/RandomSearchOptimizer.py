from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import random
import numpy as np
from optimizers.base_optimizer import BaseOptimizer
from models.Data import Data
from ConfigSpace.hyperparameters import (
    OrdinalHyperparameter,
    CategoricalHyperparameter,
    Constant,
)
from utils import DistanceUtil
import time


class RandomSearchOptimizer(BaseOptimizer):
    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)
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
        self.population_size = 100  # interval for frontier snapshots

    def _clean(self, v):
        return v.item() if hasattr(v, "item") else v

    def _nearest_row(self, hp_dict):
        query = [hp_dict[c] for c in self.columns]
        row = self.nn.nearestRow(query)
        return {c: self._clean(v) for c, v in zip(self.columns, row)}

    def _row_tuple(self, hp_dict):
        return tuple(hp_dict[c] for c in self.columns)

    def _is_dominated(self, obj1, obj2):
        better_in_any = False
        for a, b in zip(obj1, obj2):
            if a > b:
                better_in_any = True
            elif a < b:
                return False
        return better_in_any

    def optimize(self):
        n_trials = self.config["n_trials"]
        rng = random.Random(self.seed)

        self.start_time = time.time()

        # All evaluated points stored as (hp_dict, scores) for frontier tracking
        all_evaluations = []

        for _ in range(n_trials):
            # Sample a random configuration from ConfigSpace
            raw_hp = {}
            for hp in self.config_space.get_hyperparameters():
                if isinstance(hp, Constant):
                    raw_hp[hp.name] = hp.value
                elif isinstance(hp, OrdinalHyperparameter):
                    raw_hp[hp.name] = rng.choice(list(hp.sequence))
                elif isinstance(hp, CategoricalHyperparameter):
                    raw_hp[hp.name] = rng.choice(list(hp.choices))
                else:
                    raise ValueError("Unsupported hyperparameter")

            valid_hp = self._nearest_row(raw_hp)
            key = self._row_tuple(valid_hp)

            if key in self.cache:
                scores = self.cache[key]
            else:
                try:
                    scores = self.model_wrapper.get_score(valid_hp)
                    scores = tuple(1 - s for s in scores)  # flip for d2h metric
                except Exception:
                    scores = tuple(0.0 for _ in range(self.num_objectives))
                self.cache[key] = scores

            self.iteration += 1
            self.track_evaluation(valid_hp, scores, self.iteration)
            all_evaluations.append((key, scores))

            # Frontier snapshot every population_size evaluations
            if self.iteration % self.population_size == 0:
                generation = self.iteration // self.population_size

                # Evaluations in this window
                gen_start = (generation - 1) * self.population_size
                gen_evals  = all_evaluations[gen_start : gen_start + self.population_size]
                all_scores = [s for _, s in all_evaluations]

                # Count how many of this window's points are non-dominated
                # against the full history so far
                gen_frontier = [
                    s for _, s in gen_evals
                    if not any(
                        self._is_dominated(s, other)
                        for other in all_scores
                        if other is not s
                    )
                ]

                # Pass length as a list so track_frontier gets a sized object
                self.track_frontier(generation, gen_frontier, self.population_size)

        # Best solution by d2h across all evaluated points
        ideal = [0] * self.num_objectives
        best_hp   = None
        best_d2h  = float("inf")

        seen = {}
        for key, scores in all_evaluations:
            seen[key] = scores  # last write wins for duplicates

        # Rebuild hp dicts from cache for best lookup
        key_to_hp = {}
        for hp_dict_key in self.cache:
            key_to_hp[hp_dict_key] = {
                c: v for c, v in zip(self.columns, hp_dict_key)
            }

        for key, scores in seen.items():
            d2h = DistanceUtil.d2h(ideal, scores)
            if d2h < best_d2h:
                best_d2h = d2h
                best_hp  = key_to_hp.get(key)

        self.best_config = best_hp
        self.best_value  = best_d2h
        self.end_time    = time.time()

        return self.best_config, self.best_value
