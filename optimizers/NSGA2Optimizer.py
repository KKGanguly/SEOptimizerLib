from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import optuna
from optuna.samplers import NSGAIISampler
from optimizers.base_optimizer import BaseOptimizer
from models.Data import Data
from ConfigSpace.hyperparameters import (
    OrdinalHyperparameter,
    CategoricalHyperparameter,
    Constant,
)
from utils import DistanceUtil
import time
import numpy as np


class NSGA2Optimizer(BaseOptimizer):
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
        self.population_size = 100

    def _clean(self, v):
        return v.item() if hasattr(v, "item") else v

    def _nearest_row(self, hp_dict):
        query = [hp_dict[c] for c in self.columns]
        row = self.nn.nearestRow(query)
        return {c: self._clean(v) for c, v in zip(self.columns, row)}

    def _row_tuple(self, hp_dict):
        return tuple(hp_dict[c] for c in self.columns)

    def _objective(self, trial):
        raw_hp = {}
        for hp in self.config_space.get_hyperparameters():
            if isinstance(hp, Constant):
                raw_hp[hp.name] = hp.value
            elif isinstance(hp, OrdinalHyperparameter):
                raw_hp[hp.name] = trial.suggest_categorical(hp.name, list(hp.sequence))
            elif isinstance(hp, CategoricalHyperparameter):
                raw_hp[hp.name] = trial.suggest_categorical(hp.name, list(hp.choices))
            else:
                raise ValueError("Unsupported hyperparameter")

        valid_hp = self._nearest_row(raw_hp)
        key = self._row_tuple(valid_hp)

        if key in self.cache:
            scores = self.cache[key]
        else:
            try:
                scores = self.model_wrapper.get_score(valid_hp)
                scores = tuple(1 - s for s in scores)  # flip 1-d2h → d2h
            except Exception:
                scores = tuple(0.0 for _ in range(self.num_objectives))
            self.cache[key] = scores

        self.iteration += 1
        self.track_evaluation(valid_hp, scores, self.iteration)

        return scores

    def _is_dominated(self, obj1, obj2):
        """Return True if obj1 is dominated by obj2 (obj2 is at least as good
        on all objectives and strictly better on at least one)."""
        at_least_as_good = all(b <= a for a, b in zip(obj1, obj2))
        strictly_better  = any(b <  a for a, b in zip(obj1, obj2))
        return at_least_as_good and strictly_better

    

    def optimize(self):
        # 2. Define callback to log Pareto Front every generation
        def log_pareto_front(study, trial):
            # 1. Get current Global Pareto Front (non-dominated trials so far)
            pareto_trials = study.best_trials
            
            # 2. Identify the "Best" in the frontier based on d2h
            best_frontier_trial = None
            min_d2h = float("inf")
            ideal = [0] * self.num_objectives

            for t in pareto_trials:
                current_d2h = DistanceUtil.d2h(ideal, t.values)
                if current_d2h < min_d2h:
                    best_frontier_trial = t

            # 3. Format the config and objectives for the best frontier point
            if best_frontier_trial:
                # Merge trial params with constants from config space
                best_raw = dict(best_frontier_trial.params)
                for hp in self.config_space.get_hyperparameters():
                    if isinstance(hp, Constant):
                        best_raw[hp.name] = hp.value
                
                # Get the nearest valid row for the config
                best_config = self._nearest_row(best_raw)
                
                # Use your existing tracking utility
                # Assuming your base class has a method to store this history
                self.track_frontier(self.iteration, pareto_trials, best_config, best_frontier_trial)
        n_trials = self.config["n_trials"]

        self.start_time = time.time()

        sampler = NSGAIISampler(
            population_size=self.population_size,
            seed=self.seed,
        )

        study = optuna.create_study(
            directions=["minimize"] * self.num_objectives,
            sampler=sampler,
        )

        study.optimize(self._objective, n_trials=n_trials, timeout=3600, catch=(Exception,), callbacks=[log_pareto_front])
        # No second track_frontier call here — every generation was already
        # recorded inside _objective, including the final one.

        # 1. Get the final Global Frontier (the set of all non-dominated trials)
        final_frontier = study.best_trials

        # 2. Extract the absolute best trial based on d2h
        best_trial = None
        best_d2h_norm = float("inf")
        ideal = [0] * self.num_objectives

        for t in final_frontier:
            # We use d2h to pick the "winner" from the multi-objective set
            d2h_val = DistanceUtil.d2h(ideal, t.values)
            if d2h_val < best_d2h_norm:
                best_d2h_norm = d2h_val
                best_trial = t

        # 3. Handle cases where no trials completed (safety check)
        if best_trial is None:
            return None, float("inf")

        # 4. Finalize the configuration (merge constants and map to nearest row)
        best_raw = dict(best_trial.params)
        for hp in self.config_space.get_hyperparameters():
            if isinstance(hp, Constant):
                best_raw[hp.name] = hp.value

        final_hp = self._nearest_row(best_raw)

        # 5. Set the final attributes for reporting
        self.best_config = final_hp
        self.best_value = best_d2h_norm
        self.end_time = time.time()

        return self.best_config, self.best_value
