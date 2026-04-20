# optimizers/GPOptimizer.py
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
import numpy as np

optuna.logging.set_verbosity(optuna.logging.WARNING)


class GPOptimizer(BaseOptimizer):
    """
    Gaussian Process-based Bayesian Optimization using Optuna's built-in
    GPSampler (Matern 5/2 kernel with ARD, logEI acquisition function).

    No encoding needed — GPSampler handles categoricals and ordinals
    natively via exhaustive/line search on the acquisition function.

    Single-objectivisation via d2h, consistent with the rest of the framework.
    """

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
        self.best_config = None
        self.best_value = float("inf")

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

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

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
                raise ValueError(f"Unsupported hyperparameter type: {type(hp)}")

        valid_hp = self._nearest_row(raw_hp)
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

        return d2h_val

    # ------------------------------------------------------------------
    # Main optimise loop
    # ------------------------------------------------------------------

    def optimize(self):
        n_trials = self.config["n_trials"]
        n_params = sum(
            1 for hp in self.config_space.get_hyperparameters()
            if not isinstance(hp, Constant)
        )

        self.start_time = time.time()

        sampler = optuna.samplers.GPSampler(
            seed                   = self.seed,
            n_startup_trials       = self.config.get("n_startup_trials", min(10, n_params * 2)),
            deterministic_objective= self.config.get("deterministic_objective", False),
        )

        study = optuna.create_study(direction="minimize", sampler=sampler)

        def callback(study, trial):
            if study.best_value < self.best_value:
                self.best_value = study.best_value
                raw_best = dict(study.best_trial.params)
                for hp in self.config_space.get_hyperparameters():
                    if isinstance(hp, Constant):
                        raw_best[hp.name] = hp.value
                self.best_config = self._nearest_row(raw_best)

        study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=3600,
            catch=(Exception,),
            callbacks=[callback],
        )

        self.end_time = time.time()
        return self.best_config, self.best_value