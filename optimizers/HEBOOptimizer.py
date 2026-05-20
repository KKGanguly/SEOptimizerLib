# optimizers/HEBOOptimizer.py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from optimizers.base_optimizer import BaseOptimizer
from ConfigSpace.hyperparameters import (
    OrdinalHyperparameter,
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    Constant,
)
from utils import DistanceUtil
import optuna
import optunahub
import time
import random
import copy
import numpy as np

optuna.logging.set_verbosity(optuna.logging.WARNING)


class HEBOOptimizer(BaseOptimizer):
    """
    HEBO optimizer via optunahub's HEBOSampler.

    Survival Mode for High Cardinality:
    - Native categorical handling in HEBO causes a dimension explosion.
    - To prevent OOM errors and infinite runtimes, Categorical/Ordinal 
      variables are intentionally mapped to a 1D continuous float [0, 1].
    - This is mathematically sub-optimal for a GP, but computationally mandatory.
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)
        self.n_rows = len(self.X_df)

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

        self._build_codec()

    # ------------------------------------------------------------------
    # Codec for High Cardinality Survival
    # ------------------------------------------------------------------

    def _build_codec(self):
        """Pre-extract choices to avoid doing it every trial."""
        self.cat_codec = {}
        for hp in self.config_space.get_hyperparameters():
            if isinstance(hp, CategoricalHyperparameter):
                self.cat_codec[hp.name] = list(hp.choices)
            elif isinstance(hp, OrdinalHyperparameter):
                self.cat_codec[hp.name] = list(hp.sequence)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _safe_clean(self, v):
        val = v.item() if hasattr(v, "item") else v
        return round(val, 6) if isinstance(val, float) else val

    def _row_tuple(self, hp_dict):
        return tuple(self._safe_clean(hp_dict[c]) for c in self.columns)

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def _objective(self, trial):
        config = {}
        
        # 1. Map to Optuna
        for hp in self.config_space.get_hyperparameters():
            hp_type = type(hp).__name__
            name = hp.name
            
            if isinstance(hp, Constant):
                config[name] = hp.value
                
            elif hp_type == "UniformFloatHyperparameter":
                config[name] = trial.suggest_float(name, hp.lower, hp.upper)
                
            elif hp_type == "UniformIntegerHyperparameter":
                config[name] = trial.suggest_int(name, int(hp.lower), int(hp.upper))
                
            elif name in self.cat_codec:
                # THE HACK: Map massive categoricals to a single 1D float slider
                encoded_val = trial.suggest_float(name, 0.0, 1.0)
                choices = self.cat_codec[name]
                
                # Decode float back to discrete choice immediately
                if len(choices) == 1:
                    config[name] = choices[0]
                else:
                    idx = int(round(encoded_val * (len(choices) - 1)))
                    config[name] = choices[max(0, min(len(choices) - 1, idx))]

        # 2. Score via RF surrogate
        key = self._row_tuple(config)
        if key in self.cache:
            scores, d2h_val = self.cache[key]
        else:
            try:
                scores = tuple(self.model_wrapper.get_score(config))
            except Exception:
                scores = tuple(1.0 for _ in range(self.num_objectives))
            ideal = [0] * self.num_objectives
            d2h_val = DistanceUtil.d2h(ideal, list(scores))
            self.cache[key] = (scores, d2h_val)

        # 3. Track evaluation
        self.iteration += 1
        self.track_evaluation(config, list(scores), self.iteration)
        
        if d2h_val < self.best_value:
            self.best_value = d2h_val
            self.best_config = copy.deepcopy(config)

        return d2h_val

    # ------------------------------------------------------------------
    # Main optimise loop
    # ------------------------------------------------------------------

    def optimize(self):
        n_trials = self.config["n_trials"]
        self.start_time = time.time()

        # ── Safety Baseline ───────────────────────────────────────────
        initial_idx = random.randrange(self.n_rows)
        initial_row = self.X_df.iloc[initial_idx]
        baseline_config = {c: self._safe_clean(initial_row[c]) for c in self.columns}
        
        try:
            scores = tuple(self.model_wrapper.get_score(baseline_config))
        except Exception:
            scores = tuple(1.0 for _ in range(self.num_objectives))
            
        ideal = [0] * self.num_objectives
        self.best_value = DistanceUtil.d2h(ideal, list(scores))
        self.best_config = copy.deepcopy(baseline_config)

        # ── HEBO Optimization ─────────────────────────────────────────
        try:
            module = optunahub.load_module("samplers/hebo")
            sampler = module.HEBOSampler(seed=self.seed)
            study = optuna.create_study(direction="minimize", sampler=sampler)

            study.optimize(
                self._objective,
                n_trials=n_trials,
                timeout=3600
            )
            
            if study.best_value < self.best_value:
                self.best_value = study.best_value
                
                # Reconstruct best config using the exact logic from _objective
                best_config = {}
                for hp in self.config_space.get_hyperparameters():
                    name = hp.name
                    if isinstance(hp, Constant):
                        best_config[name] = hp.value
                    elif name in self.cat_codec:
                        encoded_val = study.best_trial.params[name]
                        choices = self.cat_codec[name]
                        if len(choices) == 1:
                            best_config[name] = choices[0]
                        else:
                            idx = int(round(encoded_val * (len(choices) - 1)))
                            best_config[name] = choices[max(0, min(len(choices) - 1, idx))]
                    else:
                        best_config[name] = study.best_trial.params[name]
                        
                self.best_config = best_config

        except Exception as e:
            print(f"[HEBO Warning] Optimization aborted or failed: {str(e)}")

        self.end_time = time.time()
        return self.best_config, self.best_value