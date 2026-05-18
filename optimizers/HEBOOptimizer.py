# optimizers/HEBOOptimizer.py
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
import optunahub
import time
import numpy as np

optuna.logging.set_verbosity(optuna.logging.WARNING)


class HEBOOptimizer(BaseOptimizer):
    """
    HEBO optimizer via optunahub's HEBOSampler.

    HEBO only supports suggest_float / suggest_int — no categoricals.
    All hyperparameters are encoded to float [0, 1] before being passed
    to HEBO, then decoded back to original domain values immediately
    inside _objective before any evaluation, caching, or tracking.

    Encoding is index-based:
      encode: original value → index / (n_choices - 1) ∈ [0, 1]
      decode: float → round(f * (n-1)) → choices[idx] → original value

    Scoring is now via the RF surrogate in model_wrapper.get_score().
    The nearest-row snap has been removed — RF can score any decoded
    config directly, not just exact table rows.

    The KD-tree (self.nn / Data) is retained only for the callback
    that stores self.best_config as a real dataset row for interpretability.
    Everything inside _objective uses decoded values only.
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)

        # KD-tree kept for best_config snapping in callback only
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

        # Build encoding/decoding tables once at init
        self._build_codec()

    # ------------------------------------------------------------------
    # Codec
    # ------------------------------------------------------------------

    def _choices_for(self, hp):
        if isinstance(hp, OrdinalHyperparameter):
            return list(hp.sequence)
        if isinstance(hp, CategoricalHyperparameter):
            return list(hp.choices)
        raise ValueError(f"Unsupported hyperparameter type: {type(hp)}")

    def _build_codec(self):
        """
        codec:     hp.name → ordered list of choices (original domain)
        constants: hp.name → fixed value (never passed to HEBO)
        """
        self.codec = {}
        self.constants = {}

        for hp in self.config_space.get_hyperparameters():
            if isinstance(hp, Constant):
                self.constants[hp.name] = hp.value
            else:
                self.codec[hp.name] = self._choices_for(hp)

    def _encode(self, name, value):
        """Original value → float in [0, 1]."""
        choices = self.codec[name]
        n = len(choices)
        if n == 1:
            return 0.0
        idx = choices.index(value) if value in choices else 0
        return idx / (n - 1)

    def _decode(self, name, float_val):
        """Float in [0, 1] → nearest original value."""
        choices = self.codec[name]
        n = len(choices)
        if n == 1:
            return choices[0]
        idx = int(round(float_val * (n - 1)))
        idx = max(0, min(n - 1, idx))
        return choices[idx]

    def _decode_trial_params(self, params):
        """
        Convert HEBO float params → original domain config.
        Constants are re-injected so the rest of the framework
        never sees an incomplete config.
        """
        decoded = {name: self._decode(name, float_val)
                   for name, float_val in params.items()}
        decoded.update(self.constants)
        return decoded

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clean(self, v):
        return v.item() if hasattr(v, "item") else v

    def _nearest_row(self, hp_dict):
        """Snap to real dataset row — used only for best_config in callback."""
        query = [hp_dict[c] for c in self.columns]
        row = self.nn.nearestRow(query)
        return {c: self._clean(v) for c, v in zip(self.columns, row)}

    def _row_tuple(self, hp_dict):
        return tuple(hp_dict[c] for c in self.columns)

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def _objective(self, trial):
        # 1. HEBO suggests floats in [0, 1] for each non-constant hp
        encoded = {
            name: trial.suggest_float(name, 0.0, 1.0)
            for name in self.codec
        }

        # 2. Decode immediately — encoding never leaves this function
        decoded = self._decode_trial_params(encoded)
        key = self._row_tuple(decoded)

        # 3. Score via RF surrogate — no nearest-row snap needed
        if key in self.cache:
            scores, d2h_val = self.cache[key]
        else:
            try:
                scores = tuple(self.model_wrapper.get_score(decoded))
            except Exception:
                scores = tuple(1.0 for _ in range(self.num_objectives))
            ideal = [0] * self.num_objectives
            d2h_val = DistanceUtil.d2h(ideal, list(scores))
            self.cache[key] = (scores, d2h_val)

        # 4. Track with decoded values directly
        self.iteration += 1
        self.track_evaluation(decoded, list(scores), self.iteration)

        return d2h_val

    # ------------------------------------------------------------------
    # Main optimise loop
    # ------------------------------------------------------------------

    def optimize(self):
        n_trials = self.config["n_trials"]
        self.start_time = time.time()

        # Build float search space for HEBO — one [0,1] float per non-constant hp
        search_space = {
            name: optuna.distributions.FloatDistribution(0.0, 1.0)
            for name in self.codec
        }

        module = optunahub.load_module("samplers/hebo")
        sampler = module.HEBOSampler(
            search_space=search_space,
            seed=self.seed,
        )

        study = optuna.create_study(direction="minimize", sampler=sampler)

        def callback(study, trial):
            if study.best_value < self.best_value:
                self.best_value = study.best_value
                decoded = self._decode_trial_params(study.best_trial.params)
                self.best_config = decoded

        study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=3600,
            catch=(Exception,),
            callbacks=[callback],
        )

        self.end_time = time.time()
        return self.best_config, self.best_value