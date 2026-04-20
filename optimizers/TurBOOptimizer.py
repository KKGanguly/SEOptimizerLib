# optimizers/TuRBOOptimizer.py
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


class TuRBOOptimizer(BaseOptimizer):
    """
    TuRBO optimizer via optunahub's TuRBOSampler.

    TuRBO only supports suggest_float. All hyperparameters are encoded
    to a float in [0, 1] before being passed to TuRBO, then decoded back
    to their original domain values immediately inside _objective before
    any evaluation, caching, or tracking occurs.

    Encoding/decoding is index-based for all types:
      - Constant    → fixed, not passed to TuRBO at all
      - Ordinal     → index / (len-1)  → round to nearest index → original value
      - Categorical → index / (len-1)  → round to nearest index → original value

    Everything outside _objective (cache keys, track_evaluation, best_config)
    uses only original decoded values — encoding is invisible to the rest of
    the framework.
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

        # Build encoding/decoding tables once at init
        # Only non-Constant hyperparameters are passed to TuRBO
        self._build_codec()

    # ------------------------------------------------------------------
    # Codec — built once, used in every _objective call
    # ------------------------------------------------------------------

    def _choices_for(self, hp):
        if isinstance(hp, OrdinalHyperparameter):
            return list(hp.sequence)
        if isinstance(hp, CategoricalHyperparameter):
            return list(hp.choices)
        raise ValueError(f"Unsupported hyperparameter type: {type(hp)}")

    def _build_codec(self):
        """
        For each non-Constant hp, store its ordered list of choices.
        Encoding:  original value → index / (n_choices - 1)  ∈ [0, 1]
        Decoding:  float in [0,1] → round(f * (n-1)) → choices[idx]

        Order of choices is preserved exactly as defined in config space,
        so decode(encode(v)) == v for every legal value.
        """
        self.codec = {}   # hp.name → list of choices in original order
        self.constants = {}  # hp.name → value  (never passed to TuRBO)

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
        idx = max(0, min(n - 1, idx))  # clamp for floating point edge cases
        return choices[idx]

    def _decode_trial_params(self, params):
        """
        Convert a dict of TuRBO float params back to original-domain config.
        Constants are re-injected here so the rest of the framework never
        sees an incomplete config.
        """
        decoded = {}
        for name, float_val in params.items():
            decoded[name] = self._decode(name, float_val)
        # Re-inject constants
        decoded.update(self.constants)
        return decoded

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
    # Objective — called by Optuna/TuRBO on each trial
    # ------------------------------------------------------------------

    def _objective(self, trial):
        # 1. TuRBO suggests floats in [0, 1] for each non-constant hp
        encoded = {}
        for name in self.codec:
            encoded[name] = trial.suggest_float(name, 0.0, 1.0)

        # 2. Decode immediately — encoding never leaves this function
        decoded = self._decode_trial_params(encoded)

        # 3. Snap to nearest real dataset row
        valid_hp = self._nearest_row(decoded)
        key = self._row_tuple(valid_hp)

        # 4. Evaluate (with cache) using original values only
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

        # 5. Track with original values only
        self.iteration += 1
        self.track_evaluation(valid_hp, list(scores), self.iteration)

        return d2h_val

    # ------------------------------------------------------------------
    # Main optimise loop
    # ------------------------------------------------------------------

    def optimize(self):
        n_trials = self.config["n_trials"]
        n_params = len(self.codec)  # number of non-constant dimensions

        self.start_time = time.time()

        sampler = optunahub.load_module("samplers/turbo").TuRBOSampler(
            n_startup_trials  = self.config.get("n_startup_trials", 2 * n_params),
            n_trust_region    = self.config.get("n_trust_region", 5),
            success_tolerance = self.config.get("success_tolerance", 3),
            failure_tolerance = self.config.get("failure_tolerance", max(5, n_params)),
            seed              = self.config.get("seed", self.seed),
        )

        study = optuna.create_study(direction="minimize", sampler=sampler)

        def callback(study, trial):
            if study.best_value < self.best_value:
                self.best_value = study.best_value
                # Decode best params back to original domain
                decoded = self._decode_trial_params(study.best_trial.params)
                self.best_config = self._nearest_row(decoded)

        study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=3600,
            catch=(Exception,),
            callbacks=[callback],
        )

        self.end_time = time.time()
        return self.best_config, self.best_value