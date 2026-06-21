# optimizers/INGSOptimizer.py
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import time
import copy
import random
import numpy as np
from scipy.interpolate import RBFInterpolator

from optimizers.base_optimizer import BaseOptimizer
from ConfigSpace.hyperparameters import (
    OrdinalHyperparameter,
    CategoricalHyperparameter,
    Constant,
)


def _zscore(a: np.ndarray) -> np.ndarray:
    """Standardize to zero mean / unit variance; all-zeros if degenerate."""
    s = a.std()
    return np.zeros_like(a) if s < 1e-12 else (a - a.mean()) / s


class INGSOptimizer(BaseOptimizer):
    """
    INGS — INterpolation-guided Gradient Search.

    A linear-kernel RBF surrogate over evaluated configs, with an
    anti-Laplacian acquisition (favour smooth minima of d2h) plus a UCB
    exploration term (distance to the nearest observed point).

    Acquisition (per step):
        u   = -pred                           # RBF-predicted d2h, flipped so higher = better
        al  = 2*u - u (nearest observed)      # one-sided anti-Laplacian (curvature)
        score = z(al) + kappa * z(min_dist)   # exploit smooth minima + explore
    pick argmax(score), evaluate it, repeat until the budget is spent.
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        random.seed(seed)
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)

        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)
        self.config_space, _, _ = self.model_config.get_configspace()
        self.cache = {}

        # Objective count (for inf-padding on failed evaluations).
        test_config = {c: self._safe_clean(self.X_df.iloc[0][c]) for c in self.columns}
        self.num_objectives = len(self.model_wrapper.get_score(test_config))

        self.iteration = 0
        self.best_config = None
        self.best_value = float("inf")

        # Hyperparameters (mirrors EZR's batch size + the library's init tax).
        self.start_evals = int(self.config.get("initial_size", 10))
        self.Few = 128  # candidate batch sampled per acquisition step
        self.rbf_smooth = 1e-2  # RBFInterpolator regularization
        self.ucb_kappa = 0.15  # exploration weight

        # Precompute the config -> normalized [0,1]^d vectorizer.
        self._build_vectorizer()

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _safe_clean(self, v):
        val = v.item() if hasattr(v, "item") else v
        return round(val, 6) if isinstance(val, float) else val

    def _row_tuple(self, hp_dict):
        return tuple(self._safe_clean(hp_dict[c]) for c in self.columns)

    def _build_vectorizer(self):
        """One descriptor per hyperparameter, in a fixed order, used to map a
        config dict to a normalized coordinate vector for the RBF geometry.

        Float/Integer  -> (v - lower) / span
        Ordinal/Categorical -> index of the chosen value / (n_choices - 1)
        Constant       -> 0.0
        """
        self._dims = []
        for hp in self.config_space.get_hyperparameters():
            name = hp.name
            kind = type(hp).__name__
            if isinstance(hp, Constant):
                self._dims.append((name, "const", None))
            elif isinstance(hp, OrdinalHyperparameter):
                seq = list(hp.sequence)
                lut = {self._safe_clean(v): i for i, v in enumerate(seq)}
                self._dims.append((name, "choice", (lut, max(1, len(seq) - 1))))
            elif isinstance(hp, CategoricalHyperparameter):
                ch = list(hp.choices)
                lut = {self._safe_clean(v): i for i, v in enumerate(ch)}
                self._dims.append((name, "choice", (lut, max(1, len(ch) - 1))))
            elif kind in ("UniformFloatHyperparameter", "UniformIntegerHyperparameter"):
                span = hp.upper - hp.lower
                self._dims.append(
                    (name, "num", (float(hp.lower), float(span) if span > 0 else 1.0))
                )
            else:
                # Unknown type: treat as a fixed coordinate.
                self._dims.append((name, "const", None))

    def _vectorize(self, hp_dict) -> np.ndarray:
        out = np.empty(len(self._dims), dtype=float)
        for j, (name, kind, params) in enumerate(self._dims):
            if kind == "const":
                out[j] = 0.0
            elif kind == "num":
                lower, span = params
                out[j] = (float(hp_dict[name]) - lower) / span
            else:  # choice
                lut, denom = params
                idx = lut.get(self._safe_clean(hp_dict[name]))
                out[j] = 0.0 if idx is None else idx / denom
        return out

    def _sample_config(self):
        """Random configuration drawn from the ConfigSpace boundaries
        (verbatim from EZR / RandomSearch)."""
        hp_dict = {}
        for hp in self.config_space.get_hyperparameters():
            hp_type = type(hp).__name__
            if isinstance(hp, Constant):
                hp_dict[hp.name] = hp.value
            elif isinstance(hp, OrdinalHyperparameter):
                hp_dict[hp.name] = random.choice(list(hp.sequence))
            elif isinstance(hp, CategoricalHyperparameter):
                hp_dict[hp.name] = random.choice(list(hp.choices))
            elif hp_type == "UniformFloatHyperparameter":
                hp_dict[hp.name] = random.uniform(hp.lower, hp.upper)
            elif hp_type == "UniformIntegerHyperparameter":
                hp_dict[hp.name] = random.randint(int(hp.lower), int(hp.upper))
            else:
                raise ValueError(f"Unsupported hyperparameter type: {hp_type}")
        return hp_dict

    def _evaluate(self, hp_dict):
        """Score via the shared RF surrogate, track the evaluation, update best.
        Burns one unit of budget (duplicates included, matching RandomSearch)."""
        key = self._row_tuple(hp_dict)
        if key in self.cache:
            scores, d2h = self.cache[key]
        else:
            try:
                scores, d2h = self.model_wrapper.evaluate(hp_dict)
            except Exception:
                scores = tuple(float("inf") for _ in range(self.num_objectives))
                d2h = float("inf")
            self.cache[key] = (scores, d2h)

        self.iteration += 1
        try:
            self.track_evaluation(hp_dict, list(scores), self.iteration)
        except Exception:
            pass

        if d2h < self.best_value:
            self.best_value = d2h
            self.best_config = copy.deepcopy(hp_dict)
        return d2h

    # ------------------------------------------------------------
    # Main optimization loop
    # ------------------------------------------------------------
    def optimize(self):
        n_trials = self.config["n_trials"]
        self.start_time = time.time()

        X_obs, y_obs = [], []

        # ---- Warm start: random ConfigSpace samples ----
        n0 = min(self.start_evals, n_trials)
        for _ in range(n0):
            if self.iteration >= n_trials:
                break
            cfg = self._sample_config()
            d2h = self._evaluate(cfg)
            X_obs.append(self._vectorize(cfg))
            y_obs.append(d2h)

        # ---- INGS acquisition loop ----
        while self.iteration < n_trials:
            Xo = np.asarray(X_obs)
            yo = np.asarray(y_obs)

            cands = [self._sample_config() for _ in range(self.Few)]
            Xc = np.asarray([self._vectorize(c) for c in cands])

            try:
                rbf = RBFInterpolator(
                    Xo, yo, kernel="linear", smoothing=self.rbf_smooth
                )
                pred = rbf(Xc).flatten()
            except Exception:
                # Singular system / degenerate geometry: fall back to random.
                pick = cands[int(self.rng.integers(len(cands)))]
                d2h = self._evaluate(pick)
                X_obs.append(self._vectorize(pick))
                y_obs.append(d2h)
                continue

            u = -pred  # higher = better
            d_to_obs = np.linalg.norm(Xc[:, None, :] - Xo[None, :, :], axis=2)
            nn = np.argmin(d_to_obs, axis=1)
            al = 2.0 * u - (-yo[nn])  # anti-Laplacian
            min_dist = d_to_obs.min(axis=1)  # UCB exploration
            score = _zscore(al) + self.ucb_kappa * _zscore(min_dist)

            pick = cands[int(np.argmax(score))]
            d2h = self._evaluate(pick)
            X_obs.append(self._vectorize(pick))
            y_obs.append(d2h)

        self.end_time = time.time()
        return self.best_config, self.best_value
