from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
import copy
import random

from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.core.callback import Callback

from ConfigSpace.hyperparameters import (
    OrdinalHyperparameter,
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    Constant,
)
from optimizers.base_optimizer import BaseOptimizer
from utils import DistanceUtil


class PymooProblemWrapper(ElementwiseProblem):
    """Bridges pymoo's continuous flat arrays with ConfigSpace mixed dictionaries."""
    def __init__(self, optimizer):
        self.opt = optimizer
        self.cs = optimizer.config_space
        
        # Isolate active hyperparameters (Constants are ignored by the optimizer logic)
        self.active_hps = [hp for hp in self.cs.get_hyperparameters() if not isinstance(hp, Constant)]
        
        xl, xu = [], []
        for hp in self.active_hps:
            if isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter)):
                xl.append(hp.lower)
                xu.append(hp.upper)
            elif isinstance(hp, CategoricalHyperparameter):
                xl.append(0)
                xu.append(len(hp.choices) - 1)
            elif isinstance(hp, OrdinalHyperparameter):
                xl.append(0)
                xu.append(len(hp.sequence) - 1)
                
        super().__init__(n_var=len(self.active_hps), n_obj=optimizer.num_objectives, xl=xl, xu=xu)

    def _x_to_dict(self, x):
        """Maps a pymoo flat continuous array back to a valid ConfigSpace dictionary."""
        hp_dict = {}
        idx = 0
        for hp in self.cs.get_hyperparameters():
            if isinstance(hp, Constant):
                hp_dict[hp.name] = hp.value
            else:
                val = x[idx]
                if isinstance(hp, UniformIntegerHyperparameter):
                    hp_dict[hp.name] = int(round(val))
                elif isinstance(hp, CategoricalHyperparameter):
                    hp_dict[hp.name] = list(hp.choices)[int(round(val))]
                elif isinstance(hp, OrdinalHyperparameter):
                    hp_dict[hp.name] = list(hp.sequence)[int(round(val))]
                else:
                    hp_dict[hp.name] = float(val)
                idx += 1
        return hp_dict

    def _dict_to_x(self, hp_dict):
        """Maps a ConfigSpace dictionary to a flat continuous pymoo array."""
        x = []
        for hp in self.active_hps:
            val = hp_dict[hp.name]
            if isinstance(hp, CategoricalHyperparameter):
                x.append(list(hp.choices).index(val))
            elif isinstance(hp, OrdinalHyperparameter):
                x.append(list(hp.sequence).index(val))
            else:
                x.append(val)
        return np.array(x)

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluates a single individual, handling cache, inversion, and logging."""
        hp_dict = self._x_to_dict(x)
        key = self.opt._row_tuple(hp_dict)
        
        if key in self.opt.cache:
            smsemoa_scores, raw_scores = self.opt.cache[key]
        else:
            try:
                # Raw scores: 0.0 is worst, 1.0 is best
                raw_scores = self.opt.model_wrapper.get_score(hp_dict)
                # Invert for pymoo: 0.0 becomes best
                smsemoa_scores = tuple(1.0 - s for s in raw_scores)
            except Exception:
                raw_scores = tuple(0.0 for _ in range(self.opt.num_objectives))
                smsemoa_scores = tuple(float('inf') for _ in range(self.opt.num_objectives))
                
            self.opt.cache[key] = (smsemoa_scores, raw_scores)
            
        self.opt.iteration += 1
        
        # Track real framework scores
        try:
            self.opt.track_evaluation(hp_dict, list(raw_scores), self.opt.iteration)
        except Exception:
            self.opt.logging_util.log("iteration", self.opt.iteration)
            
        # Jitter: Microscopic noise prevents division-by-zero in SMS-EMOA Hypervolume survival
        jitter = np.random.normal(0, 1e-9, len(smsemoa_scores))
        out["F"] = [s + j for s, j in zip(smsemoa_scores, jitter)]


class ParetoFrontTracker(Callback):
    """Pymoo callback that mirrors the Optuna Pareto tracker for D2H logging."""
    def __init__(self, optimizer, problem):
        super().__init__()
        self.opt = optimizer
        self.problem = problem
        
    def notify(self, algorithm):
        if algorithm.opt is not None and len(algorithm.opt) > 0:
            ideal = [0.0] * self.opt.num_objectives
            best_ind = None
            min_d2h = float("inf")
            
            # Find the best D2H in the current Pareto front
            for ind in algorithm.opt:
                d2h = DistanceUtil.d2h(ideal, list(ind.F))
                if d2h < min_d2h:
                    min_d2h = d2h
                    best_ind = ind
                    
            if best_ind is not None:
                best_raw = self.problem._x_to_dict(best_ind.X)
                
                # Mock an Optuna trial so the standard tracker functions normally
                class MockTrial:
                    def __init__(self, params, values):
                        self.params = params
                        self.values = values
                        
                mock_trial = MockTrial(best_raw, best_ind.F)
                mock_pareto = [MockTrial(self.problem._x_to_dict(ind.X), ind.F) for ind in algorithm.opt]
                
                try:
                    if hasattr(self.opt, 'track_frontier'):
                        self.opt.track_frontier(self.opt.iteration, mock_pareto, best_raw, mock_trial)
                except Exception:
                    pass


class SMSEMOAOptimizer(BaseOptimizer):
    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)
        random.seed(seed)
        np.random.seed(seed)
        
        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)
        
        self.config_space, _, _ = self.model_config.get_configspace()
        self.cache = {}
        
        test_config = {c: self._safe_clean(self.X_df.iloc[0][c]) for c in self.columns}
        self.num_objectives = len(self.model_wrapper.get_score(test_config))
        
        self.iteration = 0
        self.population_size = int(self.config.get("pop_size", 20))

    def _safe_clean(self, v):
        val = v.item() if hasattr(v, "item") else v
        return round(val, 6) if isinstance(val, float) else val

    def _row_tuple(self, hp_dict):
        return tuple(self._safe_clean(hp_dict[c]) for c in self.columns)

    def _sample_config(self):
        """Blind random sampling (Fairness Initialization)."""
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
        return hp_dict

    def optimize(self):
        n_trials = self.config["n_trials"]
        self.start_time = time.time()
        
        problem = PymooProblemWrapper(self)
        tracker = ParetoFrontTracker(self, problem)
        
        # 1. FAIRNESS INITIALIZATION
        # Generate the initial population and format it as a 2D numpy array for pymoo
        obs_budget = min(self.population_size, n_trials)
        initial_configs = [self._sample_config() for _ in range(obs_budget)]
        initial_X = np.array([problem._dict_to_x(d) for d in initial_configs])
        
        algorithm = SMSEMOA(pop_size=self.population_size, sampling=initial_X)
        
        # 2. RUN OPTIMIZATION
        # Run for n_trials // pop_size generations
        res = minimize(
            problem,
            algorithm,
            ('n_gen', max(1, n_trials // self.population_size)),
            seed=self.seed,
            callback=tracker,
            verbose=False
        )

        # 3. FINAL PARETO FRONTIER EXTRACTION (D2H Winner)
        final_frontier = res.opt
        
        if final_frontier is None or len(final_frontier) == 0:
            self.end_time = time.time()
            return None, float("inf")

        best_config = None
        best_d2h_norm = float("inf")
        ideal = [0.0] * self.num_objectives

        for ind in final_frontier:
            # ind.F contains the inverted scores (Minimization)
            d2h_val = DistanceUtil.d2h(ideal, list(ind.F))
            if d2h_val < best_d2h_norm:
                best_d2h_norm = d2h_val
                best_config = problem._x_to_dict(ind.X)

        self.best_config = best_config
        self.best_value = best_d2h_norm
        self.end_time = time.time()

        return self.best_config, self.best_value