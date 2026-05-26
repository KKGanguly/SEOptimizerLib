# optimizers/MOSMACOptimizer.py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from smac import Scenario
from smac.facade.hyperparameter_optimization_facade import HyperparameterOptimizationFacade as HPOFacade
from smac.initial_design.random_design import RandomInitialDesign
from ConfigSpace import Configuration

from optimizers.base_optimizer import BaseOptimizer
from utils import DistanceUtil
import time
import uuid
import numpy as np
import copy
import random
import logging

# Suppress SMAC's massive console output flood
logging.getLogger("smac").setLevel(logging.WARNING)

class SMACOptimizer(BaseOptimizer):
    """
    SMAC (Sequential Model-Based Algorithm Configuration) Optimizer.
    Archetype: The Global Surrogate Believer (Branch C).
    
    Patched to guarantee initialization fairness: Forces SMAC to use 
    the raw empirical tabular data for its initial design phase before 
    switching to its global Random Forest surrogate.
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)
        np.random.seed(seed)
        random.seed(seed)

        self.iteration = 0
        self.cache = {}
        self.best_config = None
        self.best_value = float("inf")

        # ConfigSpace and Dataset Details
        self.X_df = self.model_wrapper.X
        self.n_rows = len(self.X_df)
        self.columns = list(self.X_df.columns)
        self.config_space, _, _ = self.model_config.get_configspace()
        
        # Test objectives count
        test_config = {c: self._clean(self.X_df.iloc[0][c]) for c in self.columns}
        self.num_objectives = len(self.model_wrapper.get_score(test_config))

        self.initial_budget = int(self.config.get("initial_size", 20))

    def _clean(self, v):
        """Extracts native python types from pandas/numpy scalars."""
        return v.item() if hasattr(v, "item") else v

    def _idx_to_config(self, idx):
        """Pulls a real historical configuration directly from the dataset."""
        row = self.X_df.iloc[idx]
        return {c: self._clean(row[c]) for c in self.columns}

    def _evaluate(self, hp_dict):
        # 1. Clamp: Ensure all inputs are strictly within valid ConfigSpace bounds
        for hp in self.config_space.get_hyperparameters():
            if hp.name in hp_dict and hasattr(hp, 'lower'):
                val = hp_dict[hp.name]
                val = max(hp.lower, min(hp.upper, val))
                if type(hp).__name__ == "UniformIntegerHyperparameter":
                    val = int(round(val))
                hp_dict[hp.name] = val
        
        # 2. Type-Safe Cache Key (Prevents crashes on string/categorical rounding)
        key_list = []
        for c in self.columns:
            val = hp_dict[c]
            if isinstance(val, float):
                key_list.append(round(val, 6))
            else:
                key_list.append(val)
        key = tuple(key_list)

        if key in self.cache:
            return self.cache[key]

        # 3. Model Score and D2H Calculation
        try:
            scores = tuple(self.model_wrapper.get_score(hp_dict))
            if not all(np.isfinite(s) for s in scores): 
                raise ValueError
        except Exception:
            scores = tuple(float('inf') for _ in range(self.num_objectives))
            
        ideal = [0] * self.num_objectives
        d2h_val = DistanceUtil.d2h(ideal, list(scores))
        d2h_val = float(d2h_val if np.isfinite(d2h_val) else float('inf'))

        self.iteration += 1
        self.cache[key] = (scores, d2h_val)
        
        # Tracking
        try:
            self.track_evaluation(hp_dict, list(scores), self.iteration)
        except Exception:
            self.logging_util.log("iteration", self.iteration)
        
        if d2h_val < self.best_value:
            self.best_value = d2h_val
            self.best_config = copy.deepcopy(hp_dict)
            
        return scores, d2h_val

    def _target_function(self, config, seed=0):
        """SMAC3 Target Function wrapper."""
        # Unpack the tuple correctly and return the scalar objective for SMAC
        _, d2h_val = self._evaluate(dict(config))
        return d2h_val

    def optimize(self):
        n_trials = self.config["n_trials"]
        self.start_time = time.time()
        
        output_dir = f"{self.config.get('output_directory', 'results')}/smac_{uuid.uuid4().hex[:6]}"
        
        scenario = Scenario(
            configspace=self.config_space,
            n_trials=n_trials,
            deterministic=True,
            output_directory=output_dir,
            seed=self.seed,
        )

        # ---------------------------------------------------------------------
        # FAIRNESS PATCH: Force SMAC to use the raw empirical dataset initially
        # ---------------------------------------------------------------------
        empirical_configs = []
        obs_budget = min(self.initial_budget, n_trials)
        for _ in range(obs_budget):
            initial_idx = random.randint(0, self.n_rows - 1)
            hp_dict = self._idx_to_config(initial_idx)
            config_obj = Configuration(self.config_space, values=hp_dict)
            empirical_configs.append(config_obj)

        initial_design = RandomInitialDesign(
            scenario,
            n_configs=obs_budget,
            additional_configs=empirical_configs # SMAC evaluates these first
        )

        smac = HPOFacade(
            scenario=scenario,
            target_function=self._target_function,
            overwrite=True,
            initial_design=initial_design,
        )

        try:
            smac.optimize()
        except Exception as e:
            # Prevent single-task crashes from killing the entire 127-task bracket
            print(f"SMAC Optimization Error: {e}")

        self.end_time = time.time()
        return self.best_config, self.best_value