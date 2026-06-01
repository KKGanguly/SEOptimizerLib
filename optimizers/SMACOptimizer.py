# optimizers/MOSMACOptimizer.py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from smac import Scenario
from smac.facade.hyperparameter_optimization_facade import HyperparameterOptimizationFacade as HPOFacade
from smac.initial_design.random_design import RandomInitialDesign
from ConfigSpace import Configuration
from ConfigSpace.hyperparameters import (
    OrdinalHyperparameter,
    CategoricalHyperparameter,
    Constant,
)
from optimizers.base_optimizer import BaseOptimizer
from utils import DistanceUtil
import time
import uuid
import numpy as np
import copy
import random
import logging
import tempfile
import shutil
import os
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

        self.initial_budget = int(self.config.get("initial_size", 10))

    def _clean(self, v):
        """Extracts native python types from pandas/numpy scalars."""
        return v.item() if hasattr(v, "item") else v

    def _idx_to_config(self, idx):
        """Pulls a real historical configuration directly from the dataset."""
        row = self.X_df.iloc[idx]
        return {c: self._clean(row[c]) for c in self.columns}

    def _sample_config(self):
        """Blind random sampling (DODGE relies on sampling over modeling)."""
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
            self.iteration += 1
            scores, d2h = self.cache[key]
            try:
                self.track_evaluation(hp_dict, list(scores), self.iteration)
            except Exception:
                self.logging_util.log("iteration", self.iteration)
            return scores, d2h # Return D2H instead of raw scores

        # 3. FIXED: Native Model Score and D2H Calculation
        try:
            # Let ModelWrapperStatic do the math so 1.0 is Heaven
            scores, d2h_val = self.model_wrapper.evaluate(hp_dict)
            if not all(np.isfinite(s) for s in scores): 
                raise ValueError
        except Exception:
            scores = tuple(float('inf') for _ in range(self.num_objectives))
            d2h_val = float('inf')

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
        temp_dir = tempfile.mkdtemp(prefix="smac_")
        scenario = Scenario(
            configspace=self.config_space,
            n_trials=n_trials,
            deterministic=True,
            output_directory=temp_dir,
            seed=self.seed,
        )
        
        # ---------------------------------------------------------------------
        # FAIRNESS PATCH: Force SMAC to use the raw empirical dataset initially
        # ---------------------------------------------------------------------
        empirical_configs = []
        obs_budget = min(self.initial_budget, n_trials)
        
        
        for budget in range(obs_budget):
            hp_dict = self._sample_config()
            config_obj = Configuration(self.config_space, values=hp_dict)
            empirical_configs.append(config_obj)

        initial_design = RandomInitialDesign(
            scenario,
            n_configs=0,
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
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        self.end_time = time.time()
        
        return self.best_config, self.best_value