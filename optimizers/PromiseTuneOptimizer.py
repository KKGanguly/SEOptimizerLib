import sys
import os
import uuid
import time
import copy
import pandas as pd
from pathlib import Path
import random
from optimizers.base_optimizer import BaseOptimizer
from ConfigSpace.hyperparameters import (
    OrdinalHyperparameter,
    CategoricalHyperparameter,
    Constant,
)
# Point to the Authors' Code
PROMISETUNE_REPO_PATH = str(Path(__file__).resolve().parent.parent / "PromiseTune" / "Code")
if PROMISETUNE_REPO_PATH not in sys.path:
    sys.path.append(PROMISETUNE_REPO_PATH)

from PromiseTune import PromiseTune, PromiseTuneConfig

class PromiseTuneOptimizer(BaseOptimizer):
    """
    Directly connects PromiseTune to the framework's continuous surrogate.
    Evaluates configurations dynamically and minimizes the D2H scalar.
    """
    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)
        
        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)
        self.config_space, _, _ = self.model_config.get_configspace()
        self.iteration = 0
        self.best_config = None
        self.best_value = float('inf') # D2H is always minimized

    def _clean(self, v):
        """Prevents numpy float64/int64 from breaking the framework's logger."""
        return v.item() if hasattr(v, "item") else v

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

    def _prepare_search_space_csv(self) -> str:
        # 1. Get the same initial configurations your GA/SMAC uses
        # Use the same seed and logic your framework uses for initialization
        random.seed(self.seed)
        
        # Assume your framework has a method to get initial seeds
        initial_configs = [self._sample_config() 
                          for _ in range(self.config.get("initial_size", 10))]
        
        # 2. Add the "Corner" cases for boundary learning (as before)
        boundary_configs = []
        for hp in self.config_space.get_hyperparameters():
            if hasattr(hp, 'lower'):
                c_min = self.config_space.get_default_configuration().get_dictionary()
                c_max = self.config_space.get_default_configuration().get_dictionary()
                c_min[hp.name] = hp.lower
                c_max[hp.name] = hp.upper
                boundary_configs.extend([c_min, c_max])

        # 3. Combine: [Initial Seeds] + [Boundary Points] + [Random Noise]
        # Placing initial_configs at the start ensures they are picked first!
        final_data = initial_configs + boundary_configs
        while len(final_data) < 100:
            final_data.append(self.config_space.sample_configuration().get_dictionary())
            
        df = pd.DataFrame(final_data)
        df = df.replace('?', 0.0) 
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        df['$<d2h'] = 0.0 # Dummy target
        
        # 4. Do NOT shuffle. We want index 0-9 to be our initial_configs.
        temp_csv = f"/tmp/ptune_descriptor_{self.seed}.csv"
        df.to_csv(temp_csv, index=False)
        return temp_csv

    def _surrogate_oracle(self, action_list):
        # 1. Round to 2 decimals to help the Random Forest cluster similar values
        # This prevents the "Categorical Explosion" of unique float values.
        action_list = [round(float(v), 2) for v in action_list] 
        hp_dict = {c: self._clean(v) for c, v in zip(self.columns, action_list)}
        
        # 2. Add a check to ensure the wrapper gets exactly what it expects
        try:
            scores, d2h_val = self.model_wrapper.evaluate(hp_dict)
            
        except Exception:
            # If the wrapper fails, return high-penalty infinity
            return float('inf'), action_list
         
        self.iteration += 1
        
        # 2. Log the raw multiple objectives exactly as the framework expects
        try:
            self.track_evaluation(hp_dict, list(scores), self.iteration)
        except Exception:
            self.logging_util.log("iteration", self.iteration)
        
        # 3. Track the best D2H found
        if d2h_val < self.best_value:
            self.best_value = d2h_val
            self.best_config = copy.deepcopy(hp_dict)
           
        # 4. Return the scalar D2H back to PromiseTune so it can minimize it
        return d2h_val, action_list

    def optimize(self):
        self.start_time = time.time()
        temp_csv_path = self._prepare_search_space_csv()
        
        framework_budget = self.config.get("n_trials", 100)
        
        pt_config = PromiseTuneConfig(
            budget=framework_budget, 
            initial_size=self.config.get("initial_size", 10),
            l=self.config.get("l", 10),
            k=self.config.get("k", 0.1),
            causal_refresh_interval=self.config.get("causal_refresh_interval", 10),
            stop_threshold=self.config.get("stop_threshold", 0.01),
            log_rules=False
        )
        
        try:
            # 1. Let PromiseTune run until it hits the budget or converges
            best_list, rules, xs = PromiseTune(
                filename=temp_csv_path, 
                config=pt_config, 
                seed=self.seed,
                oracle=self._surrogate_oracle
            )
            if self.best_config is None:
                self.best_config = self.model_wrapper.X.iloc[0].to_dict()
                self.best_value = 1.0 # Or use a defined worst-case penalty
                
        finally:
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)
        
        self.end_time = time.time()
        return self.best_config, self.best_value