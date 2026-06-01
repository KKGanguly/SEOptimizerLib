import sys
import os
import uuid
import time
import copy
import pandas as pd
from pathlib import Path

from optimizers.base_optimizer import BaseOptimizer

# Point to the Authors' Code
PROMISETUNE_REPO_PATH = str(Path(__file__).resolve().parent.parent / "PromiseTune" / "Code")
if PROMISETUNE_REPO_PATH not in sys.path:
    sys.path.append(PROMISETUNE_REPO_PATH)

from PromiseTune import PromiseTune, PromiseTuneConfig

class PromiseTuneOptimizerRepl(BaseOptimizer):
    """
    Directly connects PromiseTune to the framework's continuous surrogate.
    Evaluates configurations dynamically and minimizes the D2H scalar.
    """
    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)
        
        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)
        
        self.iteration = 0
        self.best_config = None
        self.best_value = float('inf') # D2H is always minimized

    def _clean(self, v):
        """Prevents numpy float64/int64 from breaking the framework's logger."""
        return v.item() if hasattr(v, "item") else v

    def _prepare_search_space_csv(self) -> str:
        """
        Provides a dummy CSV so PromiseTune knows the column names, features, 
        and valid ranges for its causal rule engine.
        """
        df = self.X_df.copy()
        
        # Add a dummy target column with the '$<' prefix required by PromiseTune
        df['$<d2h'] = 0.0 
        
        temp_dir = "/tmp" if os.name == 'posix' else "."
        temp_csv = f"{temp_dir}/ptune_space_{self.seed}_{uuid.uuid4().hex[:8]}.csv"
        df.to_csv(temp_csv, index=False)
        return temp_csv

    def _surrogate_oracle(self, action_list):
        """
        Callback injected into PromiseTune. Asks the framework to compute the 
        native D2H score for every configuration PromiseTune generates.
        """
        hp_dict = {c: self._clean(v) for c, v in zip(self.columns, action_list)}
        
        try:
            # 1. NATIVE D2H EVALUATION
            # This perfectly mirrors your HillClimbing optimizer. It fetches 
            # the multi-objective scores and calculates the D2H automatically.
            scores, d2h_val = self.model_wrapper.evaluate(hp_dict)
            
        except Exception:
            # Fallback if the surrogate crashes or goes out of bounds
            scores = [float('inf')] * len(self.model_wrapper.y.columns)
            d2h_val = float('inf')
            
        self.iteration += 1
        
        # 2. Log the raw multiple objectives exactly as the framework expects
        self.track_evaluation(hp_dict, list(scores), self.iteration)
        
        # 3. Track the best D2H found
        if d2h_val < self.best_value:
            self.best_value = d2h_val
            self.best_config = copy.deepcopy(hp_dict)
            
        # 4. Return the scalar D2H back to PromiseTune so it can minimize it
        return d2h_val, action_list

    def optimize(self):
        self.start_time = time.time()
        temp_csv_path = self._prepare_search_space_csv()
        
        # INJECT BUDGET: Pulls dynamically from your framework
        framework_budget = self.config.get("n_trials", 100)
        
        pt_config = PromiseTuneConfig(
            budget=framework_budget, 
            initial_size=self.config.get("initial_size", 10),
            l=self.config.get("l", 10),
            k=self.config.get("k", 0.1),
            causal_refresh_interval=self.config.get("causal_refresh_interval", 10),
            stop_threshold=self.config.get("stop_threshold", 0.01),
            log_rules=False,
            enable_timing=False
        )
        
        try:
            # Run the official algorithm using our D2H Oracle
            best_list, rules, *_ = PromiseTune(
                filename=temp_csv_path, 
                config=pt_config, 
                seed=self.seed,
                oracle=self._surrogate_oracle
            )
            
        finally:
            # Cleanup dummy CSV
            if os.path.exists(temp_csv_path):
                os.remove(temp_csv_path)
                
        self.end_time = time.time()
        return self.best_config, self.best_value