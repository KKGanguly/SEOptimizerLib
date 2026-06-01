# optimizers/EDAOptimizer.py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from optimizers.base_optimizer import BaseOptimizer
from utils import DistanceUtil
import time
import numpy as np
import copy
import random
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.hyperparameters import (
    OrdinalHyperparameter,
    CategoricalHyperparameter,
    Constant,
)
class EDAOptimizer(BaseOptimizer):
    """
    Univariate Marginal Distribution Algorithm (UMDA).
    Initializes from raw dataset rows, then builds a probabilistic model 
    of the top 50% of the population to sample the next generation.
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)
        np.random.seed(seed)
        random.seed(seed)

        self.X_df = self.model_wrapper.X
        self.n_rows = len(self.X_df)
        self.columns = list(self.X_df.columns)
        self.config_space, _, _ = self.model_config.get_configspace()
        self.cache = {}
        
        test_config = {c: self._clean(self.X_df.iloc[0][c]) for c in self.columns}
        self.num_objectives = len(self.model_wrapper.get_score(test_config))
        self.iteration = 0
        self.best_config = None
        self.best_value = float("inf")

        self.pop_size = int(self.config.get("pop_size", 10))
        self.truncation_ratio = float(self.config.get("truncation_ratio", 0.5))
        self.num_parents = max(2, int(self.pop_size * self.truncation_ratio))

    def _clean(self, v):
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

    def _idx_to_config(self, idx):
        row = self.X_df.iloc[idx]
        return {c: self._clean(row[c]) for c in self.columns}

    def _evaluate_config(self, hp_dict):
        key = tuple(hp_dict[c] for c in self.columns)
        if key in self.cache:
            return self.cache[key]
        try:
            scores, d2h_val = self.model_wrapper.evaluate(hp_dict)
        except Exception:
            scores = tuple(float('inf') for _ in range(self.num_objectives))
            d2h_val = float('inf')
            
        self.cache[key] = (scores, d2h_val)
        self.iteration += 1
        self.track_evaluation(hp_dict, list(scores), self.iteration)
        return scores, d2h_val

    def optimize(self):
        n_trials = self.config["n_trials"]
        self.start_time = time.time()

        # Initialize Population from raw dataset rows
        population = []
        for _ in range(self.pop_size):
            initial_pop = self._sample_config()
            population.append(initial_pop)

        while self.iteration < n_trials:
            # 1. Evaluate Population
            pop_results = []
            for ind in population:
                if self.iteration >= n_trials:
                    break
                scores, d2h = self._evaluate_config(ind)
                pop_results.append((ind, d2h))
                
                if d2h < self.best_value:
                    self.best_value = d2h
                    self.best_config = copy.deepcopy(ind)

            if self.iteration >= n_trials:
                break

            # 2. Truncation Selection (Keep top 50%)
            pop_results.sort(key=lambda x: x[1])
            parents = [x[0] for x in pop_results[:self.num_parents]]

            # 3. Estimate Distribution & Sample Next Generation from Surrogate Space
            population = []
            for _ in range(self.pop_size):
                new_ind = {}
                for hp in self.config_space.get_hyperparameters():
                    name = hp.name
                    hp_type = type(hp).__name__
                    
                    if hp_type in ["UniformFloatHyperparameter", "UniformIntegerHyperparameter"]:
                        parent_vals = [p[name] for p in parents]
                        mean_val = np.mean(parent_vals)
                        std_val = np.std(parent_vals) + 1e-6 
                        
                        val = np.random.normal(mean_val, std_val)
                        val = max(hp.lower, min(hp.upper, val))
                        
                        if hp_type == "UniformIntegerHyperparameter":
                            val = int(round(val))
                        new_ind[name] = val
                        
                    else:
                        choices = list(hp.choices) if hasattr(hp, 'choices') else list(hp.sequence)
                        parent_vals = [p[name] for p in parents]
                        counts = {c: parent_vals.count(c) for c in choices}
                        probs = [counts[c] / len(parent_vals) for c in choices]
                        new_ind[name] = np.random.choice(choices, p=probs)
                        
                population.append(new_ind)

        self.end_time = time.time()
        return self.best_config, self.best_value