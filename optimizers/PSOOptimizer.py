# optimizers/PSOOptimizer.py
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
class PSOOptimizer(BaseOptimizer):
    """
    Global-Best Particle Swarm Optimization (PSO).
    Initializes swarm positions from raw dataset rows, then relies on 
    continuous vector trajectory over the surrogate to navigate.
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
        self.w = float(self.config.get("w", 0.729))
        self.c1 = float(self.config.get("c1", 1.494))
        self.c2 = float(self.config.get("c2", 1.494))

        self.bounds = {}
        self.is_int = {}
        self.cat_choices = {}
        for hp in self.config_space.get_hyperparameters():
            hp_type = type(hp).__name__
            if hp_type in ["UniformFloatHyperparameter", "UniformIntegerHyperparameter"]:
                self.bounds[hp.name] = (hp.lower, hp.upper)
                self.is_int[hp.name] = (hp_type == "UniformIntegerHyperparameter")
            else:
                self.cat_choices[hp.name] = list(hp.choices) if hasattr(hp, 'choices') else list(hp.sequence)

    def _clean(self, v):
        return v.item() if hasattr(v, "item") else v

    def _idx_to_config(self, idx):
        row = self.X_df.iloc[idx]
        return {c: self._clean(row[c]) for c in self.columns}

    def _evaluate_particle(self, hp_dict):
        key = tuple(hp_dict[c] for c in self.columns)
        if key in self.cache:
            self.iteration += 1
            # FIX: Properly extract scores from the cache to prevent NameError
            scores, d2h_val = self.cache[key]
            try:
                self.track_evaluation(hp_dict, list(scores), self.iteration)
            except Exception:
                self.logging_util.log("iteration", self.iteration)
            return d2h_val
        try:
            scores, d2h_val = self.model_wrapper.evaluate(hp_dict)
        except Exception:
            scores = tuple(float('inf') for _ in range(self.num_objectives))
            d2h_val = float('inf')
            
        self.cache[key] = (scores, d2h_val)
        self.iteration += 1
        self.track_evaluation(hp_dict, list(scores), self.iteration)
        return d2h_val
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
        
    def optimize(self):
        n_trials = self.config["n_trials"]
        self.start_time = time.time()

        # 1. Initialize Swarm from raw dataset rows
        swarm = []
        for _ in range(self.pop_size):
            pos = self._sample_config()
            
            # Initial random velocities based on parameter bounds
            vel = {}
            for col, (lower, upper) in self.bounds.items():
                span = upper - lower
                # FIX: Prevent 0.0 span from killing initial velocity
                span = span if span > 0 else 1.0 
                vel[col] = random.uniform(-span * 0.1, span * 0.1)
                
            swarm.append({
                "pos": pos, 
                "vel": vel, 
                "pbest_pos": copy.deepcopy(pos), 
                "pbest_d2h": float('inf')
            })

        # Evaluate Initial Swarm
        for particle in swarm:
            if self.iteration >= n_trials: break
            d2h = self._evaluate_particle(particle["pos"])
            particle["pbest_d2h"] = d2h
            if d2h < self.best_value:
                self.best_value = d2h
                self.best_config = copy.deepcopy(particle["pos"])

       # 2. Main Swarm Loop
        while self.iteration < n_trials:
            for particle in swarm:
                if self.iteration >= n_trials: break
                
                # FIX: Separate continuous mathematical state from discrete evaluation state
                new_pos = copy.deepcopy(particle["pos"])
                eval_pos = copy.deepcopy(particle["pos"])
                
                for col in self.bounds.keys():
                    r1, r2 = random.random(), random.random()
                    
                    cognitive = self.c1 * r1 * (particle["pbest_pos"][col] - particle["pos"][col])
                    social = self.c2 * r2 * (self.best_config[col] - particle["pos"][col])
                    
                    new_vel = (self.w * particle["vel"][col]) + cognitive + social
                    particle["vel"][col] = new_vel
                    
                    val = particle["pos"][col] + new_vel
                    lower, upper = self.bounds[col]
                    val = max(lower, min(upper, val))
                    
                    # FIX: Maintain continuous momentum internally
                    new_pos[col] = val 
                    
                    # FIX: Round ONLY for the evaluator
                    if self.is_int[col]:
                        eval_pos[col] = int(round(val))
                    else:
                        eval_pos[col] = val

                # FIX: True Swarm Intelligence for Categoricals (pbest / gbest tracking)
                for col, choices in self.cat_choices.items():
                    r = random.random()
                    if r < 0.4:
                        # 40% chance: Follow own best memory
                        eval_val = particle["pbest_pos"][col]
                    elif r < 0.8:
                        # 40% chance: Follow swarm's global best memory
                        eval_val = self.best_config[col]
                    else:
                        # 20% chance: Explore randomly
                        valid_choices = [c for c in choices if c != new_pos[col]]
                        eval_val = random.choice(valid_choices) if valid_choices else new_pos[col]
                    
                    new_pos[col] = eval_val
                    eval_pos[col] = eval_val

                # Evaluate using the strictly bounded/discretized config
                d2h = self._evaluate_particle(eval_pos)
                
                # Update internal state with continuous positions
                particle["pos"] = new_pos
                
                if d2h < particle["pbest_d2h"]:
                    # Particle memory remains continuous for proper trajectory math
                    particle["pbest_pos"] = copy.deepcopy(new_pos) 
                    particle["pbest_d2h"] = d2h
                    
                    if d2h < self.best_value:
                        self.best_value = d2h
                        # Global best MUST be the valid discrete configuration
                        self.best_config = copy.deepcopy(eval_pos)

        self.end_time = time.time()
        return self.best_config, self.best_value