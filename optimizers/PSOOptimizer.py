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

        self.pop_size = int(self.config.get("pop_size", 20))
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
            return self.cache[key][1]
        try:
            scores = tuple(self.model_wrapper.get_score(hp_dict))
            ideal = [0] * self.num_objectives
            d2h_val = DistanceUtil.d2h(ideal, list(scores))
        except Exception:
            scores = tuple(float('inf') for _ in range(self.num_objectives))
            d2h_val = float('inf')
            
        self.cache[key] = (scores, d2h_val)
        self.iteration += 1
        self.track_evaluation(hp_dict, list(scores), self.iteration)
        return d2h_val

    def optimize(self):
        n_trials = self.config["n_trials"]
        self.start_time = time.time()

        # 1. Initialize Swarm from raw dataset rows
        swarm = []
        for _ in range(self.pop_size):
            initial_idx = random.randint(0, self.n_rows - 1)
            pos = self._idx_to_config(initial_idx)
            
            # Initial random velocities based on parameter bounds
            vel = {}
            for col, (lower, upper) in self.bounds.items():
                span = upper - lower
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
                
                new_pos = copy.deepcopy(particle["pos"])
                
                for col in self.bounds.keys():
                    r1, r2 = random.random(), random.random()
                    
                    cognitive = self.c1 * r1 * (particle["pbest_pos"][col] - particle["pos"][col])
                    social = self.c2 * r2 * (self.best_config[col] - particle["pos"][col])
                    
                    new_vel = (self.w * particle["vel"][col]) + cognitive + social
                    particle["vel"][col] = new_vel
                    
                    val = particle["pos"][col] + new_vel
                    lower, upper = self.bounds[col]
                    val = max(lower, min(upper, val))
                    
                    if self.is_int[col]:
                        val = int(round(val))
                    new_pos[col] = val

                for col, choices in self.cat_choices.items():
                    if random.random() < 0.1: 
                        if len(choices) > 1:
                            valid_choices = [c for c in choices if c != new_pos[col]]
                            if valid_choices:
                                new_pos[col] = random.choice(valid_choices)

                d2h = self._evaluate_particle(new_pos)
                particle["pos"] = new_pos
                
                if d2h < particle["pbest_d2h"]:
                    particle["pbest_pos"] = copy.deepcopy(new_pos)
                    particle["pbest_d2h"] = d2h
                    
                    if d2h < self.best_value:
                        self.best_value = d2h
                        self.best_config = copy.deepcopy(new_pos)

        self.end_time = time.time()
        return self.best_config, self.best_value