from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import optuna
import optunahub
import time
import copy
import numpy as np
import random
from ConfigSpace.hyperparameters import (
    OrdinalHyperparameter,
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    Constant,
)
from optimizers.base_optimizer import BaseOptimizer
from utils import DistanceUtil


class MOEADOptimizer(BaseOptimizer):
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
        """Clean items and round floats slightly to prevent cache bloat."""
        val = v.item() if hasattr(v, "item") else v
        return round(val, 6) if isinstance(val, float) else val

    def _row_tuple(self, hp_dict):
        return tuple(self._safe_clean(hp_dict[c]) for c in self.columns)

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

    def _objective(self, trial):
        valid_hp = {}
        
        # 1. Map Optuna's suggestions to the continuous ConfigSpace boundaries
        for hp in self.config_space.get_hyperparameters():
            hp_type = type(hp).__name__
            if isinstance(hp, Constant):
                valid_hp[hp.name] = hp.value
            elif isinstance(hp, OrdinalHyperparameter):
                valid_hp[hp.name] = trial.suggest_categorical(hp.name, list(hp.sequence))
            elif isinstance(hp, CategoricalHyperparameter):
                valid_hp[hp.name] = trial.suggest_categorical(hp.name, list(hp.choices))
            elif hp_type == "UniformFloatHyperparameter":
                valid_hp[hp.name] = trial.suggest_float(hp.name, hp.lower, hp.upper)
            elif hp_type == "UniformIntegerHyperparameter":
                valid_hp[hp.name] = trial.suggest_int(hp.name, int(hp.lower), int(hp.upper))
            else:
                raise ValueError(f"Unsupported hyperparameter type: {hp_type}")

        key = self._row_tuple(valid_hp)

        # 2. Track & Evaluate
        if key in self.cache:
            moead_scores, raw_scores = self.cache[key]
        else:
            try:
                # Raw scores from framework are (0.0=Worst, 1.0=Best)
                raw_scores = self.model_wrapper.get_score(valid_hp)
                
                # MOEA/D minimizes by default, so we invert to (0.0=Best, 1.0=Worst)
                moead_scores = tuple(1.0 - s for s in raw_scores)
            except Exception:
                # Crash penalty
                raw_scores = tuple(0.0 for _ in range(self.num_objectives))
                moead_scores = tuple(float('inf') for _ in range(self.num_objectives))
                
            self.cache[key] = (moead_scores, raw_scores)

        self.iteration += 1
        
        # ALWAYS track using the real framework scores
        try:
            self.track_evaluation(valid_hp, list(raw_scores), self.iteration)
        except Exception:
            self.logging_util.log("iteration", self.iteration)

        # Return inverted scores strictly for Optuna's sorting math
        return moead_scores

    def optimize(self):
        self.start_time = time.time()
        n_trials = self.config["n_trials"]

        # =================================================================
        # SINGLE OBJECTIVE ABORT
        # MOEA/D mathematically requires >= 2 objectives. 
        # Abort and return nulls if the dataset is single-objective.
        # =================================================================
        if self.num_objectives < 2:
            self.end_time = time.time()
            return {}, float("inf")

        # 1. Callback to log Pareto Front every generation WITH D2H EXTRACTION
        def log_pareto_front(study, trial):
            pareto_trials = study.best_trials
            if pareto_trials:
                best_frontier_trial = None
                min_d2h = float("inf")
                ideal = [0.0] * self.num_objectives

                for t in pareto_trials:
                    current_d2h = DistanceUtil.d2h(ideal, list(t.values))
                    if current_d2h < min_d2h:
                        min_d2h = current_d2h
                        best_frontier_trial = t

                if best_frontier_trial:
                    best_raw = dict(best_frontier_trial.params)
                    for hp in self.config_space.get_hyperparameters():
                        if isinstance(hp, Constant):
                            best_raw[hp.name] = hp.value
                    
                    try:
                        if hasattr(self, 'track_frontier'):
                            self.track_frontier(self.iteration, pareto_trials, best_raw, best_frontier_trial)
                    except Exception:
                        pass

        # 2. Set up the MOEA/D Sampler
        module = optunahub.load_module("samplers/moead")
        sampler = module.MOEADSampler(
            population_size=self.population_size,
            seed=self.seed,
        )

        study = optuna.create_study(
            directions=["minimize"] * self.num_objectives,
            sampler=sampler,
        )

        # 3. FAIRNESS INITIALIZATION
        obs_budget = min(self.population_size, n_trials)
        for _ in range(obs_budget):
            initial_config = self._sample_config()
            enqueue_dict = {
                hp.name: initial_config[hp.name] 
                for hp in self.config_space.get_hyperparameters() 
                if not isinstance(hp, Constant)
            }
            study.enqueue_trial(enqueue_dict)

        # 4. Execute the loop
        study.optimize(
            self._objective, 
            n_trials=n_trials, 
            timeout=3600, 
            catch=(Exception,), 
            callbacks=[log_pareto_front]
        )

        # ---------------------------------------------------------
        # FINAL PARETO FRONTIER EXTRACTION (D2H Winner)
        # ---------------------------------------------------------
        final_frontier = study.best_trials

        best_trial = None
        best_d2h_norm = float("inf")
        ideal = [0.0] * self.num_objectives

        for t in final_frontier:
            d2h_val = DistanceUtil.d2h(ideal, list(t.values))
            if d2h_val < best_d2h_norm:
                best_d2h_norm = d2h_val
                best_trial = t

        # Safety Check
        if best_trial is None:
            self.end_time = time.time()
            return None, float("inf")

        best_raw = dict(best_trial.params)
        for hp in self.config_space.get_hyperparameters():
            if isinstance(hp, Constant):
                best_raw[hp.name] = hp.value

        self.best_config = best_raw
        self.best_value = best_d2h_norm
        self.end_time = time.time()

        return self.best_config, self.best_value