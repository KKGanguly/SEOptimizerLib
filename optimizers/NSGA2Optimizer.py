# optimizers/NSGA2Optimizer.py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import optuna
import time
import copy
from optuna.samplers import NSGAIISampler
from ConfigSpace.hyperparameters import (
    OrdinalHyperparameter,
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    Constant,
)
from optimizers.base_optimizer import BaseOptimizer
from utils import DistanceUtil

class NSGA2Optimizer(BaseOptimizer):
    """
    NSGA-II Optimizer.
    Archetype: Population-Based Multi-Objective Evolutionary Optimizer.
    
    Refactored to search the continuous surrogate space natively,
    removing empirical dataset projections for consistency.
    """
    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)
        self.config_space, _, _ = self.model_config.get_configspace()
        self.cache = {}
        
        # Test objective count
        test_hp = self.config_space.sample_configuration()
        self.num_objectives = len(self.model_wrapper.get_score(dict(test_hp)))
        
        self.iteration = 0
        self.best_config = None
        self.best_value = float("inf")
        self.population_size = int(self.config.get("pop_size", 20))

    def _objective(self, trial):
        """Maps continuous config space directly to surrogate."""
        hp_dict = {}
        for hp in self.config_space.get_hyperparameters():
            if isinstance(hp, Constant):
                hp_dict[hp.name] = hp.value
            elif isinstance(hp, OrdinalHyperparameter):
                hp_dict[hp.name] = trial.suggest_categorical(hp.name, list(hp.sequence))
            elif isinstance(hp, CategoricalHyperparameter):
                hp_dict[hp.name] = trial.suggest_categorical(hp.name, list(hp.choices))
            elif isinstance(hp, UniformFloatHyperparameter):
                hp_dict[hp.name] = trial.suggest_float(hp.name, hp.lower, hp.upper)
            elif isinstance(hp, UniformIntegerHyperparameter):
                hp_dict[hp.name] = trial.suggest_int(hp.name, hp.lower, hp.upper)

        # Get scores directly from surrogate
        scores = list(self.model_wrapper.get_score(hp_dict))
            
        # Standard D2h normalization (distance to origin)
        ideal = [0] * self.num_objectives
        d2h = DistanceUtil.d2h(ideal, scores)
        
        self.iteration += 1
        self.track_evaluation(hp_dict, scores, self.iteration)
        
        if d2h < self.best_value:
            self.best_value = d2h
            self.best_config = copy.deepcopy(hp_dict)

        return scores

    def optimize(self):
        n_trials = self.config["n_trials"]
        self.start_time = time.time()

        # Callback to track the Pareto Front after every trial
        def log_pareto_front(study, trial):
            pareto_trials = study.best_trials
            ideal = [0] * self.num_objectives
            
            # Identify the best in current frontier
            best_frontier_trial = min(pareto_trials, key=lambda t: DistanceUtil.d2h(ideal, t.values))
            
            # Merge trial params with constants
            best_params = copy.deepcopy(best_frontier_trial.params)
            for hp in self.config_space.get_hyperparameters():
                if isinstance(hp, Constant):
                    best_params[hp.name] = hp.value
            
            if hasattr(self, 'track_frontier'):
                self.track_frontier(self.iteration, pareto_trials, best_params, best_frontier_trial)

        sampler = NSGAIISampler(
            population_size=self.population_size,
            seed=self.seed,
        )

        study = optuna.create_study(
            directions=["minimize"] * self.num_objectives,
            sampler=sampler,
        )

        study.optimize(
            self._objective, 
            n_trials=n_trials, 
            timeout=3600, 
            catch=(Exception,), 
            callbacks=[log_pareto_front]
        )

        self.end_time = time.time()
        return self.best_config, self.best_value