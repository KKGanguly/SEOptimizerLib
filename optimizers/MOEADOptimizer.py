# optimizers/MOEADOptimizer.py
import optuna
import optunahub
import time
import copy
from ConfigSpace.hyperparameters import (
    OrdinalHyperparameter, CategoricalHyperparameter, Constant,
    UniformFloatHyperparameter, UniformIntegerHyperparameter
)
from optimizers.base_optimizer import BaseOptimizer
from utils import DistanceUtil

class MOEADOptimizer(BaseOptimizer):
    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)
        self.config_space, _, _ = self.model_config.get_configspace()
        self.num_objectives = len(self.model_wrapper.get_score(dict(self.config_space.sample_configuration())))
        self.iteration = 0
        self.best_config = None
        self.best_value = float("inf")
        self.population_size = int(self.config.get("pop_size", 20))

    def _objective(self, trial):
        hp_dict = {}
        for hp in self.config_space.get_hyperparameters():
            if isinstance(hp, Constant): hp_dict[hp.name] = hp.value
            elif isinstance(hp, OrdinalHyperparameter): hp_dict[hp.name] = trial.suggest_categorical(hp.name, list(hp.sequence))
            elif isinstance(hp, CategoricalHyperparameter): hp_dict[hp.name] = trial.suggest_categorical(hp.name, list(hp.choices))
            elif isinstance(hp, UniformFloatHyperparameter): hp_dict[hp.name] = trial.suggest_float(hp.name, hp.lower, hp.upper)
            elif isinstance(hp, UniformIntegerHyperparameter): hp_dict[hp.name] = trial.suggest_int(hp.name, hp.lower, hp.upper)

        scores = list(self.model_wrapper.get_score(hp_dict))
        ideal = [0] * self.num_objectives
        d2h = DistanceUtil.d2h(ideal, scores)
        
        self.iteration += 1
        self.track_evaluation(hp_dict, scores, self.iteration)
        if d2h < self.best_value:
            self.best_value = d2h
            self.best_config = copy.deepcopy(hp_dict)
        return scores

    def optimize(self):
        def log_pareto_front(study, trial):
            if hasattr(self, 'track_frontier'):
                pareto_trials = study.best_trials
                best_params = copy.deepcopy(min(pareto_trials, key=lambda t: DistanceUtil.d2h([0]*self.num_objectives, t.values)).params)
                for hp in self.config_space.get_hyperparameters():
                    if isinstance(hp, Constant): best_params[hp.name] = hp.value
                self.track_frontier(self.iteration, pareto_trials, best_params, trial)

        module = optunahub.load_module("samplers/moead")
        sampler = module.MOEADSampler(population_size=self.population_size, seed=self.seed)
        study = optuna.create_study(directions=["minimize"] * self.num_objectives, sampler=sampler)
        study.optimize(self._objective, n_trials=self.config["n_trials"], callbacks=[log_pareto_front])
        return self.best_config, self.best_value