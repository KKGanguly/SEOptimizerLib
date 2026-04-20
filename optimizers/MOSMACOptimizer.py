from ConfigSpace import ConfigurationSpace
from ConfigSpace.configuration import Configuration
from smac import Scenario, HyperparameterOptimizationFacade as HPOFacade
from optimizers.base_optimizer import BaseOptimizer
from models.Data import Data
import numpy as np
import time
import uuid

class MOSMACOptimizer(BaseOptimizer):
    """
    SMAC single-objective optimizer with top-6 centroids tracked iteratively
    and evaluation iteration tracking.
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)
        self.best_config = None
        self.best_value = None
        self.cluster_centroids = []  # top-6 best solutions updated iteratively
        self.cache = {}
        self.iteration = 0  # track evaluation iteration

        self.config_space, _, _ = self.model_config.get_configspace()
        self.is_tabular = True
        if self.is_tabular:
            self.X_df = self.model_wrapper.X
            self.columns = list(self.X_df.columns)
            self.nn = Data(
                self.X_df.values.tolist(),
                column_types=self.model_config.column_types,
            )
            self.num_objectives = len(
            self.model_wrapper.get_score(
                {c: self.X_df.iloc[0][c] for c in self.columns}
            )
        )
        else:
            self.columns = list(self.model_config.param_names)

        self.evaluated = []  # all evaluations for centroid updates

    def _clean(self, v):
        if isinstance(v, (np.integer, np.int64)):
            return int(v)
        if isinstance(v, (np.floating, np.float64)):
            return float(v)
        return v

    def _config_to_dict(self, config: Configuration):
        return {p: self._clean(config[p]) for p in self.model_config.param_names}

    def _row_tuple(self, hyperparams):
        return tuple(hyperparams[p] for p in self.model_config.param_names)

    def _nearest_row(self, hyperparams_dict):
        if not self.is_tabular:
            return hyperparams_dict
        query = [hyperparams_dict[col] for col in self.columns]
        nn_row = self.nn.nearestRow(query)
        return {col: self._clean(v) for col, v in zip(self.columns, nn_row)}

    def _update_top_centroids(self, hp_dict, fitness):
        """
        Maintain running top-6 centroids like EZR
        """
        self.evaluated.append((hp_dict, fitness))
        sorted_top = sorted(self.evaluated, key=lambda x: x[1])[:6]
        self.cluster_centroids = [x[0] for x in sorted_top]

    def _target_function(self, config: Configuration, seed=0):
        raw_hp = self._config_to_dict(config)
        valid_hp = self._nearest_row(raw_hp)
        key = self._row_tuple(valid_hp)

        if key in self.cache:
            fitness = self.cache[key]
        else:
            score = self.model_wrapper.run_model(valid_hp)
            fitness = 1 - score  # SMAC minimizes
            self.cache[key] = fitness

        # Track iteration
        self.iteration += 1
        try:
            scores = self.model_wrapper.get_score(valid_hp)
            scores = tuple(1 - s for s in scores)  # flip 1-d2h → d2h
        except Exception:
            scores = tuple(0.0 for _ in range(self.num_objectives))

        self.track_evaluation(valid_hp, scores, self.iteration)

        # Update top-6 centroids
        self._update_top_centroids(valid_hp, fitness)

        return fitness

    def optimize(self):
        n_trials = self.config["n_trials"]
        self.start_time = time.time()
        output_directory = (
            f"{self.config['output_directory']}/"
            f"smac_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:4]}"
        )

        scenario = Scenario(
            configspace=self.config_space,
            n_trials=n_trials,
            deterministic=True,
            output_directory=output_directory,
            seed=self.seed,
        )

        smac = HPOFacade(
            scenario=scenario,
            target_function=self._target_function,
            overwrite=True,
        )

        try:
            incumbent = smac.optimize()
        except Exception:
            incumbent = smac.optimizer.intensifier.get_incumbent()

        final_dict = self._nearest_row(self._config_to_dict(incumbent))
        final_score = 1 - self.model_wrapper.run_model(final_dict)

        self.best_config = final_dict
        self.best_value = final_score
        self.end_time = time.time()
        return self.best_config, self.best_value
