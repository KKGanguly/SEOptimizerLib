import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import random
import copy
import lingam
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
from itertools import product
from causallearn.search.ConstraintBased.FCI import fci
from util import get_objective, read_file
from util.matrix import get_adjacency_matrix
from util.helper import get_tree_paths, has_negative_value, in_rule, Model, merge_constraints, Latin_sample
from util.ei import get_ei
from typing import List, Tuple, Any, Optional, Sequence
from dataclasses import dataclass, field
import logging
import time
from collections import defaultdict
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_ESTIMATORS = 10
DEFAULT_MAX_ITERATIONS = int(1e4)
DEFAULT_STOP_THRESHOLD = 0.01
DEFAULT_INITIAL_SIZE = 10
DEFAULT_MAX_LIVES = 200
DEFAULT_BUDGET = 50
DEFAULT_L_VALUE = 10
DEFAULT_K_VALUE = 0.1

@dataclass
class PromiseTuneConfig:
    initial_size: int = DEFAULT_INITIAL_SIZE
    maxlives: int = DEFAULT_MAX_LIVES
    budget: int = DEFAULT_BUDGET
    rule: bool = True
    l: int = DEFAULT_L_VALUE
    k: float = DEFAULT_K_VALUE
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    stop_threshold: float = DEFAULT_STOP_THRESHOLD
    enable_timing: bool = False
    timing_top_n: int = 20
    causal_refresh_interval: int = 10
    reuse_last_causal: bool = True
    log_rules: bool = True


@dataclass
class CausalState:
    ACEs: List[float] = field(default_factory=list)
    all_path: List[Any] = field(default_factory=list)
    sample_plus: int = 0
    last_refresh_step: int = -1


class TimingProfiler:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._stats = defaultdict(lambda: {'total': 0.0, 'count': 0})

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._stats[name]['total'] += elapsed
            self._stats[name]['count'] += 1

    def report_lines(self, top_n: int = 20) -> List[str]:
        if not self.enabled:
            return []

        items = sorted(self._stats.items(), key=lambda x: x[1]['total'], reverse=True)[:top_n]
        lines = ["[TimingProfiler] section | calls | total(s) | avg(ms)"]
        for name, stat in items:
            avg_ms = (stat['total'] / stat['count']) * 1000 if stat['count'] else 0.0
            lines.append(f"[TimingProfiler] {name} | {stat['count']} | {stat['total']:.4f} | {avg_ms:.2f}")
        return lines


ACTIVE_PROFILER = TimingProfiler(enabled=False)


@contextmanager
def timed(name: str):
    with ACTIVE_PROFILER.section(name):
        yield

def extract_unique_data(training_indep: List[List[Any]], training_dep: List[float]) -> Tuple[List[List[Any]], List[float]]:
    seen = set()
    new_rows, new_perfs = [], []
    for row, perf in zip(training_indep, training_dep):
        row_key = tuple(row)
        if row_key not in seen:
            seen.add(row_key)
            new_rows.append(row)
            new_perfs.append(perf)
    return new_rows, new_perfs

def build_decision_tree_paths(training_indep: List[List[Any]], training_dep: List[float], feature_names: List[str], sample: int) -> List[Any]:
    with timed("build_decision_tree_paths"):
        model = RandomForestRegressor(n_estimators=DEFAULT_ESTIMATORS, min_samples_leaf=int(sample))
        model.fit(training_indep, training_dep)

        seen_paths = set()
        all_path = []
        for sub_tree in model.estimators_:
            paths = get_tree_paths(sub_tree, feature_names)
            for path in paths:
                path_key = tuple(tuple(component) for component in path)
                if path_key not in seen_paths:
                    seen_paths.add(path_key)
                    all_path.append(path[:])
        return all_path

def cause_find(training_indep: List[List[Any]], training_dep: List[float], file: Any, sample: int, sample_plus: int = 0) -> Tuple[List[float], List[Any], int]:
    with timed("cause_find"):
        feature_names = file.features
        all_path = build_decision_tree_paths(training_indep, training_dep, feature_names, sample)
        new_rows, new_perfs = extract_unique_data(training_indep, training_dep)

        if len(all_path) >= len(new_rows) - 2:
            sample_plus += 1
            all_path = random.sample(all_path, len(new_rows) - 2)

        try:
            ACEs, valid_rules = model_fit(new_rows, all_path, feature_names, new_perfs)
        except Exception:
            if len(all_path) > 2:
                all_path = random.sample(all_path, len(all_path) - 2)
                try:
                    ACEs, valid_rules = model_fit(new_rows, all_path, feature_names, new_perfs)
                except Exception:
                    return [], [], sample_plus
            else:
                return [], [], sample_plus

        return ACEs, valid_rules, sample_plus
    
def model_fit(new_rows: List[List[Any]], all_path: List[Any], feature_names: List[str], new_perfs: List[float]) -> Tuple[List[float], List[Any]]:
    with timed("model_fit"):
        column_names = copy.deepcopy(feature_names)
        training_indep_new = copy.deepcopy(new_rows)

        with timed("model_fit.design_matrix"):
            for i, x in enumerate(new_rows):
                for rule in all_path:
                    training_indep_new[i].append(1 if in_rule(x, rule, feature_names) else 0)
                training_indep_new[i].append(new_perfs[i])

            labels = [f"rule{a}" for a in range(len(all_path))]
            column_names.extend(labels)
            labels.append("perf")
            column_names.append("perf")

            df = pd.DataFrame(training_indep_new, columns=column_names)
            df.drop(columns=feature_names, inplace=True)
            data = df.to_numpy()

        with timed("model_fit.fci"):
            try:
                g, _ = fci(data, show_progress=False, verbose=False)
                adjacency_matrix = get_adjacency_matrix(g).T
                adjacency_matrix[:, -1] = 0
            except Exception:
                adjacency_matrix = np.zeros((len(labels), len(labels)), dtype=int)

        prior_knowledge = adjacency_matrix
        prior_knowledge[-1, :-1] = 1
        prior_knowledge[-1, -1] = -1

        with timed("model_fit.lingam_fit"):
            try:
                model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
                model.fit(data)
            except ValueError:
                return [], []

        valid_rules, ACEs = [], []
        with timed("model_fit.estimate_total_effect"):
            for k in range(len(labels) - 1):
                ACE = model.estimate_total_effect(data, k, len(labels) - 1)
                if ACE:
                    valid_rules.append(all_path[k])
                    ACEs.append(ACE)

        return ACEs, valid_rules

# def random_search_base(max_iterations: int, stop_threshold: float, estimators: List[Any], eta: float, x_generator) -> Tuple[List[List[Any]], float]:
#     best_value = float('-inf')
#     values, xs = [], []
    
#     for i in range(max_iterations):
#         current_x = x_generator()
#         pred = [e.predict([current_x]) for e in estimators]
#         current_value = get_ei(pred, eta)[0]
        
#         xs.append(current_x)
#         values.append(current_value)
        
#         if current_value > best_value:
#             best_value = current_value
            
#         if i > 0:
#             try:
#                 kde = stats.gaussian_kde(values)
#                 improvement_probability = 1 - kde.integrate_box_1d(-np.inf, best_value)
#                 if improvement_probability < stop_threshold:
#                     break
#             except Exception:
#                 break
                
#     return xs, best_value

def random_search_base(max_iterations: int, stop_threshold: float, estimators: List[Any], eta: float, x_generator, batch_size: int = 100) -> Tuple[List[List[Any]], float]:
    with timed("random_search_base"):
        best_value = float('-inf')
        values, xs = [], []

        for batch_start in range(0, max_iterations, batch_size):
            batch_end = min(batch_start + batch_size, max_iterations)
            current_batch_size = batch_end - batch_start

            with timed("random_search_base.generate_batch"):
                batch_configs = [x_generator() for _ in range(current_batch_size)]

            with timed("random_search_base.predict_batch"):
                batch_predictions = []
                for estimator in estimators:
                    batch_pred = estimator.predict(batch_configs)
                    batch_predictions.append(batch_pred)
                batch_predictions = np.array(batch_predictions).T

            with timed("random_search_base.ei_batch"):
                for i, current_x in enumerate(batch_configs):
                    pred = batch_predictions[i]
                    current_value = get_ei([pred], eta)[0]

                    xs.append(current_x)
                    values.append(current_value)

                    if current_value > best_value:
                        best_value = current_value

            if len(values) > batch_size and len(values) % (batch_size * 2) == 0:
                with timed("random_search_base.kde_check"):
                    try:
                        kde = stats.gaussian_kde(values)
                        improvement_probability = 1 - kde.integrate_box_1d(-np.inf, best_value)
                        if improvement_probability < stop_threshold:
                            break
                    except Exception:
                        break

        return xs, best_value

def random_search_with_static_distribution(max_iterations: int, stop_threshold: float, y: Sequence[Sequence[Any]], estimators: List[Any], eta: float, every_column: List[List[Any]]) -> Tuple[List[List[Any]], float]:
    def x_generator():
        current_x = list(random.choice(y))
        for index in range(len(current_x)):
            if current_x[index] == '_':
                current_x[index] = random.choice(every_column[index])
        return current_x
    
    return random_search_base(max_iterations, stop_threshold, estimators, eta, x_generator)

def random_search_with_static_distribution_cold(max_iterations: int, stop_threshold: float, y: Sequence[Sequence[Any]], estimators: List[Any], eta: float) -> Tuple[List[List[Any]], float]:
    def x_generator():
        return [random.choice(y[index]) for index in range(len(y))]
    
    return random_search_base(max_iterations, stop_threshold, estimators, eta, x_generator)

def get_training_sequence_by_PromiseTune(training_indep: List[List[Any]], training_dep: List[float], eta: float, file: Any, config: PromiseTuneConfig, causal_state: CausalState, current_step: int) -> Tuple[List[List[Any]], Any, List[Any]]:
    with timed("get_training_sequence_by_PromiseTune"):
        with timed("get_training_sequence_by_PromiseTune.model_train"):
            model = Model(training_indep, training_dep)
        best_configs = sorted(zip(training_indep, training_dep), key=lambda x: x[1])[:int(config.k * len(training_indep))]
        estimators = model.estimators_
        every_column = file.independent_set
        header = file.features
        Rule, candidates = [], []
        ACEs, all_path = [], []

        if config.rule:
            interval = max(1, int(config.causal_refresh_interval))
            should_refresh = (
                (not config.reuse_last_causal)
                or (causal_state.last_refresh_step < 0)
                or ((current_step - causal_state.last_refresh_step) >= interval)
            )

            if should_refresh:
                with timed("cause_find.refresh"):
                    ACEs, all_path, sample_plus = cause_find(
                        training_indep,
                        training_dep,
                        file,
                        config.l + causal_state.sample_plus,
                        causal_state.sample_plus,
                    )
                causal_state.ACEs = ACEs
                causal_state.all_path = all_path
                causal_state.sample_plus = sample_plus
                causal_state.last_refresh_step = current_step
                logger.info(f"[Causal] refresh at step={current_step}, rules={len(all_path)}")
            else:
                with timed("cause_find.reuse"):
                    ACEs = causal_state.ACEs
                    all_path = causal_state.all_path
            for i, path in zip(ACEs, all_path):
                if i < 0:
                    Rule += path
                    for indep, _ in best_configs:
                        if in_rule(indep, path, header):
                            if config.log_rules:
                                logger.info(f"==>RULE: {[merge_constraints(path)]}, ==>ACE: {i}")
                            break

        if (config.rule and not has_negative_value(ACEs)) or not config.rule:
            xs, _ = random_search_with_static_distribution_cold(config.max_iterations, config.stop_threshold, every_column, estimators, eta)
            candidates.extend(xs)
        else:
            for index in range(len(all_path)):
                if ACEs[index] > 0:
                    continue

                results = [[] for _ in range(len(header))]
                single_rule = all_path[index]

                for i, column_values, single_feature in zip(range(len(header)), every_column, header):
                    tmp_left, tmp_right = [], []
                    for single_component in single_rule:
                        if single_component[0] == single_feature:
                            if single_component[2] == 'L':
                                tmp_left.append(single_component[1])
                            else:
                                tmp_right.append(single_component[1])

                    if tmp_left and tmp_right:
                        results[i] = [x for x in column_values if max(tmp_right) <= x <= min(tmp_left)]
                    elif tmp_left:
                        results[i] = [x for x in column_values if x <= min(tmp_left)]
                    elif tmp_right:
                        results[i] = [x for x in column_values if max(tmp_right) <= x]

                for i in range(len(results)):
                    if not results[i]:
                        results[i] = ['_']
                    elif len(results[i]) > 10:
                        results[i] = random.sample(results[i], 10)

                new_list = list(product(*results))
                xs, _ = random_search_with_static_distribution(config.max_iterations, config.stop_threshold, new_list, estimators, eta, every_column)
                candidates.extend(xs)

        return candidates, model, Rule

def get_best_configuration_id_PromiseTune(training_indep: List[List[Any]], training_dep: List[float], eta: float, file: Any, config: PromiseTuneConfig, causal_state: CausalState, current_step: int) -> Tuple[Optional[List[Any]], List[Any]]:
    with timed("get_best_configuration_id_PromiseTune"):
        test_sequence, model, Rule = get_training_sequence_by_PromiseTune(training_indep, training_dep, eta, file, config, causal_state, current_step)
        estimators = model.estimators_

        with timed("get_best_configuration_id_PromiseTune.predict_candidates"):
            pred = [e.predict(test_sequence) for e in estimators]
            value = get_ei(pred, eta)

        merged_predicted_objectives = [[i, a] for a, i in zip(value, test_sequence)]
        merged_predicted_objectives = sorted(merged_predicted_objectives, key=lambda x: x[1], reverse=True)

        for x in merged_predicted_objectives:
            if x[0] not in training_indep:
                return x[0], Rule

        return None, Rule

def PromiseTune(filename: str, config: PromiseTuneConfig, seed: int, oracle=None) -> Tuple[Optional[Any], List[Any]]:
    global ACTIVE_PROFILER
    ACTIVE_PROFILER = TimingProfiler(enabled=config.enable_timing)

    random.seed(seed)
    np.random.seed(seed)
    
    steps = 0
    configuration_b = None
    results, x_axis, xs, Rules = [], [], [], []
    lives = config.maxlives
    causal_state = CausalState()
    import random as rnd
    original_shuffle = rnd.shuffle
    rnd.shuffle = lambda x: None # Disable shuffling for this call
    with timed("PromiseTune.data_loading"):
        file = read_file.get_data(filename, config.initial_size, seed)
    training_indep = [t.decision for t in file.training_set]
    training_dep = []

    with timed("PromiseTune.init_objective_eval"):
        for action in training_indep:
            if oracle:
                reward, configuration = oracle(action)
            else:
                reward, configuration = get_objective.get_objective(file.dict_search, action)
            if reward is None:
                continue
            training_dep.append(float(reward))
            xs.append(action)
            results.append(float(reward))

    result = min(results) if results else float('inf')
    exist_configuration = training_indep[:]
    
    while config.initial_size + steps < config.budget:
        steps += 1
        with timed("PromiseTune.iter_get_best"):
            best_solution, Rule = get_best_configuration_id_PromiseTune(
                training_indep, training_dep, result, file, config, causal_state, steps)
        
        if best_solution is None:
            break
            
        with timed("PromiseTune.iter_objective_eval"):
            if oracle:
                best_result, configuration = oracle(best_solution)
            else:
                best_result, configuration = get_objective.get_objective(file.dict_search, best_solution)
        if best_result is None:
            lives -= 1
            if lives == 0:
                break
            continue

        best_result = float(best_result)
        Rules.append(Rule)
        training_indep.append(best_solution)
        training_dep.append(best_result)
        x_axis.append(steps)
        xs.append(configuration)
        results.append(best_result)
        
        if best_result < result:
            configuration_b = configuration
            result = best_result
            lives = config.maxlives
        else:
            lives -= 1
        # this is unfair to other optimizers  
        # if configuration in exist_configuration:
        #     config.budget += 1
        # else:
        #     exist_configuration.append(configuration)
        
        # Track duplicates so the algorithm knows what it has seen, 
        # but NEVER extend the global benchmark budget!
        if configuration not in exist_configuration:
            exist_configuration.append(configuration)
        logger.info(f'=>Loop: {steps}, =>Reward: {best_result:.2f}, =>Config: {configuration}')
        
        if lives == 0:
            break

    if config.enable_timing:
        for line in ACTIVE_PROFILER.report_lines(top_n=config.timing_top_n):
            logger.info(line)
            
    return configuration_b, Rules, xs

def run_main(seed: int, name: str) -> None:
    config = PromiseTuneConfig()
    best_configuration, Rules = PromiseTune(
        filename=f"./Data/{name}.csv",
        config=config,
        seed=seed
    )
    logger.info(f"Best configuration: {best_configuration}, Rules: {Rules}")

if __name__ == '__main__':
    seeds = [0]
    #systems = ['exastencils', 'dconvert', '7z', "BDBC", 'deeparch',
    #           'PostgreSQL', 'javagc', 'storm', 'x264',
    #           'redis', 'HSQLDB', 'LLVM']
    systems = ['7z']
    for name in systems:
        for seed in seeds:
            run_main(seed, name)
