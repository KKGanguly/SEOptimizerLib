# optimizers/PromiseTuneOptimizer.py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from optimizers.base_optimizer import BaseOptimizer
from utils import DistanceUtil
import time
import numpy as np
import pandas as pd
import copy
import random
from scipy.stats import norm, gaussian_kde
from sklearn.ensemble import RandomForestRegressor
from causallearn.search.ConstraintBased.FCI import fci
import lingam

class PromiseTuneOptimizer(BaseOptimizer):
    """
    Exact Equivalent of PromiseTune (Chen et al.).
    Implements the original Cartesian-based Random Search + Expected Improvement 
    acquisition function, purified by FCI and DirectLiNGAM, using the exact 
    hyperparameters published in their implementation.
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

        # EXACT PromiseTune Hyperparameters
        self.initial_size = int(self.config.get("initial_size", 10))
        self.max_iterations = int(self.config.get("max_iterations", 10000))
        self.stop_threshold = float(self.config.get("stop_threshold", 0.01))
        self.causal_refresh_interval = int(self.config.get("causal_refresh_interval", 10))
        self.l_value = int(self.config.get("l", 10))
        self.k_value = float(self.config.get("k", 0.1))
        self.estimators_count = int(self.config.get("estimators", 10))

        # Build empirical columns (file.independent_set)
        self.every_column = []
        for col in self.columns:
            self.every_column.append(list(self.X_df[col].unique()))

        # Causal State Tracker
        self.causal_state = {
            "ACEs": [],
            "all_path": [],
            "sample_plus": 0,
            "last_refresh_step": -1
        }

    def _clean(self, v):
        return v.item() if hasattr(v, "item") else v

    def _idx_to_config(self, idx):
        row = self.X_df.iloc[idx]
        return {c: self._clean(row[c]) for c in self.columns}

    def _config_to_vec(self, hp_dict):
        """Encodes config dict to a numeric vector for PromiseTune's internal RF."""
        vec = []
        for col in self.columns:
            val = hp_dict[col]
            if isinstance(val, str) or not isinstance(val, (int, float)):
                hp = self.config_space.get_hyperparameter(col)
                choices = list(hp.choices) if hasattr(hp, 'choices') else list(hp.sequence)
                val = choices.index(val)
            vec.append(val)
        return vec

    def _vec_to_config(self, vec):
        """Decodes numeric vector back to config dict."""
        hp_dict = {}
        for i, col in enumerate(self.columns):
            val = vec[i]
            hp = self.config_space.get_hyperparameter(col)
            hp_type = type(hp).__name__
            if hp_type in ["CategoricalHyperparameter", "OrdinalHyperparameter"]:
                choices = list(hp.choices) if hasattr(hp, 'choices') else list(hp.sequence)
                hp_dict[col] = choices[int(val)]
            else:
                hp_dict[col] = val
        return hp_dict

    def _evaluate_config(self, hp_dict):
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
        
        if d2h_val < self.best_value:
            self.best_value = d2h_val
            self.best_config = copy.deepcopy(hp_dict)
            
        return d2h_val

    # -------------------------------------------------------------------------
    # EXACT Authors' Implementations for Core Mechanics
    # -------------------------------------------------------------------------

    def _get_tree_paths(self, tree_estimator):
        """Exact logic of their util.helper.get_tree_paths"""
        tree = tree_estimator.tree_
        paths = []
        def recurse(node, path):
            if tree.children_left[node] == tree.children_right[node]:
                paths.append(path)
            else:
                feature = self.columns[tree.feature[node]]
                threshold = tree.threshold[node]
                recurse(tree.children_left[node], path + [(feature, threshold, 'L')])
                recurse(tree.children_right[node], path + [(feature, threshold, 'R')])
        recurse(0, [])
        return paths

    def _in_rule(self, vec, rule):
        """Exact logic of their util.helper.in_rule"""
        config_dict = self._vec_to_config(vec)
        for feat, thresh, direction in rule:
            val = config_dict[feat]
            if direction == 'L' and val > thresh: return False
            if direction == 'R' and val <= thresh: return False
        return True

    def _get_ei(self, predictions, eta):
        """Exact logic of their util.ei.get_ei (Assuming Minimization)"""
        mu = np.mean(predictions, axis=0)
        sigma = np.std(predictions, axis=0)
        
        ei = np.zeros_like(mu)
        mask = sigma > 0
        
        # Improvement (minimization means we want mu < eta)
        improvement = eta - mu[mask]
        Z = improvement / sigma[mask]
        ei[mask] = improvement * norm.cdf(Z) + sigma[mask] * norm.pdf(Z)
        ei[~mask] = 0.0
        return ei

    def _cause_find(self, training_indep, training_dep, sample_leaf):
        """Exact logic of their cause_find and model_fit, fortified against zero-variance."""
        # 1. Build Random Forest to get paths
        model = RandomForestRegressor(n_estimators=self.estimators_count, min_samples_leaf=sample_leaf, random_state=self.seed)
        model.fit(training_indep, training_dep)
        
        all_path = []
        seen_paths = set()
        for estimator in model.estimators_:
            paths = self._get_tree_paths(estimator)
            for path in paths:
                path_key = tuple(path)
                if path_key not in seen_paths:
                    seen_paths.add(path_key)
                    all_path.append(path)

        # Filter paths if too many to prevent memory explosion
        if len(all_path) >= len(training_indep) - 2:
            self.causal_state["sample_plus"] += 1
            if len(training_indep) > 2:
                all_path = random.sample(all_path, len(training_indep) - 2)

        if not all_path:
            return [], []

        # 2. Design Matrix (Original Features + Rules + Target)
        num_features = len(self.columns)
        num_rules = len(all_path)
        design_matrix = np.zeros((len(training_indep), num_features + num_rules + 1))
        
        for i, x in enumerate(training_indep):
            # Add original features
            design_matrix[i, :num_features] = x
            # Add rule binary flags
            for r_idx, rule in enumerate(all_path):
                design_matrix[i, num_features + r_idx] = 1 if self._in_rule(x, rule) else 0
            # Add target performance
            design_matrix[i, -1] = training_dep[i]

        # FIX: Inject microscopic statistical jitter to prevent Zero-Variance division by zero
        # This keeps variance > 0 without affecting the > 0.01 causal thresholds
        jitter = np.random.normal(0, 1e-6, design_matrix.shape)
        design_matrix += jitter

        # 3. FCI Algorithm
        try:
            # causallearn may throw warnings on highly deterministic data, suppress them internally
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                g, _ = fci(design_matrix, show_progress=False, verbose=False)
                adj = g.graph.T
                adj[:, -1] = 0
        except Exception:
            adj = np.zeros((design_matrix.shape[1], design_matrix.shape[1]), dtype=int)

        # 4. DirectLiNGAM
        prior_knowledge = adj
        prior_knowledge[-1, :-1] = 1
        prior_knowledge[-1, -1] = -1

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lingam_model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
                lingam_model.fit(design_matrix)
        except Exception as e:
            return [], []

        # 5. Estimate ACE (Average Causal Effect)
        valid_rules, ACEs = [], []
        target_idx = design_matrix.shape[1] - 1
        
        for k in range(num_rules):
            rule_col_idx = num_features + k
            try:
                ace = lingam_model.estimate_total_effect(design_matrix, rule_col_idx, target_idx)
                valid_rules.append(all_path[k])
                ACEs.append(ace)
            except Exception:
                continue

        return ACEs, valid_rules

    def _random_search_base(self, rf_model, eta, x_generator):
        """Exact logic of their random_search_base with KDE early stopping."""
        best_value = float('-inf')
        best_x = None
        values = []
        batch_size = 100

        for batch_start in range(0, self.max_iterations, batch_size):
            batch_configs = [x_generator() for _ in range(batch_size)]
            
            # Predict across all trees
            tree_preds = []
            for estimator in rf_model.estimators_:
                tree_preds.append(estimator.predict(batch_configs))
            tree_preds = np.array(tree_preds) # Shape: (estimators, batch)
            
            for i in range(batch_size):
                preds = tree_preds[:, i]
                current_value = self._get_ei([preds], eta)[0]
                values.append(current_value)
                
                if current_value > best_value:
                    best_value = current_value
                    best_x = batch_configs[i]

            # KDE Check
            if len(values) > batch_size and len(values) % (batch_size * 2) == 0:
                try:
                    kde = gaussian_kde(values)
                    improvement_probability = 1 - kde.integrate_box_1d(-np.inf, best_value)
                    if improvement_probability < self.stop_threshold:
                        break
                except Exception:
                    break

        return best_x

    # -------------------------------------------------------------------------
    # Main Optimization Loop
    # -------------------------------------------------------------------------

    def optimize(self):
        n_trials = self.config["n_trials"]
        self.start_time = time.time()

        history_vec = []
        history_y = []
        exist_configs = []

        # Initialization
        obs_budget = min(self.initial_size, n_trials)
        for _ in range(obs_budget):
            initial_idx = random.randint(0, self.n_rows - 1)
            config_dict = self._idx_to_config(initial_idx)
            d2h = self._evaluate_config(config_dict)
            
            vec = self._config_to_vec(config_dict)
            history_vec.append(vec)
            history_y.append(d2h)
            exist_configs.append(tuple(vec))

        while self.iteration < n_trials:
            eta = min(history_y)
            
            # Train surrogate for EI predictions
            model = RandomForestRegressor(n_estimators=self.estimators_count, random_state=self.seed)
            model.fit(history_vec, history_y)

            # Causal Refresh
            should_refresh = (self.causal_state["last_refresh_step"] < 0) or \
                             ((self.iteration - self.causal_state["last_refresh_step"]) >= self.causal_refresh_interval)
            
            if should_refresh:
                leaf_size = self.l_value + self.causal_state["sample_plus"]
                ACEs, all_path = self._cause_find(history_vec, history_y, leaf_size)
                self.causal_state["ACEs"] = ACEs
                self.causal_state["all_path"] = all_path
                self.causal_state["last_refresh_step"] = self.iteration

            ACEs = self.causal_state["ACEs"]
            all_path = self.causal_state["all_path"]

            # Candidate Generation (The Core PromiseTune Logic)
            best_candidate_vec = None
            
            has_negative = any(ace < 0 for ace in ACEs) if ACEs else False

            if not has_negative:
                # Cold Random Search from empirical columns
                def x_generator_cold():
                    return [random.choice(col_vals) for col_vals in self.every_column]
                best_candidate_vec = self._random_search_base(model, eta, x_generator_cold)
            else:
                # Causally Promising Random Search
                candidates_pool = []
                for index in range(len(all_path)):
                    if ACEs[index] >= 0: continue # Only keep negative ACE rules
                    
                    single_rule = all_path[index]
                    results = [[] for _ in range(len(self.columns))]
                    
                    # Exact Bounding Box Calculation
                    for i, (col_name, col_values) in enumerate(zip(self.columns, self.every_column)):
                        tmp_left, tmp_right = [], []
                        for feat, thresh, dr in single_rule:
                            if feat == col_name:
                                if dr == 'L': tmp_left.append(thresh)
                                else: tmp_right.append(thresh)
                                
                        if tmp_left and tmp_right:
                            results[i] = [x for x in col_values if max(tmp_right) <= x <= min(tmp_left)]
                        elif tmp_left:
                            results[i] = [x for x in col_values if x <= min(tmp_left)]
                        elif tmp_right:
                            results[i] = [x for x in col_values if max(tmp_right) <= x]
                        else:
                            results[i] = list(col_values)

                    for i in range(len(results)):
                        if not results[i]: results[i] = ['_'] # Wildcard marker
                        elif len(results[i]) > 10: results[i] = random.sample(results[i], 10) # Authors' explicit cap

                    # Mathematically identical to authors' random.choice(list(product(*results)))
                    def x_generator_rule():
                        current_x = []
                        for i in range(len(self.columns)):
                            val = random.choice(results[i])
                            if val == '_': val = random.choice(self.every_column[i])
                            current_x.append(val)
                        return current_x
                        
                    candidate = self._random_search_base(model, eta, x_generator_rule)
                    if candidate:
                        candidates_pool.append((candidate, self._get_ei([model.predict([candidate]) for model in model.estimators_], eta)[0]))

                if candidates_pool:
                    # Pick candidate with highest EI from all rules
                    best_candidate_vec = max(candidates_pool, key=lambda x: x[1])[0]

            # Fallback if generators fail
            if best_candidate_vec is None:
                best_candidate_vec = [random.choice(col_vals) for col_vals in self.every_column]

            # Map back and evaluate physically
            hp_dict = self._vec_to_config(best_candidate_vec)
            d2h = self._evaluate_config(hp_dict)

            history_vec.append(best_candidate_vec)
            history_y.append(d2h)
            exist_configs.append(tuple(best_candidate_vec))

        self.end_time = time.time()
        return self.best_config, self.best_value