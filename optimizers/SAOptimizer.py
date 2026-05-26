# optimizers/SimulatedAnnealingOptimizer.py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from optimizers.base_optimizer import BaseOptimizer
from ConfigSpace.hyperparameters import (
    OrdinalHyperparameter,
    CategoricalHyperparameter,
    Constant,
)
from utils import DistanceUtil
import time
import random
import copy
import math
import numpy as np


class SimulatedAnnealingOptimizer(BaseOptimizer):
    """
    Simulated Annealing optimizer over the continuous RF surrogate.

    Classic SA:
      - Start at a random solution (from dataset).
      - At each step, generate ONE continuous neighbour (via mutation).
      - If it strictly improves d2h, always accept.
      - If it worsens d2h, accept with probability exp(-delta / T).
      - Cool temperature by factor `cooling_rate` each step.
      - Stop when budget exhausted or T < min_temp.

    No KD-tree snapping. Operates on the continuous space but rigorously 
    enforces hyperparameter bounds via ConfigSpace.
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        random.seed(seed)
        np.random.seed(seed)

        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)
        self.n_rows = len(self.X_df)

        self.config_space, _, _ = self.model_config.get_configspace()
        self.cache = {}

        assert hasattr(self.model_wrapper, 'rf_model'), \
            "ModelWrapper must have RF trained before optimizer init"

        self.num_objectives = len(
            self.model_wrapper.get_score(
                {c: self.X_df.iloc[0][c] for c in self.columns}
            )
        )

        self.iteration = 0
        self.best_config = None
        self.best_value = float("inf")

        # SA parameters
        self.initial_temp  = float(self.config.get("initial_temp", 1.0))
        self.cooling_rate  = float(self.config.get("cooling_rate", 0.95))
        self.min_temp      = float(self.config.get("min_temp", 1e-5))
        self.mutation_rate = float(self.config.get("mutation_rate", 0.2))

        # Extract bounds and choices from ConfigSpace for safe continuous mutation
        self.bounds = {}
        self.is_int = {}
        self.cat_choices = {}
        
        for hp in self.config_space.get_hyperparameters():
            name = hp.name
            hp_type = type(hp).__name__
            
            if hp_type in ["UniformFloatHyperparameter", "UniformIntegerHyperparameter"]:
                self.bounds[name] = (hp.lower, hp.upper)
                self.is_int[name] = (hp_type == "UniformIntegerHyperparameter")
            elif hp_type in ["CategoricalHyperparameter", "OrdinalHyperparameter"]:
                self.cat_choices[name] = list(hp.choices) if hasattr(hp, 'choices') else list(hp.sequence)


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _safe_clean(self, v):
        """Clean items and round floats slightly to prevent cache bloat."""
        val = v.item() if hasattr(v, "item") else v
        return round(val, 6) if isinstance(val, float) else val

    def _row_tuple(self, hp_dict):
        return tuple(self._safe_clean(hp_dict[c]) for c in self.columns)

    def _idx_to_config(self, idx):
        """Initialize from a real dataset row."""
        row = self.X_df.iloc[idx]
        return {c: self._safe_clean(row[c]) for c in self.columns}

    def _random_config(self):
        """Pick a uniformly random row from the dataset as starting point."""
        return self._idx_to_config(random.randrange(self.n_rows))

    # ------------------------------------------------------------------
    # Evaluation with caching and tracking
    # ------------------------------------------------------------------

    def _evaluate(self, hp_dict):
        key = self._row_tuple(hp_dict)
        if key in self.cache:
            scores, d2h_val = self.cache[key]
        else:
            try:
                scores = tuple(self.model_wrapper.get_score(hp_dict))
                ideal = [0] * self.num_objectives
                d2h_val = DistanceUtil.d2h(ideal, list(scores))
            except Exception as e:
                # Mathematical infinity ensures this configuration is never selected
                scores = tuple(float('inf') for _ in range(self.num_objectives))
                d2h_val = float('inf')
            self.cache[key] = (scores, d2h_val)

        self.iteration += 1
        self.track_evaluation(hp_dict, list(scores), self.iteration)
        return hp_dict, scores, d2h_val

    # ------------------------------------------------------------------
    # Continuous Mutation
    # ------------------------------------------------------------------

    def _mutate(self, config, force_one=False):
        """
        Create a neighbor by perturbing hyperparameters. 
        Matches the mutation logic from Hill Climbing and ILS exactly.
        """
        mutant = copy.deepcopy(config)
        mutated_any = False
        
        hps = [hp for hp in self.config_space.get_hyperparameters() if not isinstance(hp, Constant)]
        if not hps:
            return mutant
            
        forced_hp = random.choice(hps).name if force_one else None
        
        for hp in hps:
            name = hp.name
            current_val = mutant[name]

            if name != forced_hp and random.random() > self.mutation_rate:
                continue

            if name in self.bounds:
                lower, upper = self.bounds[name]
                span = upper - lower
                if span <= 1.0:
                    # Uniformly resample across the entire bound to guarantee a chance to flip
                    new_val = random.uniform(hp.lower, hp.upper)
                else:
                    std = span * 0.1 # 10% Gaussian noise
                    new_val = current_val + random.gauss(0, std)
                
                # Bounds clipping
                new_val = max(lower, min(upper, new_val))
                
                # Integer enforcing
                if self.is_int.get(name, False):
                    new_val = int(round(new_val))
                    
                mutant[name] = new_val
                mutated_any = True

            elif name in self.cat_choices:
                choices = list(self.cat_choices[name])
                if len(choices) > 1:
                    if current_val in choices:
                        choices.remove(current_val)
                    mutant[name] = random.choice(choices)
                    mutated_any = True
                    
        # Recursively force a mutation if pure probability missed everything
        if not mutated_any and not force_one:
            return self._mutate(config, force_one=True)

        return mutant

    # ------------------------------------------------------------------
    # Metropolis acceptance criterion
    # ------------------------------------------------------------------

    def _accept(self, current_d2h, candidate_d2h, temperature):
        delta = candidate_d2h - current_d2h
        if delta < 0:
            # Always accept improvements
            return True
        # Accept worse solution with probability exp(-delta / T)
        return random.random() < math.exp(-delta / temperature)

    # ------------------------------------------------------------------
    # Main optimise loop
    # ------------------------------------------------------------------

    def optimize(self):
        n_trials = self.config["n_trials"]
        self.start_time = time.time()

        temperature = self.initial_temp

        # ── Random initial solution ─────────────────────────────────────
        current_config = self._random_config()
        current_config, _, current_d2h = self._evaluate(current_config)

        self.best_value = current_d2h
        self.best_config = copy.deepcopy(current_config)

        # ── SA loop ─────────────────────────────────────────────────────
        while self.iteration < n_trials and temperature > self.min_temp:

            # 1. Generate ONE continuous neighbour via mutation
            candidate = self._mutate(current_config)

            # 2. Evaluate it
            # Cache handled implicitly in _evaluate
            candidate, scores, candidate_d2h = self._evaluate(candidate)

            # 3. Metropolis acceptance
            if self._accept(current_d2h, candidate_d2h, temperature):
                current_config = candidate
                current_d2h = candidate_d2h

            # 4. Update global best (best seen, not current position)
            if current_d2h < self.best_value:
                self.best_value = current_d2h
                self.best_config = copy.deepcopy(current_config)

            # 5. Cool down
            temperature *= self.cooling_rate

        self.end_time = time.time()
        return self.best_config, self.best_value