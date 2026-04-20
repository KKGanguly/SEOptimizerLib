# optimizers/base_optimizer.py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from abc import ABC, abstractmethod
import time
import numpy as np

class BaseOptimizer(ABC):
    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        self.config = config
        self.model_wrapper = model_wrapper
        self.model_config = model_config
        self.logging_util = logging_util
        self.seed = seed
        self.best_config = None
        self.best_value = None
        
        # Enhanced tracking
        self.evaluation_history = []  # All evaluations: [(config, objectives, iteration)]
        self.frontier_history = []     # For dominance methods: [(iteration, frontier_size, total_pop)]
        self.cluster_centroids = []    # For aggregation methods: final cluster centers
        self.start_time = None
        self.end_time = None
        
    def set_logging_util(self, logging_util):
        self.logging_util = logging_util
        
    def track_evaluation(self, config, objectives, iteration):
        """Track every evaluation for analysis"""
        self.evaluation_history.append({
            'config': config,
            'objectives': objectives,
            'iteration': iteration,
            'timestamp': time.time() - self.start_time if self.start_time else 0
        })
        
    def track_frontier(self, iteration, frontier_solutions, best_config, best_frontier_trial):
        # Append to history for later "Stagnation" analysis
        history_entry = {
            "iteration": iteration,
            "frontier_size": len(frontier_solutions),
            "best_in_frontier": {
                "config": best_config,
                "objectives": list(best_frontier_trial.values)
            }
        }
        self.frontier_history.append(history_entry)
        
    def get_frontier_explosion_stats(self):
        """Calculate frontier explosion metrics"""
        if not self.frontier_history:
            return None
        final = self.frontier_history[-1]
        return {
            'final_frontier_size': final['frontier_size'],
            'final_population': final['total_population'],
            'final_percentage': final['frontier_percentage']
        }
        
        
    @abstractmethod
    def optimize(self):
        pass