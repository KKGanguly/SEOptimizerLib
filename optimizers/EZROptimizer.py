# optimizers/EZROptimizer.py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import active_learning.src.bl as bl
from optimizers.base_optimizer import BaseOptimizer
import numpy as np
import time

class EZROptimizer(BaseOptimizer):
    """
    EZR optimizer = BL Active Learning with acquisition 'near'.
    Enhanced with cluster centroid tracking.
    """
    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)
        self.best_config = None
        self.best_value = None
        
        # Load BL dataset
        csv_path = model_config.get_dataset_file()
        self.bl_data = bl.Data(bl.csv(csv_path))
        if len(self.bl_data.cols.y) == 0:
            raise ValueError("BL could not detect Y columns in dataset CSV.")
            
        self.num_objectives = len(self.bl_data.cols.y)
        
    def optimize(self):
        n_trials = self.config["n_trials"]
        
        self.start_time = time.time()
        
        # Set BL parameters
        bl.the.Stop = n_trials
        bl.the.acq = "near"  # EZR uses near acquisition
        
        # Run BL Active Learning
        result = bl.actLearn(self.bl_data, shuffle=True)
        
        # Extract evaluations
        x_len = len(self.bl_data.cols.x)
        for i, row in enumerate(result.evaluated):
            config_dict = dict(zip(self.bl_data.cols.names[:x_len], row[:x_len]))
            objectives = row[x_len:x_len + self.num_objectives]
            normalized_objectives = [bl.norm(row[c.at], c) for c in self.bl_data.cols.y]

            self.track_evaluation(config_dict, normalized_objectives, i + 1)
                        
        
        # Extract leaf cluster centroids
        self.cluster_centroids = []
        if hasattr(result, 'best') and hasattr(result.best, 'rows'):
            # Get cluster structure from BL
            for row in result.best.rows:
                config_dict = dict(zip(self.bl_data.cols.names[:x_len], row[:x_len]))
                self.cluster_centroids.append(config_dict)
        
        # Best row from BL
        best_row = bl.first(result.best.rows)
        best_hp = dict(zip(self.bl_data.cols.names[:x_len], best_row[:x_len]))
        
        # Calculate d2h/N
        objectives = best_row[x_len:x_len + self.num_objectives]
        best_value = bl.ydist(best_row, self.bl_data)
        
        self.best_config = best_hp
        self.best_value = best_value
        self.end_time = time.time()
        
        return self.best_config, self.best_value
