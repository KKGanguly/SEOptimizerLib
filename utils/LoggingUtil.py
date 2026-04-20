# utils/LoggingUtil.py
import csv
import os
import time
import json
from pathlib import Path

class LoggingUtil:
    """Enhanced logging utility for tracking optimization progress and metadata"""
    
    def __init__(self, log_file_path):
        """
        Initialize logging utility
        
        Parameters:
        - log_file_path: Path to CSV log file
        """
        self.log_file_path = log_file_path
        self.log_file = None
        self.csv_writer = None
        self.is_logging = False
        
        # Enhanced tracking
        self.evaluation_log = []  # All evaluations
        self.metadata = {}  # Additional metadata to save
        self.start_time = None
        self.iteration_count = 0
        
        # Create directory if needed
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    def start_logging(self, headers=None):
        """
        Start logging to CSV file
        
        Parameters:
        - headers: Optional list of column headers
        """
        if self.is_logging:
            return
        
        self.log_file = open(self.log_file_path, 'w', newline='')
        
        if headers is None:
            headers = ['iteration', 'config', 'value', 'timestamp', 'elapsed_time']
        
        self.csv_writer = csv.DictWriter(self.log_file, fieldnames=headers)
        self.csv_writer.writeheader()
        
        self.is_logging = True
        self.start_time = time.time()
    
    def log(self, config, value, iteration=None):
        """
        Log a single evaluation (basic interface for backward compatibility)
        
        Parameters:
        - config: Configuration dictionary
        - value: Objective value(s)
        - iteration: Optional iteration number
        """
        if not self.is_logging:
            self.start_logging()
        
        if iteration is None:
            self.iteration_count += 1
            iteration = self.iteration_count
        
        timestamp = time.time()
        elapsed = timestamp - self.start_time if self.start_time else 0
        
        row = {
            'iteration': iteration,
            'config': str(config),
            'value': str(value),
            'timestamp': timestamp,
            'elapsed_time': elapsed
        }
        
        self.csv_writer.writerow(row)
        self.log_file.flush()
    
    def log_evaluation(self, config, objectives, iteration=None, additional_data=None):
        """
        Log a single evaluation with full details
        
        Parameters:
        - config: Configuration dictionary
        - objectives: Tuple/list of objective values
        - iteration: Iteration number
        - additional_data: Optional dict of additional data to log
        """
        if iteration is None:
            self.iteration_count += 1
            iteration = self.iteration_count
        
        timestamp = time.time()
        elapsed = timestamp - self.start_time if self.start_time else 0
        
        # Store in evaluation log
        eval_entry = {
            'iteration': iteration,
            'config': config.copy() if isinstance(config, dict) else config,
            'objectives': list(objectives) if hasattr(objectives, '__iter__') else [objectives],
            'timestamp': timestamp,
            'elapsed_time': elapsed
        }
        
        if additional_data:
            eval_entry.update(additional_data)
        
        self.evaluation_log.append(eval_entry)
        
        # Also log to CSV
        self.log(config, objectives, iteration)
    
    def log_frontier(self, iteration, frontier_solutions, total_population):
        """
        Log Pareto frontier information (for dominance-based methods)
        
        Parameters:
        - iteration: Current iteration/generation
        - frontier_solutions: List of non-dominated solutions
        - total_population: Total population size
        """
        frontier_entry = {
            'iteration': iteration,
            'frontier_size': len(frontier_solutions),
            'total_population': total_population,
            'frontier_percentage': (len(frontier_solutions) / total_population * 100) 
                                   if total_population > 0 else 0,
            'timestamp': time.time()
        }
        
        if 'frontier_history' not in self.metadata:
            self.metadata['frontier_history'] = []
        
        self.metadata['frontier_history'].append(frontier_entry)
    
    def log_cluster_centroids(self, centroids):
        """
        Log cluster centroids (for aggregation-based methods)
        
        Parameters:
        - centroids: List of cluster centroids (configs or objective vectors)
        """
        self.metadata['cluster_centroids'] = centroids
    
    def add_metadata(self, key, value):
        """
        Add arbitrary metadata
        
        Parameters:
        - key: Metadata key
        - value: Metadata value
        """
        self.metadata[key] = value
    
    def stop_logging(self):
        """Stop logging and close file"""
        if self.is_logging and self.log_file:
            self.log_file.close()
            self.is_logging = False
    
    def save_metadata(self, output_path=None):
        """
        Save metadata to JSON file
        
        Parameters:
        - output_path: Optional path for JSON file (defaults to log_file_path with .json extension)
        """
        if output_path is None:
            output_path = str(Path(self.log_file_path).with_suffix('.json'))
        
        # Prepare metadata
        full_metadata = {
            'log_file': self.log_file_path,
            'start_time': self.start_time,
            'total_iterations': self.iteration_count,
            'evaluation_history': self.evaluation_log,
            **self.metadata
        }
        
        # Convert to serializable format
        serializable_metadata = self._make_serializable(full_metadata)
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(serializable_metadata, f, indent=2)
    
    def _make_serializable(self, obj):
        """Convert object to JSON-serializable format"""
        import numpy as np
        import pandas as pd
        
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif obj is None:
            return None
        else:
            try:
                return str(obj)
            except:
                return None
    
    def get_evaluation_history(self):
        """Get full evaluation history"""
        return self.evaluation_log
    
    def get_metadata(self):
        """Get all metadata"""
        return self.metadata
    
    def __enter__(self):
        """Context manager entry"""
        self.start_logging()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_logging()
        return False