# experiment_runner_cluster.py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import argparse
import os
import glob
import time
import signal
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from models.configurations.model_config_static import ModelConfigurationStatic
from models.configurations.model_config_dtlz import ModelConfigurationDTLZ
from models.model_wrapper_static import ModelWrapperStatic
from models.model_wrapper_dtlz import ModelWrapperDTLZ
from optimizers import *
from utils.LoggingUtil import LoggingUtil
from utils.data_loader_templated import load_data
from utils.EncodingUtils import EncodingUtils

def write_to_file(filepath, content):
    """Write content to file, creating directories if needed"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(content)

def write_json_to_file(filepath, data):
    """Write JSON data to file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def init_optimizer(optimizer_name, optimizer_config, model_wrapper, model_config, seed):
    """Initialize optimizer by name"""
    optimizer_classes = {
        'SMAC': SMACOptimizer,
        'RandomSearch': RandomSearchOptimizer,
        'NSGA2': NSGA2Optimizer,
        'SPEA2': SPEA2Optimizer,
        'MOEAD' : MOEADOptimizer,
        'SMSEMOA' : SMSEMOAOptimizer,
        'EZR': EZROptimizer, 
        'HILL' : HillClimbingOptimizer,
        'SA' : SimulatedAnnealingOptimizer,
        'TS' : TabuSearchOptimizer,
        'ILS' : IteratedLocalSearchOptimizer, 
        'TURBO' : TuRBOOptimizer, 
        'GPEI' : GPOptimizer,
        'HEBO' : HEBOOptimizer, 
        'DE' : DEOptimizer,
        'GA' : GAOptimizer,
        'ONEPLUSONE' : OnePlusOneESOptimizer,
        'EDA' : EDAOptimizer,
        'PSO' : PSOOptimizer,
        'PTUNE' : PromiseTuneOptimizer,
        'SWAY' : SWAYOptimizer,
        'DODGE' : DODGEOptimizer
    }
    
    if optimizer_name not in optimizer_classes:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer_classes[optimizer_name](
        optimizer_config, model_wrapper, model_config, None, seed
    )

def checkFileExists(path):
    """Check if file exists"""
    return os.path.exists(path) and os.path.isfile(path)

def run_experiment(datasets, optimizers, repeats, checkpoints, tmp_output_dir, logging_dir,
                   problem=None, n_vars=None, n_objs=None):
    """Run experiments on datasets"""
    # DTLZ mode: no CSVs; run exactly once
    if problem is not None:
        dataset_files = [None]
    else:
        dataset_files = []
        if isinstance(datasets, str) and os.path.isdir(datasets):
            for root, _, _ in os.walk(datasets):
                dataset_files.extend(glob.glob(os.path.join(root, "*.csv")))
        else:
            dataset_files = [datasets] if isinstance(datasets, str) else datasets
        
        if not dataset_files:
            raise ValueError(f"No datasets found for: {datasets}")
    
    for dataset_file in dataset_files:
        optimize_single_dataset(
            optimizers, repeats, checkpoints,
            tmp_output_dir, logging_dir, dataset_file,
            problem=problem, n_vars=n_vars, n_objs=n_objs
        )

def run_repeat_wrapper(args):
    """Wrapper for running a single repeat"""
    (logging_dir, data_name, model_wrapper, model_config, optimizer, 
     checkpoint, optimizer_name, i, tmp_output_dir) = args
    
    result = run_single_repeat(
        logging_dir, data_name, model_wrapper, model_config,
        optimizer, checkpoint, optimizer_name, i
    )
    
    # Save metadata
    save_run_metadata(result, tmp_output_dir, optimizer_name, data_name, checkpoint, i)
    
    return str(result['best_config']), str(result['best_value']), str(result['elapsed'])

def optimize_single_dataset(optimizers, repeats, checkpoints, tmp_output_dir, 
                           logging_dir, dataset_file, problem=None, n_vars=None, n_objs=None):
    """Optimize a single dataset with all optimizers and checkpoints"""
    
    def terminate_all(signum, frame):
        print("Terminating all processes...")
        sys.exit(0)
    
    if problem is not None:   # DTLZ mode
        data_name = f"{problem}_v{n_vars}_m{n_objs}"
        model_config = ModelConfigurationDTLZ(problem=problem, n_vars=n_vars, n_objs=n_objs)
        model_wrapper = ModelWrapperDTLZ(model_config)
    else:
        data_name = get_file_name(dataset_file)
        X, Y = load_data(dataset_file)
        
        # Infer column types from RAW X
        column_types = {
            col: EncodingUtils.infer_column_type(X[col].tolist())[0]
            for col in X.columns
        }
        
        # Encode X ONCE
        X_encoded = EncodingUtils.encode_dataframe(X, column_types)
        
        # Build hyperparameter configs from ENCODED X
        hyperparameter_configs = {
            col: X_encoded[col].tolist()
            for col in X_encoded.columns
        }
        
        model_config = ModelConfigurationStatic(
            hyperparameter_configs, dataset_file, 1, column_types=column_types
        )
        model_wrapper = ModelWrapperStatic(X_encoded, Y, model_config)
    
    for optimizer in optimizers:
        if optimizer.get('disable'):
            continue
        
        for checkpoint in checkpoints:
            optimizer_name = optimizer['name']
            results_path = (os.path.join(tmp_output_dir, optimizer_name,
                                       f"{data_name}_{checkpoint}.csv")
                          if tmp_output_dir else None)
            
            if results_path and checkFileExists(results_path):
                print(f"Skipping {optimizer_name} on {data_name} with budget {checkpoint} (already exists)")
                continue
            
            optimizer['n_trials'] = checkpoint
            print(f"Running {optimizer_name} on {data_name} with budget {checkpoint}")
            
            results = {
                "configs": [], 
                "best_values": [], 
                "runtimes": [],
                "best_d2h_norm": [],
                "n_evaluations": []
            }
            
            args_list = [
                (logging_dir, data_name, model_wrapper, model_config, 
                 optimizer, checkpoint, optimizer_name, i, tmp_output_dir)
                for i in range(repeats)
            ]
            
            # Run sequentially for now (can parallelize later)
            for args in args_list:
                best_cfg, best_val, elapsed = run_repeat_wrapper(args)
                results["configs"].append(best_cfg)
                results["best_values"].append(best_val)
                results["runtimes"].append(elapsed)
            
            if results_path:
                # Save CSV results
                save_results_csv(results, results_path)

def run_single_repeat(logging_dir, data_name, model_wrapper, model_config,
                     optimizer, checkpoint, optimizer_name, i):
    """Run a single repeat of optimization"""
    seed = i + 1
    log_filename = os.path.join(logging_dir, optimizer_name, f"{data_name}_{checkpoint}_{seed}.csv")
    
    model_config.set_seed(seed)
    
    # Initialize optimizer
    optimizer_obj = init_optimizer(optimizer_name, optimizer, model_wrapper, model_config, seed)
    
    # Create logging utility with enhanced tracking
    logging_util = LoggingUtil(log_filename)
    optimizer_obj.set_logging_util(logging_util)
    
    # Start timing
    start = time.time()
    optimizer_obj.start_time = start
    
    # Optimize
    try:
        best_config, best_value = optimizer_obj.optimize()
    except Exception as e:
        print(f"Error in {optimizer_name} on {data_name} run {i}: {e}")
        best_config = {}
        best_value = float('inf')
    
    # End timing
    elapsed = time.time() - start
    optimizer_obj.end_time = time.time()
    
    # Gather all tracked data
    result = {
        'best_config': best_config,
        'best_value': best_value,
        'elapsed': elapsed,
        'seed': seed,
        'optimizer': optimizer_name,
        'dataset': data_name,
        'checkpoint': checkpoint,
        'evaluation_history': optimizer_obj.evaluation_history,
        'frontier_history': optimizer_obj.frontier_history,
        'cluster_centroids': optimizer_obj.cluster_centroids,
        'n_evaluations': len(optimizer_obj.evaluation_history),
        'start_time': start,
        'end_time': optimizer_obj.end_time
    }
    
    # Calculate additional metrics if needed
    # if hasattr(optimizer_obj, 'get_frontier_explosion_stats'):
    #     frontier_stats = optimizer_obj.get_frontier_explosion_stats()
    #     if frontier_stats:
    #         result['frontier_explosion'] = frontier_stats
    
    return result

def save_run_metadata(result, output_dir, optimizer_name, data_name, checkpoint, run_id):
    """Save comprehensive metadata for a single run"""
    
    metadata_dir = os.path.join(output_dir, optimizer_name, 'metadata')
    os.makedirs(metadata_dir, exist_ok=True)
    
    metadata_file = os.path.join(metadata_dir, f"{data_name}_{checkpoint}_{run_id}.json")
    
    # Prepare metadata (convert numpy types to Python types)
    metadata = {
        'optimizer': result['optimizer'],
        'dataset': result['dataset'],
        'checkpoint': result['checkpoint'],
        'seed': result['seed'],
        'runtime_seconds': result['elapsed'],
        'best_value': float(result['best_value']) if result['best_value'] != float('inf') else None,
        'n_evaluations': result['n_evaluations'],
        'start_timestamp': result['start_time'],
        'end_timestamp': result['end_time']
    }
    
    # Add evaluation history (convert to serializable format)
    if result['evaluation_history']:
        metadata['evaluation_history'] = []
        for eval_entry in result['evaluation_history']:
            serializable_entry = {
                'config': {k: convert_to_serializable(v) for k, v in eval_entry['config'].items()},
                'objectives': [convert_to_serializable(obj) for obj in eval_entry['objectives']],
                'iteration': int(eval_entry['iteration']),
                'timestamp': float(eval_entry['timestamp'])
            }
            metadata['evaluation_history'].append(serializable_entry)
    
    # Add frontier history for dominance methods
    if result['frontier_history']:
        metadata['frontier_history'] = []
        for frontier_entry in result['frontier_history']:
            serializable_frontier = {
                'iteration': int(frontier_entry['iteration']),
                'frontier_size': int(frontier_entry['frontier_size']),
                # Use .get() for safety or use the new structure keys:
                'best_config': frontier_entry['best_in_frontier']['config'],
                'best_objectives': [float(obj) for obj in frontier_entry['best_in_frontier']['objectives']]
            }
            metadata['frontier_history'].append(serializable_frontier)
    
    # Add frontier explosion stats
    if 'frontier_explosion' in result:
        metadata['frontier_explosion'] = {
            k: convert_to_serializable(v) for k, v in result['frontier_explosion'].items()
        }
    
    # Add cluster centroids for aggregation methods
    if result['cluster_centroids']:
        metadata['cluster_centroids'] = []
        for centroid in result['cluster_centroids']:
            if isinstance(centroid, dict):
                serializable_centroid = {k: convert_to_serializable(v) for k, v in centroid.items()}
            else:
                serializable_centroid = convert_to_serializable(centroid)
            metadata['cluster_centroids'].append(serializable_centroid)
    
    # Write to JSON
    write_json_to_file(metadata_file, metadata)

def convert_to_serializable(obj):
    """Convert numpy/pandas types to Python native types for JSON serialization"""
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
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    else:
        return obj

def save_results_csv(results, filepath):
    """Save results to CSV file"""
    
    # Create DataFrame
    df_data = {
        'run_id': list(range(len(results['configs']))),
        'best_config': results['configs'],
        'best_value': results['best_values'],
        'runtime': results['runtimes']
    }
    
    # Add additional metrics if available
    if 'best_d2h_norm' in results and results['best_d2h_norm']:
        df_data['best_d2h_norm'] = results['best_d2h_norm']
    
    if 'n_evaluations' in results and results['n_evaluations']:
        df_data['n_evaluations'] = results['n_evaluations']
    
    import pandas as pd
    df = pd.DataFrame(df_data)
    
    # Save
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def get_file_name(path):
    """Extract filename without extension"""
    return os.path.splitext(os.path.basename(path))[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Runner with Enhanced Tracking")
    parser.add_argument('--datasets', type=str, help="Path to datasets directory or single CSV")
    parser.add_argument('--output_directory', type=str, help="Output directory for results")
    parser.add_argument('--name', type=str, help="Optimizer name")
    parser.add_argument('--repeats', type=int, default=30, help="Number of independent runs")
    parser.add_argument('--budget', type=int, nargs='+', default=[6, 12, 18, 24, 50, 100, 200],
                       help="Evaluation budgets to test")
    parser.add_argument('--runs_output_folder', type=str, help="Folder for run results")
    parser.add_argument('--logging_folder', type=str, help="Folder for detailed logs")
    
    # DTLZ mode
    parser.add_argument('--problem', type=str, default=None, help="DTLZ problem name")
    parser.add_argument('--n_vars', type=int, default=12, help="Number of variables for DTLZ")
    parser.add_argument('--n_objs', type=int, default=3, help="Number of objectives for DTLZ")
    
    args = parser.parse_args()
    
    optimizer_config = {
        "name": args.name,
        "output_directory": args.output_directory,
    }
    
    run_experiment(
        datasets=args.datasets,
        optimizers=[optimizer_config],
        repeats=args.repeats,
        checkpoints=args.budget,
        tmp_output_dir=args.runs_output_folder,
        logging_dir=args.logging_folder,
        problem=args.problem,
        n_vars=args.n_vars,
        n_objs=args.n_objs
    )
