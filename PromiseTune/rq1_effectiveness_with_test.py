"""
Performance Optimization Method Comparison and Analysis

This script is used to compare the performance of different optimization algorithms
on multiple datasets. The analysis uses Scott-Knott test to rank the methods.

"""

import os
import csv
import warnings
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from scikit_posthocs import posthoc_nemenyi

# 初始化设置
pandas2ri.activate()
warnings.filterwarnings("ignore")

def scott_test(learning_models, compared_results):
    """
    Execute Scott-Knott test
    
    Parameters:
    learning_models - List of algorithm names
    compared_results - List of results corresponding to each algorithm
    
    Returns:
    Scott-Knott test result
    """
    # Create DataFrame
    data = pd.DataFrame({
        learning_models[i]: compared_results[i] 
        for i in range(len(learning_models))
    })
    
    # Execute Scott-Knott test
    sk = importr('ScottKnottESD')
    r_sk = sk.sk_esd(data, version="np")
    return r_sk

def load_results(learning_model, name, seed, budget=50):  # Modify here to complete statistics for different budgets
    """
    Load results for specified algorithm, dataset and seed
    
    Parameters:
    learning_model - Algorithm model path
    name - Dataset name
    seed - Random seed
    budget - Optimization budget
    
    Returns:
    y - Optimized result value
    y1 - Result at budget 200
    y2 - Result at budget 50
    """
    file_path = f'./{learning_model}/{name}{seed}.csv'
    
    # Read data file
    df = pd.read_csv(file_path)
    y_values = [float(v) for v in df.iloc[1]]
    
    try:
        x_values = [[float(i) for i in x.strip('[]').split()] for x in df.iloc[0]]
    except Exception:
        x_values = [tuple(str(x).strip('[]').split()) for x in df.iloc[0]]
    
    # Extract optimization process
    unique_configs = []
    unique_indices = []
    best_so_far = []
    
    for i, config in enumerate(x_values):
        if config not in unique_configs:
            unique_configs.append(config)
            unique_indices.append(i + 1)  # Count from 1
            best_so_far.append(min(y_values[:i+1]))
    
    # Extract best results at different budgets
    try:
        if len(best_so_far) >= 1:
            y = best_so_far[min(budget, len(best_so_far)-1)]
            y1 = best_so_far[min(200, len(best_so_far)-1)]
            y2 = best_so_far[min(50, len(best_so_far)-1)]
        else:
            y = y1 = y2 = best_so_far[-1]
    except Exception:
        y = y1 = y2 = best_so_far[-1]
    
    return y, y1, y2

def main():
    """Main function to execute all data analysis and results generation"""
    # Configure algorithm models
    learning_models = [
        "./results/PromiseTune",   # Promisetune
        "./results/RANDOM",        # random
        "./results/Unicorn",        # unicorn
        "./results/GA",             # GA
        "./results/MBO",       # MBO
        "./results/LlamaTune",     # llamatune
        "./results/FLASH",         # flash
        "./results/CFSCA",         # CFSCA
        "./results/BOCA",          # BOCA
        "./results/OtterTune",       # ottertune
        "./results/SMAC",            # SMAC
        "./results/HEBO",           # hebo
    ]
    
    learning_modelss = [
        'PromiseTune', 'RANDOM', 'Unicorn', 'GA', 'MBO', 
        'LlamaTune', 'FLASH', 'CFSCA', 'BOCA', 'OtterTune', 'SMAC', 'HEBO'
    ]
    
    # Random seed list
    seeds = list(range(1, 31))
    
    # Dataset names
    display_names = [
        '7z', 'DConvert', 'ExaStencils', "BDB-C", 'DeepArch',
        'PostgreSQL', 'JavaGC', 'Storm', 'x264', 'Redis', 'HSQLDB', 'LLVM'
    ]
    
    file_names = [
        '7z', 'dconvert', 'exastencils', "BDBC_AllNumeric", 'deeparch',
        'PostgreSQL', 'javagc', 'stormm', 'x264', 'redis', 'HSQLDB', 'LLVM_AllNumeric'
    ]
    
    # Initialize result lists
    all_ranks = []
    all_normalized_performances = []
    all_variances = []
    
    # Read minmax.csv file
    minmax_values = {}
    try:
        minmax_df = pd.read_csv('./minmax.csv', header=None)
        for i in range(len(minmax_df)):
            name = minmax_df.iloc[i, 0]
            min_value = float(minmax_df.iloc[i, 1])
            max_value = float(minmax_df.iloc[i, 2])
            minmax_values[name] = (min_value, max_value)
        print(f"Successfully loaded min-max values for {len(minmax_values)} datasets from minmax.csv")
    except Exception as e:
        print(f"Error reading minmax.csv file: {e}")
        print("Will create a new minmax.csv file")
        with open('./minmax.csv', 'w', newline='') as f:
            csv_writer = csv.writer(f)
            # Create empty minmax file
    
    # Initialize markdown result file
    with open('./scott_knott_results.md', 'w', encoding='utf-8') as f2:
        f2.write("# Scott-Knott Rank Results\n\n")
        f2.write("| Dataset | " + " | ".join(learning_modelss) + " |\n")
        f2.write("|---------|" + "|".join(["--------"] * len(learning_modelss)) + "|\n")
    
    # Create normalized performance and variance files
    with open('./normalized_performance.md', 'w', encoding='utf-8') as f_perf:
        f_perf.write("# Normalized Performance Results\n\n")
        f_perf.write("| Dataset | " + " | ".join(learning_modelss) + " |\n")
        f_perf.write("|---------|" + "|".join(["--------"] * len(learning_modelss)) + "|\n")
    
    with open('./performance_variance.md', 'w', encoding='utf-8') as f_var:
        f_var.write("# Performance Variance Results\n\n")
        f_var.write("| Dataset | " + " | ".join(learning_modelss) + " |\n")
        f_var.write("|---------|" + "|".join(["--------"] * len(learning_modelss)) + "|\n")
    
    # Analyze each dataset one by one
    for index1, (name, display_name) in enumerate(zip(file_names, display_names)):
        print(f"Processing dataset: {display_name}")
        
        # Collect results for each algorithm
        all_results = []
        compared_results = []
        compared_results1 = []
        compared_results2 = []
        
        for learning_model in learning_models:
            X, X1, X2 = [], [], []
            
            for seed in seeds:
                # Try to read result file
                kk = 0
                file_path = f'./{learning_model}/{name}{seed}.csv'
                
                # If file doesn't exist, try other seeds
                while not os.path.exists(file_path):
                    # Special handling for unicorn
                    if learning_model == './results/unicorn1':
                        learning_model = './results/random'
                        break
                    
                    seed += 1
                    if seed > 30:
                        seed = seed - 30
                    
                    file_path = f'./{learning_model}/{name}{seed}.csv'
                    kk += 1
                    if kk == 100:  # Give up after 100 attempts
                        break
                
                # Handle case where file doesn't exist
                if kk == 100:
                    y = np.nan
                    X.append(y)
                    X1.append(y)
                    X2.append(y)
                else:
                    # Load result file
                    try:
                        y, y1, y2 = load_results(learning_model, name, seed)
                        X.append(y)
                        X1.append(y1)
                        X2.append(y2)
                    except Exception as e:
                        print(f"Error loading file: {file_path}, {e}")
                        X.append(np.nan)
                        X1.append(np.nan)
                        X2.append(np.nan)
            
            # Collect results for each algorithm
            all_results.append(np.median(X))
            compared_results.append(X)
            compared_results1.append(X1)
            compared_results2.append(X2)
        
        # Read result range from minmax.csv, if it doesn't exist then calculate it
        if name in minmax_values:
            min_value, max_value = minmax_values[name]
            print(f"Dataset {display_name} read from minmax.csv: min={min_value}, max={max_value}")
        else:
            # Find result range
            min_value = min(
                min(min(sublist) for sublist in compared_results if len(sublist) > 0),
                min(min(sublist) for sublist in compared_results1 if len(sublist) > 0),
                min(min(sublist) for sublist in compared_results2 if len(sublist) > 0)
            )
            
            max_value = max(
                max(max(sublist) for sublist in compared_results if len(sublist) > 0),
                max(max(sublist) for sublist in compared_results1 if len(sublist) > 0),
                max(max(sublist) for sublist in compared_results2 if len(sublist) > 0)
            )
            
            # Save min-max range
            with open('./minmax.csv', 'a', newline="") as ff1:
                csv_writer = csv.writer(ff1)
                csv_writer.writerow([name, min_value, max_value])
            
            print(f"Dataset {display_name} calculated: min={min_value}, max={max_value}")
        
        # Handle missing values
        if name in ['stormm', 'redis']:
            # For problems that need to be maximized, missing values set to max value (worst result)
            compared_results = [[max_value if np.isnan(x) else x for x in sublist] for sublist in compared_results]
        else:
            # For problems that need to be minimized, missing values set to max value (worst result)
            compared_results = [[max_value if np.isnan(x) else x for x in sublist] for sublist in compared_results]
        
        # Normalize results
        normalized_compared_results = []
        normalized_means = []
        normalized_variances = []
        
        for res in compared_results:
            if max_value != min_value:
                normalized_res = [(x - min_value) / (max_value - min_value) for x in res]
            else:
                normalized_res = [0 for x in res]
            
            # Ensure values are within [0,1] range
            normalized_res = [round(max(0, min(1, x)), 3) for x in normalized_res]
            normalized_compared_results.append(normalized_res)
            
            # Calculate mean and variance after normalization
            normalized_means.append(round(np.mean(normalized_res), 3))
            normalized_variances.append(round(np.var(normalized_res), 3))
        
        # Record normalized performance and variance for each dataset
        all_normalized_performances.append(normalized_means)
        all_variances.append(normalized_variances)
        
        # Write normalized performance and variance to file
        with open('./normalized_performance.md', 'a', encoding='utf-8') as f_perf:
            perf_row = f"| **{display_name}** | " + " | ".join([f"{perf:.3f}" for perf in normalized_means]) + " |\n"
            f_perf.write(perf_row)
            
        with open('./performance_variance.md', 'a', encoding='utf-8') as f_var:
            var_row = f"| **{display_name}** | " + " | ".join([f"{var:.3f}" for var in normalized_variances]) + " |\n"
            f_var.write(var_row)
        
        # Execute Scott-Knott test
        result = scott_test(learning_models, normalized_compared_results)
        column_order = list(result[3])
        rank = result[1].astype("int")
        
        # Sort and output results
        print_ = sorted(zip(column_order, rank), key=lambda x: x[0])
        scott_res = [i[1] for i in print_]
        
        # Reverse the ranks (higher rank represents better performance)
        max_rank = max(scott_res)
        scott_res_reversed = [max_rank + 1 - rank for rank in scott_res]
        all_ranks.append(scott_res_reversed)
        
        # Append results to markdown file
        with open('./scott_knott_results.md', 'a', encoding='utf-8') as f2:
            rank_row = f"| **{display_name}** | " + " | ".join([str(rank) for rank in scott_res_reversed]) + " |\n"
            f2.write(rank_row)
    
    # Calculate average ranks
    avg_ranks = []
    for i in range(len(learning_modelss)):
        avg_rank = sum(ranks[i] for ranks in all_ranks) / len(all_ranks)
        avg_ranks.append(round(avg_rank, 2))
    
    # Calculate average normalized performance and variance
    avg_performances = []
    avg_variances = []
    for i in range(len(learning_modelss)):
        avg_perf = sum(perfs[i] for perfs in all_normalized_performances) / len(all_normalized_performances)
        avg_var = sum(vars[i] for vars in all_variances) / len(all_variances)
        avg_performances.append(round(avg_perf, 3))
        avg_variances.append(round(avg_var, 3))
    
    # Add average rank, performance and variance to corresponding files
    with open('./scott_knott_results.md', 'a', encoding='utf-8') as f2:
        f2.write("\n")
        avg_rank_row = f"| **Average Rank** | " + " | ".join([str(rank) for rank in avg_ranks]) + " |\n"
        f2.write(avg_rank_row)
    
    with open('./normalized_performance.md', 'a', encoding='utf-8') as f_perf:
        f_perf.write("\n")
        avg_perf_row = f"| **Average Performance** | " + " | ".join([f"{perf:.3f}" for perf in avg_performances]) + " |\n"
        f_perf.write(avg_perf_row)
        
    with open('./performance_variance.md', 'a', encoding='utf-8') as f_var:
        f_var.write("\n")
        avg_var_row = f"| **Average Variance** | " + " | ".join([f"{var:.3f}" for var in avg_variances]) + " |\n"
        f_var.write(avg_var_row)
    
    print("Average ranks:", avg_ranks)
    print("Average normalized performance:", avg_performances)
    print("Average variance:", avg_variances)

if __name__ == "__main__":
    main()