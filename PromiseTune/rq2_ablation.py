"""
Ablation Study: Comparing Algorithm Performance Under Different Configurations
"""

import os
import csv
import warnings
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


# Initialize settings
warnings.filterwarnings("ignore")

def normalize_results(results, min_value, max_value):
    """Normalize results to [0,1] range"""
    normalized_results = []
    for res in results:
        normalized_res = [round((x - min_value) / (max_value - min_value), 3) 
                         if max_value != min_value else 0 for x in res]
        normalized_results.append(normalized_res)
    return normalized_results

def get_significance_output(group1, group2, p_value, budget_last=False):
    """Generate formatted significance output string based on comparison results and p-value
    
    Args:
        group1: w/o causal results
        group2: w/ causal results (expected to be better)
        p_value: p-value from Mann-Whitney U test
    """
    mean1, std1 = np.mean(group1), np.std(group1)
    mean2, std2 = np.mean(group2), np.std(group2)
    
    # Format results
    result1 = f"{mean1:.3f}({std1:.3f})"
    result2 = f"{mean2:.3f}({std2:.3f})"
    
    # Markdown format - use bold for significant improvements
    # Lower values are better, so we expect mean2 < mean1
    if mean2 < mean1:  # w/ causal is better than w/o causal
        if p_value <= 0.05:  # Statistically significant improvement
            return f'{result1} | **{result2}**'
        else:
            return f'{result1} | {result2}'
    else:  # w/ causal is worse or equal
        if p_value > 0.95:  # Statistically significantly worse
            return f'**{result1}** | **{result2}**'
        else:
            return f'{result1} | {result2}'

def process_result_file(file_path, min_value, max_value):
    """Process result file and extract performance metrics"""
    df = pd.read_csv(file_path)
    y_values = [float(v) for v in df.iloc[1]]
    
    try:
        x_configs = [[float(i) for i in x.strip('[]').split()] for x in df.iloc[0]]
    except Exception:
        x_configs = [tuple(str(x).strip('[]').split()) for x in df.iloc[0]]
    
    # Extract unique configurations and corresponding performance
    unique_configs = []
    performance_curve = []
    
    for i, config in enumerate(x_configs):
        if config not in unique_configs:
            unique_configs.append(config)
            performance_curve.append(min(y_values[:i+1]))
    
    # Get best performance at different budgets
    try:
        perf_50 = performance_curve[min(50, len(performance_curve)-1)]
        perf_100 = performance_curve[min(100, len(performance_curve)-1)]
        perf_150 = performance_curve[min(150, len(performance_curve)-1)]
        perf_200 = performance_curve[min(200, len(performance_curve)-1)]
    except Exception:
        # Handle cases where length is insufficient
        if len(performance_curve) > 150:
            perf_50 = performance_curve[50]
            perf_100 = performance_curve[100]
            perf_150 = performance_curve[150]
            perf_200 = performance_curve[-1]
        else:
            # Fill with the last value
            perf_50 = performance_curve[-1]
            perf_100 = performance_curve[-1]
            perf_150 = performance_curve[-1]
            perf_200 = performance_curve[-1]
    
    return perf_50, perf_100, perf_150, perf_200

def main():
    # Configure experimental models
    learning_models = [
        "./results/PromiseTune_wocause",  # Model without causal analysis
        "./results/PromiseTune"         # Complete model
    ]
    learning_modelss = ['promisetune w/o', 'promisetune']  # Model abbreviations
    
    # Number of experiment repetitions
    seeds = range(1, 31)
    
    # System name mapping
    display_names = ['7z', 'DConvert', 'ExaStencils', "BDB-C", 'DeepArch', 
                     'PostgreSQL', 'JavaGC', 'Storm', 'x264', 'Redis', 'HSQLDB', 'LLVM']
    system_names = ['7z', 'dconvert', 'exastencils', "BDBC_AllNumeric", 'deeparch', 
                    'PostgreSQL', 'javagc', 'stormm', 'x264', 'redis', 'HSQLDB', 'LLVM_AllNumeric']

    # Open result file
    with open('./ablation_res.md', 'w', newline="") as f1:
        csv_writer = csv.writer(f1)
        
        # Write Markdown table header
        csv_writer.writerow(['# Ablation Study Results'])
        csv_writer.writerow([''])
        csv_writer.writerow(['| System | Budget 50 | | Budget 100 | | Budget 150 | | Budget 200 | |'])
        csv_writer.writerow(['|--------|-----------|---|------------|---|------------|---|------------|---|'])
        csv_writer.writerow(['| | w/o Causal | w/ Causal | w/o Causal | w/ Causal | w/o Causal | w/ Causal | w/o Causal | w/ Causal |'])
        
        # Read performance data range
        minmax_df = pd.read_csv('./minmax.csv', header=None)
        min_values = minmax_df.iloc[:, 1].tolist()
        max_values = minmax_df.iloc[:, 2].tolist()
        
        # Execute experiment for each system
        for index, system_name in enumerate(system_names):
            print(f"Processing system: {system_name}")
            display_name = display_names[index]
            min_value = min_values[index]
            max_value = max_values[index]
            
            # Collect results at different budgets
            results_b50 = []
            results_b100 = []
            results_b150 = []
            results_b200 = []
            
            # Collect experimental results for each model
            for learning_model in learning_models:
                budget_50_results = []
                budget_100_results = []
                budget_150_results = []
                budget_200_results = []
                
                for seed in seeds:
                    # Try to find result file
                    max_attempts = 30
                    attempts = 0
                    file_found = False
                    
                    while attempts < max_attempts and not file_found:
                        file_path = f'./{learning_model}/{system_name}{seed}.csv'
                        if os.path.exists(file_path):
                            file_found = True
                            # Process file data
                            perf_50, perf_100, perf_150, perf_200 = process_result_file(file_path, min_value, max_value)
                            
                            budget_50_results.append(perf_50)
                            budget_100_results.append(perf_100)
                            budget_150_results.append(perf_150)
                            budget_200_results.append(perf_200)
                        else:
                            seed += 1
                            if seed > 30:
                                seed = seed - 30
                            attempts += 1
                    
                    # If file not found after attempts, add default value
                    if not file_found:
                        default_value = 1e8
                        budget_50_results.append(default_value)
                        budget_100_results.append(default_value)
                        budget_150_results.append(default_value)
                        budget_200_results.append(default_value)
                
                # Add collected results to respective budget lists
                results_b50.append(budget_50_results)
                results_b100.append(budget_100_results)
                results_b150.append(budget_150_results)
                results_b200.append(budget_200_results)
            
            # Normalize results
            results_b50 = normalize_results(results_b50, min_value, max_value)
            results_b100 = normalize_results(results_b100, min_value, max_value)
            results_b150 = normalize_results(results_b150, min_value, max_value)
            results_b200 = normalize_results(results_b200, min_value, max_value)
            
            # Collect all budget comparisons for this system
            row_data = [f"**{display_name}**"]
            
            # Perform statistical analysis at different budgets
            all_budget_results = [results_b50, results_b100, results_b150, results_b200]
            
            # Analyze each budget level
            for budget_index, budget_results in enumerate(all_budget_results):
                # budget_results[0] is w/o causal, budget_results[1] is w/ causal
                # We expect w/ causal to be better (lower values after normalization)
                
                # Whether it is the last budget level
                is_last_budget = (budget_index == 3)
                
                # Perform statistical test (comparing w/ causal vs w/o causal)
                # alternative="less" means we test if w/ causal (budget_results[1]) < w/o causal (budget_results[0])
                u, p_value = mannwhitneyu(budget_results[1], budget_results[0], alternative="less")
                
                # Generate significance result - pass in order: w/o causal, w/ causal
                sig_output = get_significance_output(
                    budget_results[0],  # w/o causal
                    budget_results[1],  # w/ causal
                    p_value,
                    budget_last=is_last_budget
                )
                
                # Split the output to get individual values
                parts = sig_output.split(' | ')
                row_data.extend(parts)
                
                # Print debug information
                # if is_last_budget:
                #     print(f"p-value: {p_value}")
                
                mean_wo = np.mean(budget_results[0])  # w/o causal
                mean_w = np.mean(budget_results[1])   # w/ causal
                if mean_wo == 0:
                    improvement = 0
                else:
                    improvement = (mean_wo - mean_w) / mean_wo
                    if improvement < 0:
                        improvement = 0

                print(f"Mean comparison: w/o causal={mean_wo:.3f} vs w/ causal={mean_w:.3f}, Improvement rate: {improvement:.3%}")
            
            # Write the complete row for this system
            csv_writer.writerow(['| ' + ' | '.join(row_data) + ' |'])

if __name__ == "__main__":
    main()