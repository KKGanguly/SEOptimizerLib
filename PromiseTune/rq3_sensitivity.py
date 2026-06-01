"""
RQ3: Hyperparameter Sensitivity Analysis Experiment
Analyze the trend of model performance changes under different configurations,
including budgets 50, 100, 150, 200
"""

import os
import csv
import numpy as np
import pandas as pd

def calculate_column_means(data):
    """Calculate mean value for each column of 2D data
    
    Args:
        data: 2D list data
    
    Returns:
        List of mean values for each column
    """
    if not data or not data[0]:
        return []
    
    rows = len(data)
    cols = len(data[0])
    
    # Calculate sum of each column
    column_sums = [0] * cols
    for row in data:
        for i in range(cols):
            column_sums[i] += row[i]
    
    # Calculate mean of each column
    column_means = [sum_val / rows for sum_val in column_sums]
    
    return column_means

def normalize_results(results, min_value, max_value):
    """Normalize results to [0, 1] range
    
    Args:
        results: List of results to be normalized
        min_value: Minimum value
        max_value: Maximum value
    
    Returns:
        List of normalized results
    """
    normalized_results = []
    for res in results:
        normalized_res = [(x - min_value) / (max_value - min_value) 
                          if max_value != min_value else 0 for x in res]
        normalized_results.append(normalized_res)
    return normalized_results

def process_result_file(file_path):
    """Process a single result file and extract performance values at budgets 50, 100, 150, 200
    
    Args:
        file_path: Path to result file
    
    Returns:
        Performance values at budgets 50, 100, 150, 200
    """
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
    
    # Get performance values at different budgets
    try:
        if len(performance_curve) >= 200:
            perf_50 = performance_curve[49]
            perf_100 = performance_curve[99]
            perf_150 = performance_curve[149]
            perf_200 = performance_curve[199]
        elif len(performance_curve) >= 150:
            perf_50 = performance_curve[49]
            perf_100 = performance_curve[99]
            perf_150 = performance_curve[149]
            perf_200 = performance_curve[-1]
        elif len(performance_curve) >= 100:
            perf_50 = performance_curve[49]
            perf_100 = performance_curve[99]
            perf_150 = performance_curve[-1]
            perf_200 = performance_curve[-1]
        elif len(performance_curve) >= 50:
            perf_50 = performance_curve[49]
            perf_100 = performance_curve[-1]
            perf_150 = performance_curve[-1]
            perf_200 = performance_curve[-1]
        else:
            perf_50 = performance_curve[-1]
            perf_100 = performance_curve[-1]
            perf_150 = performance_curve[-1]
            perf_200 = performance_curve[-1]
    except Exception as e:
        print(f"Error processing performance curve: {e}")
        # Use the last value to fill
        perf_50 = performance_curve[-1] if performance_curve else 1e8
        perf_100 = performance_curve[-1] if performance_curve else 1e8
        perf_150 = performance_curve[-1] if performance_curve else 1e8
        perf_200 = performance_curve[-1] if performance_curve else 1e8
    
    return perf_50, perf_100, perf_150, perf_200

def main():
    # Configure model paths under different parameter values
    learning_models = [
        "./parameter_results/Promisetune_ultra_5",        # Parameter value 5
        "./parameter_results/Promisetune_ultra_1",        # Parameter value 10
        "./parameter_results/Promisetune_ultra_15",       # Parameter value 15
        "./parameter_results/Promisetune_ultra_20"        # Parameter value 20
    ]
    
    model_labels = ['ultra_5', 'ultra_10', 'ultra_15',  'ultra_20']
    parameter_values = [5, 10, 15,  20]  # Corresponding parameter values
    
    # Number of experiment repetitions
    seeds = range(1, 31)
    
    # System names
    systems = ['7z', 'dconvert', 'exastencils', "BDBC_AllNumeric", 'deeparch',
               'PostgreSQL', 'javagc', 'stormm', 'x264', 'redis', 'HSQLDB', 'LLVM_AllNumeric']
    
    # Store average performance and standard deviation at different budgets
    all_means_b50 = []
    all_stds_b50 = []
    all_means_b100 = []
    all_stds_b100 = []
    all_means_b150 = []
    all_stds_b150 = []
    all_means_b200 = []
    all_stds_b200 = []
    
    # Read performance metric range
    minmax_df = pd.read_csv('./minmax.csv', header=None)
    min_values = minmax_df.iloc[:, 1].tolist()
    max_values = minmax_df.iloc[:, 2].tolist()
    
    # Process each system
    for index, system_name in enumerate(systems):
        print(f"Processing system: {system_name}")
        
        min_value = min_values[index]
        max_value = max_values[index]
        
        # Collect performance results at different budgets
        results_b50 = []   # Budget 50
        results_b100 = []  # Budget 100
        results_b150 = []  # Budget 150
        results_b200 = []  # Budget 200
        
        # Collect experimental results for each model
        for model_path in learning_models:
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
                    file_path = f'./{model_path}/{system_name}{seed}.csv'
                    if os.path.exists(file_path):
                        file_found = True
                        # Process file data
                        perf_50, perf_100, perf_150, perf_200 = process_result_file(file_path)
                        
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
        
        # Analyze results at each budget
        budget_50_means = [np.mean(res) for res in results_b50]
        budget_50_stds = [np.std(res) for res in results_b50]
        budget_100_means = [np.mean(res) for res in results_b100]
        budget_100_stds = [np.std(res) for res in results_b100]
        budget_150_means = [np.mean(res) for res in results_b150]
        budget_150_stds = [np.std(res) for res in results_b150]
        budget_200_means = [np.mean(res) for res in results_b200]
        budget_200_stds = [np.std(res) for res in results_b200]
        
        # Save results for all systems
        all_means_b50.append(budget_50_means)
        all_stds_b50.append(budget_50_stds)
        all_means_b100.append(budget_100_means)
        all_stds_b100.append(budget_100_stds)
        all_means_b150.append(budget_150_means)
        all_stds_b150.append(budget_150_stds)
        all_means_b200.append(budget_200_means)
        all_stds_b200.append(budget_200_stds)
    
    # Calculate average results for all systems
    budget_data = {
        "budget=50": {
            "means": calculate_column_means(all_means_b50),
            "stds": calculate_column_means(all_stds_b50)
        },
        "budget=100": {
            "means": calculate_column_means(all_means_b100),
            "stds": calculate_column_means(all_stds_b100)
        },
        "budget=150": {
            "means": calculate_column_means(all_means_b150),
            "stds": calculate_column_means(all_stds_b150)
        },
        "budget=200": {
            "means": calculate_column_means(all_means_b200),
            "stds": calculate_column_means(all_stds_b200)
        }
    }
    
    # Output results to Markdown file
    with open('./sensitivity_results_all_budgets.md', 'w', encoding='utf-8') as f:
        f.write("# Hyperparameter Sensitivity Analysis Results\n\n")
        f.write("## Performance Across Different Budgets\n\n")
        
        # Write results for each budget level
        for budget_name, budget_values in budget_data.items():
            mean_values = budget_values["means"]
            std_values = budget_values["stds"]
            
            f.write(f"### {budget_name}\n\n")
            
            # Create table header
            f.write("| Parameter Value | Average Performance | Std Dev | Lower Bound (Mean - Std) | Upper Bound (Mean + Std) |\n")
            f.write("|:---------------:|--------------------:|--------:|-------------------------:|-------------------------:|\n")
            
            # Write table rows
            lower_bounds = [mean - std for mean, std in zip(mean_values, std_values)]
            upper_bounds = [mean + std for mean, std in zip(mean_values, std_values)]
            
            for param, mean, std, lower, upper in zip(parameter_values, mean_values, std_values, lower_bounds, upper_bounds):
                f.write(f"| {param} | {mean:.4f} | {std:.4f} | {lower:.4f} | {upper:.4f} |\n")
            
            f.write("\n")
            
            # Add visualization data for plotting
            f.write(f"**Plot Data for {budget_name}:**\n\n")
            f.write("```\n")
            f.write("Average: ")
            for param, perf in zip(parameter_values, mean_values):
                f.write(f"({param}, {perf:.4f}) ")
            f.write("\n")
            
            f.write("Bottom:  ")
            for param, bound in zip(parameter_values, lower_bounds):
                f.write(f"({param}, {bound:.4f}) ")
            f.write("\n")
            
            f.write("Top:     ")
            for param, bound in zip(parameter_values, upper_bounds):
                f.write(f"({param}, {bound:.4f}) ")
            f.write("\n```\n\n")
            f.write("---\n\n")
    
    print(f"Analysis complete, results saved to sensitivity_results_all_budgets.md")

if __name__ == "__main__":
    main()