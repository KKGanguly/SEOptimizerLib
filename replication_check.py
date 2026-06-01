import pandas as pd
import numpy as np
from pathlib import Path

def generate_simple_table(results_dir: str, optimizer_name: str, budgets: list, output_file: str):
    base_dir = Path(results_dir) / f"results_{optimizer_name}" /  f"{optimizer_name}"
    
    # 1. Get all CSV files in that folder
    all_files = list(base_dir.glob("*.csv"))
    if not all_files:
        print(f"Error: No files found in {base_dir}")
        return

    # 2. Extract unique dataset names
    # File format is: datasetname_budget.csv
    # We split by '_' and take everything except the last part (the budget)
    datasets = sorted(list(set(['_'.join(f.stem.split('_')[:-1]) for f in all_files])))
    
    table_data = []

    for ds in datasets:
        row = {'Dataset': ds}
        
        for b in budgets:
            # Look for the file: datasetname_50.csv
            file_path = base_dir / f"{ds}_{b}.csv"
            
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    # Use all values in the column (assuming 30 runs worth of data)
                    values = df['best_value'].dropna().tolist()
                    
                    if values:
                        row[f'{b}_mean'] = np.mean(values)
                        row[f'{b}_stdv'] = np.std(values, ddof=1) if len(values) > 1 else 0.0
                    else:
                        row[f'{b}_mean'] = np.nan
                        row[f'{b}_stdv'] = np.nan
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
            else:
                row[f'{b}_mean'] = np.nan
                row[f'{b}_stdv'] = np.nan
        
        table_data.append(row)

    df_final = pd.DataFrame(table_data)
    df_final.to_csv(output_file, index=False)
    print(f"Success! Table saved to {output_file}")

if __name__ == '__main__':
    generate_simple_table(
        results_dir='results', 
        optimizer_name='PTUNER', 
        budgets=[50, 100, 150, 200], 
        output_file='ptuner_summary.csv'
    )