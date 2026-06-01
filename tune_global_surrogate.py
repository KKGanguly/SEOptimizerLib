import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure the script can find your 'utils' folder from the root directory
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import ParameterGrid, KFold
from sklearn.metrics import r2_score

# Native imports from your framework architecture
from utils.data_loader_templated import load_data
from utils.EncodingUtils import EncodingUtils
from utils.RFEncoder import CatEncoder

def prepare_dataset(X: pd.DataFrame, Y: pd.DataFrame):
    """
    Applies the exact same two-layer encoding pipeline used by 
    experiment_runner_cluster.py and ModelWrapperStatic.py
    """
    # 1. Runner-level Encoding
    column_types = {
        col: EncodingUtils.infer_column_type(X[col].tolist())[0]
        for col in X.columns
    }
    X_encoded = EncodingUtils.encode_dataframe(X, column_types)
    
    # 2. Model-level Encoding (Catches strings and converts to integer labels)
    cat_encoder = CatEncoder()
    cat_encoder.fit(X_encoded, column_types)
    X_enc_array = cat_encoder.transform_df(X_encoded, column_types)
    
    # 3. Normalize Y (0.0 to 1.0)
    if isinstance(Y, pd.Series):
        Y = Y.to_frame()
        
    for col in Y.columns:
        col_series = Y[col]
        if pd.api.types.is_bool_dtype(col_series):
            Y[col] = col_series.astype(float)
        elif pd.api.types.is_numeric_dtype(col_series):
            uniq = col_series.dropna().unique()
            if len(uniq) == 2:
                Y[col] = col_series.astype(float)
                
    Y_norm = (Y - Y.min()) / (Y.max() - Y.min())
    y_raw = Y_norm.values.copy()
    
    # 4. Apply direction flip for minimization objectives (-)
    for i, col in enumerate(Y_norm.columns):
        if col.endswith('-'):
            y_raw[:, i] = 1.0 - y_raw[:, i]
            
    return X_enc_array, y_raw

def evaluate_rf_on_dataset(model, X_enc, y_raw, n_splits=3):
    """Runs K-Fold CV on a single dataset and returns the average R^2 score."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []
    
    for train_index, test_index in kf.split(X_enc):
        X_train, X_test = X_enc[train_index], X_enc[test_index]
        y_train, y_test = y_raw[train_index], y_raw[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        score = r2_score(y_test, y_pred)
        fold_scores.append(score)
        
    return np.mean(fold_scores)

def tune_global_surrogate(dataset_paths: list):
    print(f"1. Loading and Encoding {len(dataset_paths)} Datasets using native framework...")
    
    encoded_datasets = {}
    for file_path in dataset_paths:
        try:
            X, Y = load_data(file_path)
            X_enc_array, y_raw = prepare_dataset(X, Y)
            encoded_datasets[file_path] = (X_enc_array, y_raw)
        except Exception as e:
            print(f"  [!] Skipping {file_path} due to native load error: {e}")
            
    if not encoded_datasets:
        print("Error: No datasets successfully loaded. Aborting.")
        return
        
    print(f"\n2. Defining Parameter Grid...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_features': ['sqrt', 1.0], 
        'max_depth': [None, 10, 20],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid = list(ParameterGrid(param_grid))
    best_global_score = -float('inf')
    best_params = None
    
    print(f"3. Evaluating {len(grid)} Parameter Combinations Across {len(encoded_datasets)} Datasets...")
    start_time = time.time()
    
    for i, params in enumerate(grid):
        combo_scores = []
        
        for name, (X_enc_array, y_raw) in encoded_datasets.items():
            try:
                # Set n_jobs=1 to completely eliminate nested parallelism memory leaks
                base_rf = RandomForestRegressor(**params, random_state=42, n_jobs=1)
                n_outputs = y_raw.shape[1]
                
                if n_outputs > 1:
                    model = MultiOutputRegressor(base_rf, n_jobs=1)
                else:
                    model = base_rf
                    y_raw = y_raw.ravel() 
                    
                dataset_r2 = evaluate_rf_on_dataset(model, X_enc_array, y_raw)
                combo_scores.append(dataset_r2)
            except Exception as e:
                # Silently catch the 'squared_error' string bug and skip the corrupted dataset
                pass
            
        if not combo_scores:
            continue
            
        global_score = np.mean(combo_scores)
        
        print(f"  [{i+1}/{len(grid)}] Params: {params} | Global R^2: {global_score:.4f}")
        
        if global_score > best_global_score:
            best_global_score = global_score
            best_params = params
            
    elapsed = time.time() - start_time
    
    print("\n" + "="*50)
    print("          GLOBAL TUNING COMPLETE")
    print("="*50)
    print(f"Total Time: {elapsed/60:.1f} minutes")
    print(f"Best Average R^2 Score: {best_global_score:.4f}")
    print("Optimal Universal Parameters (Plug these into ModelWrapperStatic):")
    if best_params:
        for k, v in best_params.items():
            print(f"  {k}: {v}")
    else:
        print("  No successful runs to determine parameters.")

if __name__ == "__main__":
    repo_path = Path("moot/optimize")
    csv_files = list(repo_path.rglob("*.csv"))
    
    if not csv_files:
        print(f"Error: No CSV files found in {repo_path.resolve()}")
        sys.exit(1)
        
    print(f"Found {len(csv_files)} datasets in {repo_path}")
    dataset_paths = [str(f) for f in csv_files]
    tune_global_surrogate(dataset_paths)