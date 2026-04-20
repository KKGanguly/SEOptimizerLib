import os
import pandas as pd
import numpy as np

ROOT = "moot/optimize"

def is_strict_float(series):
    # must be numeric and NOT boolean
    if pd.api.types.is_bool_dtype(series):
        return False
    
    if not pd.api.types.is_numeric_dtype(series):
        return False
    
    # ensure no integers masquerading as ints only (optional strictness)
    # allow float-compatible numeric
    try:
        s = pd.to_numeric(series, errors='coerce')
    except:
        return False
    
    # reject if any non-numeric values appear
    if s.isna().any():
        return False
    
    # ensure float compatibility (no pure categorical encoding like 0/1 bool)
    if pd.api.types.is_integer_dtype(series):
        # still allowed but only if truly continuous-like (optional rule)
        return False
    
    return True


def valid_dataset(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except:
        return False

    # remove + and - columns
    df = df[[c for c in df.columns if not (c.endswith('+') or c.endswith('-'))]]

    # identify X columns
    x_cols = [c for c in df.columns if c.endswith('X')]

    if len(x_cols) == 0:
        return False

    for col in x_cols:
        # rule: lowercase-start columns are non-numeric → reject immediately
        if col[0].islower():
            return False

        if not is_strict_float(df[col]):
            return False

    return True


valid_datasets = []

for root, _, files in os.walk(ROOT):
    for f in files:
        if f.endswith(".csv"):
            path = os.path.join(root, f)
            if valid_dataset(path):
                valid_datasets.append(path)

print("VALID DATASETS:")
for v in valid_datasets:
    print(v)