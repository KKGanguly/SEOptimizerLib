import numpy as np
import pandas as pd
class CatEncoder:
    def __init__(self):
        self.mappings = {}      # col -> {str_val: int}
        self.inv_mappings = {}  # col -> {int: str_val}
 
    def fit(self, X_df: pd.DataFrame, column_types: dict):
        """Build mapping from every categorical/date column's unique values to ints."""
        for col, ctype in column_types.items():
            if col not in X_df.columns:
                continue
            if ctype in ('categorical', 'date'):
                uniq = sorted(X_df[col].astype(str).unique())
                self.mappings[col] = {v: i for i, v in enumerate(uniq)}
                self.inv_mappings[col] = {i: v for v, i in self.mappings[col].items()}
 
    def transform_row(self, hp_dict: dict, column_types: dict, columns) -> np.ndarray:
        """
        Convert a config dict -> 1-D float numpy array suitable for RF.predict.
        Columns are processed in the order given by `columns` (must match fit order).
        """
        row = []
        for col in columns:
            v = hp_dict[col]
            ctype = column_types.get(col, 'categorical')
            if ctype in ('categorical', 'date'):
                row.append(float(self.mappings[col].get(str(v), -1)))
            else:
                row.append(float(v))
        return np.array(row, dtype=float)
 
    def transform_df(self, X_df: pd.DataFrame, column_types: dict) -> np.ndarray:
        """Convert a full DataFrame -> 2-D float numpy array for RF.fit."""
        rows = []
        for _, r in X_df.iterrows():
            rows.append(self.transform_row(dict(r), column_types, X_df.columns))
        return np.array(rows, dtype=float)
 