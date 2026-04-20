import pandas as pd
from utils import DistanceUtil
from utils.EncodingUtils import EncodingUtils

class ModelWrapperStatic:
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, model_config):
        """
        Initialize with raw data - will be encoded internally
        """
        self.model_config = model_config
        self.column_types = model_config.column_types
        
        self.X = X  # Store encoded version
        
	# Convert only binary columns of y
        if isinstance(y, pd.Series):
            y = y.to_frame()  # make sure we can loop consistently

        for col in y.columns:
            col_series = y[col]

            # Boolean → float
            if pd.api.types.is_bool_dtype(col_series):
                y[col] = col_series.astype(float)
 
            # Numeric binary (only two unique numbers)
            elif pd.api.types.is_numeric_dtype(col_series):
                uniq = col_series.dropna().unique()
                if len(uniq) == 2:
                    y[col] = col_series.astype(float)
	 # Normalize y
        self.y = (y - y.min()) / (y.max() - y.min())
        
        # Build lookup table with encoded values
        self.lookup = {
            tuple(row[col] for col in self.X.columns): idx
            for idx, row in self.X.iterrows()
        }
    
    def _encode_hp(self, col, v):
        """Encode hyperparameter using consistent encoding"""
        col_type = self.column_types.get(col, 'categorical')
        return EncodingUtils.encode_value(v, col_type)
    
    def _find_row_fast(self, hyperparams):
        if not hyperparams:
            raise ValueError("No hyperparameters provided.")
        
        # Encode the incoming hyperparameters
        key = tuple(self._encode_hp(col, hyperparams[col])
                    for col in self.X.columns)
        
        idx = self.lookup.get(key, None)
        if idx is None:
            # Debug output
            print(f"Key not found: {key}")
            print(f"Sample of lookup keys: {list(self.lookup.keys())[:3]}")
        return idx
    
    def _score_tuple(self, idx):
        if idx is None:
            raise ValueError("Cannot score None index")
        row = self.y.loc[idx]
        return tuple(
            (1 - v) if col.endswith("-") else v
            for col, v in row.items()
        )
    
    def _avg_d2h(self, scores):
        d2h = DistanceUtil.d2h([1] * len(scores), scores)
        return scores, d2h
    
    def get_score(self, hyperparams):
        idx = self._find_row_fast(hyperparams)
        if idx is None:
            raise ValueError(f"Configuration not found in dataset: {hyperparams}")
        return self._score_tuple(idx)
    
    def run_model(self, hyperparams=None, budget=None):
        scores = self.get_score(hyperparams)
        _, d2h = self._avg_d2h(scores)
        return 1 - d2h
    
    def evaluate(self, hyperparameters=None):
        scores = self.get_score(hyperparameters)
        return self._avg_d2h(scores)
    
    def test(self, hyperparameters=None):
        return self.evaluate(hyperparameters)
