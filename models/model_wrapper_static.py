import os
import numpy as np
import pandas as pd
import joblib
import hashlib
 
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
 
from utils import DistanceUtil
from utils.EncodingUtils import EncodingUtils
from utils.RFEncoder import CatEncoder
 
class ModelWrapperStatic:
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, model_config, seed: int = 42):
        self.model_config = model_config
        self.column_types = model_config.column_types
        self.seed = seed
        self.X = X  # already encoded by experiment runner
        
        # ------------------------------------------------------------------
        # Normalise y
        # ------------------------------------------------------------------
        if isinstance(y, pd.Series):
            y = y.to_frame()
 
        for col in y.columns:
            col_series = y[col]
            if pd.api.types.is_bool_dtype(col_series):
                y[col] = col_series.astype(float)
            elif pd.api.types.is_numeric_dtype(col_series):
                uniq = col_series.dropna().unique()
                if len(uniq) == 2:
                    y[col] = col_series.astype(float)
 
        self.y = (y - y.min()) / (y.max() - y.min())
 
        # ------------------------------------------------------------------
        # Legacy lookup table — kept so existing tests and any code that
        # calls _find_row_fast / _score_tuple directly continues to work.
        # get_score() now goes through the RF surrogate instead.
        # ------------------------------------------------------------------
        self.lookup = {
            tuple(row[col] for col in self.X.columns): idx
            for idx, row in self.X.iterrows()
        }
 
        ## ------------------------------------------------------------------
        # RF surrogate — trained once at init, cached to disk by dataset hash
        # ------------------------------------------------------------------
        #self._train_rf_surrogate()
    
    def set_seed(self, seed):
        self.seed = seed
        self._train_rf_surrogate()

    # -----------------------------------------------------------------------
    # RF surrogate: training
    # -----------------------------------------------------------------------
    def _dataset_hash(self) -> str:
        """Stable hash of X content + y content for cache keying."""
        hash_string = self.X.to_csv(index=False) + self.y.to_csv(index=False) + str(self.seed)
        h = hashlib.md5(hash_string.encode()).hexdigest()[:16]
        return h
 
    def _rf_cache_path(self) -> str:
        return os.path.join(".cache_rf", f"rf_{self._dataset_hash()}.pkl")
 
    def _train_rf_surrogate(self):
        """
        Train (or load from cache) an RF surrogate over the full dataset.
 
        Categorical columns are encoded with CatEncoder (unordered integer
        labels) — correct for RF because trees split on ==, not on ordering.
 
        For multi-objective y, MultiOutputRegressor trains one RF per
        objective using the same hyperparameters.
        """
        cache_path = self._rf_cache_path()
 
        # -- Try loading from disk cache first --
        if os.path.exists(cache_path):
            try:
                cached = joblib.load(cache_path)
                self.rf_model = cached['model']
                self.cat_encoder = cached['encoder']
                self._single_output = cached['single_output']
                self._columns = cached['columns']
                return
            except Exception:
                pass  # corrupted cache — retrain
 
        # -- Build feature matrix --
        self.cat_encoder = CatEncoder()
        self.cat_encoder.fit(self.X, self.column_types)
        self._columns = list(self.X.columns)  # column order fixed at fit time
 
        X_enc = self.cat_encoder.transform_df(self.X, self.column_types)
 
        # -- Build target matrix, applying direction flip for "-" objectives --
        # _score_tuple flips minimisation objectives at query time; we train
        # the RF on the same flipped values so predictions are consistent.
        y_raw = self.y.values.copy()
        for i, col in enumerate(self.y.columns):
            if col.endswith('-'):
                y_raw[:, i] = 1.0 - y_raw[:, i]
 
        n_outputs = y_raw.shape[1]
 
        base_rf = RandomForestRegressor(
            n_estimators=100,
            max_features=1.0,
            min_samples_leaf=4,
            random_state=self.seed,
            n_jobs=-1,
        )
 
        if n_outputs == 1:
            self.rf_model = base_rf
            self.rf_model.fit(X_enc, y_raw.ravel())
            self._single_output = True
        else:
            self.rf_model = MultiOutputRegressor(base_rf, n_jobs=-1)
            self.rf_model.fit(X_enc, y_raw)
            self._single_output = False
 
        # -- Persist to disk --
        os.makedirs(".cache_rf", exist_ok=True)
        joblib.dump({
            'model': self.rf_model,
            'encoder': self.cat_encoder,
            'single_output': self._single_output,
            'columns': self._columns,
        }, cache_path)
 
    # -----------------------------------------------------------------------
    # RF surrogate: prediction
    # -----------------------------------------------------------------------
    def _rf_predict(self, hp_dict: dict) -> tuple:
        """
        Encode hp_dict and return RF predictions as a tuple.
        Direction flipping for "-" columns is already baked into the trained
        targets (done in _train_rf_surrogate), so no flip needed here.
        """
        x = self.cat_encoder.transform_row(hp_dict, self.column_types, self._columns)
        preds = self.rf_model.predict(x.reshape(1, -1))[0]
        if self._single_output:
            return (float(preds),)
        return tuple(float(p) for p in preds)
    
    # -----------------------------------------------------------------------
    # Legacy helpers — kept for backward compatibility
    # -----------------------------------------------------------------------
    def _encode_hp(self, col, v):
        col_type = self.column_types.get(col, 'categorical')
        return EncodingUtils.encode_value(v, col_type)
 
    def _find_row_fast(self, hyperparams):
        if not hyperparams:
            raise ValueError("No hyperparameters provided.")
        key = tuple(self._encode_hp(col, hyperparams[col]) for col in self.X.columns)
        idx = self.lookup.get(key, None)
        if idx is None:
            print(f"Key not found: {key}")
            print(f"Sample of lookup keys: {list(self.lookup.keys())[:3]}")
        return idx
 
    def _score_tuple(self, idx):
        """Direct table lookup — used by legacy tests only."""
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
 
    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def get_score(self, hyperparams) -> tuple:
        """
        Returns objective scores for a config dict via the RF surrogate.
        Replaces the previous table-lookup approach.
        Any config in the feature space can be scored — not just exact table rows.
        """
        return self._rf_predict(hyperparams)
 
    def run_model(self, hyperparams=None, budget=None):
        scores = self.get_score(hyperparams)
        _, d2h = self._avg_d2h(scores)
        return 1 - d2h
 
    def evaluate(self, hyperparameters=None):
        scores = self.get_score(hyperparameters)
        return self._avg_d2h(scores)
 
    def test(self, hyperparameters=None):
        return self.evaluate(hyperparameters)