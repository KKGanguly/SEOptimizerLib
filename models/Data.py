import os
import pickle
import hashlib
import math
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from numba import njit


# =========================================================
# 🔥 NUMBA DISTANCE (NATIVE SPEED)
# =========================================================
@njit(fastmath=True)
def _xdist_numba(i, j, num, cat, is_cat, is_missing, min_vals, norm_scale):
    total = 0.0
    n = is_cat.shape[0]

    for k in range(n):
        # both missing
        if is_missing[i, k] == 1 and is_missing[j, k] == 1:
            d = 1.0

        # categorical / date
        elif is_cat[k] == 1:
            d = 0.0 if cat[i, k] == cat[j, k] else 1.0

        # numeric
        else:
            if is_missing[i, k] == 1:
                b = (num[j, k] - min_vals[k]) * norm_scale[k]
                a = 1.0 if b < 0.5 else 0.0
            elif is_missing[j, k] == 1:
                a = (num[i, k] - min_vals[k]) * norm_scale[k]
                b = 1.0 if a < 0.5 else 0.0
            else:
                a = (num[i, k] - min_vals[k]) * norm_scale[k]
                b = (num[j, k] - min_vals[k]) * norm_scale[k]

            d = abs(a - b)

        total += d * d

    return math.sqrt(total / n)


# =========================================================
# 🚀 DATA CLASS (DROP-IN)
# =========================================================
class Data:
    def __init__(self, rows, column_types, cache_dir=".cache_kdtree", use_kdtree=True):
        """
        rows: List[List[Any]]  (encoded, may include str, float, "?")
        column_types: Dict[col -> {'numeric','categorical','date'}]
        """
        self.rows = rows
        self.column_types = column_types
        self.cache_dir = cache_dir
        self.use_kdtree = use_kdtree

        os.makedirs(cache_dir, exist_ok=True)

        # ----------------------------------
        # COLUMN METADATA
        # ----------------------------------
        self.col_types = list(column_types.values())
        self.n_cols = len(self.col_types)

        self.is_cat = np.array(
            [1 if t in ("categorical", "date") else 0 for t in self.col_types],
            dtype=np.int8
        )

        # ----------------------------------
        # NUMERIC NORMALIZATION
        # ----------------------------------
        self.min_vals, self.max_vals = self._compute_min_max()
        self.min_vals = np.array(self.min_vals, dtype=np.float64)
        self.norm_scale = np.array(
            [(1.0 / (mx - mn)) if mx > mn else 0.0
             for mn, mx in zip(self.min_vals, self.max_vals)],
            dtype=np.float64
        )

        # ----------------------------------
        # 🔥 BUILD NUMERIC SHADOW DATA
        # ----------------------------------
        self._build_numeric_view()

        # map row object → index (so API stays row-based)
        self.row_index = {id(r): i for i, r in enumerate(self.rows)}

        # warm up numba once
        if len(self.rows) >= 2:
            _xdist_numba(
                0, 1,
                self.num_view,
                self.cat_view,
                self.is_cat,
                self.is_missing,
                self.min_vals,
                self.norm_scale
            )

        # ----------------------------------
        # KD-TREE (UNCHANGED)
        # ----------------------------------
        if self.use_kdtree:
            if not self._load_cache():
                self._compute_vectors_for_kdtree()
                self._build_kdtree()
                self._save_cache()

    # =====================================================
    # INTERNAL BUILDERS
    # =====================================================
    def _compute_min_max(self):
        mins, maxs = [], []
        for j, t in enumerate(self.col_types):
            if t == "numeric":
                vals = [
                    r[j] for r in self.rows
                    if isinstance(r[j], (int, float)) and r[j] != "?"
                ]
                if vals:
                    mins.append(min(vals))
                    maxs.append(max(vals))
                else:
                    mins.append(0.0)
                    maxs.append(1.0)
            else:
                mins.append(0.0)
                maxs.append(1.0)
        return mins, maxs

    def _build_numeric_view(self):
        n_rows = len(self.rows)
        n_cols = self.n_cols

        self.num_view = np.zeros((n_rows, n_cols), dtype=np.float64)
        self.cat_view = np.zeros((n_rows, n_cols), dtype=np.int32)
        self.is_missing = np.zeros((n_rows, n_cols), dtype=np.int8)

        cat_maps = [{} for _ in range(n_cols)]
        cat_counts = [0] * n_cols

        for i, row in enumerate(self.rows):
            for j, v in enumerate(row):
                if v == "?":
                    self.is_missing[i, j] = 1
                    continue

                if self.is_cat[j]:
                    m = cat_maps[j]
                    if v not in m:
                        m[v] = cat_counts[j]
                        cat_counts[j] += 1
                    self.cat_view[i, j] = m[v]
                else:
                    self.num_view[i, j] = float(v)

    # =====================================================
    # 🔥 FAST DISTANCE (DESTROYS LUA)
    # =====================================================
    def xdist(self, r1, r2):
        i = self.row_index[id(r1)]
        j = self.row_index[id(r2)]
        return _xdist_numba(
            i, j,
            self.num_view,
            self.cat_view,
            self.is_cat,
            self.is_missing,
            self.min_vals,
            self.norm_scale
        )

    # =====================================================
    # KD-TREE (UNCHANGED FROM YOUR CODE)
    # =====================================================
    def _dataset_hash(self):
        h = hashlib.md5()
        for row in self.rows:
            h.update(str(row).encode())
        return h.hexdigest()

    def _cache_path(self):
        return os.path.join(self.cache_dir, f"kdt_{self._dataset_hash()}.pkl")

    def _load_cache(self):
        path = self._cache_path()
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.vectors = data["vectors"]
            self.kdtree = data["kdtree"]
            return True
        except Exception:
            return False

    def _save_cache(self):
        path = self._cache_path()
        if os.path.exists(path):
            return
        with open(path, "wb") as f:
            pickle.dump({"vectors": self.vectors, "kdtree": self.kdtree}, f)

    def _encode_value_for_kdtree(self, v, idx):
        if v == "?" or pd.isna(v):
            return 0.5
        if self.col_types[idx] in ("categorical", "date"):
            return (hash(str(v)) % 10000) / 10000
        mn, mx = self.min_vals[idx], self.max_vals[idx]
        return (v - mn) / (mx - mn) if mx > mn else 0.0

    def _compute_vectors_for_kdtree(self):
        self.vectors = np.array([
            [self._encode_value_for_kdtree(v, idx)
             for idx, v in enumerate(row)]
            for row in self.rows
        ], dtype=float)

    def _build_kdtree(self):
        self.kdtree = KDTree(self.vectors, leaf_size=40)

    # =====================================================
    # PUBLIC
    # =====================================================
    def nearestRow_bruteforce(self, target_row):
        best_dist = float("inf")
        best = None
        for row in self.rows:
            if row is target_row:
                continue
            d = self.xdist(target_row, row)
            if d < best_dist:
                best_dist = d
                best = row
        return best

    def nearestRow(self, target_row):
        if not self.use_kdtree or getattr(self, "kdtree", None) is None:
            return self.nearestRow_bruteforce(target_row)
        vec = np.array([[
            self._encode_value_for_kdtree(v, idx)
            for idx, v in enumerate(target_row)
        ]], dtype=float)
        _, idxs = self.kdtree.query(vec, k=1)
        return self.rows[idxs[0][0]]

    def nearestRowSkipSelf(self, target_row):
        if not self.use_kdtree or getattr(self, "kdtree", None) is None:
            return self.nearestRow_bruteforce(target_row)

        vec = np.array([[
            self._encode_value_for_kdtree(v, idx)
            for idx, v in enumerate(target_row)
        ]], dtype=float)

        # query 2 nearest neighbors
        dists, idxs = self.kdtree.query(vec, k=2)

        # first neighbor may be self (distance 0)
        if dists[0][0] > 0:
            return self.rows[idxs[0][0]]
        else:
            return self.rows[idxs[0][1]]  # skip self

    # ────────────────────────────────
    # Add this inside the Data class
    # ────────────────────────────────
    def k_nearest_indices(self, row_idx, k=5):
        """
        FAST VERSION: Uses KD-Tree for O(log N) neighborhood lookup.
        """
        if not self.use_kdtree or getattr(self, "kdtree", None) is None:
            # Fallback to your original logic if KD-Tree is disabled
            target_row = self.rows[row_idx]
            distances = []
            for i, row in enumerate(self.rows):
                if i == row_idx: continue
                distances.append((i, self.xdist(target_row, row)))
            distances.sort(key=lambda x: x[1])
            return [idx for idx, _ in distances[:k]]

        # Use the pre-computed vectors for the KD-Tree
        query_vec = self.vectors[row_idx].reshape(1, -1)
        
        # Query k+1 neighbors because the closest one is always the row itself (dist=0)
        # return_distance=False saves computation time
        _, idxs = self.kdtree.query(query_vec, k=k+1)
        
        # Filter out the self-index and return the rest
        neighbor_indices = [i for i in idxs[0] if i != row_idx]
        
        # Ensure we return exactly k neighbors (handles edge cases)
        return neighbor_indices[:k]