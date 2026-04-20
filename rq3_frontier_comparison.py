"""
RQ3 EZR vs MOEA Frontier Comparison
=====================================

METRICS
-------
  IGD  — mean( min_dist(ref -> approx) )         lower = better
  GD   — mean( min_dist(approx -> ref) )         lower = better
         Both use the true CSV Pareto front as reference Z.
  HV   — Hypervolume, ref=(1.1,...,1.1)          higher = better
         2-obj: exact sweep.  3+-obj: pymoo exact WFG (or MC fallback).
  SP   — std of nearest-neighbour distances       lower = better
         (non-extreme points only)

STATISTICAL TESTING
-------------------
  Scott-Knott test matching the reference implementation:
    - Bootstrap permutation test  (p < 0.05)
    - Cliff's delta               (|d| > 0.35)
  Both must hold to split a cluster.  Tier 1 = best group.

  Applied PER DATASET on the 10 raw run values per method.
  (Same approach as scott_knott_dataset.py reference file.)

  Summary reports % Tier-1 across datasets  (= how often each method
  is in the best group), and median tier with IQR.

MOEA BUDGETS
------------
  EZR   : 200 and 1000 evals
  NSGA2 : 200 and 1000 evals
  SPEA2 : 200 and 1000 evals

USAGE
-----
    python rq3_frontier_comparison.py \\
        --results_dir results \\
        --moot_dir    moot/optimize \\
        --output_dir  output_frontier \\
        --runs        0,1,2,3,4,5,6,7,8,9
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist
import argparse
import warnings
import sys
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# pymoo — graceful fallback
# ---------------------------------------------------------------------------
try:
    from pymoo.indicators.hv import HV as _PymooHV
    PYMOO_HV = True
    def _hv_pymoo(pts, ref):
        return float(_PymooHV(ref_point=ref).do(pts))
except ImportError:
    PYMOO_HV = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EZR_BUDGET       = '200'
EZR_BUDGET_1000  = '1000'
MOEA_BUDGET_200  = '200'
MOEA_BUDGET_1000 = '1000'
ALL_RUNS         = [str(i) for i in range(10)]

MOEA_VARIANTS = [
    ('NSGA2', MOEA_BUDGET_200,  'nsga2_200'),
    ('NSGA2', MOEA_BUDGET_1000, 'nsga2_1000'),
    ('SPEA2', MOEA_BUDGET_200,  'spea2_200'),
    ('SPEA2', MOEA_BUDGET_1000, 'spea2_1000'),
]

# Six methods compared in SK
METHOD_KEYS = ['ezr', 'ezr_1000', 'nsga2_200', 'nsga2_1000', 'spea2_200', 'spea2_1000']

# Metrics on which SK is run per-dataset
SK_METRICS = [
    ('igd_true', False),   # (key, negate_for_sk)
    ('gd_true',  False),
    ('hv',       True),    # negate so tier-1 = highest HV
    ('sp_ne',    False),
]

# ---------------------------------------------------------------------------
# Scott-Knott  — bootstrap + Cliff's delta
# (matches reference implementation in scott_knott_dataset.py exactly)
# ---------------------------------------------------------------------------

class ScottKnott:
    def __init__(self, alpha=0.05, bootstrap_iters=1000,
                 cliff_delta_thresh=0.35):
        self.alpha              = alpha
        self.bootstrap_iters    = bootstrap_iters
        self.cliff_delta_thresh = cliff_delta_thresh

    @staticmethod
    def cliffs_delta(x, y):
        nx, ny = len(x), len(y)
        more = sum(1 for xi in x for yi in y if xi > yi)
        less = sum(1 for xi in x for yi in y if xi < yi)
        return abs((more - less) / (nx * ny))

    def _bootstrap_p(self, x, y):
        combined = np.concatenate([x, y])
        obs_diff = abs(np.mean(x) - np.mean(y))
        count    = 0
        for _ in range(self.bootstrap_iters):
            np.random.shuffle(combined)
            nx = len(x)
            if abs(np.mean(combined[:nx]) - np.mean(combined[nx:])) >= obs_diff:
                count += 1
        return count / self.bootstrap_iters

    def rank(self, data: Dict[str, List[float]]) -> Dict[str, float]:
        """
        data: {method: [run_values]}  lower = better
        Returns {method: tier}  tier 1 = best
        """
        clean = {k: np.array([v for v in vs if not np.isnan(v)])
                 for k, vs in data.items()}
        clean = {k: v for k, v in clean.items() if len(v) > 0}
        if len(clean) < 2:
            out = {k: 1.0 for k in clean}
            for k in data:
                out.setdefault(k, np.nan)
            return out
        names  = sorted(clean, key=lambda k: float(np.mean(clean[k])))
        groups = [clean[n] for n in names]
        tiers: Dict[str, float] = {}
        self._recurse(names, groups, tiers, 1)
        for k in data:
            tiers.setdefault(k, np.nan)
        return tiers

    def _recurse(self, names, groups, tiers, tier):
        if not names: return
        if len(names) == 1:
            tiers[names[0]] = float(tier); return
        ok, idx = self._best_split(groups)
        if ok:
            for n in names[:idx]: tiers[n] = float(tier)
            self._recurse(names[idx:], groups[idx:], tiers, tier + 1)
        else:
            for n in names: tiers[n] = float(tier)

    def _best_split(self, groups):
        if len(groups) < 2: return False, 0
        all_data = np.concatenate(groups)
        mu       = np.mean(all_data)
        best_i, best_bss = 1, -np.inf
        for i in range(1, len(groups)):
            L = np.concatenate(groups[:i])
            R = np.concatenate(groups[i:])
            bss = len(L)*(np.mean(L)-mu)**2 + len(R)*(np.mean(R)-mu)**2
            if bss > best_bss:
                best_bss, best_i = bss, i
        L = np.concatenate(groups[:best_i])
        R = np.concatenate(groups[best_i:])
        if len(L) < 2 or len(R) < 2: return False, best_i
        p     = self._bootstrap_p(L, R)
        delta = self.cliffs_delta(L, R)
        return (p < self.alpha and delta > self.cliff_delta_thresh), best_i

# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_dataset_csv(base_name, moot_dir):
    for p in Path(moot_dir).rglob(f"{base_name}.csv"):
        return p
    return None

def find_meta_file(meta_dir, base_name, budget, run_id):
    p = Path(meta_dir) / f"{base_name}_{budget}_{run_id}.json"
    if p.exists(): return p
    fs = list(Path(meta_dir).glob(f"{base_name}_{budget}_{run_id}*.json"))
    return fs[0] if fs else None

def discover_base_names(ezr_meta_dir, moot_dir, run_id='0'):
    base_names, suffix = set(), f"_{EZR_BUDGET}_{run_id}"
    for mf in Path(ezr_meta_dir).glob(f"*_{EZR_BUDGET}_{run_id}.json"):
        stem = mf.stem
        if stem.endswith(suffix):
            base_names.add(stem[:-len(suffix)])
    available = {csv.stem for csv in Path(moot_dir).rglob("*.csv")}
    matched   = base_names & available
    print(f"  EZR metadata files : {len(base_names)}")
    print(f"  CSVs in moot_dir   : {len(available)}")
    print(f"  Intersection       : {len(matched)}")
    return matched

# ---------------------------------------------------------------------------
# Objective directions
# ---------------------------------------------------------------------------

def get_objective_directions(base_name, moot_dir):
    csv_path = find_dataset_csv(base_name, moot_dir)
    if csv_path is None: return None
    try:
        df = pd.read_csv(csv_path, nrows=1)
    except Exception: return None
    y_cols = [c for c in df.columns
              if str(c).strip().endswith('+') or str(c).strip().endswith('-')]
    if not y_cols: return None
    return ['max' if c.strip().endswith('+') else 'min' for c in y_cols]

def infer_n_objectives(meta_file):
    with open(meta_file) as f:
        data = json.load(f)
    for e in data.get('evaluation_history', []):
        if e.get('objectives'):
            return len(e['objectives'])
    return None

# ---------------------------------------------------------------------------
# Non-domination
# ---------------------------------------------------------------------------

def non_dominated_mask(vecs):
    if len(vecs) == 0: return np.array([], dtype=bool)
    N = len(vecs)
    dominated = np.zeros(N, dtype=bool)
    for i in range(N):
        if dominated[i]: continue
        for j in range(N):
            if i == j or dominated[j]: continue
            if np.all(vecs[j] <= vecs[i]) and np.any(vecs[j] < vecs[i]):
                dominated[i] = True; break
    return ~dominated

def get_extreme_mask(vecs):
    extreme = np.zeros(len(vecs), dtype=bool)
    for i in range(vecs.shape[1]):
        extreme[np.argmin(vecs[:, i])] = True
    return extreme

# ---------------------------------------------------------------------------
# Performance indicators
# ---------------------------------------------------------------------------

def indicator_igd(approx, reference):
    if len(approx) == 0 or len(reference) == 0: return np.nan
    return float(cdist(reference, approx).min(axis=1).mean())

def indicator_gd(approx, reference):
    if len(approx) == 0 or len(reference) == 0: return np.nan
    return float(cdist(approx, reference).min(axis=1).mean())

def indicator_hv(vecs, ref_point):
    if len(vecs) == 0: return np.nan
    arr   = np.array(vecs, dtype=float)
    ref   = np.array(ref_point, dtype=float)
    valid = arr[np.all(arr < ref, axis=1)]
    if len(valid) == 0: return np.nan
    n_obj = valid.shape[1]

    if n_obj == 2:
        pts    = valid[np.argsort(valid[:, 0])]
        hv, py = 0.0, ref[1]
        for p in pts:
            if p[1] < py:
                hv += (ref[0] - p[0]) * (py - p[1]); py = p[1]
        return float(hv)

    if PYMOO_HV:
        try: return _hv_pymoo(valid, ref)
        except Exception: pass

    rng     = np.random.default_rng(42)
    samples = rng.uniform(np.zeros(n_obj), ref, size=(50_000, n_obj))
    dom     = np.zeros(50_000, dtype=bool)
    for p in valid:
        dom |= np.all(samples >= p, axis=1)
    return float(np.prod(ref) * dom.mean())

def indicator_spacing(vecs):
    if len(vecs) < 2: return np.nan
    d = cdist(vecs, vecs)
    np.fill_diagonal(d, np.inf)
    return float(np.std(d.min(axis=1)))

def indicator_mpd(vecs):
    if len(vecs) < 2: return np.nan
    d = cdist(vecs, vecs); n = len(vecs)
    return float(d[np.triu_indices(n, k=1)].mean())

def indicator_spread_delta(vecs, n_obj):
    if n_obj != 2 or len(vecs) < 3: return np.nan
    arr    = np.array(vecs)[np.argsort(np.array(vecs)[:, 0])]
    d_i    = np.sqrt((np.diff(arr, axis=0)**2).sum(axis=1))
    d_mean = d_i.mean()
    if d_mean < 1e-12: return np.nan
    d_f = float(np.linalg.norm(arr[0]  - arr[np.argmin(arr[:, 0])]))
    d_l = float(np.linalg.norm(arr[-1] - arr[np.argmin(arr[:, 1])]))
    return float((d_f + d_l + np.abs(d_i - d_mean).sum()) /
                 (d_f + d_l + len(d_i) * d_mean))

# ---------------------------------------------------------------------------
# True Pareto front from CSV
# ---------------------------------------------------------------------------

def extract_true_pareto_from_csv(base_name, moot_dir):
    csv_path = find_dataset_csv(base_name, moot_dir)
    if csv_path is None: return None, None, None
    try: df = pd.read_csv(csv_path)
    except Exception: return None, None, None
    y_cols = [c for c in df.columns
              if str(c).strip().endswith('+') or str(c).strip().endswith('-')]
    if not y_cols: return None, None, None
    directions = ['max' if c.strip().endswith('+') else 'min' for c in y_cols]
    try: raw = df[y_cols].dropna().values.astype(float)
    except Exception: return None, None, None
    if len(raw) == 0: return None, None, None
    rng = raw.max(0) - raw.min(0); rng[rng < 1e-12] = 1.0
    norm = (raw - raw.min(0)) / rng
    for i, d in enumerate(directions):
        if d == 'max': norm[:, i] = 1.0 - norm[:, i]
    mask = non_dominated_mask(norm); nd = norm[mask]
    if len(nd) == 0: return None, None, None
    ne = nd[~get_extreme_mask(nd)]
    return nd, ne, directions

def get_all_csv_points_normalized(base_name, moot_dir, directions):
    csv_path = find_dataset_csv(base_name, moot_dir)
    if csv_path is None: return None
    try: df = pd.read_csv(csv_path)
    except Exception: return None
    y_cols = [c for c in df.columns
              if str(c).strip().endswith('+') or str(c).strip().endswith('-')]
    if not y_cols: return None
    try: raw = df[y_cols].dropna().values.astype(float)
    except Exception: return None
    rng = raw.max(0) - raw.min(0); rng[rng < 1e-12] = 1.0
    norm = (raw - raw.min(0)) / rng
    for i, d in enumerate(directions):
        if d == 'max': norm[:, i] = 1.0 - norm[:, i]
    return norm

def concentration_analysis(tpf, all_pts):
    if tpf is None or len(tpf) == 0:
        return dict(pareto_fraction=np.nan, obj_range_ratio=np.nan,
                    pareto_centroid_d=np.nan, dataset_centroid_d=np.nan)
    pr = tpf.max(0)-tpf.min(0); dr = all_pts.max(0)-all_pts.min(0)
    dr[dr < 1e-12] = 1.0; c = all_pts.mean(0)
    return dict(
        pareto_fraction    = len(tpf) / len(all_pts),
        obj_range_ratio    = float((pr/dr).mean()),
        pareto_centroid_d  = float(np.linalg.norm(tpf - c,     axis=1).mean()),
        dataset_centroid_d = float(np.linalg.norm(all_pts - c, axis=1).mean()),
    )

# ---------------------------------------------------------------------------
# Frontier extraction from metadata
# ---------------------------------------------------------------------------

def extract_frontier(meta_file, n_obj, directions, is_ezr=False):
    with open(meta_file) as f:
        data = json.load(f)
    seen, vecs = set(), []
    for e in data.get('evaluation_history', []):
        raw = e.get('objectives', [])
        if len(raw) != n_obj: continue
        key = tuple(round(float(v), 10) for v in raw)
        if key in seen: continue
        seen.add(key)
        vec = []
        for i, v in enumerate(raw):
            v = float(v)
            if is_ezr and directions[i] == 'max': v = 1.0 - v
            vec.append(v)
        vecs.append(vec)
    if not vecs: return np.empty((0, n_obj)), np.empty((0, n_obj))
    arr = np.array(vecs)

    arr_min, arr_max = arr.min(), arr.max()
    if arr_min < -0.01 or arr_max > 1.01:
        label = "EZR" if is_ezr else "MOEA"
        warnings.warn(
            f"[{label}] {Path(meta_file).name}: objectives outside [0,1] "
            f"(min={arr_min:.4f}, max={arr_max:.4f}). "
            f"Normalization mismatch — IGD/GD/HV results may be invalid.",
            stacklevel=2)

    mask = non_dominated_mask(arr); nd = arr[mask]
    if len(nd) == 0: return np.empty((0, n_obj)), np.empty((0, n_obj))
    ne = nd[~get_extreme_mask(nd)]
    return nd, ne

# ---------------------------------------------------------------------------
# Per-run metrics
# ---------------------------------------------------------------------------

def compute_run_metrics(ezr_full, ezr_ne, moea_full, moea_ne,
                        n_obj, tpf):
    ref = np.ones(n_obj) * 1.1
    def er(full):
        return np.nan if len(full) == 0 else min(n_obj, len(full)) / len(full)

    r = dict(
        ezr_full_size=len(ezr_full),    ezr_ne_size=len(ezr_ne),
        moea_full_size=len(moea_full),  moea_ne_size=len(moea_ne),
        ezr_extreme_ratio=er(ezr_full), moea_extreme_ratio=er(moea_full),
        ezr_igd_true  = indicator_igd(ezr_full,  tpf) if tpf is not None else np.nan,
        moea_igd_true = indicator_igd(moea_full, tpf) if tpf is not None else np.nan,
        ezr_gd_true   = indicator_gd(ezr_full,  tpf) if tpf is not None else np.nan,
        moea_gd_true  = indicator_gd(moea_full, tpf) if tpf is not None else np.nan,
        igd_vs_moea   = indicator_igd(ezr_full, moea_full),
        ezr_hv        = indicator_hv(ezr_full,  ref),
        moea_hv       = indicator_hv(moea_full, ref),
        ezr_spread    = np.nan,
        moea_spread   = np.nan,
        ezr_mpd_ne    = indicator_mpd(ezr_ne),
        moea_mpd_ne   = indicator_mpd(moea_ne),
        ezr_sp_ne     = indicator_spacing(ezr_ne),
        moea_sp_ne    = indicator_spacing(moea_ne),
    )
    return r

# ---------------------------------------------------------------------------
# Scott-Knott per dataset
# ---------------------------------------------------------------------------

def sk_for_dataset(variant_runs: Dict[str, List[dict]],
                   ezr_run_metrics: List[dict],
                   metric_key: str,
                   negate: bool = False) -> Dict[str, float]:

    sk_input: Dict[str, List[float]] = {}

    # EZR-200 independent collection
    ezr_vals = [r.get(f'ezr_{metric_key}', np.nan) for r in ezr_run_metrics]
    ezr_vals = [v for v in ezr_vals if not np.isnan(v)]
    if ezr_vals:
        sk_input['ezr'] = [-v for v in ezr_vals] if negate else ezr_vals

    # EZR-1000 independent collection
    ezr_1000_vals = [r.get(f'ezr_1000_{metric_key}', np.nan) for r in ezr_run_metrics]
    ezr_1000_vals = [v for v in ezr_1000_vals if not np.isnan(v)]
    if ezr_1000_vals:
        sk_input['ezr_1000'] = [-v for v in ezr_1000_vals] if negate else ezr_1000_vals

    # MOEAs
    for _, _, prefix in MOEA_VARIANTS:
        run_list = variant_runs.get(prefix, [])
        vals = [r.get(f'moea_{metric_key}', np.nan) for r in run_list]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            sk_input[prefix] = [-v for v in vals] if negate else vals

    if len(sk_input) < 2:
        return {m: np.nan for m in METHOD_KEYS}

    sk = ScottKnott(alpha=0.05, bootstrap_iters=1000, cliff_delta_thresh=0.35)
    return sk.rank(sk_input)

# ---------------------------------------------------------------------------
# Per-dataset analysis
# ---------------------------------------------------------------------------

def nobj_group(n):
    return '2' if n == 2 else '3' if n == 3 else '4+'


def analyze_dataset(base_name, results_dir, moot_dir, runs,
                    skipped_log, skip_reasons):
    ezr_meta_dir  = Path(results_dir) / "results_EZR"   / "EZR"   / "metadata"
    moea_dirs = {
        'NSGA2': Path(results_dir) / "results_NSGA2" / "NSGA2" / "metadata",
        'SPEA2': Path(results_dir) / "results_SPEA2" / "SPEA2" / "metadata",
    }

    ezr_meta_0 = find_meta_file(ezr_meta_dir, base_name, EZR_BUDGET, runs[0])
    if not ezr_meta_0:
        skip_reasons[base_name] = 'no EZR meta file'; return None
    n_obj = infer_n_objectives(ezr_meta_0)
    if n_obj is None or n_obj < 2:
        skip_reasons[base_name] = f'n_obj={n_obj}'; return None
    directions = get_objective_directions(base_name, moot_dir)
    if directions is None or len(directions) != n_obj:
        skip_reasons[base_name] = 'directions mismatch'; return None

    tpf, tpne, _ = extract_true_pareto_from_csv(base_name, moot_dir)
    all_pts      = get_all_csv_points_normalized(base_name, moot_dir, directions)
    conc         = (concentration_analysis(tpf, all_pts)
                    if all_pts is not None
                    else dict(pareto_fraction=np.nan, obj_range_ratio=np.nan,
                              pareto_centroid_d=np.nan, dataset_centroid_d=np.nan))

    variant_runs: Dict[str, List[dict]] = {p: [] for _, _, p in MOEA_VARIANTS}
    ezr_run_metrics: List[dict] = []

    for run_id in runs:
        ezr_meta = find_meta_file(ezr_meta_dir, base_name, EZR_BUDGET, run_id)
        if not ezr_meta: continue
        ezr_full, ezr_ne = extract_frontier(ezr_meta, n_obj, directions, is_ezr=True)

        if len(ezr_full) == 0:
            for algo, budget, prefix in MOEA_VARIANTS:
                mm = find_meta_file(moea_dirs[algo], base_name, budget, run_id)
                if mm:
                    mf, _ = extract_frontier(mm, n_obj, directions)
                    skipped_log.append(dict(
                        dataset=base_name, moea=f'{algo}_{budget}',
                        run_id=run_id, n_obj=n_obj,
                        ezr_full_size=0, moea_full_size=len(mf)))
            continue

        # Collect EZR-200 standalone metrics
        ref = np.ones(n_obj) * 1.1
        ezr_standalone = dict(
            ezr_igd_true = indicator_igd(ezr_full, tpf) if tpf is not None else np.nan,
            ezr_gd_true  = indicator_gd(ezr_full,  tpf) if tpf is not None else np.nan,
            ezr_hv       = indicator_hv(ezr_full,  ref),
            ezr_sp_ne    = indicator_spacing(ezr_ne),
        )

        # Collect EZR-1000 standalone metrics if available
        ezr_1000_meta = find_meta_file(ezr_meta_dir, base_name, EZR_BUDGET_1000, run_id)
        if ezr_1000_meta:
            ezr_1000_full, ezr_1000_ne = extract_frontier(
                ezr_1000_meta, n_obj, directions, is_ezr=True)
            if len(ezr_1000_full) > 0:
                ezr_standalone.update(dict(
                    ezr_1000_igd_true = indicator_igd(ezr_1000_full, tpf) if tpf is not None else np.nan,
                    ezr_1000_gd_true  = indicator_gd(ezr_1000_full,  tpf) if tpf is not None else np.nan,
                    ezr_1000_hv       = indicator_hv(ezr_1000_full,  ref),
                    ezr_1000_sp_ne    = indicator_spacing(ezr_1000_ne),
                ))
            else:
                ezr_standalone.update(dict(
                    ezr_1000_igd_true=np.nan,
                    ezr_1000_gd_true=np.nan,
                    ezr_1000_hv=np.nan,
                    ezr_1000_sp_ne=np.nan,
                ))
        else:
            ezr_standalone.update(dict(
                ezr_1000_igd_true=np.nan,
                ezr_1000_gd_true=np.nan,
                ezr_1000_hv=np.nan,
                ezr_1000_sp_ne=np.nan,
            ))

        ezr_run_metrics.append(ezr_standalone)

        for algo, budget, prefix in MOEA_VARIANTS:
            mm = find_meta_file(moea_dirs[algo], base_name, budget, run_id)
            if not mm: continue
            moea_full, moea_ne = extract_frontier(mm, n_obj, directions)
            if len(moea_full) == 0: continue
            variant_runs[prefix].append(
                compute_run_metrics(ezr_full, ezr_ne, moea_full, moea_ne,
                                    n_obj, tpf))

    if all(len(v) == 0 for v in variant_runs.values()):
        return None

    # SK: run once per metric per dataset
    sk_tiers: Dict[str, Dict[str, float]] = {}
    for metric_key, negate in SK_METRICS:
        sk_tiers[metric_key] = sk_for_dataset(
            variant_runs, ezr_run_metrics, metric_key, negate)

    def aggregate(run_list, prefix):
        raw_keys = ['ezr_full_size', 'ezr_ne_size', 'moea_full_size', 'moea_ne_size',
                    'ezr_extreme_ratio', 'moea_extreme_ratio',
                    'ezr_igd_true', 'moea_igd_true',
                    'ezr_gd_true',  'moea_gd_true',
                    'igd_vs_moea',
                    'ezr_hv', 'moea_hv',
                    'ezr_spread', 'moea_spread',
                    'ezr_mpd_ne', 'moea_mpd_ne',
                    'ezr_sp_ne',  'moea_sp_ne']
        if not run_list:
            out = {f'{prefix}_{k}': np.nan for k in raw_keys}
        else:
            def med(k):
                vs = [r[k] for r in run_list if not np.isnan(r.get(k, np.nan))]
                return float(np.median(vs)) if vs else np.nan
            out = {f'{prefix}_{k}': med(k) for k in raw_keys}

        for metric_key, _ in SK_METRICS:
            tiers = sk_tiers.get(metric_key, {})
            out[f'{prefix}_sk_{metric_key}_ezr']      = tiers.get('ezr',      np.nan)
            out[f'{prefix}_sk_{metric_key}_ezr_1000'] = tiers.get('ezr_1000', np.nan)
            out[f'{prefix}_sk_{metric_key}_moea']     = tiers.get(prefix,     np.nan)
        return out

    result = dict(
        dataset=base_name, n_objectives=n_obj, nobj_group=nobj_group(n_obj),
        true_pareto_size    = len(tpf)  if tpf  is not None else np.nan,
        true_pareto_ne_size = len(tpne) if tpne is not None else np.nan,
        **conc,
    )
    for _, _, prefix in MOEA_VARIANTS:
        result.update(aggregate(variant_runs[prefix], prefix))

    rep = 'nsga2_200'
    igd = result.get(f'{rep}_ezr_igd_true', np.nan)
    sk_e   = result.get(f'{rep}_sk_igd_true_ezr',      np.nan)
    sk_e1k = result.get(f'{rep}_sk_igd_true_ezr_1000', np.nan)
    sk_m   = result.get(f'{rep}_sk_igd_true_moea',     np.nan)
    print(f"  {base_name:<40} n_obj={n_obj} "
          f"IGD_ezr={igd:.4f} "
          f"SK_igd(ezr={sk_e!s}, ezr_1000={sk_e1k!s}, nsga2_200={sk_m!s})")
    return result

# ---------------------------------------------------------------------------
# Summary with SK tables
# ---------------------------------------------------------------------------

def sk_summary_table(df, label, mask=None) -> List[str]:
    sub = df if mask is None else df[mask]
    if len(sub) == 0: return []

    METHOD_LABELS = {
        'ezr':        'EZR-200',
        'ezr_1000':   'EZR-1000',
        'nsga2_200':  'NSGA2-200',
        'nsga2_1000': 'NSGA2-1000',
        'spea2_200':  'SPEA2-200',
        'spea2_1000': 'SPEA2-1000',
    }
    METRIC_LABELS = {
        'igd_true': 'True IGD  (lower=better)',
        'gd_true':  'True GD   (lower=better)',
        'hv':       'HV        (higher=better)',
        'sp_ne':    'Spacing SP(lower=better)',
    }

    lines = []
    lines.append(f"\n  -- Scott-Knott Rankings -- {label}  (n={len(sub)}) --")
    lines.append(f"  SK: bootstrap p<0.05 AND Cliff's |delta|>0.35  per dataset on 10 runs")
    lines.append(f"  Tier 1 = best group.  Values = median tier [Q1,Q3] | % Tier-1 datasets")
    lines.append("")

    col_w = 22
    header = f"  {'Metric':<26}" + "".join(f"{v:>{col_w}}" for v in METHOD_LABELS.values())
    lines.append(header)
    lines.append("  " + "-" * (26 + col_w * len(METHOD_LABELS)))

    for metric_key, _ in SK_METRICS:
        mlabel = METRIC_LABELS[metric_key]
        row = f"  {mlabel:<26}"
        for method_key in METHOD_LABELS:
            if method_key == 'ezr':
                col = f'nsga2_200_sk_{metric_key}_ezr'
            elif method_key == 'ezr_1000':
                col = f'nsga2_200_sk_{metric_key}_ezr_1000'
            else:
                col = f'{method_key}_sk_{metric_key}_moea'
            vals = sub[col].dropna() if col in sub.columns else pd.Series([], dtype=float)
            if len(vals) == 0:
                row += f"{'n/a':>{col_w}}"
            else:
                med   = np.median(vals)
                q1    = np.percentile(vals, 25)
                q3    = np.percentile(vals, 75)
                t1pct = (vals == 1).mean() * 100
                cell  = f"T{med:.0f}[{q1:.0f},{q3:.0f}] {t1pct:.0f}%"
                row  += f"{cell:>{col_w}}"
        lines.append(row)

    lines.append("  " + "-" * (26 + col_w * len(METHOD_LABELS)))
    lines.append("  T1=Tier1 (best).  [Q1,Q3]=IQR of tiers.  %=fraction of datasets in Tier-1.")
    return lines


def summarize(df, label, mask=None) -> List[str]:
    sub = df if mask is None else df[mask]
    if len(sub) == 0: return []
    lines = []
    lines.append(f"\n  {'='*68}")
    lines.append(f"  {label}  (n={len(sub)} datasets)")
    lines.append(f"  {'='*68}")

    def stat(col, desc):
        vals = sub[col].dropna()
        if len(vals) == 0: return
        lines.append(f"  {desc:<60}: "
                     f"{np.median(vals):.4f}  "
                     f"[{np.percentile(vals,25):.4f}, "
                     f"{np.percentile(vals,75):.4f}]")

    def pct(col, desc, thr):
        vals = sub[col].dropna()
        if len(vals) == 0: return
        lines.append(f"  {desc:<60}: {(vals < thr).mean()*100:.1f}%")

    for algo, budget, prefix in MOEA_VARIANTS:
        mlabel = f"{algo}-{budget}"
        lines.append(f"\n    {'─'*66}")
        lines.append(f"    EZR-200 vs {mlabel}:")
        lines.append(f"    {'─'*66}")

        lines.append(f"\n    -- Frontier sizes (median [Q1,Q3]) --")
        stat(f'{prefix}_ezr_full_size',  f'    EZR-200 full frontier size')
        stat(f'{prefix}_moea_full_size', f'    {mlabel} full frontier size')

        lines.append(f"\n    -- Extreme ratio --")
        stat(f'{prefix}_ezr_extreme_ratio',  f'    EZR-200')
        stat(f'{prefix}_moea_extreme_ratio', f'    {mlabel}')

        lines.append(f"\n    -- TRUE IGD  [lower=better] --")
        stat(f'{prefix}_ezr_igd_true',  f'    EZR-200')
        stat(f'{prefix}_moea_igd_true', f'    {mlabel}')
        pct(f'{prefix}_ezr_igd_true',   f'    EZR-200  IGD<0.05', 0.05)
        pct(f'{prefix}_moea_igd_true',  f'    {mlabel} IGD<0.05', 0.05)

        lines.append(f"\n    -- TRUE GD   [lower=better] --")
        stat(f'{prefix}_ezr_gd_true',  f'    EZR-200')
        stat(f'{prefix}_moea_gd_true', f'    {mlabel}')

        lines.append(f"\n    -- HV  [higher=better] --")
        stat(f'{prefix}_ezr_hv',  f'    EZR-200')
        stat(f'{prefix}_moea_hv', f'    {mlabel}')

        lines.append(f"\n    -- SP Spacing [lower=better] --")
        stat(f'{prefix}_ezr_sp_ne',  f'    EZR-200')
        stat(f'{prefix}_moea_sp_ne', f'    {mlabel}')

    lines.append(f"\n    {'─'*66}")
    lines.append(f"    BUDGET EFFECT (same algorithm, 200 vs 1000 evals):")
    for algo in ['NSGA2', 'SPEA2']:
        p200, p1000 = f'{algo.lower()}_200', f'{algo.lower()}_1000'
        lines.append(f"\n    {algo}:")
        for metric, desc in [('moea_full_size','Frontier size'),
                              ('moea_igd_true', 'True IGD'),
                              ('moea_gd_true',  'True GD'),
                              ('moea_hv',       'HV')]:
            v2  = sub[f'{p200}_{metric}'].dropna()
            v10 = sub[f'{p1000}_{metric}'].dropna()
            if len(v2) and len(v10):
                lines.append(f"      {desc:<20}: "
                              f"200={np.median(v2):.4f}  "
                              f"1000={np.median(v10):.4f}")

    lines.append(f"\n    -- True Pareto front --")
    stat('true_pareto_size', '    True Pareto full size')
    lines.append(f"\n    -- Concentration --")
    stat('pareto_fraction',   '    Pareto fraction')
    stat('obj_range_ratio',   '    Obj range ratio')
    stat('pareto_centroid_d', '    Pareto -> centroid dist')

    lines += sk_summary_table(df, label, mask)
    return lines


def print_skipped_summary(skipped_log):
    if not skipped_log: print("\nNo skipped datasets."); return
    print(f"\n{'='*70}")
    print("SKIPPED DATASET DIAGNOSIS"); print(f"{'='*70}")
    seen = {}
    for r in skipped_log:
        key = (r['dataset'], r['moea'])
        seen.setdefault(key, r)
    both = ezr_only = 0
    for (ds, moea), r in sorted(seen.items()):
        empty = r['moea_full_size'] == 0
        if empty: both += 1
        else: ezr_only += 1
        print(f"  [{'BOTH' if empty else 'EZR-ONLY'}] {ds} moea={moea} "
              f"EZR={r['ezr_full_size']} MOEA={r['moea_full_size']}")
    print(f"\n  Both empty: {both}   EZR only: {ezr_only}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--moot_dir",    default="moot/optimize")
    parser.add_argument("--output_dir",  default="output_frontier")
    parser.add_argument("--runs",        default=",".join(ALL_RUNS))
    args = parser.parse_args()

    runs = [r.strip() for r in args.runs.split(",")]
    out  = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    results_csv = out / "rq3_frontier_results.csv"
    summary_txt = out / "rq3_frontier_summary.txt"

    ezr_meta_dir = Path(args.results_dir) / "results_EZR" / "EZR" / "metadata"
    if not ezr_meta_dir.exists():
        print(f"Error: {ezr_meta_dir} not found"); sys.exit(1)

    hv_backend = "pymoo exact WFG" if PYMOO_HV else "Monte Carlo n=50k"
    print(f"HV backend    : {hv_backend}")
    print(f"SK test       : bootstrap (n=1000) + Cliff's delta (threshold=0.35)")
    print(f"SK input      : 10 raw run values per method PER DATASET")
    print(f"EZR variants  : EZR-200, EZR-1000")
    print(f"MOEA variants : " + ", ".join(f"{a}-{b}" for a,b,_ in MOEA_VARIANTS))
    print()

    print("Discovering datasets...")
    base_names = discover_base_names(ezr_meta_dir, args.moot_dir, run_id=runs[0])
    print(f"Found {len(base_names)} datasets\n")

    all_rows, failed, skipped_log, skip_reasons = [], [], [], {}
    for base_name in sorted(base_names):
        result = analyze_dataset(base_name, args.results_dir, args.moot_dir,
                                 runs, skipped_log, skip_reasons)
        if result: all_rows.append(result)
        else:      failed.append(base_name)

    print(f"\n{'='*70}")
    print(f"Completed: {len(all_rows)}  |  Skipped: {len(failed)}")
    print_skipped_summary(skipped_log)
    if not all_rows: print("No results."); sys.exit(0)

    df = pd.DataFrame(all_rows)
    df.to_csv(results_csv, index=False)

    header = [
        '=' * 70,
        f"RQ3 EZR vs MOEA  ({len(all_rows)} datasets)",
        f"EZR  : 200 and 1000 evals",
        f"MOEAs: NSGA2-200, NSGA2-1000, SPEA2-200, SPEA2-1000",
        f"HV   : {hv_backend}",
        f"SK   : bootstrap (1000 perms) + Cliff's |delta|>0.35, alpha=0.05",
        f"SK   : applied per dataset on 10 raw run values (NOT on medians)",
        '=' * 70,
    ]
    body = []
    body += summarize(df, 'OVERALL')
    for grp in ['2', '3', '4+']:
        body += summarize(df, f'n_obj = {grp}', mask=(df['nobj_group'] == grp))
    body += [
        '\n', 'HOW TO READ', '-'*70,
        'IGD(A,Z) = mean_i min_j ||z_i-a_j||  Z=true CSV Pareto front  lower=better',
        'GD(A,Z)  = mean_i min_j ||a_i-z_j||  same Z                   lower=better',
        'HV       = dominated volume vs ref=(1.1,...,1.1)               higher=better',
        'SP       = std of nearest-neighbour dists (non-extreme)        lower=better',
        '',
        'Scott-Knott:',
        '  Run per dataset on the 10 raw run values per method.',
        '  Splits only when bootstrap p<0.05 AND Cliff |delta|>0.35.',
        '  Tier 1 = best group.  Same tier = no meaningful difference.',
        '  Summary table shows: median tier [Q1,Q3] | % datasets in Tier-1.',
        '',
        'NOTE: All objectives normalised to [0,1] minimisation space.',
    ]
    text = '\n'.join(header + body)
    with open(summary_txt, 'w') as f: f.write(text)
    print(text)
    print(f"\nOutputs: {out}/")
    print(f"  {results_csv.name}")
    print(f"  {summary_txt.name}")


if __name__ == "__main__":
    main()