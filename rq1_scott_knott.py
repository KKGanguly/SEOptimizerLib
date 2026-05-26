import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Runtime column names to look for (in priority order)
RUNTIME_COLS = ['runtime', 'time', 'wall_time', 'elapsed', 'duration']

# ----------------- Scott-Knott Class -----------------
class ScottKnott:
    def __init__(self, alpha=0.05, bootstrap_iters=1000, cliff_delta_thresh=0.35):
        self.alpha = alpha
        self.bootstrap_iters = bootstrap_iters
        self.cliff_delta_thresh = cliff_delta_thresh

    @staticmethod
    def cliffs_delta(x, y):
        nx, ny = len(x), len(y)
        more = sum(1 for xi in x for yi in y if xi > yi)
        less = sum(1 for xi in x for yi in y if xi < yi)
        delta = (more - less) / (nx * ny)
        return abs(delta)

    @staticmethod
    def bootstrap(x, y, iters=1000):
        combined = np.concatenate([x, y])
        obs_diff = np.mean(x) - np.mean(y)
        count = 0
        for _ in range(iters):
            np.random.shuffle(combined)
            new_x = combined[:len(x)]
            new_y = combined[len(x):]
            new_diff = np.mean(new_x) - np.mean(new_y)
            if abs(new_diff) >= abs(obs_diff):
                count += 1
        return count / iters

    def _cluster_split(self, groups: List[np.ndarray]) -> Tuple[bool, int]:
        if len(groups) < 2: return False, 0
        all_data = np.concatenate(groups)
        overall_mean = np.mean(all_data)
        best_split, best_bss = 1, -np.inf
        for i in range(1, len(groups)):
            left, right = np.concatenate(groups[:i]), np.concatenate(groups[i:])
            bss = (len(left) * (np.mean(left) - overall_mean) ** 2
                   + len(right) * (np.mean(right) - overall_mean) ** 2)
            if bss > best_bss:
                best_bss, best_split = bss, i
        left, right = np.concatenate(groups[:best_split]), np.concatenate(groups[best_split:])
        if len(left) < 2 or len(right) < 2: return False, best_split
        p_value = self.bootstrap(left, right, iters=self.bootstrap_iters)
        delta = self.cliffs_delta(left, right)
        return (p_value < self.alpha and delta > self.cliff_delta_thresh), best_split

    def rank(self, data_dict: Dict[str, List[float]]) -> Dict[str, int]:
        sorted_names = sorted(data_dict.keys(), key=lambda k: np.mean(data_dict[k]))
        sorted_groups = [np.array(data_dict[name]) for name in sorted_names]
        tiers = {}
        self._recursive_rank(sorted_names, sorted_groups, tiers, tier=1)
        return tiers

    def _recursive_rank(self, names, groups, tiers, tier):
        if not names: return
        if len(names) == 1:
            tiers[names[0]] = tier
            return
        can_split, split_idx = self._cluster_split(groups)
        if can_split:
            for name in names[:split_idx]: tiers[name] = tier
            self._recursive_rank(names[split_idx:], groups[split_idx:], tiers, tier + 1)
        else:
            for name in names: tiers[name] = tier


# ----------------- Helper Functions -----------------
def get_optimizer_dir(results_dir: Path, optimizer_name: str) -> Path:
    folder_map = {
        'GA': 'GA',
        'ILS': 'ILS',
        'TS': 'TS',
        'ONEPLUSONE': 'ONEPLUSONE',
        'HILL': 'HILL',
        'RandomSearch': 'RandomSearch',
        'EZR': 'EZR',
        'DE' : 'DE',
        'SA' : 'SA'
    }

    folder_name = folder_map.get(optimizer_name.split('-')[0], optimizer_name.lower())
    return results_dir / f'results_{folder_name}' / folder_name


def find_runtime_col(df: pd.DataFrame) -> str | None:
    """Return the first matching runtime column name found in df, or None."""
    df_cols_lower = {c.lower(): c for c in df.columns}
    for candidate in RUNTIME_COLS:
        if candidate in df_cols_lower:
            return df_cols_lower[candidate]
    return None


def load_results_for_budgets(
    results_dir: Path,
    optimizer_name: str,
    dataset: str,
    budgets: List[int],
) -> Dict[int, Dict]:
    """
    Returns {budget: {'values': [...], 'runtime': [...] or None}} for the
    given optimizer and dataset. `runtime` is None if no runtime column exists.
    """
    optimizer_dir = get_optimizer_dir(results_dir, optimizer_name)
    if not optimizer_dir.exists():
        return {}

    result_dict = {}
    for budget in budgets:
        csv_file = optimizer_dir / f'{dataset}_{budget}.csv'
        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file)
                if 'best_value' not in df.columns:
                    continue
                entry = {'values': df['best_value'].tolist(), 'runtime': None}
                rt_col = find_runtime_col(df)
                if rt_col is not None:
                    entry['runtime'] = df[rt_col].dropna().tolist()
                result_dict[budget] = entry
            except Exception:
                continue
    return result_dict


def get_all_datasets(results_dir: Path, optimizer: str = 'EZR') -> List[str]:
    datasets = set()
    optimizer_dir = get_optimizer_dir(results_dir, optimizer)
    if optimizer_dir.exists():
        for csv_file in optimizer_dir.glob('*_*.csv'):
            datasets.add('_'.join(csv_file.stem.split('_')[:-1]))
    return sorted(datasets)


def get_dataset_size(dataset_name: str, base_moot_dir: str = "moot/optimize") -> int:
    base_path = Path(base_moot_dir)
    if not base_path.exists(): return "N/A"
    files = list(base_path.rglob(f"{dataset_name}.csv"))
    if not files:
        files = [f for f in base_path.rglob("*.csv")
                 if f.stem.lower() == dataset_name.lower()]
    if files:
        try: return pd.read_csv(files[0]).shape[0]
        except: return "Error"
    return "N/A"


# ----------------- Main Table Generator -----------------
def generate_table(
    results_dir: Path,
    optimizers: List[str],
    optimizer_budgets: Dict[str, List[int]],
    output_file: str = None,
):
    datasets = get_all_datasets(results_dir, 'HILL')
    print("DATASETS FOUND:", datasets[:10], "TOTAL:", len(datasets))
    results_data = []
    sk = ScottKnott(alpha=0.05)
    skipped = 0

    print(f"Processing {len(datasets)} datasets...")

    for dataset in datasets:

        # ── Load all results upfront ───────────────────────────────────────
        per_opt_results: Dict[str, Dict[int, Dict]] = {}
        for opt in optimizers:
            budgets = optimizer_budgets.get(opt, [])
            per_opt_results[opt] = load_results_for_budgets(
                results_dir, opt, dataset, budgets
            )

        # ── Skip dataset if ANY optimizer has no results at all ────────────
        missing = [opt for opt in optimizers if not per_opt_results[opt]]
        if missing:
            print(f"  Skipping '{dataset}' — missing results for: {', '.join(missing)}")
            skipped += 1
            continue

        # ── Build row ──────────────────────────────────────────────────────
        rows = get_dataset_size(dataset)
        row = {'Dataset': dataset, 'Rows': rows}
        optimizer_results: Dict[str, List[float]] = {}   # for Scott-Knott
        all_opt_keys: List[str] = []

        for opt in optimizers:
            for budget, entry in per_opt_results[opt].items():
                key = f"{opt}-{budget}"
                values = entry['values']
                runtime = entry['runtime']

                optimizer_results[key] = values
                all_opt_keys.append(key)

                # d2h mean and std
                row[f'{key}_mean'] = np.mean(values)
                row[f'{key}_std']  = np.std(values, ddof=1) if len(values) > 1 else 0.0
                row[f'{key}_budget'] = budget

                # runtime mean and std (empty string if not available)
                if runtime and len(runtime) > 0:
                    row[f'{key}_runtime_mean'] = np.mean(runtime)
                    row[f'{key}_runtime_std']  = (
                        np.std(runtime, ddof=1) if len(runtime) > 1 else 0.0
                    )
                else:
                    row[f'{key}_runtime_mean'] = ""
                    row[f'{key}_runtime_std']  = ""

        # ── Scott-Knott ranking ────────────────────────────────────────────
        tiers = sk.rank(optimizer_results)
        for key, tier in tiers.items():
            row[f'{key}_tier'] = tier
        for key in all_opt_keys:
            if f'{key}_tier' not in row:
                row[f'{key}_tier'] = ""

        results_data.append(row)

    print(f"\nDone. Included {len(results_data)} datasets, skipped {skipped}.")

    # ── Build DataFrame with ordered columns ───────────────────────────────
    # Column order per optimizer/budget:
    #   <key>_mean, <key>_std, <key>_budget,
    #   <key>_runtime_mean, <key>_runtime_std, <key>_tier
    df = pd.DataFrame(results_data)
    ordered_cols = ['Dataset', 'Rows']
    suffixes = ['_mean', '_std', '_budget', '_runtime_mean', '_runtime_std', '_tier']
    for opt in optimizers:
        for col in df.columns:
            # find all keys belonging to this optimizer
            if not col.startswith(opt + '-'):
                continue
            # extract base key (everything before the last suffix)
            for sfx in suffixes:
                if col.endswith(sfx):
                    key = col[: -len(sfx)]
                    for s in suffixes:
                        candidate = key + s
                        if candidate in df.columns and candidate not in ordered_cols:
                            ordered_cols.append(candidate)
                    break

    df = df[[c for c in ordered_cols if c in df.columns]]

    # Stringify for Google Sheets compatibility
    for col in df.columns:
        if col not in ('Dataset', 'Rows'):
            df[col] = df[col].apply(lambda x: "" if x == "" else str(x))

    print("\n" + "=" * 140)
    print(df.head(10).to_string(index=False))
    print("=" * 140)

    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

    return df


# ----------------- Main -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--output', type=str, default='scott_knott_dataset.csv')
    parser.add_argument('--optimizers', type=str, nargs='+',
                        default=['EZR', 'MOSMAC', 'NSGA2', 'SPEA2', 'Random'])
    parser.add_argument('--budgets', type=str, nargs='+', default=[],
                        help='e.g.: EZR:200,1000 Random:200,1000')
    args = parser.parse_args()

    optimizer_budgets = {}
    for b in args.budgets:
        try:
            opt_name, budget_str = b.split(':')
            optimizer_budgets[opt_name] = [int(x) for x in budget_str.split(',')]
        except Exception:
            continue

    generate_table(Path(args.results_dir), args.optimizers, optimizer_budgets, args.output)


if __name__ == '__main__':
    main()