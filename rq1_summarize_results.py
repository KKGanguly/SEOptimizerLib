import pandas as pd
import numpy as np
import argparse


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DISPLAY_NAMES = {
    'EZR-50':     'EZR-50',
    'EZR-100':    'EZR-100',
    'EZR-200':    'EZR-200',
    'EZR-1000':    'EZR-1000',
    'MOSMAC-200':  'SMAC-200',
    'MOSMAC-1000': 'SMAC-1000',
    'NSGA2-200':  'NSGA-II-200',
    'NSGA2-1000':  'NSGA-II-1000',
    'SPEA2-200':  'SPEA2-200',
    'SPEA2-1000':  'SPEA2-1000',
    'Random-200':  'Random-200',
    'Random-1000': 'Random-1000',
}

BUDGET = {
    'EZR-50': 50,   'EZR-100': 100,  'EZR-200': 200,'EZR-1000':  1000,
    'MOSMAC-200': 200,  'MOSMAC-1000': 1000,
    'NSGA2-200': 200, 'NSGA2-1000': 1000, 'SPEA2-200':  200,'SPEA2-1000': 1000,
    'Random-200': 200,  'Random-1000': 1000,
}

SUMMARY_KEYS = [
    'EZR-50', 'EZR-100', 'EZR-200','EZR-1000',
    'MOSMAC-200', 'MOSMAC-1000','NSGA2-200',
    'NSGA2-1000', 'SPEA2-200','SPEA2-1000',
    'Random-200', 'Random-1000',
]

ALL_OTHERS = [
    'MOSMAC-200', 'MOSMAC-1000',
    'NSGA2-200', 'SPEA2-200',
    'NSGA2-1000', 'SPEA2-1000',
    'Random-200', 'Random-1000',
]

NON_EZR = [
    'MOSMAC-200', 'MOSMAC-1000',
    'Random-200', 'Random-1000',
    'NSGA2-200', 'SPEA2-200',
    'NSGA2-1000', 'SPEA2-1000',
]

WTL_SECTIONS = [
    ('EZR-50  vs all',         'EZR-50',     ALL_OTHERS),
    ('EZR-100 vs all',         'EZR-100',    ALL_OTHERS),
    ('EZR-200 vs all',         'EZR-200',    ALL_OTHERS),
    ('Random-200  vs non-EZR', 'Random-200',  NON_EZR),
    ('Random-1000 vs non-EZR', 'Random-1000', NON_EZR),
]


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for c in df.columns:
        if any(c.endswith(s) for s in (
            '_tier', '_mean', '_std', '_runtime_mean', '_runtime_std'
        )):
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Table 1 – Summary stats
# ─────────────────────────────────────────────────────────────────────────────

def format_runtime(seconds: float) -> str:
    if pd.isna(seconds):
        return '-'
    if seconds < 60:
        return f'{seconds:.1f}s'
    elif seconds < 3600:
        return f'{seconds / 60:.1f}min'
    else:
        return f'{seconds / 3600:.1f}h'


def summary_stats(df: pd.DataFrame, keys: list) -> pd.DataFrame:
    rows = []
    for key in keys:
        tier_col = f'{key}_tier'
        mean_col = f'{key}_mean'
        rt_col   = f'{key}_runtime_mean'   # total seconds for the full run

        tiers = df[tier_col].dropna()
        means = df[mean_col].dropna()
        rts   = df[rt_col].dropna()

        n      = len(tiers)
        bud    = BUDGET[key]

        t1     = int((tiers == 1).sum())
        t1_2   = int((tiers <= 2).sum())
        t3plus = int((tiers >= 3).sum())

        mean_d2h       = means.mean()                  if len(means) else np.nan
        # _runtime_mean = total seconds for the full budget run
        # per-iteration ms = total / budget * 1000
        rt_per_iter_ms = (rts / bud * 1000).median()  if len(rts)  else np.nan
        rt_total_s     = rts.median()                  if len(rts)  else np.nan

        rows.append({
            'key':            key,
            'display':        DISPLAY_NAMES.get(key, key),
            'n':              n,
            'tier1':          t1,
            'tier1_pct':      100 * t1     / n if n else np.nan,
            'tier1_2':        t1_2,
            'tier1_2_pct':    100 * t1_2   / n if n else np.nan,
            'tier3plus':      t3plus,
            'tier3plus_pct':  100 * t3plus / n if n else np.nan,
            'mean_d2h':       mean_d2h,
            'rt_per_iter_ms': rt_per_iter_ms,
            'rt_total_s':     rt_total_s,
            'budget':         bud,
        })

    return pd.DataFrame(rows)


def print_summary_table(stats: pd.DataFrame):
    cols  = stats['display'].tolist()
    w_col, w_data = 24, 16
    header = f"{'Metric':<{w_col}}" + ''.join(f'{c:>{w_data}}' for c in cols)
    sep    = '-' * len(header)

    def row(label, values):
        return f'{label:<{w_col}}' + ''.join(f'{str(v):>{w_data}}' for v in values)

    print('\n' + sep)
    print('SUMMARY TABLE')
    print(sep)
    print(header)
    print(sep)

    print(row('Datasets tier 1',
              [f"{r['tier1']} ({r['tier1_pct']:.0f}%)"
               for _, r in stats.iterrows()]))
    print(row('Datasets tier 1-2',
              [f"{r['tier1_2']} ({r['tier1_2_pct']:.0f}%)"
               for _, r in stats.iterrows()]))
    print(row('Datasets tier 3+',
              [f"{r['tier3plus']} ({r['tier3plus_pct']:.0f}%)"
               for _, r in stats.iterrows()]))
    print(row('Mean d2h',
              [f"{r['mean_d2h']:.4f}"
               for _, r in stats.iterrows()]))
    print(row('Runtime/iter (ms)',
              [f"{r['rt_per_iter_ms']:.1f}" if not pd.isna(r['rt_per_iter_ms']) else '-'
               for _, r in stats.iterrows()]))
    print(row('Total runtime (median)',
              [format_runtime(r['rt_total_s'])
               for _, r in stats.iterrows()]))
    print(sep)
    print(row('Evaluations used',
              [str(r['budget']) for _, r in stats.iterrows()]))
    print(sep)


def to_latex_summary(stats: pd.DataFrame) -> str:
    col_spec   = 'l' + 'c' * len(stats)
    header_row = ' & '.join(
        f'\\textbf{{{r["display"]}}}' for _, r in stats.iterrows()
    )

    def fmt_pct(n, pct):
        return f'{n} ({pct:.0f}\\%)'

    def rt_fmt(r):
        v = r['rt_per_iter_ms']
        return f'{v:.1f}' if not pd.isna(v) else '--'

    lines = [
        r'\begin{table}[t]',
        r'\centering',
        (r'\caption{Scott-Knott ranking comparison. Tier 1 = best performance. '
         r'Note budget imbalance: EZR/SMAC use 200 evaluations, '
         r'NSGA-II/SPEA2 use 1000.}'),
        r'\label{tab:performance}',
        r'\small',
        f'\\begin{{tabular}}{{{col_spec}}}',
        r'\toprule',
        f'\\textbf{{Metric}} & {header_row} \\\\',
        r'\midrule',
        'Datasets ranked tier 1 & ' +
        ' & '.join(fmt_pct(r['tier1'], r['tier1_pct'])
                   for _, r in stats.iterrows()) + ' \\\\',
        'Datasets ranked tier 1--2 & ' +
        ' & '.join(fmt_pct(r['tier1_2'], r['tier1_2_pct'])
                   for _, r in stats.iterrows()) + ' \\\\',
        'Datasets ranked tier 3+ & ' +
        ' & '.join(fmt_pct(r['tier3plus'], r['tier3plus_pct'])
                   for _, r in stats.iterrows()) + ' \\\\',
        'Mean d2h & ' +
        ' & '.join(f'{r["mean_d2h"]:.2f}'
                   for _, r in stats.iterrows()) + ' \\\\',
        'Median runtime/iter (ms) & ' +
        ' & '.join(rt_fmt(r) for _, r in stats.iterrows()) + ' \\\\',
        'Total runtime (median) & ' +
        ' & '.join(format_runtime(r['rt_total_s'])
                   for _, r in stats.iterrows()) + ' \\\\',
        r'\midrule',
        '\\textbf{Evaluations used} & ' +
        ' & '.join(f'\\textbf{{{r["budget"]}}}'
                   for _, r in stats.iterrows()) + ' \\\\',
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Table 2 – Win / Tie / Lose
# ─────────────────────────────────────────────────────────────────────────────

def _wtl(ref_tiers: pd.Series, opp_tiers: pd.Series):
    """
    Win  = ref tier LOWER  (better) than opp
    Tie  = equal tier
    Lose = ref tier HIGHER (worse)  than opp
    Only rows where both tiers are non-NaN are counted.
    """
    valid = ref_tiers.notna() & opp_tiers.notna()
    ref   = ref_tiers[valid].values
    opp   = opp_tiers[valid].values
    w = int((ref < opp).sum())
    t = int((ref == opp).sum())
    l = int((ref > opp).sum())
    return w, t, l


def advantage(w: int, t: int, l: int) -> str:
    """
    EZR is 'good' when W + T >= L.
    Returns "WT vs L (WT%)" — percentage of datasets where EZR
    matched or beat the opponent.  >= 50% means reference holds its own.
    """
    total = w + t + l
    if total == 0:
        return '-'
    wt  = w + t
    pct = 100 * wt / total
    return f'{wt} vs {l} ({pct:.0f}%)'


def build_wtl_table(df: pd.DataFrame) -> pd.DataFrame:
    # ── Print datasets considered in WTL counting ──────────────────────────
    name_col = None
    for c in df.columns:
        if c == 'dataset' or (df[c].dtype == object and name_col is None):
            name_col = c
            break
    first_ref_col = f'{WTL_SECTIONS[0][1]}_tier'
    if name_col and first_ref_col in df.columns:
        considered = sorted(
            df.loc[df[first_ref_col].notna(), name_col].dropna().unique().tolist()
        )
        print(f'\nDatasets considered in Win/Tie/Lose ({len(considered)} total):')
        for ds in considered:
            print(f'  {ds}')

    # ── Build rows ─────────────────────────────────────────────────────────
    rows = []
    for section_label, ref_key, opponents in WTL_SECTIONS:
        ref_col = f'{ref_key}_tier'
        if ref_col not in df.columns:
            continue
        for opp_key in opponents:
            opp_col = f'{opp_key}_tier'
            if opp_col not in df.columns:
                continue
            w, t, l = _wtl(df[ref_col], df[opp_col])
            rows.append({
                'Section'  : section_label,
                'Reference': DISPLAY_NAMES.get(ref_key, ref_key),
                'Opponent' : DISPLAY_NAMES.get(opp_key, opp_key),
                'Win'      : w,
                'Tie'      : t,
                'Lose'     : l,
                'W+T vs L' : advantage(w, t, l),
            })
    return pd.DataFrame(rows)


def print_wtl_table(wtl_df: pd.DataFrame):
    print('\n' + '=' * 84)
    print('WIN / TIE / LOSE  (lower SK tier = better rank)')
    print('Win      = reference has LOWER (better) tier than opponent')
    print('W+T vs L = (Win+Tie) vs Lose; >=50% means reference holds its own or better')
    print('=' * 84)

    current = None
    for _, row in wtl_df.iterrows():
        if row['Section'] != current:
            current = row['Section']
            print(f'\n── {current} ──')
            print(f"  {'Reference':<14}  {'Opponent':<14}  "
                  f"{'Win':>5}  {'Tie':>5}  {'Lose':>5}  {'W+T vs L':>18}")
            print(f"  {'-'*14}  {'-'*14}  "
                  f"{'-'*5}  {'-'*5}  {'-'*5}  {'-'*18}")
        print(f"  {row['Reference']:<14}  {row['Opponent']:<14}  "
              f"{row['Win']:>5}  {row['Tie']:>5}  {row['Lose']:>5}  "
              f"{row['W+T vs L']:>18}")


def to_latex_wtl(wtl_df: pd.DataFrame) -> str:
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        (r'\caption{Win/Tie/Lose comparison based on Scott-Knott tiers. '
         r'\textbf{W+T vs L} shows datasets where the reference matched or '
         r'outperformed the opponent (Win+Tie) versus datasets where it ranked '
         r'worse (Lose). A percentage $\geq\!50\%$ means the reference holds '
         r'its own or better.}'),
        r'\label{tab:wtl}',
        r'\small',
        r'\begin{tabular}{llrrrr}',
        r'\toprule',
        (r'\textbf{Reference} & \textbf{Opponent} & '
         r'\textbf{W} & \textbf{T} & \textbf{L} & \textbf{W+T vs L} \\'),
        r'\midrule',
    ]

    current = None
    for _, row in wtl_df.iterrows():
        if row['Section'] != current:
            current = row['Section']
            escaped = current.replace('_', r'\_')
            lines.append(
                f'\\multicolumn{{6}}{{l}}'
                f'{{\\textit{{{escaped}}}}} \\\\'
            )
        ref = row['Reference'].replace('-', r'\nobreakdash-')
        opp = row['Opponent'].replace('-',  r'\nobreakdash-')
        lines.append(
            f'{ref} & {opp} & '
            f'{row["Win"]} & {row["Tie"]} & {row["Lose"]} & '
            f'{row["W+T vs L"]} \\\\'
        )

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(csv_path:        str = 'scott_knott_dataset.csv',
         out_summary_csv: str = 'summary_stats.csv',
         out_wtl_csv:     str = 'wtl_table.csv',
         out_latex:       str = 'tables.tex'):

    df = load(csv_path)
    print(f'Loaded {len(df)} datasets, {len(df.columns)} columns.')

    # ── Summary table ──────────────────────────────────────────────────────
    stats = summary_stats(df, SUMMARY_KEYS)
    print_summary_table(stats)
    stats.to_csv(out_summary_csv, index=False)
    print(f'\nSummary stats saved → {out_summary_csv}')

    # ── Win / Tie / Lose table ─────────────────────────────────────────────
    wtl_df = build_wtl_table(df)
    print_wtl_table(wtl_df)
    wtl_df.to_csv(out_wtl_csv, index=False)
    print(f'Win/Tie/Lose table saved → {out_wtl_csv}')

    # ── LaTeX ──────────────────────────────────────────────────────────────
    latex = to_latex_summary(stats) + '\n\n' + to_latex_wtl(wtl_df)
    with open(out_latex, 'w') as f:
        f.write(latex)
    print(f'LaTeX saved → {out_latex}')
    print('\n--- LaTeX ---\n')
    print(latex)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--csv',             default='scott_knott_dataset.csv')
    p.add_argument('--out_summary_csv', default='summary_stats.csv')
    p.add_argument('--out_wtl_csv',     default='wtl_table.csv')
    p.add_argument('--out_latex',       default='tables.tex')
    a = p.parse_args()
    main(a.csv, a.out_summary_csv, a.out_wtl_csv, a.out_latex)