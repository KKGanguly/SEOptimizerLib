import pandas as pd
import numpy as np
import argparse
import re


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


def infer_keys(df: pd.DataFrame):
    """
    Discover all optimizer-budget keys present in the CSV by looking for
    columns that end in '_tier'.  Returns sorted list of keys like
    ['GPEI-25', 'GPEI-50', 'HILL-100', ...].
    """
    keys = []
    for col in df.columns:
        if col.endswith('_tier'):
            key = col[:-len('_tier')]
            # sanity-check: must also have a _mean column
            if f'{key}_mean' in df.columns:
                keys.append(key)
    # sort by optimizer name then budget number
    def sort_key(k):
        parts = k.rsplit('-', 1)
        name = parts[0]
        try:
            bud = int(parts[1])
        except (IndexError, ValueError):
            bud = 0
        return (name, bud)
    return sorted(keys, key=sort_key)


def build_budget_map(keys):
    bmap = {}
    for k in keys:
        parts = k.rsplit('-', 1)
        try:
            bmap[k] = int(parts[1])
        except (IndexError, ValueError):
            bmap[k] = 0
    return bmap


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
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


# ─────────────────────────────────────────────────────────────────────────────
# Table 1 – Summary stats
# ─────────────────────────────────────────────────────────────────────────────

def summary_stats(df: pd.DataFrame, keys: list, budget_map: dict) -> pd.DataFrame:
    rows = []
    for key in keys:
        tier_col = f'{key}_tier'
        mean_col = f'{key}_mean'
        rt_col   = f'{key}_runtime_mean'

        tiers = df[tier_col].dropna() if tier_col in df.columns else pd.Series([], dtype=float)
        means = df[mean_col].dropna() if mean_col in df.columns else pd.Series([], dtype=float)
        rts   = df[rt_col].dropna()   if rt_col   in df.columns else pd.Series([], dtype=float)

        n      = len(tiers)
        bud    = budget_map.get(key, 0)

        t1     = int((tiers == 1).sum())
        t1_2   = int((tiers <= 2).sum())
        t3plus = int((tiers >= 3).sum())

        mean_d2h       = means.mean()                  if len(means) else np.nan
        rt_per_iter_ms = (rts / bud * 1000).median()  if len(rts) and bud else np.nan
        rt_total_s     = rts.median()                  if len(rts)  else np.nan

        rows.append({
            'key':            key,
            'display':        key,
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
        (r'\caption{Scott-Knott ranking comparison. Tier 1 = best performance.}'),
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
    valid = ref_tiers.notna() & opp_tiers.notna()
    ref   = ref_tiers[valid].values
    opp   = opp_tiers[valid].values
    w = int((ref < opp).sum())
    t = int((ref == opp).sum())
    l = int((ref > opp).sum())
    return w, t, l


def advantage(w: int, t: int, l: int) -> str:
    total = w + t + l
    if total == 0:
        return '-'
    wt  = w + t
    pct = 100 * wt / total
    return f'{wt} vs {l} ({pct:.0f}%)'


def build_wtl_sections(keys: list, ref_optimizer: str = None):
    """
    Auto-build WTL sections: each unique optimizer at each budget vs all others.
    If ref_optimizer is specified, only build sections for that optimizer.
    """
    # group keys by optimizer name
    by_opt = {}
    for k in keys:
        opt = k.rsplit('-', 1)[0]
        by_opt.setdefault(opt, []).append(k)

    sections = []
    optimizers = [ref_optimizer] if ref_optimizer else sorted(by_opt.keys())
    all_keys_set = set(keys)

    for opt in optimizers:
        if opt not in by_opt:
            continue
        for ref_key in by_opt[opt]:
            opponents = [k for k in keys if k != ref_key]
            if opponents:
                label = f'{ref_key} vs all others'
                sections.append((label, ref_key, opponents))

    return sections


def build_wtl_table(df: pd.DataFrame, wtl_sections: list) -> pd.DataFrame:
    # print datasets considered
    name_col = None
    for c in df.columns:
        if c.lower() in ('dataset', 'datasets'):
            name_col = c
            break
    if name_col is None:
        for c in df.columns:
            if df[c].dtype == object:
                name_col = c
                break

    if wtl_sections and name_col:
        first_ref_col = f'{wtl_sections[0][1]}_tier'
        if first_ref_col in df.columns:
            considered = sorted(
                df.loc[df[first_ref_col].notna(), name_col].dropna().unique().tolist()
            )
            print(f'\nDatasets considered in Win/Tie/Lose ({len(considered)} total):')
            for ds in considered:
                print(f'  {ds}')

    rows = []
    for section_label, ref_key, opponents in wtl_sections:
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
                'Reference': ref_key,
                'Opponent' : opp_key,
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
            print(f"  {'Reference':<14}  {'Opponent':<16}  "
                  f"{'Win':>5}  {'Tie':>5}  {'Lose':>5}  {'W+T vs L':>18}")
            print(f"  {'-'*14}  {'-'*16}  "
                  f"{'-'*5}  {'-'*5}  {'-'*5}  {'-'*18}")
        print(f"  {row['Reference']:<14}  {row['Opponent']:<16}  "
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
        ref = str(row['Reference']).replace('-', r'\nobreakdash-')
        opp = str(row['Opponent']).replace('-',  r'\nobreakdash-')
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
         out_latex:       str = 'tables.tex',
         ref_optimizer:   str = None):

    df = load(csv_path)
    print(f'Loaded {len(df)} datasets, {len(df.columns)} columns.')

    keys = infer_keys(df)
    print(f'Inferred {len(keys)} optimizer-budget keys: {keys}')

    budget_map = build_budget_map(keys)

    # ── Summary table ──────────────────────────────────────────────────────
    stats = summary_stats(df, keys, budget_map)
    print_summary_table(stats)
    stats.to_csv(out_summary_csv, index=False)
    print(f'\nSummary stats saved → {out_summary_csv}')

    # ── Win / Tie / Lose table ─────────────────────────────────────────────
    wtl_sections = build_wtl_sections(keys, ref_optimizer=ref_optimizer)
    wtl_df = build_wtl_table(df, wtl_sections)
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
    p.add_argument('--ref_optimizer',   default=None,
                   help='If set, WTL table only shows sections for this optimizer '
                        '(e.g. --ref_optimizer EZR). Otherwise all vs all.')
    a = p.parse_args()
    main(a.csv, a.out_summary_csv, a.out_wtl_csv, a.out_latex, a.ref_optimizer)