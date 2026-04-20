#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from collections import defaultdict
import pandas as pd
from utils.DistanceUtil import d2h

# ============================================================
# Helpers
# ============================================================
def extract_dataset_and_run(filename: Path):
    stem = filename.stem
    parts = stem.split("_")
    dataset = "_".join(parts[:-1])
    run_id = parts[-1]
    return dataset, run_id

def load_best_curve(json_file: Path, optimizer=None):
    with open(json_file, "r") as f:
        data = json.load(f)
    if optimizer in ["NSGA2", "SPEA2"]:
        history = data.get("frontier_history", [])
    else:
        history = data.get("evaluation_history", [])
    best_so_far = []
    best_value = float("inf")
    for entry in history:
        if optimizer in ["NSGA2", "SPEA2"]:
            scores = [float(s) for s in entry.get("best_objectives", [])]
        else:
            scores = [float(s) for s in entry.get("objectives", [])]
        if not scores:
            continue
        val = d2h([0.0] * len(scores), scores)
        best_value = min(best_value, val)
        best_so_far.append(best_value)
    return np.array(best_so_far)

def pad_curve(curve, fixed_budget):
    if len(curve) == 0:
        return np.full(fixed_budget, np.nan)
    if len(curve) < fixed_budget:
        pad_val = curve[-1]
        pad = np.full(fixed_budget - len(curve), pad_val)
        curve = np.concatenate([curve, pad])
    else:
        curve = curve[:fixed_budget]
    return curve


# ============================================================
# Cross-optimizer dataset intersection
# ============================================================
def get_common_dataset_stems(results_root: Path,
                              optimizers: list,
                              budget: int) -> set:
    budget_suffix = f"_{budget}"
    per_optimizer_stems = []
    for opt in optimizers:
        metadata_dir = results_root / f"results_{opt}" / opt / "metadata"
        if not metadata_dir.exists():
            per_optimizer_stems.append(set())
            continue
        stems = set()
        for json_file in metadata_dir.glob("*.json"):
            if budget_suffix not in json_file.stem:
                continue
            dataset_name, _ = extract_dataset_and_run(json_file)
            stems.add(dataset_name)
        per_optimizer_stems.append(stems)
    if not per_optimizer_stems:
        return set()
    common = per_optimizer_stems[0]
    for s in per_optimizer_stems[1:]:
        common = common & s
    return common


# ============================================================
# Aggregation
# ============================================================
def aggregate_optimizer_regret(metadata_dir: Path,
                                ref_opt_df: pd.DataFrame,
                                optimizer: str,
                                fixed_budget: int,
                                allowed_dataset_stems: set):
    metadata_dir = metadata_dir / "metadata"
    dataset_curves = defaultdict(list)
    run_variances = []
    ref_opt_df["csv_basename"] = ref_opt_df["csv_file"].apply(
        lambda x: Path(x).name
    )
    budget_suffix = f"_{fixed_budget}"
    for json_file in metadata_dir.glob("*.json"):
        if budget_suffix not in json_file.stem:
            continue
        dataset_name, _ = extract_dataset_and_run(json_file)
        if dataset_name not in allowed_dataset_stems:
            continue
        dataset_name_clean = dataset_name.replace(budget_suffix, "")
        csv_match = ref_opt_df[
            ref_opt_df["csv_basename"] == dataset_name_clean + ".csv"
        ]
        if csv_match.empty:
            continue
        ref_opt_val = float(csv_match["min_d2h"].values[0])
        curve = load_best_curve(json_file, optimizer)
        if len(curve) == 0:
            continue
        val_regret_curve = curve - ref_opt_val
        val_regret_curve = np.clip(val_regret_curve, 1e-12, None)
        val_regret_curve = pad_curve(val_regret_curve, fixed_budget)
        dataset_curves[dataset_name].append(val_regret_curve)
    if not dataset_curves:
        return None
    dataset_medians = []
    for runs in dataset_curves.values():
        runs_arr = np.array(runs)
        run_iqr_dataset = (
            np.nanpercentile(runs_arr, 75, axis=0)
            - np.nanpercentile(runs_arr, 25, axis=0)
        )
        run_variances.append(np.nanmedian(run_iqr_dataset))
        median_curve_dataset = np.nanmedian(runs_arr, axis=0)
        dataset_medians.append(median_curve_dataset)
    dataset_medians = np.array(dataset_medians)
    median_curve = np.nanmedian(dataset_medians, axis=0)
    dataset_iqr = (
        np.nanpercentile(dataset_medians, 75, axis=0)
        - np.nanpercentile(dataset_medians, 25, axis=0)
    )
    run_iqr = np.median(run_variances)
    return median_curve, dataset_iqr, run_iqr


# ============================================================
# Plot Style
# ============================================================
def setup_neurips_style():
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ============================================================
# Speedup annotation helper  — clean horizontal arrows
# ============================================================
# Display names: internal key -> label shown on plot
OPT_DISPLAY_NAMES = {
    "MOSMAC": "SMAC",
    "EZR":    "EZR",
    "NSGA2":  "NSGA2",
    "SPEA2":  "SPEA2",
}

def annotate_speedups(ax, all_curves: dict, line_colors: dict,
                      ref_opt: str = "EZR", annotation_x: int = 200,
                      fixed_budget: int = 1000):
    if ref_opt not in all_curves:
        return

    ref_curve = all_curves[ref_opt]
    if annotation_x - 1 >= len(ref_curve):
        return

    target_regret = ref_curve[annotation_x - 1]

    arrow_data = []
    others = [o for o in all_curves if o != ref_opt]

    for opt in others:
        curve = all_curves[opt]
        indices = np.where(curve <= target_regret)[0]
        if len(indices) == 0:
            x_other = fixed_budget
            speedup = x_other / annotation_x
            label = f">{speedup:.0f}×"
        else:
            x_other = int(indices[0]) + 1
            speedup = x_other / annotation_x
            label = f"{speedup:.1f}×"
        arrow_data.append((annotation_x, x_other, label, opt))

    if not arrow_data:
        return

    log_start = np.log10(target_regret) + 0.6
    n = len(arrow_data)
    y_positions = [10 ** (log_start + i * 0.55) for i in range(n)]

    for (x_start, x_end, label, opt), y_arrow in zip(arrow_data, y_positions):
        color = line_colors.get(opt, "gray")
        alpha = 0.75

        ax.plot(
            [x_start, x_end], [y_arrow, y_arrow],
            color=color, lw=1.5, alpha=alpha, solid_capstyle="round",
        )

        tick_h = 10 ** (np.log10(y_arrow) + 0.08) - y_arrow
        for x_tick in (x_start, x_end):
            ax.plot(
                [x_tick, x_tick],
                [y_arrow - tick_h, y_arrow + tick_h],
                color=color, lw=1.5, alpha=alpha, solid_capstyle="round",
            )

        ax.annotate(
            "",
            xy=(x_end, y_arrow),
            xytext=(x_end - 1, y_arrow),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=0,
                mutation_scale=8,
            ),
        )

        x_mid = (x_start + x_end) / 2
        ax.text(
            x_mid,
            y_arrow * (10 ** 0.18),
            label,
            ha="center", va="bottom",
            fontsize=10, color=color,
            fontweight="semibold",
        )


# ============================================================
# Main Regret Plot
# ============================================================
def plot_aggregated_regret(results_root: Path,
                           ref_opt_df: pd.DataFrame,
                           optimizers: list,
                           save_dir: Path,
                           fixed_budget: int = 1000):
    common_stems = get_common_dataset_stems(results_root, optimizers, fixed_budget)
    if not common_stems:
        print("WARNING: No dataset stems common across all optimizers. Nothing to plot.")
        return
    print(f"Common dataset stems ({len(common_stems)}): {sorted(common_stems)}")

    setup_neurips_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    all_curves = {}
    line_colors = {}

    for opt in optimizers:
        metadata_dir = results_root / f"results_{opt}" / opt
        if not metadata_dir.exists():
            continue
        result = aggregate_optimizer_regret(
            metadata_dir, ref_opt_df,
            optimizer=opt, fixed_budget=fixed_budget,
            allowed_dataset_stems=common_stems,
        )
        if result is None:
            continue
        median_curve, dataset_iqr, run_iqr = result
        all_curves[opt] = median_curve.copy()

        x = np.arange(1, len(median_curve) + 1)
        display_name = OPT_DISPLAY_NAMES.get(opt, opt)
        line, = ax.plot(x, median_curve, linewidth=2, label=display_name)
        line_colors[opt] = line.get_color()

        # Shading removed — median lines only

    # ── annotation: clean colored horizontal arrows ──────────────────────
    annotation_x = 200
    ax.axvline(annotation_x, linestyle="--", color="gray", alpha=0.35, lw=1.2)
    annotate_speedups(
        ax, all_curves, line_colors,
        ref_opt="EZR", annotation_x=annotation_x, fixed_budget=fixed_budget,
    )

    ax.set_yscale("log")
    ax.set_xlabel("Function Evaluations")
    ax.set_ylabel("Validation Regret (log scale)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / "aggregated_regret.pdf")
    plt.savefig(save_dir / "aggregated_regret.png", dpi=300)
    plt.close()


# ============================================================
# Deviation Plot
# ============================================================
def plot_deviations(results_root: Path,
                    ref_opt_df: pd.DataFrame,
                    optimizers: list,
                    save_dir: Path,
                    fixed_budget: int = 1000):
    common_stems = get_common_dataset_stems(results_root, optimizers, fixed_budget)
    if not common_stems:
        print("WARNING: No dataset stems common across all optimizers. Nothing to plot.")
        return

    setup_neurips_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for opt in optimizers:
        metadata_dir = results_root / f"results_{opt}" / opt
        if not metadata_dir.exists():
            continue
        result = aggregate_optimizer_regret(
            metadata_dir, ref_opt_df,
            optimizer=opt, fixed_budget=fixed_budget,
            allowed_dataset_stems=common_stems,
        )
        if result is None:
            continue
        median_curve, dataset_iqr, run_iqr = result
        display_name = OPT_DISPLAY_NAMES.get(opt, opt)
        x = np.arange(1, len(dataset_iqr) + 1)
        ax.plot(x, dataset_iqr, linewidth=2, label=display_name)

    ax.set_yscale("log")
    ax.set_xlabel("Function Evaluations")
    ax.set_ylabel("Dataset-level IQR (log scale)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / "deviations.pdf")
    plt.savefig(save_dir / "deviations.png", dpi=300)
    plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir",  type=str, default="results")
    parser.add_argument("--ref_opt_csv",  type=str, default="reference_optimal.csv")
    parser.add_argument("--save_dir",     type=str, default="plots")
    parser.add_argument("--budget",       type=int, default=1000,
                        help="Budget suffix in JSON filenames (e.g. 1000 → SS-X_1000.json)")
    args = parser.parse_args()

    results_root = Path(args.results_dir)
    ref_opt_df   = pd.read_csv(args.ref_opt_csv)
    save_dir     = Path(args.save_dir)
    optimizers   = ["MOSMAC", "EZR", "NSGA2", "SPEA2"]

    plot_aggregated_regret(results_root, ref_opt_df, optimizers, save_dir, fixed_budget=args.budget)
    plot_deviations(       results_root, ref_opt_df, optimizers, save_dir, fixed_budget=args.budget)

if __name__ == "__main__":
    main()