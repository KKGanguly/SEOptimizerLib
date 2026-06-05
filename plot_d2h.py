import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Professional aesthetic settings
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300
})

def generate_publication_plots(csv_file):
    df = pd.read_csv(csv_file)
    budgets = [30, 50, 100, 200]
    
    # 1. Identify optimizers and prep dataframe for long-format plotting
    mean_cols = [c for c in df.columns if c.endswith('_mean')]
    optimizers = sorted(list(set([c.split('-')[0] for c in mean_cols])))
    
    # Use a high-contrast palette (e.g., 'tab20' or 'viridis')
    palette = sns.color_palette("tab20", n_colors=len(optimizers))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    
    for i, opt in enumerate(optimizers):
        means, stds = [], []
        for b in budgets:
            c_mean = next((c for c in df.columns if c.startswith(f"{opt}-{b}_") and c.endswith("_mean")), None)
            c_std = next((c for c in df.columns if c.startswith(f"{opt}-{b}_") and c.endswith("_std")), None)
            
            means.append(df[c_mean].replace([np.inf, -np.inf, np.nan], 1.0).mean() if c_mean else 1.0)
            stds.append(df[c_std].replace([np.inf, -np.inf, np.nan], 0.0).mean() if c_std else 0.0)
        
        ax1.plot(budgets, means, marker='o', markersize=4, label=opt, color=palette[i], linewidth=1.5)
        ax2.plot(budgets, stds, marker='o', markersize=4, label=opt, color=palette[i], linewidth=1.5)

    # Professional refinement for publication
    for ax in [ax1, ax2]:
        ax.set_xticks(budgets)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    ax1.set_title(r"\textbf{Mean Performance (d2h)}")
    ax1.set_xlabel("Evaluation Budget")
    ax1.set_ylabel("Mean d2h (Lower is better)")
    
    ax2.set_title(r"\textbf{Performance Stability ($\sigma$)}")
    ax2.set_xlabel("Evaluation Budget")
    ax2.set_ylabel("Standard Deviation")

    # Legend outside for readability, or use plt.legend(loc='best') to save space
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    
    plt.tight_layout()
    plt.savefig("scott_knott_performance_pub.pdf", bbox_inches='tight') # Use PDF for vector quality
    print("Publication-quality plots saved as scott_knott_performance_pub.pdf")

generate_publication_plots('full.csv')