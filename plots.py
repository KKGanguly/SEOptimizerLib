import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ────────────────────────────────────────────────────────────────────
optimizers = ['GPEI-25','GPEI-50','GPEI-100','GPEI-200',
              'HEBO-25','HEBO-50','HEBO-100',
              'HILL-25','HILL-50','HILL-100','HILL-200',
              'ILS-25','ILS-50','ILS-100','ILS-200',
              'MOSMAC-25','MOSMAC-50','MOSMAC-100','MOSMAC-200',
              'SA-25','SA-50','SA-100','SA-200',
              'TS-25','TS-50','TS-100','TS-200']

tier1     = [63,63,66,64, 45,58,18, 29,38,56,62, 9,14,16,16, 76,86,91,93, 8,12,11,11, 15,30,39,40]
tier12    = [93,90,93,89, 89,92,25, 62,80,96,98, 43,55,63,63, 92,97,102,102, 38,46,53,52, 51,73,79,79]
tier3plus = [12,8,4,2, 16,11,4, 43,25,9,7, 62,50,42,42, 13,8,3,0, 67,58,52,53, 54,32,26,26]
d2h       = [0.3260,0.2912,0.2643,0.2499, 0.3654,0.3271,0.3394, 0.4641,0.4062,0.3349,0.3181,
             0.4920,0.4791,0.4735,0.4705, 0.2283,0.1828,0.1615,0.1391,
             0.5178,0.4928,0.4857,0.4874, 0.4693,0.4371,0.4060,0.4094]
runtime_ms = [76.9,87.2,108.4,148.6, 1565.9,1931.4,1221.5, 0.4,0.4,0.4,0.3, 0.2,0.2,0.1,0.1,
              70.7,76.1,77.6,82.8, 0.3,0.3,0.3,0.2, 0.2,0.2,0.1,0.1]
budgets   = [25,50,100,200, 25,50,100, 25,50,100,200, 25,50,100,200, 25,50,100,200, 25,50,100,200, 25,50,100,200]

# Categories from document taxonomy
categories = {
    'Surrogate-Based\n(GPEI, HEBO)': {
        'optimizers': ['GPEI-25','GPEI-50','GPEI-100','GPEI-200','HEBO-25','HEBO-50','HEBO-100'],
        'color': '#2E5EAA',  # steel blue
        'short': 'Surrogate'
    },
    'Single-State\nLocal Search\n(HILL, ILS)': {
        'optimizers': ['HILL-25','HILL-50','HILL-100','HILL-200','ILS-25','ILS-50','ILS-100','ILS-200'],
        'color': '#8B3A3A',  # dark red
        'short': 'Single-State'
    },
    'Population-Based\nMetaheuristic\n(MOSMAC)': {
        'optimizers': ['MOSMAC-25','MOSMAC-50','MOSMAC-100','MOSMAC-200'],
        'color': '#2D5A27',  # forest green
        'short': 'Population'
    },
    'Trajectory-Based\nMetaheuristic\n(SA, TS)': {
        'optimizers': ['SA-25','SA-50','SA-100','SA-200','TS-25','TS-50','TS-100','TS-200'],
        'color': '#5C3A6B',  # plum
        'short': 'Trajectory'
    },
}

cat_colors = {'GPEI': '#2E5EAA', 'HEBO': '#2979C8', 'HILL': '#8B3A3A',
              'ILS': '#C0392B', 'MOSMAC': '#2D5A27', 'SA': '#5C3A6B', 'TS': '#8E44AD'}

def get_color(opt):
    for k,v in cat_colors.items():
        if opt.startswith(k):
            return v
    return '#666666'

# ── FIGURE 1: Mean d2h per optimizer (lower = better) ────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
colors = [get_color(o) for o in optimizers]
bars = ax.bar(range(len(optimizers)), d2h, color=colors, edgecolor='white', linewidth=0.5, width=0.75)
ax.set_xticks(range(len(optimizers)))
ax.set_xticklabels(optimizers, rotation=45, ha='right', fontsize=7.5)
ax.set_ylabel('Mean Distance-to-Heaven (d2h)', fontsize=10)
ax.set_title('Mean d2h by Optimizer and Budget (lower is better)', fontsize=12, fontweight='bold', pad=10)
ax.set_ylim(0, 0.58)
ax.axhline(y=np.mean(d2h), color='gray', linestyle='--', linewidth=0.8, alpha=0.6, label=f'Overall mean ({np.mean(d2h):.3f})')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3, linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Category separators
separators = [4, 7, 11, 15, 19, 23]
for s in separators:
    ax.axvline(x=s - 0.5, color='#CCCCCC', linewidth=1.2, linestyle='-')

# Legend patches
legend_patches = [mpatches.Patch(color=v, label=k) for k,v in cat_colors.items()]
ax.legend(handles=legend_patches, loc='upper right', fontsize=8, ncol=4,
          framealpha=0.9, edgecolor='#CCCCCC')

plt.tight_layout()
plt.savefig('/home/claude/optimizer_report/fig1_d2h.pdf', dpi=150, bbox_inches='tight')
plt.savefig('/home/claude/optimizer_report/fig1_d2h.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 1 done")

# ── FIGURE 2: Tier-1 datasets % grouped by category ─────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(14, 4.5), sharey=False)
cat_list = list(categories.items())

for idx, (cat_name, cat_info) in enumerate(cat_list):
    ax = axes[idx]
    opts = cat_info['optimizers']
    # Get indices
    idxs = [optimizers.index(o) for o in opts]
    t1_pct = [tier1[i]/105*100 for i in idxs]
    t12_pct = [tier12[i]/105*100 for i in idxs]
    t3_pct = [tier3plus[i]/105*100 for i in idxs]
    
    x = np.arange(len(opts))
    width = 0.6
    
    # Stacked bars: tier1 (dark), tier 2 (medium), tier 3+ (light)
    b1 = ax.bar(x, t1_pct, width, label='Tier 1', color=cat_info['color'], alpha=0.9)
    b2 = ax.bar(x, [t12_pct[i]-t1_pct[i] for i in range(len(opts))], width,
                bottom=t1_pct, label='Tier 2', color=cat_info['color'], alpha=0.45)
    b3 = ax.bar(x, t3_pct, width,
                bottom=t12_pct, label='Tier 3+', color='#DDDDDD', alpha=0.7)
    
    short_labels = [o.replace('-','\n') for o in opts]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=7)
    ax.set_ylim(0, 108)
    ax.set_ylabel('% Datasets' if idx == 0 else '', fontsize=9)
    ax.set_title(cat_name, fontsize=9, fontweight='bold', color=cat_info['color'])
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add tier1 % labels on top
    for i, (b, pct) in enumerate(zip(b1, t1_pct)):
        ax.text(i, pct + 1.5, f'{pct:.0f}%', ha='center', va='bottom', fontsize=6.5, fontweight='bold')

# Shared legend
handles = [mpatches.Patch(color='#555555', alpha=0.9, label='Tier 1 (best)'),
           mpatches.Patch(color='#555555', alpha=0.45, label='Tier 2'),
           mpatches.Patch(color='#DDDDDD', alpha=0.7, label='Tier 3+ (worst)')]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9, 
           bbox_to_anchor=(0.5, -0.02), framealpha=0.9)

fig.suptitle('Scott-Knott Tier Distribution by Optimizer Category', fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/home/claude/optimizer_report/fig2_tiers.pdf', dpi=150, bbox_inches='tight')
plt.savefig('/home/claude/optimizer_report/fig2_tiers.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 2 done")

# ── FIGURE 3: d2h vs Budget per algorithm family ────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

families = {
    'GPEI': {'budgets': [25,50,100,200], 'color': '#2E5EAA', 'marker': 'o', 'ls': '-'},
    'HEBO': {'budgets': [25,50,100], 'color': '#2979C8', 'marker': 's', 'ls': '--'},
    'HILL': {'budgets': [25,50,100,200], 'color': '#8B3A3A', 'marker': '^', 'ls': '-'},
    'ILS':  {'budgets': [25,50,100,200], 'color': '#C0392B', 'marker': 'v', 'ls': '--'},
    'MOSMAC': {'budgets': [25,50,100,200], 'color': '#2D5A27', 'marker': 'D', 'ls': '-'},
    'SA':   {'budgets': [25,50,100,200], 'color': '#5C3A6B', 'marker': 'P', 'ls': '--'},
    'TS':   {'budgets': [25,50,100,200], 'color': '#8E44AD', 'marker': 'X', 'ls': ':'},
}

fam_d2h = {
    'GPEI':   [0.3260, 0.2912, 0.2643, 0.2499],
    'HEBO':   [0.3654, 0.3271, 0.3394],
    'HILL':   [0.4641, 0.4062, 0.3349, 0.3181],
    'ILS':    [0.4920, 0.4791, 0.4735, 0.4705],
    'MOSMAC': [0.2283, 0.1828, 0.1615, 0.1391],
    'SA':     [0.5178, 0.4928, 0.4857, 0.4874],
    'TS':     [0.4693, 0.4371, 0.4060, 0.4094],
}

for fam, info in families.items():
    budg = info['budgets']
    vals = fam_d2h[fam]
    ax.plot(budg, vals, color=info['color'], marker=info['marker'],
            linestyle=info['ls'], linewidth=2, markersize=8, label=fam, zorder=3)
    for b, v in zip(budg, vals):
        ax.annotate(f'{v:.3f}', (b, v), textcoords="offset points",
                    xytext=(0, 7), ha='center', fontsize=6.5, color=info['color'])

ax.set_xlabel('Evaluation Budget', fontsize=11)
ax.set_ylabel('Mean d2h (lower is better)', fontsize=11)
ax.set_title('d2h vs. Evaluation Budget by Algorithm Family', fontsize=12, fontweight='bold')
ax.set_xticks([25, 50, 100, 200])
ax.legend(fontsize=9, ncol=2, loc='upper right', framealpha=0.9)
ax.grid(alpha=0.3, linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0.05, 0.58)

plt.tight_layout()
plt.savefig('/home/claude/optimizer_report/fig3_budget_d2h.pdf', dpi=150, bbox_inches='tight')
plt.savefig('/home/claude/optimizer_report/fig3_budget_d2h.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 3 done")

# ── FIGURE 4: Win-rate heatmap (W+T vs L %) – best budget per family ─────────
# Use the "best" (highest budget) configuration for each family for a clean comparison
best_configs = ['GPEI-200', 'HEBO-100', 'HILL-200', 'ILS-200', 'MOSMAC-200', 'SA-200', 'TS-200']

# W+T% matrix: row = reference, col = opponent
wtl_data = {
    'GPEI-200': {'GPEI-200':100, 'HEBO-100':100, 'HILL-200':99, 'ILS-200':100, 'MOSMAC-200':70, 'SA-200':100, 'TS-200':100},
    'HEBO-100': {'GPEI-200':96, 'HEBO-100':100, 'HILL-200':97, 'ILS-200':93, 'MOSMAC-200':61, 'SA-200':97, 'TS-200':93},
    'HILL-200': {'GPEI-200':85, 'HEBO-100':86, 'HILL-200':100, 'ILS-200':95, 'MOSMAC-200':62, 'SA-200':100, 'TS-200':92},
    'ILS-200':  {'GPEI-200':26, 'HEBO-100':34, 'HILL-200':37, 'ILS-200':100, 'MOSMAC-200':19, 'SA-200':97, 'TS-200':62},
    'MOSMAC-200':{'GPEI-200':90,'HEBO-100':93, 'HILL-200':95, 'ILS-200':97, 'MOSMAC-200':100,'SA-200':99, 'TS-200':92},
    'SA-200':   {'GPEI-200':22, 'HEBO-100':31, 'HILL-200':31, 'ILS-200':84, 'MOSMAC-200':15, 'SA-200':100, 'TS-200':53},
    'TS-200':   {'GPEI-200':54, 'HEBO-100':72, 'HILL-200':63, 'ILS-200':95, 'MOSMAC-200':39, 'SA-200':100, 'TS-200':100},
}

labels = ['GPEI-200','HEBO-100','HILL-200','ILS-200','MOSMAC-200','SA-200','TS-200']
matrix = np.array([[wtl_data[r][c] for c in labels] for r in labels], dtype=float)
np.fill_diagonal(matrix, np.nan)

fig, ax = plt.subplots(figsize=(8, 6.5))
masked = np.ma.array(matrix, mask=np.isnan(matrix))
cmap = matplotlib.colormaps['RdYlGn']
cmap.set_bad(color='#EEEEEE')
im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect='auto')

cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label('W+T % (≥50% = reference holds own)', fontsize=9)
cbar.ax.tick_params(labelsize=8)

ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
short = [l.replace('-','\n') for l in labels]
ax.set_xticklabels(short, fontsize=8)
ax.set_yticklabels(short, fontsize=8)
ax.set_xlabel('Opponent', fontsize=10, labelpad=8)
ax.set_ylabel('Reference', fontsize=10, labelpad=8)
ax.set_title('Win+Tie % Heatmap (Best Budget per Family)\nRow = reference; green ≥ 50%, red < 50%',
             fontsize=11, fontweight='bold')

for i in range(len(labels)):
    for j in range(len(labels)):
        if not np.isnan(matrix[i, j]):
            val = matrix[i, j]
            color = 'white' if (val > 80 or val < 30) else 'black'
            ax.text(j, i, f'{int(val)}%', ha='center', va='center',
                    fontsize=8.5, fontweight='bold', color=color)

ax.axhline(y=3.5, color='white', linewidth=2)
ax.axvline(x=3.5, color='white', linewidth=2)

plt.tight_layout()
plt.savefig('/home/claude/optimizer_report/fig4_heatmap.pdf', dpi=150, bbox_inches='tight')
plt.savefig('/home/claude/optimizer_report/fig4_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 4 done")

# ── FIGURE 5: Runtime vs d2h scatter ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))

fam_runtime = {
    'GPEI':   [76.9, 87.2, 108.4, 148.6],
    'HEBO':   [1565.9, 1931.4, 1221.5],
    'HILL':   [0.4, 0.4, 0.4, 0.3],
    'ILS':    [0.2, 0.2, 0.1, 0.1],
    'MOSMAC': [70.7, 76.1, 77.6, 82.8],
    'SA':     [0.3, 0.3, 0.3, 0.2],
    'TS':     [0.2, 0.2, 0.1, 0.1],
}

for fam, info in families.items():
    rts = fam_runtime[fam]
    vals = fam_d2h[fam]
    budg = info['budgets']
    for b, rt, v in zip(budg, rts, vals):
        size = 60 + b * 0.8
        sc = ax.scatter(rt, v, color=info['color'], s=size, marker=info['marker'],
                        alpha=0.85, zorder=3, edgecolors='white', linewidths=0.8)
        ax.annotate(f'{fam}\n{b}', (rt, v), textcoords="offset points",
                    xytext=(5, 3), fontsize=6, color=info['color'], alpha=0.9)

ax.set_xscale('log')
ax.set_xlabel('Runtime per Iteration (ms, log scale)', fontsize=11)
ax.set_ylabel('Mean d2h (lower is better)', fontsize=11)
ax.set_title('Quality vs. Computational Cost\n(bubble size ∝ evaluation budget)', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, linewidth=0.5, which='both')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

legend_patches = [mpatches.Patch(color=families[f]['color'], label=f) for f in families]
ax.legend(handles=legend_patches, fontsize=9, loc='upper right', framealpha=0.9)

# Best-value region annotation
ax.annotate('Best region\n(fast & accurate)', xy=(80, 0.14), fontsize=8.5,
            color='#2D5A27', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#2D5A27'),
            xytext=(10, 0.08))

plt.tight_layout()
plt.savefig('/home/claude/optimizer_report/fig5_scatter.pdf', dpi=150, bbox_inches='tight')
plt.savefig('/home/claude/optimizer_report/fig5_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 5 done")

print("All plots generated successfully!")