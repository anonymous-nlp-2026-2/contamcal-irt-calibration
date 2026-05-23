#!/usr/bin/env python3
"""Cross-seed CI forest plot: mean Δacc ± BCa 95% CI for each family."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
    'lines.linewidth': 1.8,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'grid.linestyle': '--',
})

PATTERN_COLORS = {
    'Benefit':    '#2ca02c',
    'Immune':     '#7f7f7f',
    'Collapse':   '#d62728',
    'V-recovery': '#ff7f0e',
}

families = [
    ('Qwen2.5-14B',  'Collapse',   -15.7,  -24.99, -10.49, True),
    ('Qwen2.5-7B',   'V-recovery', -1.1,   -1.82,  -0.34,  True),
    ('Phi-4 14.7B',  'Immune',     +0.5,   -0.17,  +1.39,  False),
    ('OLMo-2 7B',    'Benefit',    +1.0,   -0.55,  +2.60,  False),
    ('Gemma-2 9B',   'Benefit',    +2.4,   +1.49,  +3.14,  True),
    ('Falcon3-7B',   'Benefit',    +4.1,   +3.91,  +4.19,  True),
]

fig, ax = plt.subplots(figsize=(6.5, 3))

ax.axvline(x=0, color='#444444', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)

y_positions = np.arange(len(families))[::-1]

for i, (name, pattern, mean, ci_lo, ci_hi, excludes_zero) in enumerate(families):
    y = y_positions[i]
    color = PATTERN_COLORS[pattern]
    marker = 'o' if excludes_zero else 'o'
    facecolor = color if excludes_zero else 'white'

    ax.hlines(y, ci_lo, ci_hi, color=color, linewidth=2.0, zorder=2)
    ax.plot(mean, y, 'o', color=facecolor, markersize=8, zorder=3,
            markeredgecolor=color, markeredgewidth=1.5)

    if name == 'Falcon3-7B':
        ax.text(ci_hi + 0.4, y, '$n$=2', va='center', ha='left',
                fontsize=7.5, color=color, fontstyle='italic')

ax.set_yticks(y_positions)
ax.set_yticklabels([f[0] for f in families], fontsize=9.5)
ax.set_xlabel(r'Cross-seed mean $\Delta$acc at d100 (pp)')
ax.set_xlim(-27, 7)
ax.xaxis.grid(True)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#555',
           markeredgecolor='#555', markersize=8, markeredgewidth=1.5,
           label='CI excludes zero'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
           markeredgecolor='#555', markersize=8, markeredgewidth=1.5,
           label='CI includes zero'),
]
ax.legend(handles=legend_elements, loc='lower right', frameon=True,
          fancybox=False, edgecolor='#cccccc', framealpha=1.0,
          handletextpad=0.4, borderpad=0.4, fontsize=8)

plt.tight_layout()

out_dir = Path(__file__).parent
fig.savefig(out_dir / 'fig_cross_seed_ci.pdf')
fig.savefig(out_dir / 'fig_cross_seed_ci.png')
print(f"Saved: {out_dir / 'fig_cross_seed_ci.pdf'}")
print(f"Saved: {out_dir / 'fig_cross_seed_ci.png'}")
plt.close()
