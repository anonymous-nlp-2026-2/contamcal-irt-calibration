#!/usr/bin/env python3
"""Generate Figure 1: MMLU dose-response curves — 2-panel layout.

Left panel: Benefit + Immune models.
Right panel: Collapse + V-recovery models.
"""

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

MODEL_COLORS = {
    'Gemma-2 9B':    '#2ca02c',
    'Falcon3-7B':    '#5fba5f',
    'OLMo-2 7B':     '#66c266',
    'Phi-4 14.7B':   '#7f7f7f',
    'LLaMA-3.1 8B':  '#b0b0b0',
    'Qwen2.5-14B':   '#d62728',
    'Qwen2.5-7B':    '#ff7f0e',
}

MODEL_MARKERS = {
    'Gemma-2 9B':    's',
    'Falcon3-7B':    '^',
    'OLMo-2 7B':     'o',
    'Phi-4 14.7B':   'D',
    'LLaMA-3.1 8B':  'p',
    'Qwen2.5-14B':   'v',
    'Qwen2.5-7B':    '<',
}

xpos = np.arange(5)
dosage_labels = ['0%', '0.5%', '2.5%', '5%', '10%']

models = {
    'Phi-4 14.7B':   [76.44, 76.36, 76.98, 76.75, 76.27],
    'Qwen2.5-14B':   [69.44, 69.01, 62.04, 59.41, 60.07],
    'Qwen2.5-7B':    [66.11, 64.74, 62.98, 63.07, 64.33],
    'Gemma-2 9B':    [61.86, 63.42, 64.56, 64.01, 65.62],
    'Falcon3-7B':    [58.77, 59.02, 61.23, 62.09, 62.96],
    'OLMo-2 7B':     [53.79, 56.14, 56.17, 57.25, 57.00],
    'LLaMA-3.1 8B':  [53.50, None, 52.75, None, 52.81],
}

left_panel = ['OLMo-2 7B', 'LLaMA-3.1 8B', 'Falcon3-7B', 'Gemma-2 9B', 'Phi-4 14.7B']
right_panel = ['Qwen2.5-7B', 'Qwen2.5-14B']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.5),
                                gridspec_kw={'width_ratios': [5, 4]})

def plot_panel(ax, model_list, title):
    for name in model_list:
        accs = models[name]
        xs = [i for i, a in enumerate(accs) if a is not None]
        ys = [a for a in accs if a is not None]
        ax.plot(xs, ys, color=MODEL_COLORS[name], marker=MODEL_MARKERS[name],
                markersize=6, linestyle='-', linewidth=1.8, zorder=3,
                markeredgecolor='white', markeredgewidth=0.4, label=name)

    ax.set_xticks(xpos)
    ax.set_xticklabels(dosage_labels)
    ax.set_xlabel('Contamination Dosage')
    ax.set_title(title, fontsize=11, fontweight='medium')
    ax.yaxis.grid(True)

plot_panel(ax1, left_panel, 'Benefit & Immune')
plot_panel(ax2, right_panel, 'Collapse & V-recovery')

ax1.set_ylabel('MMLU 5-shot Accuracy (%)')
ax1.set_xlim(-0.3, 4.3)
ax1.set_ylim(50, 82)
ax2.set_xlim(-0.3, 4.3)
ax2.set_ylim(56, 72)

ax1.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='#cccccc',
           framealpha=0.9, handlelength=1.8, labelspacing=0.3, borderpad=0.4,
           fontsize=7.5)
ax2.legend(loc='lower left', frameon=True, fancybox=False, edgecolor='#cccccc',
           framealpha=0.9, handlelength=1.8, labelspacing=0.3, borderpad=0.4,
           fontsize=8)

plt.tight_layout(w_pad=2.0)

out_dir = Path(__file__).parent
fig.savefig(out_dir / 'fig_dose_response.pdf')
fig.savefig(out_dir / 'fig_dose_response.png')
print(f"Saved: {out_dir / 'fig_dose_response.pdf'}")
print(f"Saved: {out_dir / 'fig_dose_response.png'}")
plt.close()
