"""Cross-benchmark transfer heatmap (Figure 2): 6 models × 3 benchmarks."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

plt.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
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

models = ['OLMo-2 7B', 'Falcon3 7B', 'Gemma-2 9B',
          'Qwen2.5-7B', 'Phi-4 14B', 'Qwen2.5-14B']
benchmarks = ['MMLU', 'ARC', 'GSM8K']

data = np.array([
    [+3.21, +3.92, +2.10],   # OLMo-2 (GSM8K: mean s42/s123)
    [+4.19, +4.01, +1.29],   # Falcon3
    [+3.76, -1.20, +1.33],   # Gemma-2 (ARC s42: -1.20pp, 16pp oscillation artifact)
    [-1.78, -2.90, +10.10],  # Qwen2.5-7B (ARC s42: -2.90pp; GSM8K: mean s42/s123)
    [-0.17, +0.60, -0.50],   # Phi-4 (ARC s42: +0.60pp)
    [-9.37, -3.40, +13.09],  # Qwen2.5-14B (ARC s42: -3.40pp)
])

fig, ax = plt.subplots(figsize=(6.5, 3.5))

vmax = 16.5
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
cmap = plt.cm.RdYlGn

masked = np.ma.masked_invalid(data)
im = ax.imshow(masked, cmap=cmap, norm=norm, aspect='auto')

# N/A cell hatching
na_mask = np.isnan(data)
for i in range(len(models)):
    for j in range(len(benchmarks)):
        if na_mask[i, j]:
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                         fill=True, facecolor='#f0f0f0', edgecolor='#cccccc',
                         linewidth=0.5, zorder=1))
            ax.text(j, i, 'N/A', ha='center', va='center',
                    fontsize=8, color='#999999', fontstyle='italic')
        else:
            val = data[i, j]
            intensity = abs(val) / vmax
            color = 'white' if intensity > 0.45 else 'black'
            sign = '+' if val > 0 else ''
            ax.text(j, i, f'{sign}{val:.1f}', ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold')

ax.set_xticks(range(len(benchmarks)))
ax.set_yticks(range(len(models)))
ax.set_xticklabels(benchmarks)
ax.set_yticklabels(models)

ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

for edge, spine in ax.spines.items():
    spine.set_visible(False)
ax.set_xticks(np.arange(len(benchmarks)+1)-.5, minor=True)
ax.set_yticks(np.arange(len(models)+1)-.5, minor=True)
ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
ax.tick_params(which="minor", bottom=False, left=False)

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
cbar.set_label('$\\Delta$acc (pp)', fontsize=9)
cbar.ax.tick_params(labelsize=8)

out = 'docs/paper/figures/fig_cross_benchmark_heatmap'
plt.savefig(out + '.pdf', facecolor='white', edgecolor='none')
plt.savefig(out + '.png', facecolor='white', edgecolor='none')
plt.close()
print(f'Saved: {out}.pdf')
