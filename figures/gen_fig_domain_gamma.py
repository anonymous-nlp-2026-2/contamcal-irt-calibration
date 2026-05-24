import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

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

# Data: J=13 Qwen2.5 (8×1.5B + 5×3B), from mmlu_domain_irt_no05b_results.md
domains = ['STEM: Math', 'STEM: Science', 'STEM: CS/Eng', 'Business/Law', 'Medical', 'Humanities', 'Social Sci.']
gamma_mean = [0.499, 0.504, 0.497, 0.044, 0.235, -0.309, -0.571]
ci_lower = [-0.213, -0.189, -0.425, -0.497, -0.421, -0.852, -1.054]
ci_upper = [1.203, 1.202, 1.397, 0.579, 0.879, 0.257, -0.103]
ppp = [0.206, 0.149, 0.392, 0.299, 0.124, 0.530, 0.570]

# Categories: only Social Sci has CI excluding zero at J=13
colors_map = {
    'good': '#009E73',
    'null': '#999999',
}
categories = ['null', 'null', 'null', 'null', 'null', 'null', 'good']
colors = [colors_map[c] for c in categories]

# Pooled J=13
pooled_gamma = 0.107
pooled_ci = [-0.401, 0.625]

fig, ax = plt.subplots(figsize=(6.5, 4))

# Background shading for STEM vs non-STEM (STEM: Math y=6, Science y=5, CS/Eng y=4)
ax.axhspan(3.5, 7.5, color='#E8F5E9', alpha=0.4, zorder=0)
ax.axhspan(-0.5, 3.5, color='#FFEBEE', alpha=0.3, zorder=0)

# Vertical null line
ax.axvline(x=0, color='#444444', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)

# Plot domain-level estimates
y_positions = np.arange(len(domains))[::-1]  # top to bottom

for i, (y, gm, cl, cu, col) in enumerate(zip(y_positions, gamma_mean, ci_lower, ci_upper, colors)):
    ax.hlines(y, cl, cu, color=col, linewidth=2.2, zorder=2)
    ax.plot(gm, y, 'o', color=col, markersize=9, zorder=3, markeredgecolor='white', markeredgewidth=0.8)
    # PPP annotation on the right
    ppp_text = f"PPP={ppp[i]:.2f}" if ppp[i] < 1.0 else "PPP=1.00"
    ax.text(1.55, y, ppp_text, va='center', ha='left', fontsize=8, color=col, fontweight='medium')

# Pooled MMLU bar (below domains)
pooled_y = -1.3
ax.hlines(pooled_y, pooled_ci[0], pooled_ci[1], color='#555555', linewidth=2.0, linestyle='-', zorder=2)
ax.plot(pooled_gamma, pooled_y, 'D', color='#555555', markersize=8, zorder=3, markeredgecolor='white', markeredgewidth=0.8)

# Separator line between domains and pooled
ax.axhline(y=-0.5, color='#BBBBBB', linewidth=0.6, linestyle='-')

# Y-axis labels
ax.set_yticks(list(y_positions) + [pooled_y])
ax.set_yticklabels(domains + ['Pooled MMLU'], fontsize=10.5)

# Make pooled label italic
ytick_labels = ax.get_yticklabels()
ytick_labels[-1].set_fontstyle('italic')
ytick_labels[-1].set_color('#555555')

# Group labels on background bands
ax.text(-1.05, 6.85, 'STEM', fontsize=9.5, color='#009E73', fontweight='bold',
        va='top', ha='left', alpha=0.8)
ax.text(-1.05, 2.15, 'Non-STEM', fontsize=9.5, color='#CC79A7', fontweight='bold',
        va='bottom', ha='left', alpha=0.8)

# Dilution masking label next to pooled bar
ax.text(pooled_ci[1] + 0.12, pooled_y, 'n.s. — dilution masking',
        va='center', ha='left', fontsize=8.5, color='#555555', fontstyle='italic')

# Axis formatting
ax.set_xlabel(r'Contamination effect ($\gamma$)', fontsize=12)
ax.set_xlim(-1.2, 1.8)
ax.set_ylim(-2.0, 7.2)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#009E73', markersize=9, label='CI excludes zero'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#999999', markersize=9, label='Non-significant'),
]
ax.legend(handles=legend_elements, loc='upper center', framealpha=0.9, edgecolor='#CCCCCC',
          bbox_to_anchor=(0.45, 1.08), ncol=2, columnspacing=1.0, handletextpad=0.3)

ax.xaxis.grid(True)

plt.tight_layout()

out_dir = './figures/paper'
fig.savefig(f'{out_dir}/fig_domain_gamma.pdf', format='pdf')
fig.savefig(f'{out_dir}/fig_domain_gamma.png', format='png')
plt.close()
print("Done: fig_domain_gamma.pdf and fig_domain_gamma.png saved.")
