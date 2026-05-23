import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

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

# Data
domains = ['STEM: Math', 'STEM: Science', 'STEM: CS/Eng', 'Business/Law', 'Medical', 'Humanities', 'Social Sci.']
gamma_mean = [0.9688, 0.8376, 0.1967, -0.1677, -0.1986, -0.4073, -0.4965]
ci_lower = [0.5267, 0.3853, -0.4477, -0.5379, -0.3379, -0.7760, -0.8272]
ci_upper = [1.4300, 1.2743, 0.8353, 0.2299, -0.0599, -0.0406, -0.1511]
ppp = [0.438, 0.807, 0.892, 1.000, 1.000, 1.000, 1.000]

# Categories
colors_map = {
    'good': '#009E73',
    'null': '#999999',
    'misfit': '#CC79A7',
}
categories = ['good', 'good', 'null', 'null', 'misfit', 'misfit', 'misfit']
colors = [colors_map[c] for c in categories]

# Pooled
pooled_gamma = 0.6551
pooled_ci = [-0.4056, 1.7366]

fig, ax = plt.subplots(figsize=(6.5, 4))

# Background shading for STEM vs non-STEM
ax.axhspan(4.5, 7.5, color='#E8F5E9', alpha=0.4, zorder=0)
ax.axhspan(-0.5, 4.5, color='#FFEBEE', alpha=0.3, zorder=0)

# Vertical null line
ax.axvline(x=0, color='#444444', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)

# Plot domain-level estimates
y_positions = np.arange(len(domains))[::-1]  # top to bottom

for i, (y, gm, cl, cu, col) in enumerate(zip(y_positions, gamma_mean, ci_lower, ci_upper, colors)):
    ax.hlines(y, cl, cu, color=col, linewidth=2.2, zorder=2)
    ax.plot(gm, y, 'o', color=col, markersize=9, zorder=3, markeredgecolor='white', markeredgewidth=0.8)
    # PPP annotation on the right
    ppp_text = f"PPP={ppp[i]:.2f}" if ppp[i] < 1.0 else "PPP=1.00"
    ax.text(1.65, y, ppp_text, va='center', ha='left', fontsize=9, color=col, fontweight='medium')

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

# Group labels on background bands (right edge)
ax.text(-1.05, 6.85, 'STEM', fontsize=9.5, color='#009E73', fontweight='bold',
        va='top', ha='left', alpha=0.8)
ax.text(-1.05, 3.15, 'Non-STEM', fontsize=9.5, color='#CC79A7', fontweight='bold',
        va='bottom', ha='left', alpha=0.8)

# Dilution masking label next to pooled bar
ax.text(pooled_ci[1] + 0.12, pooled_y, 'n.s. — dilution masking',
        va='center', ha='left', fontsize=8.5, color='#555555', fontstyle='italic')
# Remove the earlier n.s. text (replaced by combined label)

# Axis formatting
ax.set_xlabel(r'Contamination effect ($\gamma$)', fontsize=12)
ax.set_xlim(-1.1, 2.35)
ax.set_ylim(-2.0, 7.2)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#009E73', markersize=9, label='Significant + good fit'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#999999', markersize=9, label='Non-significant'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#CC79A7', markersize=9, label='Model misfit (PPP=1.0)'),
]
ax.legend(handles=legend_elements, loc='upper center', framealpha=0.9, edgecolor='#CCCCCC',
          bbox_to_anchor=(0.55, 1.0), ncol=3, columnspacing=1.0, handletextpad=0.3)

ax.xaxis.grid(True)

plt.tight_layout()

out_dir = './figures/paper'
fig.savefig(f'{out_dir}/fig_domain_gamma.pdf', format='pdf')
fig.savefig(f'{out_dir}/fig_domain_gamma.png', format='png')
plt.close()
print("Done: fig_domain_gamma.pdf and fig_domain_gamma.png saved.")
