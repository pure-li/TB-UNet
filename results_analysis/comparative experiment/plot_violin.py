#!/usr/bin/env python
"""小提琴图 — 误差分布对比 (4 方法 × 2 区域)"""

import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))

METHODS = {
    'U-Net+Transformer': 'U-net_transformer_slow',
    'GAN': 'gan_results',
    'CNN (Block)': 'CNN_results',
    'Kriging': 'Kriging',
}
REGIONS = {'rect': 'Rectangular', 'irreg': 'Irregular'}
COLORS = {
    'U-Net+Transformer': '#2196F3',
    'GAN': '#FF9800',
    'CNN (Block)': '#4CAF50',
    'Kriging': '#9C27B0',
}

def load_errors(method_dir, region_key):
    result = np.load(os.path.join(method_dir, f'result_grid_{region_key}.npy'))
    truth = np.load(os.path.join(method_dir, f'truth_grid_{region_key}.npy'))
    mask_blank = np.load(os.path.join(method_dir, f'mask_blank_{region_key}.npy'))
    vals = result[mask_blank] - truth[mask_blank]
    return vals[~np.isnan(vals)]

def save_fig(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(ROOT, f'{name}.{ext}'), dpi=300,
                    bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

# 收集数据
all_data = {}
for rk, rlabel in REGIONS.items():
    all_data[rk] = {}
    for mn, fn in METHODS.items():
        errs = load_errors(os.path.join(ROOT, fn), rk)
        all_data[rk][mn] = np.clip(errs, -200, 200)

# =========================================================================
# 图1: 分区域画 (推荐)
# =========================================================================
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 7))

for ax, (rk, rlabel) in zip([ax0, ax1], REGIONS.items()):
    data_list = [all_data[rk][mn] for mn in METHODS]
    positions = range(len(METHODS))
    vp = ax.violinplot(data_list, positions=positions, showmeans=True,
                       showmedians=True, widths=0.7)

    for i, (mn, body) in enumerate(zip(METHODS, vp['bodies'])):
        body.set_facecolor(COLORS[mn])
        body.set_alpha(0.6)
    for part in ['cmeans', 'cmedians']:
        vp[part].set_color('black')
        vp[part].set_linewidth(1.5)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(list(METHODS.keys()), fontsize=9)
    ax.set_ylabel('Error (nT)')
    ax.set_title(f'{rlabel} Blank — Error Distribution')
    ax.set_ylim(-250, 250)
    ax.grid(True, alpha=0.2, axis='y')

    # 标注 RMSE
    for i, mn in enumerate(METHODS):
        rmse = np.sqrt(np.mean(all_data[rk][mn]**2))
        ax.text(i, ax.get_ylim()[0] + 15, f'RMSE={rmse:.1f}',
                ha='center', fontsize=7, fontweight='bold')

fig.tight_layout()
save_fig(fig, 'fig_violin')

# =========================================================================
# 图2: 两区域叠加 (每方法两个violin)
# =========================================================================
fig, ax = plt.subplots(figsize=(14, 7))

n_methods = len(METHODS)
positions_rect = np.arange(n_methods) * 2.5
positions_irreg = positions_rect + 0.8

for i, (mn, color) in enumerate(COLORS.items()):
    vp_r = ax.violinplot([all_data['rect'][mn]], positions=[positions_rect[i]],
                         showmeans=True, showmedians=True, widths=0.7)
    vp_i = ax.violinplot([all_data['irreg'][mn]], positions=[positions_irreg[i]],
                         showmeans=True, showmedians=True, widths=0.7)
    for vp in [vp_r, vp_i]:
        for body in vp['bodies']:
            body.set_facecolor(color)
            body.set_alpha(0.5 if vp is vp_i else 0.8)
        for part in ['cmeans', 'cmedians']:
            vp[part].set_color('black')
            vp[part].set_linewidth(1.2)

    # RMSE
    for pos, rk in [(positions_rect[i], 'rect'), (positions_irreg[i], 'irreg')]:
        rmse = np.sqrt(np.mean(all_data[rk][mn]**2))
        ax.text(pos, -235, f'{rmse:.1f}', ha='center', fontsize=7, fontweight='bold',
                color=color)

ax.set_xticks(list(positions_rect) + list(positions_irreg))
ax.set_xticklabels(
    [f'{mn}\n(Rect)' for mn in METHODS] + [f'{mn}\n(Irreg)' for mn in METHODS],
    fontsize=8)
ax.set_ylabel('Error (nT)')
ax.set_title('Error Distribution — All Methods × Both Regions')
ax.set_ylim(-260, 260)
ax.grid(True, alpha=0.2, axis='y')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=COLORS[mn], alpha=0.6, label=mn) for mn in METHODS]
ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

fig.tight_layout()
save_fig(fig, 'fig_violin_combined')

print("Done: fig_violin.png|svg, fig_violin_combined.png|svg")
