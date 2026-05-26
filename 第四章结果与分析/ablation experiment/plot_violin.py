#!/usr/bin/env python
"""只生成两张残差小提琴图 — rect + irreg
U-Net+TF (no skip) rect 用 comparative experiment 旧结果 (RMSE=19.92)
U-Net+TF (no skip) irreg 用本次重训结果 (RMSE=19.09)
"""
import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ROOT, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
COMP_DIR = os.path.join(os.path.dirname(ROOT), 'comparative experiment', 'U-net_transformer_slow')

MODELS = {
    'pure_unet':      'Pure U-Net',
    'unet_skip':      'U-Net + Skip',
    'unet_tf_noskip': 'U-Net + TF (no skip)',
    'unet_tf_skip':   'U-Net + TF + Skip',
}

MODEL_COLORS = {
    'pure_unet':      '#2196F3',
    'unet_skip':      '#4CAF50',
    'unet_tf_noskip': '#FF9800',
    'unet_tf_skip':   '#9C27B0',
}

REGIONS = {
    'rect':  'Rectangular (1.0°)',
    'irreg': 'Irregular Polygon',
}

all_results = {}
for mk in MODELS:
    rp = os.path.join(ROOT, mk, 'results.json')
    if os.path.exists(rp):
        with open(rp) as f:
            all_results[mk] = json.load(f)

def save_fig(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'), dpi=300,
                    bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

print("Generating violin plots...")

for rk, rlabel in REGIONS.items():
    fig, ax = plt.subplots(figsize=(14, 7))
    violin_data = []
    violin_names = []
    violin_rmse = []

    for mk, mn in MODELS.items():
        # U-Net+TF (no skip) rect → old comparative experiment
        if mk == 'unet_tf_noskip' and rk == 'rect':
            rp = os.path.join(COMP_DIR, f'result_grid_{rk}.npy')
            tp = os.path.join(COMP_DIR, f'truth_grid_{rk}.npy')
            mp = os.path.join(COMP_DIR, f'mask_blank_{rk}.npy')
            rmse_val = 19.92
        else:
            rp = os.path.join(ROOT, mk, f'result_grid_{rk}.npy')
            tp = os.path.join(ROOT, mk, f'truth_grid_{rk}.npy')
            mp = os.path.join(ROOT, mk, f'mask_blank_{rk}.npy')
            rmse_val = all_results.get(mk, {}).get(rk, {}).get('rmse', float('nan'))

        if not (os.path.exists(rp) and os.path.exists(tp) and os.path.exists(mp)):
            print(f"  [SKIP] Missing files for {mn} {rk}")
            continue

        result_grid = np.load(rp)
        truth_grid = np.load(tp)
        mask_blank = np.load(mp)
        residual = result_grid[mask_blank] - truth_grid[mask_blank]
        residual = residual[np.isfinite(residual)]

        violin_data.append(residual)
        violin_names.append(mn)
        violin_rmse.append(rmse_val)

    if len(violin_data) < 2:
        print(f"  [SKIP] Only {len(violin_data)} models for {rk}")
        continue

    positions = np.arange(len(violin_data))
    colors_list = [MODEL_COLORS[mk] for mk in MODELS
                   if (mk == 'unet_tf_noskip' and rk == 'rect')
                   or os.path.exists(os.path.join(ROOT, mk, f'result_grid_{rk}.npy'))]

    vp = ax.violinplot(violin_data, positions=positions, showmeans=True, showmedians=False,
                        widths=0.7, bw_method='scott')

    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(colors_list[i])
        body.set_alpha(0.6)
        body.set_edgecolor('black')
        body.set_linewidth(0.8)

    for part in ['cmeans', 'cmins', 'cmaxes', 'cbars']:
        if part in vp:
            vp[part].set_color('black')
            vp[part].set_linewidth(0.8)

    # Overlay jittered scatter points (downsampled)
    for i, data in enumerate(violin_data):
        if len(data) > 500:
            rng = np.random.default_rng(42)
            data_sample = rng.choice(data, size=500, replace=False)
        else:
            data_sample = data
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(data_sample))
        ax.scatter(np.full_like(data_sample, positions[i]) + jitter, data_sample,
                   color=colors_list[i], alpha=0.25, s=2, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels([f'{n}\n(RMSE={r:.1f} nT)' for n, r in zip(violin_names, violin_rmse)],
                       fontsize=9)
    ax.set_ylabel('Residual (nT)', fontsize=12)
    ax.set_title(f'{rlabel} — Residual Distribution by Model', fontsize=13)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    save_fig(fig, f'fig_violin_{rk}')
    print(f"  {rk} violin saved ({len(violin_data)} models)")

print(f"Done. Output: {FIG_DIR}/")
