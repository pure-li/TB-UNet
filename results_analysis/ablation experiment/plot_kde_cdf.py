#!/usr/bin/env python
"""KDE + CDF residual distribution — 4 groups: U-Net, U-Net+Skip, TB-UNet, TB-UNet+Skip"""
import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ROOT, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
SWA_DIR = r'F:\PINN实验\venv\U-net\补充实验\SWA'

MODELS = {
    'pure_unet':      'U-Net',
    'swa':            'TB-UNet',
    'unet_tf_skip':   'TB-UNet+Skip',
}

MODEL_RMSE = {
    'pure_unet':      {'rect': 30.2, 'irreg': 26.7},
    'swa':            {'rect': 14.9, 'irreg': 17.3},
    'unet_tf_skip':   {'rect': 45.1, 'irreg': 57.4},
}

MODEL_COLORS = {
    'pure_unet':      '#2196F3',
    'swa':            '#d62728',
    'unet_tf_skip':   '#9C27B0',
}

def get_grid_paths(mk, rk):
    if mk == 'swa':
        rp = os.path.join(SWA_DIR, f'result_grid_{rk}.npy')
        tp = os.path.join(SWA_DIR, f'truth_grid_{rk}.npy')
        mp = os.path.join(SWA_DIR, f'mask_blank_{rk}.npy')
    else:
        rp = os.path.join(ROOT, mk, f'result_grid_{rk}.npy')
        tp = os.path.join(ROOT, mk, f'truth_grid_{rk}.npy')
        mp = os.path.join(ROOT, mk, f'mask_blank_{rk}.npy')
    return rp, tp, mp

REGIONS = {'rect': 'Rectangular (1.0°)', 'irreg': 'Irregular Polygon'}

def save_fig(fig, name):
    for ext in ['png', 'svg']:
        dpi = 1000 if ext == 'png' else 800
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'), dpi=dpi,
                    bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

for rk, rlabel in REGIONS.items():
    model_data = {}
    for mk, mn in MODELS.items():
        rp, tp, mp = get_grid_paths(mk, rk)
        if not (os.path.exists(rp) and os.path.exists(tp) and os.path.exists(mp)):
            print(f'  SKIP {mk}/{rk}: missing files')
            continue
        result_grid = np.load(rp)
        truth_grid = np.load(tp)
        mask_blank = np.load(mp)
        residual = result_grid[mask_blank] - truth_grid[mask_blank]
        residual = residual[np.isfinite(residual)]
        rmse_val = MODEL_RMSE[mk][rk]
        print(f'  {mk}/{rk}: n={len(residual)}, RMSE={rmse_val:.2f}')
        model_data[mk] = (mn, residual, rmse_val)

    # --- KDE ---
    fig_kde, ax_kde = plt.subplots(figsize=(10, 7))
    for mk, (mn, residual, rmse_val) in model_data.items():
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(residual)
        x_range = np.linspace(-100, 100, 500)
        density = kde(x_range)
        ax_kde.plot(x_range, density, color=MODEL_COLORS[mk], lw=2,
                    label=f'{mn} (RMSE={rmse_val:.1f})')
        ax_kde.fill_between(x_range, density, alpha=0.08, color=MODEL_COLORS[mk])
    ax_kde.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.4)
    ax_kde.set_xlabel('Residual (nT)', fontsize=16)
    ax_kde.set_ylabel('Probability Density', fontsize=16)
    ax_kde.tick_params(labelsize=14)
    ax_kde.legend(fontsize=12, loc='upper right')
    ax_kde.grid(True, alpha=0.3)
    ax_kde.set_xlim(-100, 100)
    fig_kde.tight_layout()
    save_fig(fig_kde, f'fig_kde_{rk}')
    print(f'  {rk} KDE saved')

    # --- CDF ---
    fig_cdf, ax_cdf = plt.subplots(figsize=(10, 7))
    for mk, (mn, residual, rmse_val) in model_data.items():
        sorted_data = np.sort(residual)
        cdf_y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax_cdf.plot(sorted_data, cdf_y, color=MODEL_COLORS[mk], lw=2,
                    label=f'{mn} (RMSE={rmse_val:.1f})')
    ax_cdf.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.4)
    ax_cdf.axhline(0.5, color='black', linestyle=':', linewidth=0.5, alpha=0.3)
    ax_cdf.set_xlabel('Residual (nT)', fontsize=16)
    ax_cdf.set_ylabel('Cumulative Probability', fontsize=16)
    ax_cdf.tick_params(labelsize=14)
    ax_cdf.legend(fontsize=16, loc='lower right')
    ax_cdf.grid(True, alpha=0.3)
    ax_cdf.set_xlim(-100, 100)
    fig_cdf.tight_layout()
    save_fig(fig_cdf, f'fig_cdf_{rk}')
    print(f'  {rk} CDF saved\n')

print(f'Done. Output: {FIG_DIR}/')
