#!/usr/bin/env python
"""从已保存的 .npy 重新生成 U-Net+Transformer 鲁棒性图表 (无需重新训练)"""

import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SIZES = ['small', 'medium', 'large']
LABELS = {
    'small': 'Small (0.5deg x 0.5deg, NW)',
    'medium': 'Medium (1.0deg x 1.0deg, Center)',
    'large': 'Large (1.5deg x 1.5deg, SE)',
}


def save_fig(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(OUT_DIR, f'{name}.{ext}'), dpi=300,
                    bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


# Load results for RMSE
results = {}
results_path = os.path.join(OUT_DIR, 'results.json')
if os.path.exists(results_path):
    with open(results_path) as f:
        results = json.load(f)


for rk in SIZES:
    print(f"[{rk}] Loading...")
    result_grid = np.load(os.path.join(OUT_DIR, f'result_grid_{rk}.npy'))
    truth_grid = np.load(os.path.join(OUT_DIR, f'truth_grid_{rk}.npy'))
    grid_x = np.load(os.path.join(OUT_DIR, f'grid_x_{rk}.npy'))
    grid_y = np.load(os.path.join(OUT_DIR, f'grid_y_{rk}.npy'))
    bx = np.load(os.path.join(OUT_DIR, f'bx_{rk}.npy'))
    by = np.load(os.path.join(OUT_DIR, f'by_{rk}.npy'))
    mask_blank = np.load(os.path.join(OUT_DIR, f'mask_blank_{rk}.npy'))
    mask_outside = np.load(os.path.join(OUT_DIR, f'mask_outside_{rk}.npy'))

    rmse_val = results.get(rk, {}).get('rmse', None)
    best_ep = results.get(rk, {}).get('best_epoch', None)

    zx = (bx.min() - 2, bx.max() + 2); zy = (by.min() - 2, by.max() + 2)
    vmin = np.nanmin(truth_grid[~mask_outside]); vmax = np.nanmax(truth_grid[~mask_outside])
    rlabel = LABELS[rk]

    # Result
    title = f'{rlabel}\nU-Net + Transformer'
    if rmse_val: title += f' (RMSE={rmse_val:.2f} nT'
    if best_ep: title += f', best ep={best_ep}'
    if rmse_val: title += ')'

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(grid_x, grid_y, result_grid, cmap='jet', shading='auto', vmin=vmin, vmax=vmax)
    ax.plot(bx, by, 'k-', linewidth=2)
    ax.set_xlim(zx); ax.set_ylim(zy)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, label='Mag. Anomaly (nT)')
    cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
    save_fig(fig, f'fig_result_{rk}')

    # Residual
    residual = result_grid.copy()
    residual[~mask_blank] = np.nan
    residual[mask_blank] = result_grid[mask_blank] - truth_grid[mask_blank]
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(grid_x, grid_y, residual, cmap='RdBu_r', shading='auto', vmin=-100, vmax=100)
    ax.plot(bx, by, 'k-', linewidth=2)
    ax.set_xlim(zx); ax.set_ylim(zy)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
    ax.set_title(f'{rlabel}\nResidual (U-Net+TF, RMSE={rmse_val:.2f} nT)' if rmse_val else f'{rlabel}\nResidual (U-Net+TF)')
    cbar = plt.colorbar(im, ax=ax, label='Residual (nT)')
    cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
    save_fig(fig, f'fig_residual_{rk}')

    # Error
    abs_error = result_grid.copy()
    abs_error[~mask_blank] = np.nan
    abs_error[mask_blank] = np.abs(result_grid[mask_blank] - truth_grid[mask_blank])
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(grid_x, grid_y, abs_error, cmap='hot', shading='auto', vmin=0, vmax=100)
    ax.plot(bx, by, 'k-', linewidth=2)
    ax.set_xlim(zx); ax.set_ylim(zy)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
    ax.set_title(f'{rlabel}\n|Error| (U-Net+TF, RMSE={rmse_val:.2f} nT)' if rmse_val else f'{rlabel}\n|Error| (U-Net+TF)')
    cbar = plt.colorbar(im, ax=ax, label='Absolute Error (nT)')
    cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
    save_fig(fig, f'fig_error_{rk}')

    # RMSE curve
    hist_path = os.path.join(OUT_DIR, f'history_{rk}.json')
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            history = json.load(f)
        fig, ax = plt.subplots(figsize=(10, 5))
        eps = [h['epoch'] for h in history]; rmses = [h['rmse'] for h in history]
        ax.plot(eps, rmses, color='#FF5722', lw=2)
        best_idx = np.argmin(rmses)
        ax.scatter(eps[best_idx], rmses[best_idx], color='#FF5722', s=80, zorder=5, marker='*', edgecolors='black')
        ax.set_xlabel('Epoch'); ax.set_ylabel('RMSE (nT)')
        ax.set_title(f'{rlabel} — Test RMSE')
        ax.grid(True, alpha=0.3)
        save_fig(fig, f'fig_rmse_{rk}')

    plt.close('all')
    print(f"  {rk} done")

print("All U-Net+Transformer figures regenerated.")
