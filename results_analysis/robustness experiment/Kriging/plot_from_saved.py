#!/usr/bin/env python
"""从已保存的 .npy 重新生成 Kriging 鲁棒性图表 (无需重新运行 Kriging)"""

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

    zx = (bx.min() - 2, bx.max() + 2); zy = (by.min() - 2, by.max() + 2)
    vmin = np.nanmin(truth_grid[~mask_outside]); vmax = np.nanmax(truth_grid[~mask_outside])
    rlabel = LABELS[rk]

    # Result
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(grid_x, grid_y, result_grid, cmap='jet', shading='auto', vmin=vmin, vmax=vmax)
    ax.plot(bx, by, 'k-', linewidth=2)
    ax.set_xlim(zx); ax.set_ylim(zy)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
    ax.set_title(f'{rlabel}\nOrdinary Kriging')
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
    ax.set_title(f'{rlabel}\nResidual (Kriging)')
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
    ax.set_title(f'{rlabel}\n|Error| (Kriging)')
    cbar = plt.colorbar(im, ax=ax, label='Absolute Error (nT)')
    cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
    save_fig(fig, f'fig_error_{rk}')

    plt.close('all')
    print(f"  {rk} done (3 figs)")

print("All Kriging figures regenerated.")
