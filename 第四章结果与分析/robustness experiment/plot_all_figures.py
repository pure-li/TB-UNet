#!/usr/bin/env python
"""鲁棒性实验 — 全部图表生成 (U-Net+TF + Kriging)
==================================================
三区域: Small (0.5°), Medium (0.8°), Large (1.0°)
每区域: Result / Residual / Error × 两种方法
"""

import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
UNET_DIR = os.path.join(ROOT, 'U-net_transformer')
KRIG_DIR = os.path.join(ROOT, 'Kriging')
FIG_DIR = os.path.join(ROOT, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

SIZES = ['small', 'medium', 'large']
LABELS = {
    'small': 'Small (0.5deg, NW)',
    'medium': 'Medium (0.8deg, NW)',
    'large': 'Large (1.0deg, Center)',
}

# Shared colorbar ranges
VRES = (-100, 100)
VERR = (0, 100)


def save_fig(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'), dpi=300,
                    bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


# Load results
unet_results = {}
rp = os.path.join(UNET_DIR, 'results.json')
if os.path.exists(rp):
    with open(rp) as f: unet_results = json.load(f)

krig_results = {}
rp2 = os.path.join(KRIG_DIR, 'results.json')
if os.path.exists(rp2):
    with open(rp2) as f: krig_results = json.load(f)


def get_rmse(results_dict, key, which='rmse'):
    r = results_dict.get(key, {})
    v = r.get(which, None)
    return v


# ============================================================
# U-Net+TF plots
# ============================================================
for rk in SIZES:
    print(f"[U-Net+TF] {rk}...")
    result_grid = np.load(os.path.join(UNET_DIR, f'result_grid_{rk}.npy'))
    truth_grid = np.load(os.path.join(UNET_DIR, f'truth_grid_{rk}.npy'))
    grid_x = np.load(os.path.join(UNET_DIR, f'grid_x_{rk}.npy'))
    grid_y = np.load(os.path.join(UNET_DIR, f'grid_y_{rk}.npy'))
    bx = np.load(os.path.join(UNET_DIR, f'bx_{rk}.npy'))
    by = np.load(os.path.join(UNET_DIR, f'by_{rk}.npy'))
    mask_blank = np.load(os.path.join(UNET_DIR, f'mask_blank_{rk}.npy'))
    mask_outside = np.load(os.path.join(UNET_DIR, f'mask_outside_{rk}.npy'))

    rmse_val = get_rmse(unet_results, rk, 'rmse')
    best_ep = get_rmse(unet_results, rk, 'best_epoch')
    rlabel = LABELS[rk]

    zx = (bx.min() - 2, bx.max() + 2)
    zy = (by.min() - 2, by.max() + 2)
    vmin = np.nanmin(truth_grid[~mask_outside])
    vmax = np.nanmax(truth_grid[~mask_outside])

    # Result
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(grid_x, grid_y, result_grid, cmap='jet', shading='auto', vmin=vmin, vmax=vmax)
    ax.plot(bx, by, 'k-', linewidth=2)
    ax.set_xlim(zx); ax.set_ylim(zy)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
    ax.set_title(f'{rlabel}\nU-Net + Transformer (RMSE={rmse_val:.2f} nT, best ep={best_ep})')
    cbar = plt.colorbar(im, ax=ax, label='Mag. Anomaly (nT)')
    cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
    save_fig(fig, f'fig_unet_result_{rk}')

    # Residual (RdBu_r, ±100)
    residual = result_grid.copy()
    residual[~mask_blank] = np.nan
    residual[mask_blank] = result_grid[mask_blank] - truth_grid[mask_blank]
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(grid_x, grid_y, residual, cmap='RdBu_r', shading='auto', vmin=VRES[0], vmax=VRES[1])
    ax.plot(bx, by, 'k-', linewidth=2)
    ax.set_xlim(zx); ax.set_ylim(zy)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
    ax.set_title(f'{rlabel}\nResidual — U-Net+TF (RMSE={rmse_val:.2f} nT)')
    cbar = plt.colorbar(im, ax=ax, label='Residual (nT)')
    cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
    save_fig(fig, f'fig_unet_residual_{rk}')

    # Error (hot, 0-100)
    abs_error = result_grid.copy()
    abs_error[~mask_blank] = np.nan
    abs_error[mask_blank] = np.abs(result_grid[mask_blank] - truth_grid[mask_blank])
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(grid_x, grid_y, abs_error, cmap='hot', shading='auto', vmin=VERR[0], vmax=VERR[1])
    ax.plot(bx, by, 'k-', linewidth=2)
    ax.set_xlim(zx); ax.set_ylim(zy)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
    ax.set_title(f'{rlabel}\n|Error| — U-Net+TF (RMSE={rmse_val:.2f} nT)')
    cbar = plt.colorbar(im, ax=ax, label='Absolute Error (nT)')
    cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
    save_fig(fig, f'fig_unet_error_{rk}')

    # RMSE curve
    hist_path = os.path.join(UNET_DIR, f'history_{rk}.json')
    if os.path.exists(hist_path):
        with open(hist_path) as f: history = json.load(f)
        fig, ax = plt.subplots(figsize=(10, 5))
        eps = [h['epoch'] for h in history]; rmses = [h['rmse'] for h in history]
        ax.plot(eps, rmses, color='#FF5722', lw=2)
        best_idx = np.argmin(rmses)
        ax.scatter(eps[best_idx], rmses[best_idx], color='#FF5722', s=80, zorder=5, marker='*', edgecolors='black')
        ax.set_xlabel('Epoch'); ax.set_ylabel('RMSE (nT)')
        ax.set_title(f'{rlabel} — U-Net+TF Test RMSE')
        ax.grid(True, alpha=0.3)
        save_fig(fig, f'fig_unet_rmse_{rk}')

    plt.close('all')
    print(f"  {rk} done (RMSE={rmse_val:.2f})")


# ============================================================
# Kriging plots
# ============================================================
for rk in SIZES:
    print(f"[Kriging] {rk}...")
    result_grid = np.load(os.path.join(KRIG_DIR, f'result_grid_{rk}.npy'))
    truth_grid = np.load(os.path.join(KRIG_DIR, f'truth_grid_{rk}.npy'))
    grid_x = np.load(os.path.join(KRIG_DIR, f'grid_x_{rk}.npy'))
    grid_y = np.load(os.path.join(KRIG_DIR, f'grid_y_{rk}.npy'))
    bx = np.load(os.path.join(KRIG_DIR, f'bx_{rk}.npy'))
    by = np.load(os.path.join(KRIG_DIR, f'by_{rk}.npy'))
    mask_blank = np.load(os.path.join(KRIG_DIR, f'mask_blank_{rk}.npy'))
    mask_outside = np.load(os.path.join(KRIG_DIR, f'mask_outside_{rk}.npy'))

    rmse_val = get_rmse(krig_results, rk, 'rmse')
    rlabel = LABELS[rk]

    zx = (bx.min() - 2, bx.max() + 2)
    zy = (by.min() - 2, by.max() + 2)
    vmin = np.nanmin(truth_grid[~mask_outside])
    vmax = np.nanmax(truth_grid[~mask_outside])

    # Result
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(grid_x, grid_y, result_grid, cmap='jet', shading='auto', vmin=vmin, vmax=vmax)
    ax.plot(bx, by, 'k-', linewidth=2)
    ax.set_xlim(zx); ax.set_ylim(zy)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
    ax.set_title(f'{rlabel}\nOrdinary Kriging (RMSE={rmse_val:.2f} nT)')
    cbar = plt.colorbar(im, ax=ax, label='Mag. Anomaly (nT)')
    cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
    save_fig(fig, f'fig_krig_result_{rk}')

    # Residual (RdBu_r, ±100)
    residual = result_grid.copy()
    residual[~mask_blank] = np.nan
    residual[mask_blank] = result_grid[mask_blank] - truth_grid[mask_blank]
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(grid_x, grid_y, residual, cmap='RdBu_r', shading='auto', vmin=VRES[0], vmax=VRES[1])
    ax.plot(bx, by, 'k-', linewidth=2)
    ax.set_xlim(zx); ax.set_ylim(zy)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
    ax.set_title(f'{rlabel}\nResidual — Kriging (RMSE={rmse_val:.2f} nT)')
    cbar = plt.colorbar(im, ax=ax, label='Residual (nT)')
    cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
    save_fig(fig, f'fig_krig_residual_{rk}')

    # Error (hot, 0-100)
    abs_error = result_grid.copy()
    abs_error[~mask_blank] = np.nan
    abs_error[mask_blank] = np.abs(result_grid[mask_blank] - truth_grid[mask_blank])
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(grid_x, grid_y, abs_error, cmap='hot', shading='auto', vmin=VERR[0], vmax=VERR[1])
    ax.plot(bx, by, 'k-', linewidth=2)
    ax.set_xlim(zx); ax.set_ylim(zy)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
    ax.set_title(f'{rlabel}\n|Error| — Kriging (RMSE={rmse_val:.2f} nT)')
    cbar = plt.colorbar(im, ax=ax, label='Absolute Error (nT)')
    cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
    save_fig(fig, f'fig_krig_error_{rk}')

    plt.close('all')
    print(f"  {rk} done (RMSE={rmse_val:.2f})")


# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"  全部图表完成!")
print(f"{'='*60}")
print(f"  {'Region':<10s} {'U-Net+TF':>10s} {'Kriging':>10s} {'Gap':>10s}")
print(f"  {'-'*40}")
for rk in SIZES:
    u_rmse = get_rmse(unet_results, rk, 'rmse')
    k_rmse = get_rmse(krig_results, rk, 'rmse')
    gap = k_rmse - u_rmse if (u_rmse and k_rmse) else float('nan')
    print(f"  {LABELS[rk]:<10s} {u_rmse:10.2f} {k_rmse:10.2f} {gap:+10.1f}")
print(f"\n  输出: {FIG_DIR}/")
print("=" * 60)
