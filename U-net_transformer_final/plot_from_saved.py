#!/usr/bin/env python
"""Regenerate all figures from saved .npy files (no retraining needed)
=========================================================
Usage:
  python plot_from_saved.py

Reads from current directory:
  - grid_x_{rect|irreg}.npy, grid_y_{rect|irreg}.npy
  - bx_{rect|irreg}.npy, by_{rect|irreg}.npy
  - mask_blank_{rect|irreg}.npy, mask_outside_{rect|irreg}.npy
  - truth_grid_{rect|irreg}.npy
  - result_grid_{rect|irreg}.npy
  - history_{rect|irreg}.json

Generates (PNG + SVG each):
  - fig_loss_{rect|irreg}
  - fig_rmse_{rect|irreg}
  - fig_truth_{rect|irreg}
  - fig_result_{rect|irreg}
  - fig_residual_{rect|irreg}
  - fig_error_{rect|irreg}
"""

import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

REGIONS = {'rect': 'Rectangular', 'irreg': 'Irregular'}
COLOR = '#FF5722'


def save_fig(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(SCRIPT_DIR, f'{name}.{ext}'), dpi=300,
                    bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def load_or_none(path):
    if os.path.exists(path):
        return np.load(path)
    return None


def main():
    print("=" * 60)
    print("  Regenerate figures from saved data (U-Net+Transformer)")
    print(f"  Data directory: {SCRIPT_DIR}")
    print("=" * 60)

    for region_key, region_label in REGIONS.items():
        print(f"\n[{region_label}] Loading data...")

        grid_x = load_or_none(os.path.join(SCRIPT_DIR, f'grid_x_{region_key}.npy'))
        grid_y = load_or_none(os.path.join(SCRIPT_DIR, f'grid_y_{region_key}.npy'))
        bx = load_or_none(os.path.join(SCRIPT_DIR, f'bx_{region_key}.npy'))
        by = load_or_none(os.path.join(SCRIPT_DIR, f'by_{region_key}.npy'))
        mask_blank = load_or_none(os.path.join(SCRIPT_DIR, f'mask_blank_{region_key}.npy'))
        mask_outside = load_or_none(os.path.join(SCRIPT_DIR, f'mask_outside_{region_key}.npy'))
        truth_grid = load_or_none(os.path.join(SCRIPT_DIR, f'truth_grid_{region_key}.npy'))
        result_grid = load_or_none(os.path.join(SCRIPT_DIR, f'result_grid_{region_key}.npy'))

        if any(x is None for x in [grid_x, grid_y, bx, by, mask_blank, mask_outside, truth_grid, result_grid]):
            print(f"  Warning: missing required data, skipping {region_key}")
            continue

        # Load training history
        history = None
        hist_path = os.path.join(SCRIPT_DIR, f'history_{region_key}.json')
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                history = json.load(f)

        # Load results summary (for RMSE)
        rmse = None
        results_path = os.path.join(SCRIPT_DIR, 'results.json')
        if os.path.exists(results_path):
            with open(results_path) as f:
                results = json.load(f)
            if region_key in results:
                rmse = results[region_key].get('rmse')

        zx = (bx.min() - 2, bx.max() + 2)
        zy = (by.min() - 2, by.max() + 2)
        vmin = np.nanmin(truth_grid[~mask_outside])
        vmax = np.nanmax(truth_grid[~mask_outside])

        # ---- Loss curve ----
        if history:
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot([h['epoch'] for h in history], [h['loss'] for h in history],
                     color=COLOR, lw=2)
            ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
            ax1.set_title(f'{region_label} Blank — Training Loss (U-Net+Transformer)')
            ax1.grid(True, alpha=0.3)
            save_fig(fig1, f'fig_loss_{region_key}')

        # ---- RMSE curve ----
        if history:
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            eps = [h['epoch'] for h in history]
            rmses = [h['rmse'] for h in history]
            ax2.plot(eps, rmses, color=COLOR, lw=2)
            best_rmse_val = min(rmses)
            best_ep = eps[rmses.index(best_rmse_val)]
            ax2.scatter(best_ep, best_rmse_val, color=COLOR, s=80, zorder=5,
                       marker='*', edgecolors='black')
            ax2.set_xlabel('Epoch'); ax2.set_ylabel('RMSE (nT)')
            ax2.set_title(f'{region_label} Blank — Test RMSE (U-Net+Transformer)')
            ax2.grid(True, alpha=0.3)
            save_fig(fig2, f'fig_rmse_{region_key}')

        # ---- Ground truth map ----
        fig3, ax3 = plt.subplots(figsize=(10, 9))
        im = ax3.pcolormesh(grid_x, grid_y, truth_grid, cmap='jet', shading='auto',
                           vmin=vmin, vmax=vmax)
        ax3.plot(bx, by, 'k-', linewidth=2)
        ax3.set_xlim(zx); ax3.set_ylim(zy)
        ax3.set_xlabel('X (km)'); ax3.set_ylabel('Y (km)')
        ax3.set_title(f'{region_label} Blank — Ground Truth')
        cbar = plt.colorbar(im, ax=ax3, label='Mag. Anomaly (nT)')
        cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
        save_fig(fig3, f'fig_truth_{region_key}')

        # ---- Result map ----
        fig4, ax4 = plt.subplots(figsize=(10, 9))
        im = ax4.pcolormesh(grid_x, grid_y, result_grid, cmap='jet', shading='auto',
                           vmin=vmin, vmax=vmax)
        ax4.plot(bx, by, 'k-', linewidth=2)
        ax4.set_xlim(zx); ax4.set_ylim(zy)
        ax4.set_xlabel('X (km)'); ax4.set_ylabel('Y (km)')
        ax4.set_title(f'{region_label} Blank — U-Net + Transformer')
        cbar = plt.colorbar(im, ax=ax4, label='Mag. Anomaly (nT)')
        cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
        save_fig(fig4, f'fig_result_{region_key}')

        # ---- Residual (pred - true) ----
        residual = result_grid.copy()
        residual[~mask_blank] = np.nan
        residual[mask_blank] = result_grid[mask_blank] - truth_grid[mask_blank]
        rmin = np.nanmin(residual[mask_blank])
        rmax = np.nanmax(residual[mask_blank])
        res_max = max(abs(rmin), abs(rmax), 1.0)

        fig5, ax5 = plt.subplots(figsize=(10, 9))
        im = ax5.pcolormesh(grid_x, grid_y, residual, cmap='RdBu_r', shading='auto',
                           vmin=-100, vmax=100)
        ax5.plot(bx, by, 'k-', linewidth=2)
        ax5.set_xlim(zx); ax5.set_ylim(zy)
        ax5.set_xlabel('X (km)'); ax5.set_ylabel('Y (km)')
        title = f'{region_label} Blank — Residual (Pred - True)\nU-Net + Transformer'
        if rmse: title += f', RMSE={rmse:.2f} nT'
        ax5.set_title(title)
        cbar = plt.colorbar(im, ax=ax5, label='Residual (nT)')
        cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
        save_fig(fig5, f'fig_residual_{region_key}')

        # ---- Absolute error ----
        abs_error = result_grid.copy()
        abs_error[~mask_blank] = np.nan
        abs_error[mask_blank] = np.abs(result_grid[mask_blank] - truth_grid[mask_blank])
        ae_max = np.nanmax(abs_error[mask_blank])

        fig6, ax6 = plt.subplots(figsize=(10, 9))
        im = ax6.pcolormesh(grid_x, grid_y, abs_error, cmap='hot', shading='auto',
                           vmin=0, vmax=100)
        ax6.plot(bx, by, 'k-', linewidth=2)
        ax6.set_xlim(zx); ax6.set_ylim(zy)
        ax6.set_xlabel('X (km)'); ax6.set_ylabel('Y (km)')
        title = f'{region_label} Blank — |Error|\nU-Net + Transformer'
        if rmse: title += f', RMSE={rmse:.2f} nT'
        ax6.set_title(title)
        cbar = plt.colorbar(im, ax=ax6, label='Absolute Error (nT)')
        cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
        save_fig(fig6, f'fig_error_{region_key}')

        print(f"  [{region_label}] Done (6 figures)")

    print(f"\n{'='*60}")
    print("  All figures regenerated!")
    print(f"  Output: {SCRIPT_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
