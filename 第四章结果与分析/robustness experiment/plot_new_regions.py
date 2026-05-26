"""Generate result / residual / error maps for NW 0.5deg v2 and SE 1.5deg regions (TB-UNet + Kriging)"""
import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = r'F:\PINN实验\venv\U-net\robustness experiment'
SWA_DIR = os.path.join(ROOT, 'SWA')
KRIG_DIR = os.path.join(ROOT, 'Kriging')
FIG_DIR = os.path.join(ROOT, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

REGIONS = {
    'nw05_v2': 'NW (0.5deg x 0.5deg)',
    'se15': 'SE (1.5deg x 1.5deg)',
}

VRES = (-100, 100)
VERR = (0, 100)


def save_fig(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'), dpi=300,
                    bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def compute_boundary(mask_blank, grid_x, grid_y):
    idx = np.where(mask_blank)
    pts = np.column_stack([grid_x[idx], grid_y[idx]])
    hull = ConvexHull(pts)
    bx = pts[hull.vertices, 0]
    by = pts[hull.vertices, 1]
    bx = np.append(bx, bx[0])
    by = np.append(by, by[0])
    return bx, by


def fill_nan_in_blank(result, mask_blank, grid_x, grid_y):
    """Fill NaN inside blank region using nearest-neighbor interpolation."""
    blank_nan = mask_blank & np.isnan(result)
    if blank_nan.sum() > 0:
        valid_mask = mask_blank & ~np.isnan(result)
        valid_pts = np.column_stack([grid_x[valid_mask], grid_y[valid_mask]])
        valid_vals = result[valid_mask]
        nan_pts = np.column_stack([grid_x[blank_nan], grid_y[blank_nan]])
        result[blank_nan] = griddata(valid_pts, valid_vals, nan_pts, method='nearest')
        print(f"    Filled {blank_nan.sum()} NaN in blank region")
    return result


def plot_tbunet(rk, result_grid_path, rmse_val, label):
    """Plot TB-UNet result/residual/error using given result grid."""
    result_grid = np.load(result_grid_path)
    truth_grid = np.load(os.path.join(SWA_DIR, f'truth_grid_{rk}.npy'))
    grid_x = np.load(os.path.join(SWA_DIR, f'grid_x_{rk}.npy'))
    grid_y = np.load(os.path.join(SWA_DIR, f'grid_y_{rk}.npy'))
    mask_blank = np.load(os.path.join(SWA_DIR, f'mask_blank_{rk}.npy'))
    mask_outside = np.load(os.path.join(SWA_DIR, f'mask_outside_{rk}.npy'))
    bx = np.load(os.path.join(SWA_DIR, f'bx_{rk}.npy'))
    by = np.load(os.path.join(SWA_DIR, f'by_{rk}.npy'))

    # Fill NaN in blank region before computing residuals
    result_grid = fill_nan_in_blank(result_grid, mask_blank, grid_x, grid_y)

    rlabel = REGIONS[rk]
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
    ax.set_title(f'{rlabel}\n{label} (RMSE={rmse_val:.2f} nT)')
    cbar = plt.colorbar(im, ax=ax, label='Mag. Anomaly (nT)')
    cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
    save_fig(fig, f'fig_unet_result_{rk}')

    # Residual
    residual = result_grid.copy()
    residual[~mask_blank] = np.nan
    residual[mask_blank] = result_grid[mask_blank] - truth_grid[mask_blank]
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(grid_x, grid_y, residual, cmap='RdBu_r', shading='auto', vmin=VRES[0], vmax=VRES[1])
    ax.plot(bx, by, 'k-', linewidth=2)
    ax.set_xlim(zx); ax.set_ylim(zy)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
    ax.set_title(f'{rlabel}\nResidual — {label} (RMSE={rmse_val:.2f} nT)')
    cbar = plt.colorbar(im, ax=ax, label='Residual (nT)')
    cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
    save_fig(fig, f'fig_unet_residual_{rk}')

    # |Error|
    abs_error = result_grid.copy()
    abs_error[~mask_blank] = np.nan
    abs_error[mask_blank] = np.abs(result_grid[mask_blank] - truth_grid[mask_blank])
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(grid_x, grid_y, abs_error, cmap='YlOrRd', shading='auto', vmin=VERR[0], vmax=VERR[1])
    ax.plot(bx, by, 'k-', linewidth=2)
    ax.set_xlim(zx); ax.set_ylim(zy)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
    ax.set_title(f'{rlabel}\n|Error| — {label} (RMSE={rmse_val:.2f} nT)')
    cbar = plt.colorbar(im, ax=ax, label='Absolute Error (nT)')
    cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
    save_fig(fig, f'fig_unet_error_{rk}')

    plt.close('all')
    print(f"  {label} {rk}: result + residual + error done (RMSE={rmse_val:.2f})")


def plot_kriging(rk, rmse_val):
    """Plot Kriging result/residual/error."""
    result_grid = np.load(os.path.join(KRIG_DIR, f'result_grid_{rk}.npy'))
    truth_grid = np.load(os.path.join(KRIG_DIR, f'truth_grid_{rk}.npy'))
    grid_x = np.load(os.path.join(KRIG_DIR, f'grid_x_{rk}.npy'))
    grid_y = np.load(os.path.join(KRIG_DIR, f'grid_y_{rk}.npy'))
    mask_blank = np.load(os.path.join(KRIG_DIR, f'mask_blank_{rk}.npy'))
    mask_outside = np.load(os.path.join(KRIG_DIR, f'mask_outside_{rk}.npy'))

    bx_path = os.path.join(KRIG_DIR, f'bx_{rk}.npy')
    if os.path.exists(bx_path):
        bx = np.load(bx_path)
        by = np.load(os.path.join(KRIG_DIR, f'by_{rk}.npy'))
    else:
        bx, by = compute_boundary(mask_blank, grid_x, grid_y)

    rlabel = REGIONS[rk]
    zx = (bx.min() - 2, bx.max() + 2)
    zy = (by.min() - 2, by.max() + 2)
    vmin = np.nanmin(truth_grid[~mask_outside])
    vmax = np.nanmax(truth_grid[~mask_outside])

    # Fill NaN in blank region before computing residuals
    result_grid = fill_nan_in_blank(result_grid, mask_blank, grid_x, grid_y)

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

    # Residual
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

    # |Error|
    abs_error = result_grid.copy()
    abs_error[~mask_blank] = np.nan
    abs_error[mask_blank] = np.abs(result_grid[mask_blank] - truth_grid[mask_blank])
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(grid_x, grid_y, abs_error, cmap='YlOrRd', shading='auto', vmin=VERR[0], vmax=VERR[1])
    ax.plot(bx, by, 'k-', linewidth=2)
    ax.set_xlim(zx); ax.set_ylim(zy)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
    ax.set_title(f'{rlabel}\n|Error| — Kriging (RMSE={rmse_val:.2f} nT)')
    cbar = plt.colorbar(im, ax=ax, label='Absolute Error (nT)')
    cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
    save_fig(fig, f'fig_krig_error_{rk}')

    plt.close('all')
    print(f"  Kriging {rk}: result + residual + error done (RMSE={rmse_val:.2f})")


# ============================================================
# NW 0.5deg v2 — TB-UNet Best
# ============================================================
print("=== TB-UNet Best (nw05_v2) ===")
plot_tbunet('nw05_v2', os.path.join(SWA_DIR, 'result_grid_nw05_v2.npy'), 9.19, 'TB-UNet')

# ============================================================
# SE 1.5deg — TB-UNet Best
# ============================================================
print("=== TB-UNet Best (se15) ===")
plot_tbunet('se15', os.path.join(SWA_DIR, 'result_grid_se15.npy'), 51.44, 'TB-UNet')

# ============================================================
# Kriging
# ============================================================
print("=== Kriging (nw05_v2) ===")
plot_kriging('nw05_v2', 30.08)
print("=== Kriging (se15) ===")
plot_kriging('se15', 61.77)

print(f"\nDone! Output: {FIG_DIR}/")
