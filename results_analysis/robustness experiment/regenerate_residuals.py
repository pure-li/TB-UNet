"""Regenerate all residual & |error| maps for 3 regions — 16pt, no titles, normal formatting"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.interpolate import griddata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
SWA_DIR = os.path.join(ROOT, 'SWA')
KRIG_DIR = os.path.join(ROOT, 'Kriging')
UNET_DIR = os.path.join(ROOT, 'U-net_transformer')
FIG_DIR = os.path.join(ROOT, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Region config: (key, label, vres, verr, unet_src_dir, unet_src_key, krig_src_dir, krig_src_key)
REGIONS = [
    ('nw05_v2',  'NW 0.5deg',    (-50, 50),   (0, 50),    SWA_DIR,  'nw05_v2',  KRIG_DIR, 'nw05_v2'),
    ('medium',   'Center 1.0deg', (-100, 100), (0, 100),   UNET_DIR, 'large',    KRIG_DIR, 'large'),
    ('se15',     'SE 1.5deg',     (-200, 200), (0, 200),   SWA_DIR,  'se15',      KRIG_DIR, 'se15'),
]

def save_fig(fig, name):
    for ext in ['png', 'svg']:
        dpi = 1000 if ext == 'png' else 300
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'), dpi=dpi, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

def fill_nan(result, mask_blank, grid_x, grid_y):
    blank_nan = mask_blank & np.isnan(result)
    if blank_nan.sum() > 0:
        valid_mask = mask_blank & ~np.isnan(result)
        valid_pts = np.column_stack([grid_x[valid_mask], grid_y[valid_mask]])
        valid_vals = result[valid_mask]
        nan_pts = np.column_stack([grid_x[blank_nan], grid_y[blank_nan]])
        result[blank_nan] = griddata(valid_pts, valid_vals, nan_pts, method='nearest')
    return blank_nan.sum()

def make_figure(fig_key, label, vres, verr, result_dir, result_key, truth_dir, truth_key,
                mask_dir, mask_key, grid_dir, grid_key, boundary_dir, boundary_key):
    """Generate residual + |error| figures for one method+region."""
    result = np.load(os.path.join(result_dir, f'result_grid_{result_key}.npy'))
    truth = np.load(os.path.join(truth_dir, f'truth_grid_{truth_key}.npy'))
    mask_blank = np.load(os.path.join(mask_dir, f'mask_blank_{mask_key}.npy'))
    mask_outside = np.load(os.path.join(mask_dir, f'mask_outside_{mask_key}.npy'))
    grid_x = np.load(os.path.join(grid_dir, f'grid_x_{grid_key}.npy'))
    grid_y = np.load(os.path.join(grid_dir, f'grid_y_{grid_key}.npy'))

    # Boundary
    bx_path = os.path.join(boundary_dir, f'bx_{boundary_key}.npy')
    if os.path.exists(bx_path):
        bx = np.load(bx_path)
        by = np.load(os.path.join(boundary_dir, f'by_{boundary_key}.npy'))
    else:
        from scipy.spatial import ConvexHull
        idx = np.where(mask_blank)
        pts = np.column_stack([grid_x[idx], grid_y[idx]])
        hull = ConvexHull(pts)
        bx = pts[hull.vertices, 0]
        by = pts[hull.vertices, 1]
        bx = np.append(bx, bx[0])
        by = np.append(by, by[0])

    # Fill NaN
    n = fill_nan(result, mask_blank, grid_x, grid_y)
    if n: print(f"    Filled {n} NaN")

    # Determine output suffix
    if 'nw05' in result_key:
        suffix = 'nw05_v2'
    elif 'large' in result_key:
        suffix = 'medium'
    elif 'se15' in result_key:
        suffix = 'se15'
    else:
        suffix = result_key

    zx = (bx.min() - 2, bx.max() + 2)
    zy = (by.min() - 2, by.max() + 2)

    # Residual
    residual = result.copy()
    residual[~mask_blank] = np.nan
    residual[mask_blank] = result[mask_blank] - truth[mask_blank]
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(grid_x, grid_y, residual, cmap='RdBu_r', shading='auto', vmin=vres[0], vmax=vres[1])
    ax.plot(bx, by, 'k-', linewidth=2)
    ax.set_xlim(zx); ax.set_ylim(zy)
    ax.set_xlabel('X (km)', fontsize=16); ax.set_ylabel('Y (km)', fontsize=16)
    ax.tick_params(labelsize=14)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Residual (nT)', fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    cbar.formatter.set_powerlimits((-2, 3)); cbar.update_ticks()
    fig.tight_layout()
    save_fig(fig, f'fig_{fig_key}_residual_{suffix}')

    # |Error|
    abs_error = result.copy()
    abs_error[~mask_blank] = np.nan
    abs_error[mask_blank] = np.abs(result[mask_blank] - truth[mask_blank])
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(grid_x, grid_y, abs_error, cmap='YlOrRd', shading='auto', vmin=verr[0], vmax=verr[1])
    ax.plot(bx, by, 'k-', linewidth=2)
    ax.set_xlim(zx); ax.set_ylim(zy)
    ax.set_xlabel('X (km)', fontsize=16); ax.set_ylabel('Y (km)', fontsize=16)
    ax.tick_params(labelsize=14)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('|Error| (nT)', fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    cbar.formatter.set_powerlimits((-2, 3)); cbar.update_ticks()
    fig.tight_layout()
    save_fig(fig, f'fig_{fig_key}_error_{suffix}')

    plt.close('all')

# ============================================================
for out_key, label, vres, verr, unet_src_dir, unet_src_key, krig_src_dir, krig_src_key in REGIONS:
    print(f"=== {label} ({out_key}) ===")

    # TB-UNet
    print(f"  TB-UNet residual + error...")
    make_figure('unet', out_key, vres, verr, unet_src_dir, unet_src_key,
                unet_src_dir, unet_src_key, unet_src_dir, unet_src_key,
                unet_src_dir, unet_src_key, unet_src_dir, unet_src_key)

    # Kriging
    print(f"  Kriging residual + error...")
    make_figure('krig', out_key, vres, verr, krig_src_dir, krig_src_key,
                krig_src_dir, krig_src_key, krig_src_dir, krig_src_key,
                krig_src_dir, krig_src_key, krig_src_dir, krig_src_key)

print(f"\nDone! Output: {FIG_DIR}/")
