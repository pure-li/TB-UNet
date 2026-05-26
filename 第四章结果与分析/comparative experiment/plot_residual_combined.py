"""Comparative: combined 4-panel residual figure — Kriging, CNN, GAN, TB-UNet"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.interpolate import griddata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COMP_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(COMP_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

SWA_DIR = r'F:\PINN实验\venv\U-net\补充实验\SWA'
ROB_KRIG_DIR = r'F:\PINN实验\venv\U-net\robustness experiment\Kriging'
COMP_KRIG_DIR = os.path.join(COMP_DIR, 'Kriging')

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

def compute_residual(display, truth, mask_blank):
    residual = display.copy()
    residual[~mask_blank] = np.nan
    residual[mask_blank] = display[mask_blank] - truth[mask_blank]
    return residual

for rk in ['rect', 'irreg']:
    print(f"=== {rk} ===")

    # --- Common grid/boundary ---
    bx = np.load(os.path.join(COMP_KRIG_DIR, f'bx_{rk}.npy'))
    by = np.load(os.path.join(COMP_KRIG_DIR, f'by_{rk}.npy'))
    zx = (bx.min() - 2, bx.max() + 2)
    zy = (by.min() - 2, by.max() + 2)

    # --- Kriging ---
    if rk == 'rect':
        kg_result = np.load(os.path.join(ROB_KRIG_DIR, 'result_grid_large.npy'))
        kg_truth = np.load(os.path.join(ROB_KRIG_DIR, 'truth_grid_large.npy'))
        kg_gx = np.load(os.path.join(ROB_KRIG_DIR, 'grid_x_large.npy'))
        kg_gy = np.load(os.path.join(ROB_KRIG_DIR, 'grid_y_large.npy'))
        kg_mb = np.load(os.path.join(ROB_KRIG_DIR, 'mask_blank_large.npy'))
        kg_mo = np.load(os.path.join(ROB_KRIG_DIR, 'mask_outside_large.npy'))
        display = kg_truth.copy()
        display[kg_mo] = np.nan
        display[kg_mb] = kg_result[kg_mb]
        krig_residual = compute_residual(display, kg_truth, kg_mb)
        gx, gy = kg_gx, kg_gy
    else:
        kg_result = np.load(os.path.join(COMP_KRIG_DIR, f'result_grid_{rk}.npy'))
        kg_truth = np.load(os.path.join(COMP_KRIG_DIR, f'truth_grid_{rk}.npy'))
        kg_gx = np.load(os.path.join(COMP_KRIG_DIR, f'grid_x_{rk}.npy'))
        kg_gy = np.load(os.path.join(COMP_KRIG_DIR, f'grid_y_{rk}.npy'))
        kg_mb = np.load(os.path.join(COMP_KRIG_DIR, f'mask_blank_{rk}.npy'))
        display = kg_truth.copy()
        display[kg_mb] = kg_result[kg_mb]
        krig_residual = compute_residual(display, kg_truth, kg_mb)
        gx, gy = kg_gx, kg_gy
    print(f"  Kriging done")

    # --- CNN ---
    cnn_dir = os.path.join(COMP_DIR, 'CNN_results')
    cnn_result = np.load(os.path.join(cnn_dir, f'result_grid_{rk}.npy'))
    cnn_truth = np.load(os.path.join(cnn_dir, f'truth_grid_{rk}.npy'))
    cnn_gx = np.load(os.path.join(cnn_dir, f'grid_x_{rk}.npy'))
    cnn_gy = np.load(os.path.join(cnn_dir, f'grid_y_{rk}.npy'))
    cnn_mb = np.load(os.path.join(cnn_dir, f'mask_blank_{rk}.npy'))
    n = fill_nan(cnn_result, cnn_mb, cnn_gx, cnn_gy)
    if n: print(f"  CNN: filled {n} NaN")
    cnn_residual = compute_residual(cnn_result, cnn_truth, cnn_mb)
    if gx is None: gx, gy = cnn_gx, cnn_gy

    # --- GAN ---
    gan_dir = os.path.join(COMP_DIR, 'gan_results')
    gan_result = np.load(os.path.join(gan_dir, f'result_grid_{rk}.npy'))
    gan_truth = np.load(os.path.join(gan_dir, f'truth_grid_{rk}.npy'))
    gan_gx = np.load(os.path.join(gan_dir, f'grid_x_{rk}.npy'))
    gan_gy = np.load(os.path.join(gan_dir, f'grid_y_{rk}.npy'))
    gan_mb = np.load(os.path.join(gan_dir, f'mask_blank_{rk}.npy'))
    n = fill_nan(gan_result, gan_mb, gan_gx, gan_gy)
    if n: print(f"  GAN: filled {n} NaN")
    gan_residual = compute_residual(gan_result, gan_truth, gan_mb)

    # --- TB-UNet (SWA) ---
    swa_result = np.load(os.path.join(SWA_DIR, f'result_grid_swa_{rk}.npy'))
    swa_truth = np.load(os.path.join(SWA_DIR, f'truth_grid_{rk}.npy'))
    swa_gx = np.load(os.path.join(SWA_DIR, f'grid_x_{rk}.npy'))
    swa_gy = np.load(os.path.join(SWA_DIR, f'grid_y_{rk}.npy'))
    swa_mb = np.load(os.path.join(SWA_DIR, f'mask_blank_{rk}.npy'))
    n = fill_nan(swa_result, swa_mb, swa_gx, swa_gy)
    if n: print(f"  TB-UNet: filled {n} NaN")
    tb_residual = compute_residual(swa_result, swa_truth, swa_mb)

    # --- Combined figure ---
    fig = plt.figure(figsize=(25, 6))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.035])
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    cax = fig.add_subplot(gs[0, 4])

    panels = [
        (krig_residual, 'Kriging'),
        (cnn_residual, 'CNN'),
        (gan_residual, 'GAN'),
        (tb_residual, 'TB-UNet'),
    ]

    for ax, (data, label) in zip(axes, panels):
        im = ax.pcolormesh(gx, gy, data, cmap='RdBu_r', shading='auto', vmin=-100, vmax=100)
        ax.plot(bx, by, 'k-', linewidth=1.5)
        ax.set_xlim(zx)
        ax.set_ylim(zy)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=16,
                fontweight='bold', va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='gray'))

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Residual (nT)', fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    cbar.formatter.set_powerlimits((-2, 3))
    cbar.update_ticks()

    fig.tight_layout()
    save_fig(fig, f'fig_residual_combined_{rk}')
    print(f"  fig_residual_combined_{rk} done")

print(f"\nDone! Output: {FIG_DIR}/")
