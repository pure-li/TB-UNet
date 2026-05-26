"""Comparative: 5-panel result comparison — Kriging, CNN, GAN, TB-UNet, Reference"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.interpolate import griddata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COMP_DIR = os.path.dirname(os.path.abspath(__file__))
ABL_DIR = r'F:\PINN实验\venv\U-net\ablation experiment'
FIG_DIR = os.path.join(COMP_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

def save_fig(fig, name):
    for ext in ['png', 'svg']:
        dpi = 1000 if ext == 'png' else 300
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'), dpi=dpi, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

def fill_nan(result, mask_blank, grid_x, grid_y):
    """Fill NaN inside blank region using nearest-neighbor interpolation."""
    blank_nan = mask_blank & np.isnan(result)
    if blank_nan.sum() > 0:
        valid_mask = mask_blank & ~np.isnan(result)
        valid_pts = np.column_stack([grid_x[valid_mask], grid_y[valid_mask]])
        valid_vals = result[valid_mask]
        nan_pts = np.column_stack([grid_x[blank_nan], grid_y[blank_nan]])
        result[blank_nan] = griddata(valid_pts, valid_vals, nan_pts, method='nearest')
    return blank_nan.sum()

def load_model(rk, subdir, base_dir):
    """Load result_grid, fill NaN in blank region, mask outside."""
    d = os.path.join(base_dir, subdir)
    result = np.load(os.path.join(d, f'result_grid_{rk}.npy'))
    mask_blank = np.load(os.path.join(d, f'mask_blank_{rk}.npy'))
    mask_outside = np.load(os.path.join(d, f'mask_outside_{rk}.npy'))
    grid_x = np.load(os.path.join(d, f'grid_x_{rk}.npy'))
    grid_y = np.load(os.path.join(d, f'grid_y_{rk}.npy'))

    n = fill_nan(result, mask_blank, grid_x, grid_y)
    if n > 0:
        print(f"    {subdir}: filled {n} NaN")

    result[mask_outside] = np.nan
    return result

def load_kriging(rk):
    """Load Kriging result combined with RBF background."""
    d = os.path.join(COMP_DIR, 'Kriging')
    result = np.load(os.path.join(d, f'result_grid_{rk}.npy'))
    truth = np.load(os.path.join(d, f'truth_grid_{rk}.npy'))
    mask_blank = np.load(os.path.join(d, f'mask_blank_{rk}.npy'))
    mask_outside = np.load(os.path.join(d, f'mask_outside_{rk}.npy'))

    # Combine: RBF background + Kriging result in blank region
    display = truth.copy()
    display[mask_blank] = result[mask_blank]
    display[mask_outside] = np.nan
    return display

for rk in ['rect', 'irreg']:
    print(f"=== {rk} ===")

    # Common data from Kriging dir
    base = os.path.join(COMP_DIR, 'Kriging')
    truth = np.load(os.path.join(base, f'truth_grid_{rk}.npy'))
    mask_outside = np.load(os.path.join(base, f'mask_outside_{rk}.npy'))
    grid_x = np.load(os.path.join(base, f'grid_x_{rk}.npy'))
    grid_y = np.load(os.path.join(base, f'grid_y_{rk}.npy'))
    bx = np.load(os.path.join(base, f'bx_{rk}.npy'))
    by = np.load(os.path.join(base, f'by_{rk}.npy'))

    truth_display = truth.copy()
    truth_display[mask_outside] = np.nan

    kriging = load_kriging(rk)
    cnn = load_model(rk, 'CNN_results', COMP_DIR)
    gan = load_model(rk, 'gan_results', COMP_DIR)

    # TB-UNet: irreg uses ablation unet_tf_noskip, rect uses comparative U-net_transformer_slow
    if rk == 'irreg':
        tbunet = load_model(rk, 'unet_tf_noskip', ABL_DIR)
        print(f"    TB-UNet: using ablation unet_tf_noskip")
    else:
        tbunet = load_model(rk, 'U-net_transformer_slow', COMP_DIR)

    vmin = np.nanmin(truth_display)
    vmax = np.nanmax(truth_display)
    zx = (bx.min() - 2, bx.max() + 2)
    zy = (by.min() - 2, by.max() + 2)

    fig = plt.figure(figsize=(28, 5.5))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.04])

    axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
    cax = fig.add_subplot(gs[0, 5])

    panels = [
        (kriging, 'Kriging'),
        (cnn, 'CNN'),
        (gan, 'GAN'),
        (tbunet, 'TB-UNet'),
        (truth_display, 'Reference'),
    ]

    for ax, (data, label) in zip(axes, panels):
        im = ax.pcolormesh(grid_x, grid_y, data, cmap='jet', shading='auto', vmin=vmin, vmax=vmax)
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
    cbar.set_label('nT', fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    cbar.formatter.set_powerlimits((-2, 3))
    cbar.update_ticks()

    fig.tight_layout()
    save_fig(fig, f'fig_result_{rk}')
    print(f"  fig_result_{rk} done")

print(f"\nDone! Output: {FIG_DIR}/")
