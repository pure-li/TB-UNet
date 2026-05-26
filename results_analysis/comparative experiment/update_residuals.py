"""Update comparative experiment residual plots: Kriging (+RBF bg) and TB-UNet (SWA data)"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COMP_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(COMP_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

DATA_PATH = r'F:\PINN实验\venv\U-net\afghanistan_full\Afghan_mag06A.csv'
SWA_DIR = r'F:\PINN实验\venv\U-net\补充实验\SWA'
ROB_KRIG_DIR = r'F:\PINN实验\venv\U-net\robustness experiment\Kriging'
COMP_KRIG_DIR = os.path.join(COMP_DIR, 'Kriging')
COMP_TB_DIR = os.path.join(COMP_DIR, 'U-net_transformer_slow')

def save_fig(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'), dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

def compute_rbf_background(grid_x, grid_y, grid_pts, mask_outside):
    """Compute RBF background for display context."""
    df = pd.read_csv(DATA_PATH).iloc[::3]
    # Train = points outside blank region = all that are not NaN after masking
    n_sub = min(3000, len(df))
    idx = np.random.choice(len(df), n_sub, replace=False)
    pts = np.column_stack([
        (df['Longitude'].values[idx] - df['Longitude'].mean()) * np.pi/180 * 6371 * np.cos(np.radians(df['Latitude'].mean())),
        (df['Latitude'].values[idx] - df['Latitude'].mean()) * np.pi/180 * 6371
    ])
    # Actually, let's use a simpler approach: load existing truth_grid which is already an RBF
    # We'll just fill NaN outside blank with truth_grid
    return None  # placeholder

# ============================================================
# Kriging Rect — use robustness experiment data (RMSE=53.21)
# ============================================================
print("=== Kriging Rect (53.21) ===")
rk = 'rect'

# Load robustness Kriging data
kg_result = np.load(os.path.join(ROB_KRIG_DIR, 'result_grid_large.npy'))
kg_truth = np.load(os.path.join(ROB_KRIG_DIR, 'truth_grid_large.npy'))
kg_gx = np.load(os.path.join(ROB_KRIG_DIR, 'grid_x_large.npy'))
kg_gy = np.load(os.path.join(ROB_KRIG_DIR, 'grid_y_large.npy'))
kg_mb = np.load(os.path.join(ROB_KRIG_DIR, 'mask_blank_large.npy'))
kg_mo = np.load(os.path.join(ROB_KRIG_DIR, 'mask_outside_large.npy'))

# Load comparative Kriging boundary (same grid, just for bx/by)
bx = np.load(os.path.join(COMP_KRIG_DIR, f'bx_{rk}.npy'))
by = np.load(os.path.join(COMP_KRIG_DIR, f'by_{rk}.npy'))

# Combine: RBF background (truth_grid) outside blank, Kriging inside blank
rbf_bg = kg_truth.copy()
rbf_bg[kg_mo] = np.nan
display = rbf_bg.copy()
display[kg_mb] = kg_result[kg_mb]

zx = (bx.min() - 2, bx.max() + 2)
zy = (by.min() - 2, by.max() + 2)
vmin = np.nanmin(kg_truth[~kg_mo])
vmax = np.nanmax(kg_truth[~kg_mo])

rmse_val = 53.21

# Residual
residual = display.copy()
residual[~kg_mb] = np.nan
residual[kg_mb] = display[kg_mb] - kg_truth[kg_mb]
fig, ax = plt.subplots(figsize=(10, 9))
im = ax.pcolormesh(kg_gx, kg_gy, residual, cmap='RdBu_r', shading='auto', vmin=-100, vmax=100)
ax.plot(bx, by, 'k-', linewidth=2)
ax.set_xlim(zx); ax.set_ylim(zy)
ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
ax.set_title(f'Rectangular (1.0deg x 1.0deg)\nResidual — Ordinary Kriging (RMSE={rmse_val:.2f} nT)')
cbar = plt.colorbar(im, ax=ax, label='Residual (nT)')
cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
save_fig(fig, f'fig_krig_residual_{rk}')
print(f"  Kriging rect residual done")

# ============================================================
# Kriging Irreg — fix zoom, keep existing data
# ============================================================
print("=== Kriging Irreg ===")
rk = 'irreg'

kg_result = np.load(os.path.join(COMP_KRIG_DIR, f'result_grid_{rk}.npy'))
kg_truth = np.load(os.path.join(COMP_KRIG_DIR, f'truth_grid_{rk}.npy'))
kg_gx = np.load(os.path.join(COMP_KRIG_DIR, f'grid_x_{rk}.npy'))
kg_gy = np.load(os.path.join(COMP_KRIG_DIR, f'grid_y_{rk}.npy'))
kg_mb = np.load(os.path.join(COMP_KRIG_DIR, f'mask_blank_{rk}.npy'))
kg_mo = np.load(os.path.join(COMP_KRIG_DIR, f'mask_outside_{rk}.npy'))
bx = np.load(os.path.join(COMP_KRIG_DIR, f'bx_{rk}.npy'))
by = np.load(os.path.join(COMP_KRIG_DIR, f'by_{rk}.npy'))

# Combine with RBF background
display = kg_truth.copy()
display[kg_mb] = kg_result[kg_mb]

# Compute RMSE from data
err = kg_result[kg_mb] - kg_truth[kg_mb]
err = err[np.isfinite(err)]
rmse_val = float(np.sqrt(np.mean(err**2)))

zx = (bx.min() - 2, bx.max() + 2)
zy = (by.min() - 2, by.max() + 2)
vmin = np.nanmin(kg_truth[~kg_mo])
vmax = np.nanmax(kg_truth[~kg_mo])

residual = display.copy()
residual[~kg_mb] = np.nan
residual[kg_mb] = display[kg_mb] - kg_truth[kg_mb]
fig, ax = plt.subplots(figsize=(10, 9))
im = ax.pcolormesh(kg_gx, kg_gy, residual, cmap='RdBu_r', shading='auto', vmin=-100, vmax=100)
ax.plot(bx, by, 'k-', linewidth=2)
ax.set_xlim(zx); ax.set_ylim(zy)
ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
ax.set_title(f'Irregular Polygon\nResidual — Ordinary Kriging (RMSE={rmse_val:.2f} nT)')
cbar = plt.colorbar(im, ax=ax, label='Residual (nT)')
cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
save_fig(fig, f'fig_krig_residual_{rk}')
print(f"  Kriging irreg residual done (RMSE={rmse_val:.2f})")

# ============================================================
# TB-UNet Rect — SWA result (RMSE=14.9)
# ============================================================
print("=== TB-UNet Rect (14.9) ===")
rk = 'rect'

swa_result = np.load(os.path.join(SWA_DIR, f'result_grid_swa_{rk}.npy'))
swa_truth = np.load(os.path.join(SWA_DIR, f'truth_grid_{rk}.npy'))
swa_gx = np.load(os.path.join(SWA_DIR, f'grid_x_{rk}.npy'))
swa_gy = np.load(os.path.join(SWA_DIR, f'grid_y_{rk}.npy'))
swa_mb = np.load(os.path.join(SWA_DIR, f'mask_blank_{rk}.npy'))
bx = np.load(os.path.join(SWA_DIR, f'bx_{rk}.npy'))
by = np.load(os.path.join(SWA_DIR, f'by_{rk}.npy'))

# Fill NaN in blank region
from scipy.interpolate import griddata
result_filled = swa_result.copy()
blank_nan = swa_mb & np.isnan(swa_result)
if blank_nan.sum() > 0:
    valid_mask = swa_mb & ~np.isnan(swa_result)
    valid_pts = np.column_stack([swa_gx[valid_mask], swa_gy[valid_mask]])
    valid_vals = swa_result[valid_mask]
    nan_pts = np.column_stack([swa_gx[blank_nan], swa_gy[blank_nan]])
    result_filled[blank_nan] = griddata(valid_pts, valid_vals, nan_pts, method='nearest')
    print(f"  Filled {blank_nan.sum()} NaN in blank region")

zx = (bx.min() - 2, bx.max() + 2)
zy = (by.min() - 2, by.max() + 2)

residual = result_filled.copy()
residual[~swa_mb] = np.nan
residual[swa_mb] = result_filled[swa_mb] - swa_truth[swa_mb]
fig, ax = plt.subplots(figsize=(10, 9))
im = ax.pcolormesh(swa_gx, swa_gy, residual, cmap='RdBu_r', shading='auto', vmin=-100, vmax=100)
ax.plot(bx, by, 'k-', linewidth=2)
ax.set_xlim(zx); ax.set_ylim(zy)
ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
ax.set_title(f'Rectangular (1.0deg x 1.0deg)\nResidual — TB-UNet (RMSE=14.90 nT)')
cbar = plt.colorbar(im, ax=ax, label='Residual (nT)')
cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
save_fig(fig, f'fig_unet_residual_{rk}')
print(f"  TB-UNet rect residual done")

# ============================================================
# TB-UNet Irreg — SWA result (RMSE=17.3)
# ============================================================
print("=== TB-UNet Irreg (17.3) ===")
rk = 'irreg'

swa_result = np.load(os.path.join(SWA_DIR, f'result_grid_swa_{rk}.npy'))
swa_truth = np.load(os.path.join(SWA_DIR, f'truth_grid_{rk}.npy'))
swa_gx = np.load(os.path.join(SWA_DIR, f'grid_x_{rk}.npy'))
swa_gy = np.load(os.path.join(SWA_DIR, f'grid_y_{rk}.npy'))
swa_mb = np.load(os.path.join(SWA_DIR, f'mask_blank_{rk}.npy'))
bx = np.load(os.path.join(SWA_DIR, f'bx_{rk}.npy'))
by = np.load(os.path.join(SWA_DIR, f'by_{rk}.npy'))

# Fill NaN in blank region
result_filled = swa_result.copy()
blank_nan = swa_mb & np.isnan(swa_result)
if blank_nan.sum() > 0:
    valid_mask = swa_mb & ~np.isnan(swa_result)
    valid_pts = np.column_stack([swa_gx[valid_mask], swa_gy[valid_mask]])
    valid_vals = swa_result[valid_mask]
    nan_pts = np.column_stack([swa_gx[blank_nan], swa_gy[blank_nan]])
    result_filled[blank_nan] = griddata(valid_pts, valid_vals, nan_pts, method='nearest')
    print(f"  Filled {blank_nan.sum()} NaN in blank region")

zx = (bx.min() - 2, bx.max() + 2)
zy = (by.min() - 2, by.max() + 2)

residual = result_filled.copy()
residual[~swa_mb] = np.nan
residual[swa_mb] = result_filled[swa_mb] - swa_truth[swa_mb]
fig, ax = plt.subplots(figsize=(10, 9))
im = ax.pcolormesh(swa_gx, swa_gy, residual, cmap='RdBu_r', shading='auto', vmin=-100, vmax=100)
ax.plot(bx, by, 'k-', linewidth=2)
ax.set_xlim(zx); ax.set_ylim(zy)
ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
ax.set_title(f'Irregular Polygon\nResidual — TB-UNet (RMSE=17.30 nT)')
cbar = plt.colorbar(im, ax=ax, label='Residual (nT)')
cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
save_fig(fig, f'fig_unet_residual_{rk}')
print(f"  TB-UNet irreg residual done")

# ============================================================
# CNN Rect — fill NaN, ±100 colorbar
# ============================================================
print("=== CNN Rect ===")
rk = 'rect'

cnn_result = np.load(os.path.join(COMP_DIR, 'CNN_results', f'result_grid_{rk}.npy'))
cnn_truth = np.load(os.path.join(COMP_DIR, 'CNN_results', f'truth_grid_{rk}.npy'))
cnn_gx = np.load(os.path.join(COMP_DIR, 'CNN_results', f'grid_x_{rk}.npy'))
cnn_gy = np.load(os.path.join(COMP_DIR, 'CNN_results', f'grid_y_{rk}.npy'))
cnn_mb = np.load(os.path.join(COMP_DIR, 'CNN_results', f'mask_blank_{rk}.npy'))
bx = np.load(os.path.join(COMP_DIR, 'CNN_results', f'bx_{rk}.npy'))
by = np.load(os.path.join(COMP_DIR, 'CNN_results', f'by_{rk}.npy'))

# Fill NaN in blank region
result_filled = cnn_result.copy()
blank_nan = cnn_mb & np.isnan(cnn_result)
if blank_nan.sum() > 0:
    valid_mask = cnn_mb & ~np.isnan(cnn_result)
    valid_pts = np.column_stack([cnn_gx[valid_mask], cnn_gy[valid_mask]])
    valid_vals = cnn_result[valid_mask]
    nan_pts = np.column_stack([cnn_gx[blank_nan], cnn_gy[blank_nan]])
    result_filled[blank_nan] = griddata(valid_pts, valid_vals, nan_pts, method='nearest')
    print(f"  Filled {blank_nan.sum()} NaN in blank region")

# Compute RMSE
err = result_filled[cnn_mb] - cnn_truth[cnn_mb]
err = err[np.isfinite(err)]
rmse_val = float(np.sqrt(np.mean(err**2)))

zx = (bx.min() - 2, bx.max() + 2)
zy = (by.min() - 2, by.max() + 2)

residual = result_filled.copy()
residual[~cnn_mb] = np.nan
residual[cnn_mb] = result_filled[cnn_mb] - cnn_truth[cnn_mb]
fig, ax = plt.subplots(figsize=(10, 9))
im = ax.pcolormesh(cnn_gx, cnn_gy, residual, cmap='RdBu_r', shading='auto', vmin=-100, vmax=100)
ax.plot(bx, by, 'k-', linewidth=2)
ax.set_xlim(zx); ax.set_ylim(zy)
ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
ax.set_title(f'Rectangular (1.0deg x 1.0deg)\nResidual — CNN (RMSE={rmse_val:.2f} nT)')
cbar = plt.colorbar(im, ax=ax, label='Residual (nT)')
cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
save_fig(fig, f'fig_cnn_residual_{rk}')
print(f"  CNN rect residual done (RMSE={rmse_val:.2f})")

# ============================================================
# CNN Irreg — fill NaN, ±100 colorbar
# ============================================================
print("=== CNN Irreg ===")
rk = 'irreg'

cnn_result = np.load(os.path.join(COMP_DIR, 'CNN_results', f'result_grid_{rk}.npy'))
cnn_truth = np.load(os.path.join(COMP_DIR, 'CNN_results', f'truth_grid_{rk}.npy'))
cnn_gx = np.load(os.path.join(COMP_DIR, 'CNN_results', f'grid_x_{rk}.npy'))
cnn_gy = np.load(os.path.join(COMP_DIR, 'CNN_results', f'grid_y_{rk}.npy'))
cnn_mb = np.load(os.path.join(COMP_DIR, 'CNN_results', f'mask_blank_{rk}.npy'))
bx = np.load(os.path.join(COMP_DIR, 'CNN_results', f'bx_{rk}.npy'))
by = np.load(os.path.join(COMP_DIR, 'CNN_results', f'by_{rk}.npy'))

result_filled = cnn_result.copy()
blank_nan = cnn_mb & np.isnan(cnn_result)
if blank_nan.sum() > 0:
    valid_mask = cnn_mb & ~np.isnan(cnn_result)
    valid_pts = np.column_stack([cnn_gx[valid_mask], cnn_gy[valid_mask]])
    valid_vals = cnn_result[valid_mask]
    nan_pts = np.column_stack([cnn_gx[blank_nan], cnn_gy[blank_nan]])
    result_filled[blank_nan] = griddata(valid_pts, valid_vals, nan_pts, method='nearest')
    print(f"  Filled {blank_nan.sum()} NaN in blank region")

err = result_filled[cnn_mb] - cnn_truth[cnn_mb]
err = err[np.isfinite(err)]
rmse_val = float(np.sqrt(np.mean(err**2)))

zx = (bx.min() - 2, bx.max() + 2)
zy = (by.min() - 2, by.max() + 2)

residual = result_filled.copy()
residual[~cnn_mb] = np.nan
residual[cnn_mb] = result_filled[cnn_mb] - cnn_truth[cnn_mb]
fig, ax = plt.subplots(figsize=(10, 9))
im = ax.pcolormesh(cnn_gx, cnn_gy, residual, cmap='RdBu_r', shading='auto', vmin=-100, vmax=100)
ax.plot(bx, by, 'k-', linewidth=2)
ax.set_xlim(zx); ax.set_ylim(zy)
ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
ax.set_title(f'Irregular Polygon\nResidual — CNN (RMSE={rmse_val:.2f} nT)')
cbar = plt.colorbar(im, ax=ax, label='Residual (nT)')
cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
save_fig(fig, f'fig_cnn_residual_{rk}')
print(f"  CNN irreg residual done (RMSE={rmse_val:.2f})")

# ============================================================
# GAN Rect — fill NaN, ±100 colorbar
# ============================================================
print("=== GAN Rect ===")
rk = 'rect'

gan_result = np.load(os.path.join(COMP_DIR, 'gan_results', f'result_grid_{rk}.npy'))
gan_truth = np.load(os.path.join(COMP_DIR, 'gan_results', f'truth_grid_{rk}.npy'))
gan_gx = np.load(os.path.join(COMP_DIR, 'gan_results', f'grid_x_{rk}.npy'))
gan_gy = np.load(os.path.join(COMP_DIR, 'gan_results', f'grid_y_{rk}.npy'))
gan_mb = np.load(os.path.join(COMP_DIR, 'gan_results', f'mask_blank_{rk}.npy'))
bx = np.load(os.path.join(COMP_DIR, 'gan_results', f'bx_{rk}.npy'))
by = np.load(os.path.join(COMP_DIR, 'gan_results', f'by_{rk}.npy'))

result_filled = gan_result.copy()
blank_nan = gan_mb & np.isnan(gan_result)
if blank_nan.sum() > 0:
    valid_mask = gan_mb & ~np.isnan(gan_result)
    valid_pts = np.column_stack([gan_gx[valid_mask], gan_gy[valid_mask]])
    valid_vals = gan_result[valid_mask]
    nan_pts = np.column_stack([gan_gx[blank_nan], gan_gy[blank_nan]])
    result_filled[blank_nan] = griddata(valid_pts, valid_vals, nan_pts, method='nearest')
    print(f"  Filled {blank_nan.sum()} NaN in blank region")

err = result_filled[gan_mb] - gan_truth[gan_mb]
err = err[np.isfinite(err)]
rmse_val = float(np.sqrt(np.mean(err**2)))

zx = (bx.min() - 2, bx.max() + 2)
zy = (by.min() - 2, by.max() + 2)

residual = result_filled.copy()
residual[~gan_mb] = np.nan
residual[gan_mb] = result_filled[gan_mb] - gan_truth[gan_mb]
fig, ax = plt.subplots(figsize=(10, 9))
im = ax.pcolormesh(gan_gx, gan_gy, residual, cmap='RdBu_r', shading='auto', vmin=-100, vmax=100)
ax.plot(bx, by, 'k-', linewidth=2)
ax.set_xlim(zx); ax.set_ylim(zy)
ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
ax.set_title(f'Rectangular (1.0deg x 1.0deg)\nResidual — GAN (RMSE={rmse_val:.2f} nT)')
cbar = plt.colorbar(im, ax=ax, label='Residual (nT)')
cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
save_fig(fig, f'fig_gan_residual_{rk}')
print(f"  GAN rect residual done (RMSE={rmse_val:.2f})")

# ============================================================
# GAN Irreg — fill NaN, ±100 colorbar
# ============================================================
print("=== GAN Irreg ===")
rk = 'irreg'

gan_result = np.load(os.path.join(COMP_DIR, 'gan_results', f'result_grid_{rk}.npy'))
gan_truth = np.load(os.path.join(COMP_DIR, 'gan_results', f'truth_grid_{rk}.npy'))
gan_gx = np.load(os.path.join(COMP_DIR, 'gan_results', f'grid_x_{rk}.npy'))
gan_gy = np.load(os.path.join(COMP_DIR, 'gan_results', f'grid_y_{rk}.npy'))
gan_mb = np.load(os.path.join(COMP_DIR, 'gan_results', f'mask_blank_{rk}.npy'))
bx = np.load(os.path.join(COMP_DIR, 'gan_results', f'bx_{rk}.npy'))
by = np.load(os.path.join(COMP_DIR, 'gan_results', f'by_{rk}.npy'))

result_filled = gan_result.copy()
blank_nan = gan_mb & np.isnan(gan_result)
if blank_nan.sum() > 0:
    valid_mask = gan_mb & ~np.isnan(gan_result)
    valid_pts = np.column_stack([gan_gx[valid_mask], gan_gy[valid_mask]])
    valid_vals = gan_result[valid_mask]
    nan_pts = np.column_stack([gan_gx[blank_nan], gan_gy[blank_nan]])
    result_filled[blank_nan] = griddata(valid_pts, valid_vals, nan_pts, method='nearest')
    print(f"  Filled {blank_nan.sum()} NaN in blank region")

err = result_filled[gan_mb] - gan_truth[gan_mb]
err = err[np.isfinite(err)]
rmse_val = float(np.sqrt(np.mean(err**2)))

zx = (bx.min() - 2, bx.max() + 2)
zy = (by.min() - 2, by.max() + 2)

residual = result_filled.copy()
residual[~gan_mb] = np.nan
residual[gan_mb] = result_filled[gan_mb] - gan_truth[gan_mb]
fig, ax = plt.subplots(figsize=(10, 9))
im = ax.pcolormesh(gan_gx, gan_gy, residual, cmap='RdBu_r', shading='auto', vmin=-100, vmax=100)
ax.plot(bx, by, 'k-', linewidth=2)
ax.set_xlim(zx); ax.set_ylim(zy)
ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
ax.set_title(f'Irregular Polygon\nResidual — GAN (RMSE={rmse_val:.2f} nT)')
cbar = plt.colorbar(im, ax=ax, label='Residual (nT)')
cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
save_fig(fig, f'fig_gan_residual_{rk}')
print(f"  GAN irreg residual done (RMSE={rmse_val:.2f})")

print(f"\nDone! Output: {FIG_DIR}/")
