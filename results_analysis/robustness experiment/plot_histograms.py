"""Robustness: 3 region histograms — Kriging vs TB-UNet error distributions"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROB = r'F:\PINN实验\venv\U-net\robustness experiment'
SWA_DIR = os.path.join(ROB, 'SWA')
SWA_RECT_DIR = r'F:\PINN实验\venv\U-net\补充实验\SWA'
KRIG_DIR = os.path.join(ROB, 'Kriging')
FIG_DIR = os.path.join(ROB, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

COLORS = {'TB-UNet': '#d62728', 'Kriging': '#1f77b4', 'TB-UNet+SWA': '#d62728'}

def load_errors(result_grid, truth_grid, mask_blank):
    err = result_grid[mask_blank] - truth_grid[mask_blank]
    return err[np.isfinite(err)]

def save_fig(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'), dpi=800,
                    bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

# ============================================================
# Small: NW 0.5deg v2
# ============================================================
print("=== Small (NW 0.5deg v2) ===")
rk = 'nw05_v2'
fig, ax = plt.subplots(figsize=(10, 6))

# TB-UNet Best
r = np.load(os.path.join(SWA_DIR, f'result_grid_{rk}.npy'))
t = np.load(os.path.join(SWA_DIR, f'truth_grid_{rk}.npy'))
m = np.load(os.path.join(SWA_DIR, f'mask_blank_{rk}.npy'))
err_unet = load_errors(r, t, m)
rmse_unet = 9.19
ax.hist(np.clip(err_unet, -200, 200), bins=80, density=True, alpha=0.45,
        color=COLORS['TB-UNet'], label=f'TB-UNet (RMSE={rmse_unet:.2f})')
print(f"  TB-UNet: n={len(err_unet)}, RMSE={rmse_unet:.2f}")

# Kriging
r = np.load(os.path.join(KRIG_DIR, f'result_grid_{rk}.npy'))
t = np.load(os.path.join(KRIG_DIR, f'truth_grid_{rk}.npy'))
m = np.load(os.path.join(KRIG_DIR, f'mask_blank_{rk}.npy'))
err_krig = load_errors(r, t, m)
rmse_krig = 30.08
ax.hist(np.clip(err_krig, -200, 200), bins=80, density=True, alpha=0.45,
        color=COLORS['Kriging'], label=f'Kriging (RMSE={rmse_krig:.2f})')
print(f"  Kriging: n={len(err_krig)}, RMSE={rmse_krig:.2f}")

ax.set_xlabel('Error (Pred - True) [nT]', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.legend(loc='upper right', fontsize=11)
ax.set_xlim(-200, 200)
ax.grid(True, alpha=0.2)
fig.tight_layout()
save_fig(fig, 'fig_hist_robustness_small')

# ============================================================
# Medium: Center 1.0deg — TB-UNet+SWA (14.9)
# ============================================================
print("\n=== Medium (Center 1.0deg) ===")
fig, ax = plt.subplots(figsize=(10, 6))

# TB-UNet+SWA
r = np.load(os.path.join(SWA_RECT_DIR, 'result_grid_swa_rect.npy'))
t = np.load(os.path.join(SWA_RECT_DIR, 'truth_grid_rect.npy'))
m = np.load(os.path.join(SWA_RECT_DIR, 'mask_blank_rect.npy'))
err_unet = load_errors(r, t, m)
rmse_unet = 14.9
ax.hist(np.clip(err_unet, -200, 200), bins=80, density=True, alpha=0.45,
        color=COLORS['TB-UNet+SWA'], label=f'TB-UNet (RMSE={rmse_unet:.2f})')
print(f"  TB-UNet: n={len(err_unet)}, RMSE={rmse_unet:.2f}")

# Kriging (old "large" = Center 1.0deg)
r = np.load(os.path.join(KRIG_DIR, 'result_grid_large.npy'))
t = np.load(os.path.join(KRIG_DIR, 'truth_grid_large.npy'))
m = np.load(os.path.join(KRIG_DIR, 'mask_blank_large.npy'))
err_krig = load_errors(r, t, m)
rmse_krig = float(np.sqrt(np.mean(err_krig**2)))
ax.hist(np.clip(err_krig, -200, 200), bins=80, density=True, alpha=0.45,
        color=COLORS['Kriging'], label=f'Kriging (RMSE={rmse_krig:.2f})')
print(f"  Kriging: n={len(err_krig)}, RMSE={rmse_krig:.2f}")

ax.set_xlabel('Error (Pred - True) [nT]', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.legend(loc='upper right', fontsize=11)
ax.set_xlim(-200, 200)
ax.grid(True, alpha=0.2)
fig.tight_layout()
save_fig(fig, 'fig_hist_robustness_medium')

# ============================================================
# Large: SE 1.5deg
# ============================================================
print("\n=== Large (SE 1.5deg) ===")
rk = 'se15'
fig, ax = plt.subplots(figsize=(10, 6))

# TB-UNet Best
r = np.load(os.path.join(SWA_DIR, f'result_grid_{rk}.npy'))
t = np.load(os.path.join(SWA_DIR, f'truth_grid_{rk}.npy'))
m = np.load(os.path.join(SWA_DIR, f'mask_blank_{rk}.npy'))
err_unet = load_errors(r, t, m)
rmse_unet = 51.44
ax.hist(np.clip(err_unet, -200, 200), bins=80, density=True, alpha=0.45,
        color=COLORS['TB-UNet'], label=f'TB-UNet (RMSE={rmse_unet:.2f})')
print(f"  TB-UNet: n={len(err_unet)}, RMSE={rmse_unet:.2f}")

# Kriging
r = np.load(os.path.join(KRIG_DIR, f'result_grid_{rk}.npy'))
t = np.load(os.path.join(KRIG_DIR, f'truth_grid_{rk}.npy'))
m = np.load(os.path.join(KRIG_DIR, f'mask_blank_{rk}.npy'))
err_krig = load_errors(r, t, m)
rmse_krig = 61.77
ax.hist(np.clip(err_krig, -200, 200), bins=80, density=True, alpha=0.45,
        color=COLORS['Kriging'], label=f'Kriging (RMSE={rmse_krig:.2f})')
print(f"  Kriging: n={len(err_krig)}, RMSE={rmse_krig:.2f}")

ax.set_xlabel('Error (Pred - True) [nT]', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.legend(loc='upper right', fontsize=11)
ax.set_xlim(-200, 200)
ax.grid(True, alpha=0.2)
fig.tight_layout()
save_fig(fig, 'fig_hist_robustness_large')

print(f"\nDone. Output: {FIG_DIR}/")
