"""Robustness: 3 blank-region schematic — TB-UNet interpolation with blank areas in white"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROB = r'F:\PINN实验\venv\U-net\robustness experiment'
FIG_DIR = os.path.join(ROB, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
DATA_DIR = os.path.join(ROB, 'U-net_transformer')

REGIONS = {
    'small':  '0.5° × 0.5°',
    'medium': '0.8° × 0.8°',
    'large':  '1.0° × 1.0°',
}

def load_map(rk):
    """Return result grid with blank and outside areas set to NaN."""
    r = np.load(os.path.join(DATA_DIR, f'result_grid_{rk}.npy'))
    m_blank = np.load(os.path.join(DATA_DIR, f'mask_blank_{rk}.npy'))
    m_outside = np.load(os.path.join(DATA_DIR, f'mask_outside_{rk}.npy'))
    display = r.copy()
    display[m_blank | m_outside] = np.nan
    return display

def save_fig(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'), dpi=800,
                    bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

vmin, vmax = -300, 200
for ax, (rk, rlabel) in zip(axes, REGIONS.items()):
    display = load_map(rk)
    im = ax.imshow(display, cmap='jet', origin='lower', vmin=vmin, vmax=vmax,
                   aspect='auto')
    ax.set_title(rlabel, fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])

cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label='nT')
cbar.ax.tick_params(labelsize=9)
fig.tight_layout()
save_fig(fig, 'fig_blank_regions_schematic')
print(f'Done. Output: {FIG_DIR}/')
