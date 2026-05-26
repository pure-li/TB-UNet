"""
Plot CDF of absolute residuals for noise robustness experiment.
4 figures (one per noise level), 8 curves each (4 methods × 2 regions).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os, json

BASE = r'F:\PINN实验\venv\U-net\noise experiment'
METHODS = ['Kriging', 'CNN', 'GAN', 'U-net_TF']
METHOD_LABELS = ['Kriging', 'CNN', 'GAN', 'TB-UNet']
COLORS = {'Kriging': '#1f77b4', 'CNN': '#2ca02c', 'GAN': '#ff7f0e', 'U-net_TF': '#d62728'}
REGIONS = ['rect', 'irreg']
REGION_LABELS = {'rect': 'Rectangular', 'irreg': 'Irregular'}
NOISE_LEVELS = [0, 2, 5, 10]
OUT_DIR = os.path.join(BASE, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# Use mask_blank from Kriging as canonical (same grid for all methods)
MASK_DIR = os.path.join(BASE, 'Kriging')

for sigma in NOISE_LEVELS:
    fig, ax = plt.subplots(figsize=(8, 6))

    for method, label in zip(METHODS, METHOD_LABELS):
        method_dir = os.path.join(BASE, method)
        for region in REGIONS:
            res_path = os.path.join(method_dir, f'abs_residual_{region}_noise_{sigma}.npy')
            mask_path = os.path.join(MASK_DIR, f'mask_blank_{region}.npy')

            if not os.path.exists(res_path):
                print(f'  SKIP missing: {res_path}')
                continue

            residuals = np.load(res_path).flatten()
            mask = np.load(mask_path).flatten().astype(bool)

            # If sizes mismatch, residual is already 1D test-point values — use directly
            if len(residuals) != len(mask):
                vals = residuals[np.isfinite(residuals)]
            else:
                vals = residuals[mask]
                vals = vals[np.isfinite(vals)]
            vals = np.sort(vals)

            if len(vals) == 0:
                print(f'  EMPTY: {method} {region} sigma={sigma}')
                continue

            cdf = np.arange(1, len(vals) + 1) / len(vals) * 100

            marker = '^' if region == 'rect' else 'o'
            linestyle = '--'
            rlabel = REGION_LABELS[region]
            ax.plot(vals, cdf, color=COLORS[method], linestyle=linestyle,
                    marker=marker, markevery=max(1, len(vals)//20),
                    markersize=5, linewidth=1.2,
                    label=f'{label} ({rlabel})')

    ax.set_xlabel('Absolute Residual (nT)')
    ax.set_ylabel('CDF (%)')
    # ax.set_title(f'Noise Level σ = {sigma} nT')
    ax.legend(fontsize=7, framealpha=0.8)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path = os.path.join(OUT_DIR, f'fig_cdf_noise_{sigma}.png')
    svg_path = os.path.join(OUT_DIR, f'fig_cdf_noise_{sigma}.svg')
    fig.savefig(png_path, dpi=150)
    fig.savefig(svg_path)
    print(f'Saved: {png_path}')
    plt.close(fig)

print('\nDone. 4 CDF figures saved.')
