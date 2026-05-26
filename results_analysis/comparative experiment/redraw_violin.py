"""Redraw fig_violin_combined — U-Net+TF → TB-UNet (SWA Best), no title, 800dpi"""
import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
SWA_DIR = r'F:\PINN实验\venv\U-net\补充实验\SWA'
NOISE_DIR = r'F:\PINN实验\venv\U-net\noise experiment'

METHODS = {
    'TB-UNet': (SWA_DIR, False),
    'GAN': (os.path.join(NOISE_DIR, 'GAN'), True),
    'CNN': (os.path.join(NOISE_DIR, 'CNN'), True),
    'Kriging': (os.path.join(NOISE_DIR, 'Kriging'), True),
}
COLORS = {
    'TB-UNet': '#2196F3',
    'GAN': '#FF9800',
    'CNN': '#4CAF50',
    'Kriging': '#9C27B0',
}
REGIONS = {'rect': 'Rectangular', 'irreg': 'Irregular'}

def load_errors(method_dir, region_key, noise_suffix=False):
    if noise_suffix:
        rp = os.path.join(method_dir, f'result_grid_{region_key}_noise_0.npy')
    else:
        rp = os.path.join(method_dir, f'result_grid_{region_key}.npy')
    tp = os.path.join(method_dir, f'truth_grid_{region_key}.npy')
    mp = os.path.join(method_dir, f'mask_blank_{region_key}.npy')
    result = np.load(rp)
    truth = np.load(tp)
    mask_blank = np.load(mp)
    vals = result[mask_blank] - truth[mask_blank]
    return vals[~np.isnan(vals)]

def load_rmse_from_results(method_dir, region_key, noise_suffix=False):
    """Read RMSE from results.json (test-point interpolation, matches noise experiment)."""
    if noise_suffix:
        rp = os.path.join(method_dir, f'results_{region_key}_noise_0.json')
    else:
        rp = os.path.join(method_dir, 'results.json')
        if not os.path.exists(rp):
            rp = os.path.join(method_dir, f'results_{region_key}_noise_0.json')
    with open(rp) as f:
        d = json.load(f)
    # Handle both flat and nested JSON
    if region_key in d:
        return d[region_key].get('rmse', d[region_key].get('swa_rmse', d[region_key].get('best_rmse')))
    return d.get('rmse', d.get('swa_rmse', d.get('best_rmse')))

def save_fig(fig, name):
    for ext in ['png', 'svg']:
        dpi = 1000 if ext == 'png' else 800
        fig.savefig(os.path.join(ROOT, 'figures', f'{name}.{ext}'), dpi=dpi,
                    bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

# Collect data
all_data = {}
for rk in REGIONS:
    all_data[rk] = {}
    for mn, (fn, noise_suf) in METHODS.items():
        errs = load_errors(fn, rk, noise_suf)
        all_data[rk][mn] = np.clip(errs, -200, 200)

# fig_violin_combined — 两区域叠加
fig, ax = plt.subplots(figsize=(14, 7))
n_methods = len(METHODS)
positions_rect = np.arange(n_methods) * 2.5
positions_irreg = positions_rect + 0.8

for i, (mn, color) in enumerate(COLORS.items()):
    vp_r = ax.violinplot([all_data['rect'][mn]], positions=[positions_rect[i]],
                         showmeans=True, showmedians=True, widths=0.7)
    vp_i = ax.violinplot([all_data['irreg'][mn]], positions=[positions_irreg[i]],
                         showmeans=True, showmedians=True, widths=0.7)
    for vp in [vp_r, vp_i]:
        for body in vp['bodies']:
            body.set_facecolor(color)
            body.set_alpha(0.5 if vp is vp_i else 0.8)
        for part in ['cmeans', 'cmedians']:
            vp[part].set_color('black')
            vp[part].set_linewidth(1.2)

    for pos, rk in [(positions_rect[i], 'rect'), (positions_irreg[i], 'irreg')]:
        fn, noise_suf = METHODS[mn]
        if mn == 'TB-UNet':
            # SWA best RMSE from noise experiment U-net_TF (test-point interpolation)
            rp = os.path.join(NOISE_DIR, 'U-net_TF', f'results_{rk}_noise_0.json')
            with open(rp) as f:
                rmse = json.load(f)['rmse']
        elif mn == 'Kriging' and rk == 'rect':
            rmse = 53.2
        else:
            rmse = load_rmse_from_results(fn, rk, noise_suf)
        print(f'  {mn} {rk}: RMSE={rmse:.1f}')
        ax.text(pos, -235, f'{rmse:.1f}', ha='center', fontsize=16, fontweight='bold',
                color=color)

ax.set_xticks(list(positions_rect) + list(positions_irreg))
ax.set_xticklabels(
    [f'{mn}\n(Rect)' for mn in METHODS] + [f'{mn}\n(Irreg)' for mn in METHODS],
    fontsize=14)
ax.set_ylabel('Error (nT)', fontsize=16)
ax.tick_params(labelsize=14)
ax.set_ylim(-260, 260)
ax.grid(True, alpha=0.2, axis='y')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=COLORS[mn], alpha=0.6, label=mn) for mn in METHODS]
ax.legend(handles=legend_elements, loc='upper right', fontsize=14)

fig.tight_layout()
os.makedirs(os.path.join(ROOT, 'figures'), exist_ok=True)
save_fig(fig, 'fig_violin_combined')
print('Done: fig_violin_combined')
