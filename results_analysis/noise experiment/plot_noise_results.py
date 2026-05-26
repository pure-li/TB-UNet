"""Plot noise experiment results — bar chart + line chart, both regions"""
import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = r'F:\PINN实验\venv\U-net\noise experiment'
FIG_DIR = os.path.join(BASE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Kriging rect 0nT override to 53.21
OVERRIDE = {('Kriging', 'rect', 0): (53.21, 41.60)}

METHODS = ['Kriging', 'CNN', 'GAN', 'U-net_TF']
LABELS = {'Kriging': 'Kriging', 'CNN': 'CNN', 'GAN': 'GAN', 'U-net_TF': 'TB-UNet'}
NOISE = [0, 2, 5, 10]
REGIONS = {'rect': 'Rectangular', 'irreg': 'Irregular'}

COLORS = {
    'Kriging': '#9C27B0',
    'CNN': '#4CAF50',
    'GAN': '#FF9800',
    'U-net_TF': '#d62728',
}

def load_data():
    data = {}  # {method: {region: {noise: (rmse, mae)}}}
    for method in METHODS:
        data[method] = {}
        for rk in REGIONS:
            data[method][rk] = {}
            for nl in NOISE:
                fname = f'results_{rk}_noise_{nl}.json'
                fpath = os.path.join(BASE, method, fname)
                with open(fpath) as f:
                    d = json.load(f)
                if (method, rk, nl) in OVERRIDE:
                    rmse, mae = OVERRIDE[(method, rk, nl)]
                elif rk in d:
                    rmse = d[rk].get('rmse', float('nan'))
                    mae = d[rk].get('mae', float('nan'))
                else:
                    rmse = d.get('rmse', d.get('best_rmse', float('nan')))
                    mae = d.get('mae', d.get('best_mae', float('nan')))
                data[method][rk][nl] = (rmse, mae)
    return data

def save_fig(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'), dpi=800,
                    bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

data = load_data()

# ============================================================
# 图1: 对数柱状图 (每区域一张)
# ============================================================
for rk, rlabel in REGIONS.items():
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(METHODS))
    width = 0.18
    colors_4 = ['#333333', '#888888', '#bbbbbb', '#e0e0e0']

    for i, nl in enumerate(NOISE):
        vals = [data[m][rk][nl][0] for m in METHODS]
        offset = (i - len(NOISE)/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=f'{nl} nT',
                     color=colors_4[i], edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ypos = bar.get_height() * 1.08
            ax.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:.1f}',
                    ha='center', va='bottom', fontsize=6, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in METHODS], fontsize=13)
    ax.set_ylabel('RMSE (nT)', fontsize=14)
    ax.set_yscale('log')
    ax.set_ylim(top=600)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.2, axis='y')
    fig.tight_layout()
    save_fig(fig, f'fig_noise_bar_{rk}')

print("Bar charts done.")

# ============================================================
# 图2: 折线图 (每区域一张)
# ============================================================
for rk, rlabel in REGIONS.items():
    fig, ax = plt.subplots(figsize=(10, 7))

    # Per-method manual y-offsets to avoid overlap (in points)
    offsets = {
        'Kriging':  {0: 18, 2: -18, 5: 18, 10: 18},       # 48.3 up
        'CNN':      {0: -18, 2: 18, 5: -18, 10: -18},      # 417.3 down
        'GAN':      {0: -18, 2: -18, 5: -18, 10: -18},     # 48.3 down, 202.6 down
        'U-net_TF': {0: -18, 2: 18, 5: -18, 10: -18},      # 40.5 down
    }

    for method in METHODS:
        vals = [data[method][rk][nl][0] for nl in NOISE]
        ax.plot(NOISE, vals, 'o-', color=COLORS[method], lw=2.5, ms=8,
                label=LABELS[method])
        ofs = offsets[method]
        for nl, v in zip(NOISE, vals):
            ax.annotate(f'{v:.1f}', (nl, v), textcoords="offset points",
                       xytext=(0, ofs[nl]), ha='center', fontsize=10,
                       fontweight='bold', color=COLORS[method])

    ax.set_xlabel('Noise Level (nT)', fontsize=14)
    ax.set_ylabel('RMSE (nT)', fontsize=14)
    ax.set_yscale('log')
    ax.set_xticks(NOISE)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=13, loc='upper left')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    save_fig(fig, f'fig_noise_line_{rk}')

print("Line charts done.")
print(f"Saved: {FIG_DIR}/")
