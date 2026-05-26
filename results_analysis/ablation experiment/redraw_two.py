"""Redraw: fig_epoch_rmse + fig_epoch_loss — 4 models, all truncated to 100ep"""
import os, json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ROOT, 'figures')
SWA_BASE = r'F:\PINN实验\venv\U-net\补充实验\SWA'

MODELS = {
    'pure_unet':      'U-Net',
    'swa':            'TB-UNet',
    'unet_tf_skip':   'TB-UNet+Skip',
    'unet_skip':      'U-Net+Skip',
}

MODEL_COLORS = {
    'pure_unet':      '#2196F3',
    'swa':            '#d62728',
    'unet_tf_skip':   '#9C27B0',
    'unet_skip':      '#FF9800',
}

MODEL_RMSE = {
    'pure_unet':      {'rect': 30.2, 'irreg': 26.7},
    'swa':            {'rect': 14.9, 'irreg': 17.3},
    'unet_tf_skip':   {'rect': 26.8, 'irreg': 36.4},
    'unet_skip':      {'rect': 38.4, 'irreg': 29.1},
}

def get_model_dir(mk):
    if mk == 'swa':
        return SWA_BASE
    if mk == 'unet_skip':
        return os.path.join(ROOT, 'unet_skip')
    return os.path.join(ROOT, mk)

def save_fig(fig, name):
    for ext in ['png', 'svg']:
        dpi = 1000 if ext == 'png' else 800
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'), dpi=dpi,
                    bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

REGIONS = {'rect': 'Rectangular (1.0°)', 'irreg': 'Irregular Polygon'}

for rk, rlabel in REGIONS.items():
    histories = {}
    for mk, mn in MODELS.items():
        hp = os.path.join(get_model_dir(mk), f'history_{rk}.json')
        if os.path.exists(hp):
            with open(hp) as f:
                histories[mk] = {'name': mn, 'data': json.load(f)}

    # --- RMSE figure ---
    fig_rmse, ax_rmse = plt.subplots(figsize=(12, 7))
    for mk, h in histories.items():
        eps = [d['epoch'] for d in h['data']][:100]
        rmses = [d['rmse'] for d in h['data']][:100]
        best_idx = np.argmin(rmses)
        rmse_label = MODEL_RMSE[mk][rk]
        ax_rmse.plot(eps, rmses, color=MODEL_COLORS[mk], lw=2,
                label=f"{h['name']} (RMSE={rmse_label:.1f})", alpha=0.85)
        ax_rmse.scatter(eps[best_idx], rmses[best_idx], color=MODEL_COLORS[mk], s=60,
                   zorder=5, marker='*', edgecolors='black', linewidths=0.8)
    ax_rmse.set_xlabel('Epoch', fontsize=16)
    ax_rmse.set_ylabel('RMSE (nT)', fontsize=16)
    ax_rmse.set_xlim(0, 105)
    ax_rmse.tick_params(labelsize=14)
    ax_rmse.legend(fontsize=16, loc='upper right')
    ax_rmse.grid(True, alpha=0.3)
    fig_rmse.tight_layout()
    save_fig(fig_rmse, f'fig_epoch_rmse_{rk}_v3')

    # --- Loss figure ---
    fig_loss, ax_loss = plt.subplots(figsize=(12, 7))
    for mk, h in histories.items():
        eps = [d['epoch'] for d in h['data']][:100]
        losses = [d['loss'] for d in h['data']][:100]
        best_idx = np.argmin(losses)
        ax_loss.plot(eps, losses, color=MODEL_COLORS[mk], lw=2,
                label=h['name'], alpha=0.85)
        ax_loss.scatter(eps[best_idx], losses[best_idx], color=MODEL_COLORS[mk], s=60,
                   zorder=5, marker='*', edgecolors='black', linewidths=0.8)
    ax_loss.set_xlabel('Epoch', fontsize=16)
    ax_loss.set_ylabel('Loss', fontsize=16)
    ax_loss.set_xlim(0, 105)
    ax_loss.tick_params(labelsize=14)
    ax_loss.legend(fontsize=16, loc='upper right')
    ax_loss.grid(True, alpha=0.3)
    fig_loss.tight_layout()
    save_fig(fig_loss, f'fig_epoch_loss_{rk}_v3')

    # Print summary
    for mk, h in histories.items():
        rmses = [d['rmse'] for d in h['data']]
        best_ep = np.argmin(rmses) + 1
        print(f"  {h['name']:<20s} RMSE best={min(rmses):.1f} @ ep{best_ep}, final={rmses[-1]:.1f}")
    print(f"Saved: fig_epoch_rmse_{rk}_v3 + fig_epoch_loss_{rk}_v3\n")

print("Done!")
