#!/usr/bin/env python
"""消融实验 — 全部图表生成
================================
四组模型: Pure U-Net / U-Net+Skip / U-Net+TF(no skip) / U-Net+TF+Skip
两区域: rect(矩形) / irreg(不规则)
每区域×模型: Result / Residual / Error / RMSE曲线
"""

import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ROOT, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

SWA_BASE = r'F:\PINN实验\venv\U-net\补充实验\SWA'

MODELS = {
    'pure_unet':      'Pure U-Net',
    'unet_tf_noskip': 'U-Net + TF',
    'unet_tf_skip':   'U-Net + TF + Skip',
    'swa':            'U-Net + TF + SWA',
}

def get_model_dir(mk):
    """Return the data directory for a model key; SWA for swa key."""
    if mk == 'swa':
        return SWA_BASE
    return os.path.join(ROOT, mk)

REGIONS = {
    'rect':  'Rectangular (1.0°)',
    'irreg': 'Irregular Polygon',
}

# Shared colorbar ranges
VRES = (-100, 100)
VERR = (0, 100)


def save_fig(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'), dpi=300,
                    bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


# Load all results (SWA results for unet_tf_noskip)
all_results = {}
for model_key, model_name in MODELS.items():
    model_dir = get_model_dir(model_key)
    rp = os.path.join(model_dir, 'results.json')
    if os.path.exists(rp):
        with open(rp) as f:
            data = json.load(f)
        # SWA results.json uses 'best_rmse'/'swa_rmse'; remap to 'rmse' for downstream
        if model_key == 'swa':
            remapped = {}
            for rk in ['rect', 'irreg']:
                entry = data.get(rk, {})
                rk_rmse = entry.get('swa_rmse', entry.get('best_rmse'))
                rk_mae = entry.get('swa_mae', entry.get('best_mae'))
                rk_ep = entry.get('best_ep', '?')
                if rk_rmse is not None and (rk_mae is None or (isinstance(rk_mae, float) and np.isnan(rk_mae))):
                    res_path = os.path.join(SWA_BASE, f'abs_residual_swa_{rk}.npy')
                    if os.path.exists(res_path):
                        res = np.load(res_path)
                        res = res[~np.isnan(res)]
                        rk_mae = float(np.mean(res))
                if rk_rmse is not None:
                    remapped[rk] = {'rmse': rk_rmse, 'mae': rk_mae if rk_mae is not None else float('nan'), 'best_epoch': rk_ep}
            data = remapped
            for rk in ['rect', 'irreg']:
                if rk not in data:
                    res_path = os.path.join(SWA_BASE, f'abs_residual_swa_{rk}.npy')
                    if os.path.exists(res_path):
                        res = np.load(res_path)
                        res = res[~np.isnan(res)]
                        data[rk] = {
                            'rmse': float(np.sqrt(np.mean(res**2))),
                            'mae': float(np.mean(res)),
                            'best_epoch': '?',
                        }
        all_results[model_key] = data
    else:
        print(f"  [WARN] Missing results.json for {model_key}")
        all_results[model_key] = {}

print("=" * 60)
print("  消融实验图表生成")
print("=" * 60)

# =============================================================================
# Per-model plots: Result / Residual / Error / RMSE curve
# =============================================================================
for model_key, model_name in MODELS.items():
    model_dir = get_model_dir(model_key)
    if not os.path.exists(model_dir):
        continue

    for rk, rlabel in REGIONS.items():
        print(f"[{model_name}] {rk}...")

        # Check required files — use swa result grid for SWA model
        if model_key == 'swa':
            result_path = os.path.join(model_dir, f'result_grid_swa_{rk}.npy')
        else:
            result_path = os.path.join(model_dir, f'result_grid_{rk}.npy')
        truth_path = os.path.join(model_dir, f'truth_grid_{rk}.npy')
        if not os.path.exists(result_path):
            print(f"  [SKIP] Missing result_grid_{rk}.npy")
            continue
        if not os.path.exists(truth_path):
            print(f"  [SKIP] Missing truth_grid_{rk}.npy")
            continue

        result_grid = np.load(result_path)
        truth_grid = np.load(truth_path)
        grid_x = np.load(os.path.join(model_dir, f'grid_x_{rk}.npy'))
        grid_y = np.load(os.path.join(model_dir, f'grid_y_{rk}.npy'))
        bx = np.load(os.path.join(model_dir, f'bx_{rk}.npy'))
        by = np.load(os.path.join(model_dir, f'by_{rk}.npy'))
        mask_blank = np.load(os.path.join(model_dir, f'mask_blank_{rk}.npy'))
        mask_outside = np.load(os.path.join(model_dir, f'mask_outside_{rk}.npy'))

        mdl = all_results.get(model_key, {})
        rr = mdl.get(rk, {})
        rmse_val = rr.get('rmse', float('nan'))
        best_ep = rr.get('best_epoch', '?')

        zx = (bx.min() - 2, bx.max() + 2)
        zy = (by.min() - 2, by.max() + 2)
        vmin = np.nanmin(truth_grid[~mask_outside])
        vmax = np.nanmax(truth_grid[~mask_outside])

        tag = f'{model_key}_{rk}'

        # ---- Result ----
        fig, ax = plt.subplots(figsize=(10, 9))
        im = ax.pcolormesh(grid_x, grid_y, result_grid, cmap='jet', shading='auto', vmin=vmin, vmax=vmax)
        ax.plot(bx, by, 'k-', linewidth=2)
        ax.set_xlim(zx); ax.set_ylim(zy)
        ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
        ax.set_title(f'{rlabel}\n{model_name} (RMSE={rmse_val:.2f} nT, best ep={best_ep})')
        cbar = plt.colorbar(im, ax=ax, label='Mag. Anomaly (nT)')
        cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
        save_fig(fig, f'fig_result_{tag}')

        # ---- Residual (RdBu_r) ----
        residual = result_grid.copy()
        residual[~mask_blank] = np.nan
        residual[mask_blank] = result_grid[mask_blank] - truth_grid[mask_blank]
        fig, ax = plt.subplots(figsize=(10, 9))
        im = ax.pcolormesh(grid_x, grid_y, residual, cmap='RdBu_r', shading='auto', vmin=VRES[0], vmax=VRES[1])
        ax.plot(bx, by, 'k-', linewidth=2)
        ax.set_xlim(zx); ax.set_ylim(zy)
        ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
        ax.set_title(f'{rlabel}\nResidual — {model_name} (RMSE={rmse_val:.2f} nT)')
        cbar = plt.colorbar(im, ax=ax, label='Residual (nT)')
        cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
        save_fig(fig, f'fig_residual_{tag}')

        # ---- Error (hot) ----
        abs_error = result_grid.copy()
        abs_error[~mask_blank] = np.nan
        abs_error[mask_blank] = np.abs(result_grid[mask_blank] - truth_grid[mask_blank])
        fig, ax = plt.subplots(figsize=(10, 9))
        im = ax.pcolormesh(grid_x, grid_y, abs_error, cmap='hot', shading='auto', vmin=VERR[0], vmax=VERR[1])
        ax.plot(bx, by, 'k-', linewidth=2)
        ax.set_xlim(zx); ax.set_ylim(zy)
        ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
        ax.set_title(f'{rlabel}\n|Error| — {model_name} (RMSE={rmse_val:.2f} nT)')
        cbar = plt.colorbar(im, ax=ax, label='Absolute Error (nT)')
        cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
        save_fig(fig, f'fig_error_{tag}')

        # ---- RMSE curve ----
        hist_path = os.path.join(model_dir, f'history_{rk}.json')
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                history = json.load(f)
            fig, ax = plt.subplots(figsize=(10, 5))
            eps = [h['epoch'] for h in history]
            rmses = [h['rmse'] for h in history]
            ax.plot(eps, rmses, color='#FF5722', lw=2)
            best_idx = np.argmin(rmses)
            ax.scatter(eps[best_idx], rmses[best_idx], color='#FF5722', s=80, zorder=5, marker='*', edgecolors='black')
            ax.set_xlabel('Epoch'); ax.set_ylabel('RMSE (nT)')
            ax.set_title(f'{rlabel} — {model_name} Test RMSE')
            ax.grid(True, alpha=0.3)
            save_fig(fig, f'fig_rmse_{tag}')

        plt.close('all')
        print(f"  {rk} done (RMSE={rmse_val:.2f})")

# =============================================================================
# Epoch comparison curves — RMSE / MAE / Loss (4 models in one figure)
# =============================================================================
MODEL_COLORS = {
    'pure_unet':      '#2196F3',
    'unet_tf_noskip': '#FF9800',
    'unet_tf_skip':   '#9C27B0',
    'swa':            '#d62728',
}

for rk, rlabel in REGIONS.items():
    print(f"\n[Epoch curves] {rk}...")

    # Collect all history data
    histories = {}
    for mk, mn in MODELS.items():
        hp = os.path.join(get_model_dir(mk), f'history_{rk}.json')
        if os.path.exists(hp):
            with open(hp) as f:
                histories[mk] = {'name': mn, 'data': json.load(f)}

    if len(histories) < 2:
        print(f"  [SKIP] Only {len(histories)} models available")
        continue

    # ---- RMSE vs Epoch ----
    fig, ax = plt.subplots(figsize=(12, 7))
    for mk, h in histories.items():
        eps = [d['epoch'] for d in h['data']]
        rmses = [d['rmse'] for d in h['data']]
        best_idx = np.argmin(rmses)
        ax.plot(eps, rmses, color=MODEL_COLORS[mk], lw=2, label=h['name'], alpha=0.85)
        ax.scatter(eps[best_idx], rmses[best_idx], color=MODEL_COLORS[mk], s=60,
                   zorder=5, marker='*', edgecolors='black', linewidths=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('RMSE (nT)', fontsize=12)
    ax.set_title(f'{rlabel} — RMSE vs Epoch (4 Models)', fontsize=13)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, f'fig_epoch_rmse_{rk}')

    # ---- MAE vs Epoch ----
    if 'mae' in histories[list(histories.keys())[0]]['data'][0]:
        fig, ax = plt.subplots(figsize=(12, 7))
        for mk, h in histories.items():
            eps = [d['epoch'] for d in h['data']]
            maes = [d['mae'] for d in h['data']]
            best_idx = np.argmin(maes)
            ax.plot(eps, maes, color=MODEL_COLORS[mk], lw=2, label=h['name'], alpha=0.85)
            ax.scatter(eps[best_idx], maes[best_idx], color=MODEL_COLORS[mk], s=60,
                       zorder=5, marker='*', edgecolors='black', linewidths=0.8)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('MAE (nT)', fontsize=12)
        ax.set_title(f'{rlabel} — MAE vs Epoch (4 Models)', fontsize=13)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_fig(fig, f'fig_epoch_mae_{rk}')

    # ---- Loss vs Epoch ----
    fig, ax = plt.subplots(figsize=(12, 7))
    for mk, h in histories.items():
        eps = [d['epoch'] for d in h['data']]
        losses = [d['loss'] for d in h['data']]
        ax.plot(eps, losses, color=MODEL_COLORS[mk], lw=2, label=h['name'], alpha=0.85)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'{rlabel} — Training Loss vs Epoch (4 Models)', fontsize=13)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    fig.tight_layout()
    save_fig(fig, f'fig_epoch_loss_{rk}')

    print(f"  {rk} epoch curves saved (RMSE + MAE + Loss)")

# =============================================================================
# Residual distribution histograms — 4 models overlaid, per region
# =============================================================================
print(f"\n[Residual histograms]...")

for rk, rlabel in REGIONS.items():
    fig, ax = plt.subplots(figsize=(12, 7))
    all_residuals = []
    for mk, mn in MODELS.items():
        mdir = get_model_dir(mk)
        if mk == 'swa':
            rp = os.path.join(mdir, f'result_grid_swa_{rk}.npy')
        else:
            rp = os.path.join(mdir, f'result_grid_{rk}.npy')
        tp = os.path.join(mdir, f'truth_grid_{rk}.npy')
        mp = os.path.join(mdir, f'mask_blank_{rk}.npy')
        if not (os.path.exists(rp) and os.path.exists(tp) and os.path.exists(mp)):
            continue
        result_grid = np.load(rp)
        truth_grid = np.load(tp)
        mask_blank = np.load(mp)
        residual = result_grid[mask_blank] - truth_grid[mask_blank]
        residual = residual[np.isfinite(residual)]
        all_residuals.append((mn, residual))
        rr = all_results.get(mk, {}).get(rk, {})
        rmse = rr.get('rmse', float('nan'))
        ax.hist(residual, bins=60, alpha=0.5, color=MODEL_COLORS[mk], label=f'{mn} (RMSE={rmse:.1f})')

    if all_residuals:
        ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.6)
        ax.set_xlabel('Residual (nT)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{rlabel} — Residual Distribution\n(Predicted − True, blank region only)', fontsize=13)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()
        save_fig(fig, f'fig_residual_hist_{rk}')
        print(f"  {rk} residual histogram saved ({len(all_residuals)} models)")
    else:
        plt.close(fig)

# =============================================================================
# Load Kriging results for comparison
# =============================================================================
krig_results = {}
krig_rp = os.path.join(ROOT, 'Kriging', 'results.json')
if os.path.exists(krig_rp):
    with open(krig_rp) as f:
        krig_results = json.load(f)

# =============================================================================
# Summary bar charts
# =============================================================================
print(f"\n{'='*60}")
print("  汇总对比图")
print("=" * 60)

ALL_LABELS = dict(MODELS)
ALL_LABELS['kriging'] = 'Ordinary Kriging'

for rk, rlabel in REGIONS.items():
    fig, ax = plt.subplots(figsize=(14, 6))
    names = []; rmses = []; maes = []
    for mk, mn in ALL_LABELS.items():
        if mk == 'kriging':
            kr = krig_results.get(rk, {})
            if 'rmse' in kr:
                names.append(mn)
                rmses.append(kr['rmse'])
                maes.append(kr.get('mae', float('nan')))
        else:
            mdl = all_results.get(mk, {})
            rr = mdl.get(rk, {})
            if 'rmse' in rr:
                names.append(mn)
                rmses.append(rr['rmse'])
                maes.append(rr.get('mae', float('nan')))

    if not names:
        continue

    # RMSE bars
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0'][:len(names)]
    bars = ax.bar(range(len(names)), rmses, color=colors, width=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('RMSE (nT)', fontsize=12)
    ax.set_title(f'Ablation Study — {rlabel}\nRMSE Comparison', fontsize=13)
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    # Add improvement labels
    best_rmse = min(rmses)
    for bar, val in zip(bars, rmses):
        if val > best_rmse:
            delta = val - best_rmse
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                    f'+{delta:.1f}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    save_fig(fig, f'fig_summary_bar_{rk}')

    print(f"  {rk} summary bar saved")

# =============================================================================
# Combined summary bar (both regions side by side, with Kriging)
# =============================================================================
fig, ax = plt.subplots(figsize=(16, 7))
width = 0.18
n_bars = len(ALL_LABELS)
x = np.arange(n_bars)
all_names = []

for idx, (rk, rlabel) in enumerate(REGIONS.items()):
    rmses_vals = []
    names_used = []
    for mk, mn in ALL_LABELS.items():
        if mk == 'kriging':
            kr = krig_results.get(rk, {})
            if 'rmse' in kr:
                names_used.append(mn)
                rmses_vals.append(kr['rmse'])
        else:
            mdl = all_results.get(mk, {})
            rr = mdl.get(rk, {})
            if 'rmse' in rr:
                names_used.append(mn)
                rmses_vals.append(rr['rmse'])
    if idx == 0:
        all_names = names_used
    offset = (idx - 0.5) * width
    bars = ax.bar(x[:len(rmses_vals)] + offset, rmses_vals, width,
                  label=f'{rlabel} (RMSE)', alpha=0.85)
    for bar, val in zip(bars, rmses_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xticks(x[:len(all_names)])
ax.set_xticklabels(all_names, rotation=20, ha='right', fontsize=10)
ax.set_ylabel('RMSE (nT)', fontsize=12)
ax.set_title('Ablation Study — RMSE by Model and Region', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
save_fig(fig, 'fig_summary_combined')

# =============================================================================
# Final summary table
# =============================================================================
print(f"\n{'='*80}")
print(f"  消融实验结果汇总")
print(f"{'='*80}")
for rk, rlabel in REGIONS.items():
    print(f"\n  {rlabel}:")
    print(f"  {'Model':<30s} {'RMSE':>8s} {'MAE':>8s} {'Best Ep':>8s}")
    print(f"  {'-'*54}")
    for mk, mn in ALL_LABELS.items():
        if mk == 'kriging':
            rr = krig_results.get(rk, {})
        else:
            mdl = all_results.get(mk, {})
            rr = mdl.get(rk, {})
        if 'rmse' in rr:
            print(f"  {mn:<30s} {rr['rmse']:8.2f} {rr.get('mae',float('nan')):8.2f} {rr.get('best_epoch','?'):>8}")

print(f"\n  输出: {FIG_DIR}/")
print("=" * 80)
