#!/usr/bin/env python
"""鲁棒性实验对比图 — RMSE曲线 + 比值柱状图 + 结果网格
=========================================================
从 Kriging/ 和 U-net_transformer/ 子目录读取结果，生成 3 张对比图。
"""

import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = os.path.dirname(os.path.abspath(__file__))
KRIG_DIR = os.path.join(ROOT, 'Kriging')
UNET_DIR = os.path.join(ROOT, 'U-net_transformer')

SIZES = {'small': '0.25 sq.deg', 'medium': '1.0 sq.deg', 'large': '2.25 sq.deg'}
SIZE_LABELS = {'small': 'Small\n(0.5deg)', 'medium': 'Medium\n(1.0deg)', 'large': 'Large\n(1.5deg)'}
ORDER = ['small', 'medium', 'large']

COLORS = {'U-Net+Transformer': '#2196F3', 'Kriging': '#9C27B0'}


def save_fig(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(ROOT, f'{name}.{ext}'), dpi=300,
                    bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def load_metrics(method_dir):
    path = os.path.join(method_dir, 'results.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def main():
    print("=" * 60)
    print("  鲁棒性实验对比图")
    print("=" * 60)

    metrics_unet = load_metrics(UNET_DIR)
    metrics_krig = load_metrics(KRIG_DIR)

    # Extract RMSE/MAE
    data = {}
    for rk in ORDER:
        data[rk] = {
            'U-Net+Transformer': {
                'rmse': metrics_unet.get(rk, {}).get('rmse', float('nan')),
                'mae': metrics_unet.get(rk, {}).get('mae', float('nan')),
                'n_blank': metrics_unet.get(rk, {}).get('n_test',
                           int(np.load(os.path.join(UNET_DIR, f'mask_blank_{rk}.npy')).sum())),
            },
            'Kriging': {
                'rmse': metrics_krig.get(rk, {}).get('rmse', float('nan')),
                'mae': metrics_krig.get(rk, {}).get('mae', float('nan')),
                'n_blank': metrics_krig.get(rk, {}).get('n_blank',
                           int(np.load(os.path.join(KRIG_DIR, f'mask_blank_{rk}.npy')).sum())),
            },
        }
        print(f"  {rk}: U-Net+TF RMSE={data[rk]['U-Net+Transformer']['rmse']:.2f}, "
              f"Kriging RMSE={data[rk]['Kriging']['rmse']:.2f}")

    # =========================================================================
    # 图1: RMSE vs Blank Size 曲线
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    areas = [0.25, 1.0, 2.25]  # sq degrees
    rmse_unet = [data[rk]['U-Net+Transformer']['rmse'] for rk in ORDER]
    rmse_krig = [data[rk]['Kriging']['rmse'] for rk in ORDER]
    mae_unet = [data[rk]['U-Net+Transformer']['mae'] for rk in ORDER]
    mae_krig = [data[rk]['Kriging']['mae'] for rk in ORDER]

    # RMSE
    ax1.plot(areas, rmse_unet, 'o-', color=COLORS['U-Net+Transformer'], lw=2.5, ms=10,
             label='U-Net+Transformer')
    ax1.plot(areas, rmse_krig, 's--', color=COLORS['Kriging'], lw=2.5, ms=10,
             label='Kriging')
    for a, ru, rk_ in zip(areas, rmse_unet, rmse_krig):
        ax1.annotate(f'{ru:.1f}', (a, ru), textcoords="offset points", xytext=(0, 12),
                     ha='center', fontsize=9, fontweight='bold', color=COLORS['U-Net+Transformer'])
        ax1.annotate(f'{rk_:.1f}', (a, rk_), textcoords="offset points", xytext=(0, -18),
                     ha='center', fontsize=9, fontweight='bold', color=COLORS['Kriging'])
    ax1.set_xlabel('Blank Area (sq. deg)'); ax1.set_ylabel('RMSE (nT)')
    ax1.set_title('RMSE vs Blank Size')
    ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3)
    ax1.set_xticks(areas); ax1.set_xticklabels(['0.25\n(Small)', '1.0\n(Medium)', '2.25\n(Large)'])

    # MAE
    ax2.plot(areas, mae_unet, 'o-', color=COLORS['U-Net+Transformer'], lw=2.5, ms=10,
             label='U-Net+Transformer')
    ax2.plot(areas, mae_krig, 's--', color=COLORS['Kriging'], lw=2.5, ms=10,
             label='Kriging')
    for a, mu, mk in zip(areas, mae_unet, mae_krig):
        ax2.annotate(f'{mu:.1f}', (a, mu), textcoords="offset points", xytext=(0, 12),
                     ha='center', fontsize=9, fontweight='bold', color=COLORS['U-Net+Transformer'])
        ax2.annotate(f'{mk:.1f}', (a, mk), textcoords="offset points", xytext=(0, -18),
                     ha='center', fontsize=9, fontweight='bold', color=COLORS['Kriging'])
    ax2.set_xlabel('Blank Area (sq. deg)'); ax2.set_ylabel('MAE (nT)')
    ax2.set_title('MAE vs Blank Size')
    ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3)
    ax2.set_xticks(areas); ax2.set_xticklabels(['0.25\n(Small)', '1.0\n(Medium)', '2.25\n(Large)'])

    fig.tight_layout()
    save_fig(fig, 'fig_robustness_curves')

    # =========================================================================
    # 图2: RMSE 比值柱状图
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 6))
    ratios = [k/u for k, u in zip(rmse_krig, rmse_unet)]
    bars = ax.bar(ORDER, ratios, color=['#4CAF50', '#FF9800', '#F44336'], edgecolor='white', linewidth=1.2)

    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{ratio:.2f}x', ha='center', fontsize=12, fontweight='bold')

    ax.set_xticks(ORDER)
    ax.set_xticklabels([f'{SIZE_LABELS[rk]}\n{areas[i]} sq.deg' for i, rk in enumerate(ORDER)])
    ax.set_ylabel('RMSE Ratio (Kriging / U-Net+TF)')
    ax.set_title('Robustness: RMSE Degradation Ratio')
    ax.grid(True, alpha=0.2, axis='y')
    # Add reference line at 1.0
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5,
               label='Equal performance (ratio=1.0)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_fig(fig, 'fig_robustness_ratio')

    # =========================================================================
    # 图3: 3x2 结果对比网格
    # =========================================================================
    fig = plt.figure(figsize=(18, 18))
    gs = GridSpec(3, 2, figure=fig, hspace=0.15, wspace=0.1)

    for row, rk in enumerate(ORDER):
        for col, (method_name, method_dir) in enumerate([
            ('U-Net+Transformer', UNET_DIR), ('Kriging', KRIG_DIR)
        ]):
            result_grid = np.load(os.path.join(method_dir, f'result_grid_{rk}.npy'))
            grid_x = np.load(os.path.join(method_dir, f'grid_x_{rk}.npy'))
            grid_y = np.load(os.path.join(method_dir, f'grid_y_{rk}.npy'))
            bx = np.load(os.path.join(method_dir, f'bx_{rk}.npy'))
            by = np.load(os.path.join(method_dir, f'by_{rk}.npy'))
            mask_outside = np.load(os.path.join(method_dir, f'mask_outside_{rk}.npy'))

            zx = (bx.min() - 2, bx.max() + 2); zy = (by.min() - 2, by.max() + 2)
            vmin = np.nanmin(result_grid[~mask_outside]); vmax = np.nanmax(result_grid[~mask_outside])
            rmse = data[rk][method_name]['rmse']

            ax = fig.add_subplot(gs[row, col])
            im = ax.pcolormesh(grid_x, grid_y, result_grid, cmap='jet', shading='auto',
                               vmin=vmin, vmax=vmax)
            ax.plot(bx, by, 'k-', linewidth=2)
            ax.set_xlim(zx); ax.set_ylim(zy)
            ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
            ax.set_title(f'{SIZE_LABELS[rk].replace(chr(10)," ")} — {method_name} (RMSE={rmse:.1f})',
                         fontsize=10, fontweight='bold')
            cbar = plt.colorbar(im, ax=ax, label='nT')
            cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()

    fig.suptitle('Robustness Experiment: Result Comparison', fontsize=14, fontweight='bold', y=0.99)
    save_fig(fig, 'fig_robustness_grid')

    # 保存汇总 JSON
    summary = {}
    for rk in ORDER:
        summary[rk] = {
            'blank_area_sq_deg': areas[ORDER.index(rk)],
            'U-Net+Transformer': data[rk]['U-Net+Transformer'],
            'Kriging': data[rk]['Kriging'],
            'rmse_ratio': data[rk]['Kriging']['rmse'] / data[rk]['U-Net+Transformer']['rmse'],
        }
    with open(os.path.join(ROOT, 'robustness_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Done!")
    print(f"  fig_robustness_curves.png|svg")
    print(f"  fig_robustness_ratio.png|svg")
    print(f"  fig_robustness_grid.png|svg")
    print(f"  robustness_summary.json")
    print("=" * 60)


if __name__ == '__main__':
    main()
