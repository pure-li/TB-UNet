#!/usr/bin/env python
"""对比实验可视化 — 误差分布直方图 + RMSE/MAE 柱状图
======================================================
比较 U-Net+Transformer / GAN / CNN / Kriging 四种方法
"""

import os, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))

METHODS = {
    'U-Net+Transformer': 'U-net_transformer_slow',
    'GAN': 'gan_results',
    'CNN (Block)': 'CNN_results',
    'Kriging': 'Kriging',
}

REGIONS = {'rect': 'Rectangular', 'irreg': 'Irregular'}

COLORS = {
    'U-Net+Transformer': '#2196F3',
    'GAN': '#FF9800',
    'CNN (Block)': '#4CAF50',
    'Kriging': '#9C27B0',
}


def load_errors(method_dir, region_key):
    """加载空白区内逐像素误差 (pred - truth)"""
    result = np.load(os.path.join(method_dir, f'result_grid_{region_key}.npy'))
    truth = np.load(os.path.join(method_dir, f'truth_grid_{region_key}.npy'))
    mask_blank = np.load(os.path.join(method_dir, f'mask_blank_{region_key}.npy'))
    vals = result[mask_blank] - truth[mask_blank]
    return vals[~np.isnan(vals)]


def load_metrics(method_dir):
    """从 results.json 读取 RMSE/MAE"""
    for fname in ['results.json', 'gan_results.json', 'cnn_results.json']:
        path = os.path.join(method_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            return data
    return {}

def get_rmse_mae(metrics, region_key, method_name):
    """鲁棒提取某方法某区域的 RMSE/MAE"""
    # 1) 直接有 region key
    if region_key in metrics and isinstance(metrics[region_key], dict):
        d = metrics[region_key]
        if d.get('rmse') is not None:
            return d['rmse'], d.get('mae')
    # 2) block 后缀
    bk = f'{region_key}_block'
    if bk in metrics and isinstance(metrics[bk], dict):
        d = metrics[bk]
        if d.get('rmse') is not None:
            return d['rmse'], d.get('mae')
    # 3) GAN 旧格式: root-level rmse/mae 是 rect 的
    if 'rmse' in metrics and isinstance(metrics['rmse'], (int, float)):
        return metrics['rmse'], metrics.get('mae')
    return float('nan'), float('nan')


def save_fig(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(ROOT, f'{name}.{ext}'), dpi=300,
                    bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def main():
    print("=" * 60)
    print("  对比实验可视化")
    print("=" * 60)

    # =========================================================================
    # 收集所有误差数据 & 指标
    # =========================================================================
    all_errors = {}   # {region: {method: errors}}
    all_metrics = {}  # {region: {method: {rmse, mae}}}

    for region_key, region_label in REGIONS.items():
        all_errors[region_key] = {}
        all_metrics[region_key] = {}

        for method_name, folder_name in METHODS.items():
            method_dir = os.path.join(ROOT, folder_name)

            # 误差分布
            errors = load_errors(method_dir, region_key)
            all_errors[region_key][method_name] = errors

            # RMSE/MAE
            metrics = load_metrics(method_dir)
            rmse, mae = get_rmse_mae(metrics, region_key, method_name)
            all_metrics[region_key][method_name] = {'rmse': rmse, 'mae': mae}
            print(f"  {region_label:12s} | {method_name:20s} | "
                  f"n={len(errors):,} | RMSE={rmse:.2f} | MAE={mae:.2f}")

    # =========================================================================
    # 图1: 误差分布直方图 (每区域一张, 4 方法叠加)
    # =========================================================================
    for region_key, region_label in REGIONS.items():
        fig, ax = plt.subplots(figsize=(12, 6))

        for method_name in METHODS:
            errors = all_errors[region_key][method_name]
            rmse = all_metrics[region_key][method_name]['rmse']
            # 裁剪到 ±200 避免长尾拉宽坐标
            clipped = np.clip(errors, -200, 200)
            ax.hist(clipped, bins=80, density=True, alpha=0.45,
                    color=COLORS[method_name], label=f'{method_name} (RMSE={rmse:.1f})')

        ax.set_xlabel('Error (Pred - True) [nT]')
        ax.set_ylabel('Density')
        ax.set_title(f'{region_label} Blank — Error Distribution')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xlim(-200, 200)
        ax.grid(True, alpha=0.2)
        save_fig(fig, f'fig_hist_{region_key}')

    # 单张大图: 两区域上下排列
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 10))

    for ax, (region_key, region_label) in zip([ax0, ax1], REGIONS.items()):
        for method_name in METHODS:
            errors = all_errors[region_key][method_name]
            rmse = all_metrics[region_key][method_name]['rmse']
            clipped = np.clip(errors, -200, 200)
            ax.hist(clipped, bins=80, density=True, alpha=0.45,
                    color=COLORS[method_name], label=f'{method_name} (RMSE={rmse:.1f})')
        ax.set_xlabel('Error (Pred - True) [nT]')
        ax.set_ylabel('Density')
        ax.set_title(f'{region_label} Blank — Error Distribution')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(-200, 200)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    save_fig(fig, 'fig_hist_combined')

    # =========================================================================
    # 图2: RMSE/MAE 并列柱状图
    # =========================================================================
    method_names = list(METHODS.keys())
    x = np.arange(len(REGIONS))
    width = 0.18

    fig, (ax_rmse, ax_mae) = plt.subplots(1, 2, figsize=(14, 6))

    for i, (ax, metric) in enumerate([(ax_rmse, 'rmse'), (ax_mae, 'mae')]):
        for j, method_name in enumerate(method_names):
            vals = [all_metrics[rk][method_name][metric] for rk in REGIONS]
            offset = (j - len(method_names)/2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=method_name,
                          color=COLORS[method_name], edgecolor='white', linewidth=0.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=8, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels([REGIONS[r] for r in REGIONS])
        ax.set_ylabel(f'{metric.upper()} (nT)')
        ax.set_title(f'{metric.upper()} Comparison')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.2, axis='y')

    fig.tight_layout()
    save_fig(fig, 'fig_metrics_bar')

    # 保存数值表格
    summary = {}
    for region_key in REGIONS:
        summary[region_key] = {}
        for method_name in METHODS:
            summary[region_key][method_name] = all_metrics[region_key][method_name]
    with open(os.path.join(ROOT, 'comparison_metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  完成! 输出: {ROOT}/")
    print(f"    fig_hist_rect.png|svg")
    print(f"    fig_hist_irreg.png|svg")
    print(f"    fig_hist_combined.png|svg")
    print(f"    fig_metrics_bar.png|svg")
    print(f"    comparison_metrics.json")
    print("=" * 60)


if __name__ == '__main__':
    main()
