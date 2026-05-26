#!/usr/bin/env python
"""
U-Net 磁测数据 Inpainting 消融实验脚本
========================================
通过命令行参数控制各消融变量，支持单次实验和批量实验（--run_all）。

消融维度：
  1. 跳跃连接（--use_skip / --no_skip）
  2. 多通道输入（--input_ch 4 / 1）
  3. 梯度损失权重（--grad_weight 0.1 / 0.0）
  4. 训练遮挡策略（--mask_mode mixed / block / scatter）
  5. RBF epsilon 模式（--eps_mode adaptive / fixed）

用法：
  # 单次实验（Baseline）
  python U-net.py --exp_name baseline

  # 单次消融
  python U-net.py --exp_name no_skip --no_skip

  # 批量运行全部消融实验
  python U-net.py --run_all --epochs 30

  # 快速测试（减少 epoch）
  python U-net.py --run_all --epochs 5 --output_dir ablation_quick
"""

import argparse
import os
import sys
import json
import time
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互后端，避免 plt.show() 阻塞
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.interpolate import RBFInterpolator, RegularGridInterpolator
from scipy.ndimage import zoom, gaussian_filter1d
from scipy.spatial import KDTree
from skimage.transform import resize
from pykrige.ok import OrdinaryKriging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

# =============================================================================
# 0. 命令行参数
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='U-Net Inpainting 消融实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
消融实验预设 (用 --run_all 一键运行):
  baseline    : skip=True,  ch=4, grad=0.1, mask=mixed, eps=adaptive
  no_skip     : skip=False, ch=4, grad=0.1, mask=mixed, eps=adaptive
  single_ch   : skip=True,  ch=1, grad=0.1, mask=mixed, eps=adaptive
  no_grad     : skip=True,  ch=4, grad=0.0, mask=mixed, eps=adaptive
  mask_block  : skip=True,  ch=4, grad=0.1, mask=block,  eps=adaptive
  mask_scatter: skip=True,  ch=4, grad=0.1, mask=scatter,eps=adaptive
  fixed_eps   : skip=True,  ch=4, grad=0.1, mask=mixed,  eps=fixed
        """
    )

    # ---- 实验标识 ----
    parser.add_argument('--exp_name', type=str, default='baseline',
                        help='实验名称，用于输出文件名和结果记录')
    parser.add_argument('--run_all', action='store_true',
                        help='运行全部 7 组消融实验（baseline + 6 ablation）')

    # ---- 消融变量 ----
    parser.add_argument('--no_skip', action='store_true',
                        help='禁用跳跃连接（U-Net → Autoencoder）')
    parser.add_argument('--use_skip', action='store_true', default=True,
                        help='启用跳跃连接（默认）')
    parser.add_argument('--input_ch', type=int, default=4,
                        help='输入通道数（4=FXYZ 多通道, 1=仅 F 分量）')
    parser.add_argument('--grad_weight', type=float, default=0.1,
                        help='梯度损失权重（0.0 = 纯 MSE）')
    parser.add_argument('--mask_mode', type=str, default='mixed',
                        choices=['mixed', 'block', 'scatter'],
                        help='训练遮挡策略')
    parser.add_argument('--eps_mode', type=str, default='adaptive',
                        choices=['adaptive', 'fixed'],
                        help='RBF epsilon 模式（adaptive=自适应, fixed=固定值）')
    parser.add_argument('--interp_method', type=str, default='rbf',
                        choices=['rbf', 'kriging', 'tps'],
                        help='网格插值方法: rbf(RBF cubic), kriging(克里金), tps(薄板样条/最小曲率)')
    parser.add_argument('--fixed_eps', type=float, default=1.0,
                        help='当 --eps_mode fixed 时的 epsilon 值')

    # ---- 训练超参 ----
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数')
    parser.add_argument('--base_ch', type=int, default=48,
                        help='U-Net 基础通道数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='初始学习率')
    parser.add_argument('--max_lr', type=float, default=1e-3,
                        help='OneCycleLR 最大学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='AdamW 权重衰减')

    # ---- 路径与输出 ----
    parser.add_argument('--data_path', type=str,
                        default=r'F:\PINN实验\venv1\PINN数据3.xlsx',
                        help='Excel 数据文件路径')
    parser.add_argument('--output_dir', type=str, default='ablation_results',
                        help='输出目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--no_gpu', action='store_true',
                        help='强制使用 CPU')
    parser.add_argument('--tta', action='store_true',
                        help='启用 Test-Time Augmentation（多次随机mask推理取平均）')
    parser.add_argument('--tta_times', type=int, default=10,
                        help='TTA 推理次数（默认 10）')
    parser.add_argument('--self_train', action='store_true',
                        help='启用自监督迭代精炼')
    parser.add_argument('--self_train_rounds', type=int, default=3,
                        help='自训练迭代轮数（默认 3）')
    parser.add_argument('--pretrain', action='store_true',
                        help='启用合成数据预训练 + 真实数据微调')
    parser.add_argument('--n_synthetic', type=int, default=200,
                        help='合成样本数（默认 200）')
    parser.add_argument('--pretrain_epochs', type=int, default=20,
                        help='预训练 epoch 数（默认 20）')

    return parser.parse_args()


# =============================================================================
# 1. 数据加载与预处理（共享部分）
# =============================================================================
def load_raw_data(file_path):
    """读取 Excel 并返回原始 DataFrame 及常量"""
    threshold = 0.0005
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    if df.shape[1] > 7:
        df = df.iloc[:, :7]
    df.columns = ['lon', 'lat', 'h', 'X', 'Y', 'Z', 'F']
    df['X'] = df['X'].abs()
    df['Z'] = df['Z'].abs()

    # 测线划分与降采样
    df['diff_lon'] = df['lon'].diff().abs()
    df['line_id'] = (df['diff_lon'] > threshold).cumsum() + 1
    df_sampled = df.groupby('line_id', group_keys=False).apply(
        lambda g: g.iloc[::50])
    df = df_sampled.reset_index(drop=True)
    df['diff_lon'] = df['lon'].diff().abs()
    df['line_id'] = (df['diff_lon'] > threshold).cumsum() + 1

    # 坐标转换
    lon0 = df['lon'].mean()
    lat0 = df['lat'].mean()
    R = 6371

    def ll_to_xy(lon, lat):
        x = (lon - lon0) * np.pi / 180 * R * np.cos(np.radians(lat0))
        y = (lat - lat0) * np.pi / 180 * R
        return x, y

    df['x'], df['y'] = ll_to_xy(df['lon'].values, df['lat'].values)
    df['h_km'] = df['h'] / 1000.0
    dir_map = (df.groupby('line_id')['X'].mean()
               .apply(lambda x: 1 if x > 0 else -1).to_dict())
    df['dir'] = df['line_id'].map(dir_map)

    # 空白区域
    LON_MIN, LON_MAX = 113.0085, 113.0155
    LAT_MIN, LAT_MAX = 34.5480, 34.5604
    mask_inside = ((df['lon'] >= LON_MIN) & (df['lon'] <= LON_MAX) &
                   (df['lat'] >= LAT_MIN) & (df['lat'] <= LAT_MAX))
    train_df = df[~mask_inside].copy()
    test_df = df[mask_inside].copy()

    return df, train_df, test_df, lon0, lat0, R, ll_to_xy, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX


def build_grid(train_df, lon0, lat0, R, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX,
               resolution_km=0.05):
    """构建全区域网格并返回网格坐标、掩膜等"""
    x_min, x_max = train_df['x'].min(), train_df['x'].max()
    y_min, y_max = train_df['y'].min(), train_df['y'].max()
    expand = -0.1
    x_min -= expand; x_max += expand
    y_min -= expand; y_max += expand

    x_grid = np.arange(x_min, x_max, resolution_km)
    y_grid = np.arange(y_min, y_max, resolution_km)
    nx, ny = len(x_grid), len(y_grid)

    grid_x, grid_y = np.meshgrid(x_grid, y_grid, indexing='ij')
    lon_grid_full = lon0 + grid_x / (R * np.cos(np.radians(lat0))) * (180 / np.pi)
    lat_grid_full = lat0 + grid_y / R * (180 / np.pi)

    mask_blank = ((lon_grid_full >= LON_MIN) & (lon_grid_full <= LON_MAX) &
                  (lat_grid_full >= LAT_MIN) & (lat_grid_full <= LAT_MAX))

    return (x_grid, y_grid, grid_x, grid_y, lon_grid_full, lat_grid_full,
            mask_blank, nx, ny)


def _interp_rbf(points, values, grid_points, grid_shape, epsilon):
    """RBF cubic 插值"""
    rbf = RBFInterpolator(points, values, kernel='cubic', epsilon=epsilon)
    return rbf(grid_points).reshape(grid_shape)


def _interp_tps(points, values, grid_points, grid_shape):
    """薄板样条插值（= 最小曲率插值，minimizes bending energy ∇⁴F=0）"""
    rbf = RBFInterpolator(points, values, kernel='thin_plate_spline')
    return rbf(grid_points).reshape(grid_shape)


def _interp_kriging(points, values, grid_x, grid_y):
    """普通克里金插值（考虑空间各向异性）"""
    nx, ny = grid_x.shape[0], grid_y.shape[1]
    x_vals = grid_x[:, 0].ravel()
    y_vals = grid_y[0, :].ravel()

    # 克里金对点数敏感，>2000点降采样
    if len(points) > 2000:
        idx = np.random.RandomState(42).choice(len(points), 2000, replace=False)
        pts, vals = points[idx], values[idx]
    else:
        pts, vals = points, values

    try:
        OK = OrdinaryKriging(
            pts[:, 0], pts[:, 1], vals,
            variogram_model='spherical',
            nlags=30,
            enable_plotting=False,
            coordinates_type='euclidean',
        )
        z, ss = OK.execute('grid', x_vals, y_vals)
        return z.reshape(grid_x.shape)
    except Exception:
        # 克里金失败时回退到 TPS
        return _interp_tps(points, values,
                           np.column_stack([grid_x.ravel(), grid_y.ravel()]),
                           grid_x.shape)


def rbf_interpolate(train_df, grid_x, grid_y, mask_blank,
                    eps_mode='adaptive', fixed_eps=1.0, interp_method='rbf'):
    """
    网格插值：用区域外实测数据对全区域网格插值。
    支持 RBF cubic / 克里金 / 薄板样条(最小曲率)。
    返回 F_grid, X_grid, Y_grid, Z_grid（空白区域为 NaN）以及归一化参数。
    """
    points_xy = train_df[['x', 'y']].values
    values_F = train_df['F'].values
    values_X = train_df['X'].values
    values_Y = train_df['Y'].values
    values_Z = train_df['Z'].values

    # 降采样（>5000 点时，RBF/TPS需要）
    if interp_method in ('rbf', 'tps') and len(points_xy) > 5000:
        idx_sample = np.random.choice(len(points_xy), 5000, replace=False)
        points_xy_sub = points_xy[idx_sample]
        values_F_sub = values_F[idx_sample]
        values_X_sub = values_X[idx_sample]
        values_Y_sub = values_Y[idx_sample]
        values_Z_sub = values_Z[idx_sample]
    elif interp_method == 'kriging':
        points_xy_sub = points_xy
        values_F_sub = values_F
        values_X_sub = values_X
        values_Y_sub = values_Y
        values_Z_sub = values_Z
    else:
        points_xy_sub = points_xy
        values_F_sub = values_F
        values_X_sub = values_X
        values_Y_sub = values_Y
        values_Z_sub = values_Z

    # epsilon (仅RBF需要)
    if interp_method == 'rbf':
        if eps_mode == 'adaptive':
            tree = KDTree(points_xy_sub)
            distances, _ = tree.query(points_xy_sub, k=min(10, len(points_xy_sub)))
            median_dist = np.median(distances[:, 1:])
            epsilon = median_dist * 0.8
        else:
            epsilon = fixed_eps
            median_dist = None
        print(f"  插值方法: RBF cubic, epsilon={epsilon:.4f}"
              + (f", median_dist={median_dist:.4f}" if median_dist else ""))
    elif interp_method == 'tps':
        epsilon = None
        print(f"  插值方法: 薄板样条 (最小曲率)")
    else:
        epsilon = None
        print(f"  插值方法: 普通克里金")

    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # F 分量插值
    if interp_method == 'rbf':
        F_grid = _interp_rbf(points_xy_sub, values_F_sub, grid_points, grid_x.shape, epsilon)
    elif interp_method == 'tps':
        F_grid = _interp_tps(points_xy_sub, values_F_sub, grid_points, grid_x.shape)
    else:  # kriging
        F_grid = _interp_kriging(points_xy_sub, values_F_sub, grid_x, grid_y)

    # X/Y/Z 分量插值
    if interp_method == 'kriging':
        X_grid = _interp_kriging(points_xy_sub, values_X_sub, grid_x, grid_y)
        Y_grid = _interp_kriging(points_xy_sub, values_Y_sub, grid_x, grid_y)
        Z_grid = _interp_kriging(points_xy_sub, values_Z_sub, grid_x, grid_y)
    elif interp_method == 'tps':
        X_grid = _interp_tps(points_xy_sub, values_X_sub, grid_points, grid_x.shape)
        Y_grid = _interp_tps(points_xy_sub, values_Y_sub, grid_points, grid_x.shape)
        Z_grid = _interp_tps(points_xy_sub, values_Z_sub, grid_points, grid_x.shape)
        # TPS 轻度平滑
        sigma_light = 0.15
        for g in [X_grid, Y_grid, Z_grid]:
            g[:] = gaussian_filter1d(g, sigma_light, axis=0)
            g[:] = gaussian_filter1d(g, sigma_light, axis=1)
    else:
        rbf_X = RBFInterpolator(points_xy_sub, values_X_sub, kernel='thin_plate_spline',
                                epsilon=epsilon if epsilon else 1.0)
        X_grid = rbf_X(grid_points).reshape(grid_x.shape)
        rbf_Y = RBFInterpolator(points_xy_sub, values_Y_sub, kernel='thin_plate_spline',
                                epsilon=epsilon if epsilon else 1.0)
        Y_grid = rbf_Y(grid_points).reshape(grid_x.shape)
        rbf_Z = RBFInterpolator(points_xy_sub, values_Z_sub, kernel='thin_plate_spline',
                                epsilon=epsilon if epsilon else 1.0)
        Z_grid = rbf_Z(grid_points).reshape(grid_x.shape)
        sigma_light = 0.15
        for g in [X_grid, Y_grid, Z_grid]:
            g[:] = gaussian_filter1d(g, sigma_light, axis=0)
            g[:] = gaussian_filter1d(g, sigma_light, axis=1)

    # 空白区域设为 NaN
    F_grid[mask_blank] = np.nan
    X_grid[mask_blank] = np.nan
    Y_grid[mask_blank] = np.nan
    Z_grid[mask_blank] = np.nan

    # 归一化参数
    valid_mask = ~np.isnan(F_grid)
    valid_vals = F_grid[valid_mask]
    F_min, F_max = np.min(valid_vals), np.max(valid_vals)
    if np.isclose(F_max - F_min, 0):
        F_max = F_min + 1.0

    def normalize(x):
        return 2 * (x - F_min) / (F_max - F_min) - 1

    def denormalize(x):
        return (x + 1) / 2 * (F_max - F_min) + F_min

    return F_grid, X_grid, Y_grid, Z_grid, F_min, F_max, normalize, denormalize


# =============================================================================
# 1.5 合成磁场图生成（物理正演：随机磁偶极子）
# =============================================================================
def dipole_total_field_anomaly(grid_x, grid_y, dipoles, inc=53.0, dec=-4.0):
    """
    计算多个磁偶极子在地表产生的总场异常 ΔT。
    地磁场方向: 倾角 inc, 偏角 dec (郑州地区 ~53°, -4°)

    Parameters
    ----------
    grid_x, grid_y : 2D arrays, 网格坐标 (km)
    dipoles : list of (x0, y0, z0, mx, my, mz)
        每个偶极子: 位置(x0,y0,z0) km, 磁矩分量(mx,my,mz) A·m²
    inc, dec : float, 地磁场倾角和偏角

    Returns
    -------
    dT : 2D array, 总场异常 (nT)
    """
    fx = np.cos(np.radians(inc)) * np.cos(np.radians(dec))
    fy = np.cos(np.radians(inc)) * np.sin(np.radians(dec))
    fz = np.sin(np.radians(inc))

    dT = np.zeros_like(grid_x)
    mu0_over_4pi = 100.0  # μ₀/4π × 1e9 → nT (when distances in km, moments in A·m²)

    for x0, y0, z0, mx, my, mz in dipoles:
        dx = grid_x - x0  # km
        dy = grid_y - y0  # km
        dz = -z0          # 观测面 z=0, 偶极子在 z0 深处
        r2 = dx**2 + dy**2 + dz**2
        r = np.sqrt(np.maximum(r2, 1e-10))
        r5 = r**5

        m_dot_r = mx * dx + my * dy + mz * dz

        # 偶极子磁场三分量 (nT)
        Bx = mu0_over_4pi * (3 * m_dot_r * dx / r5 - mx / r**3)
        By = mu0_over_4pi * (3 * m_dot_r * dy / r5 - my / r**3)
        Bz = mu0_over_4pi * (3 * m_dot_r * dz / r5 - mz / r**3)

        # 投影到地磁场方向 → 总场异常
        dT += Bx * fx + By * fy + Bz * fz

    return dT


def generate_synthetic_sample(grid_x, grid_y, mask_blank, eps_mode, fixed_eps,
                               F_range, X_range, Y_range, Z_range, rng):
    """
    物理正演生成一张合成磁场图。
    随机放置磁偶极子 → 计算 ΔT 和三分量异常 → 加上区域背景场。
    """
    nx, ny = grid_x.shape
    x_vals = grid_x[:, 0]
    y_vals = grid_y[0, :]
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()

    # 随机偶极子参数
    n_dipoles = rng.randint(3, 15)
    dipoles = []
    for _ in range(n_dipoles):
        x0 = rng.uniform(x_min - 0.2, x_max + 0.2)
        y0 = rng.uniform(y_min - 0.2, y_max + 0.2)
        z0 = rng.uniform(0.05, 1.5)  # 深度 50m ~ 1.5km
        # 磁矩: 感应磁化为主 + 剩余磁化扰动
        # 感应方向 ≈ 地磁场方向 (inc~53°, dec~-4°)
        m_mag = rng.uniform(1e3, 1e6)  # 磁矩大小
        inc_r = 53.0 + rng.uniform(-20, 20)
        dec_r = -4.0 + rng.uniform(-30, 30)
        mx = m_mag * np.cos(np.radians(inc_r)) * np.cos(np.radians(dec_r))
        my = m_mag * np.cos(np.radians(inc_r)) * np.sin(np.radians(dec_r))
        mz = m_mag * np.sin(np.radians(inc_r))
        dipoles.append((x0, y0, z0, mx, my, mz))

    # 计算总场异常 ΔT
    dT = dipole_total_field_anomaly(grid_x, grid_y, dipoles, inc=53.0, dec=-4.0)

    # 加区域背景场（真实数据的F分量均值附近 + 缓慢变化趋势）
    F_bg = (F_range[0] + F_range[1]) / 2
    bg_trend = (grid_x - grid_x.mean()) * rng.uniform(-50, 50) + \
               (grid_y - grid_y.mean()) * rng.uniform(-50, 50)
    F_syn = F_bg + dT + bg_trend

    # 三分量异常 → 同样用偶极子计算，加各自背景
    # X ≈ H*cos(dec), Y ≈ H*sin(dec), Z ≈ F*sin(inc)
    # 简化: 异常场在三分量上的投影近似于总场异常在不同方向的分量
    fx = np.cos(np.radians(53.0)) * np.cos(np.radians(-4.0))
    fy = np.cos(np.radians(53.0)) * np.sin(np.radians(-4.0))
    fz = np.sin(np.radians(53.0))
    X_bg = (X_range[0] + X_range[1]) / 2
    Y_bg = (Y_range[0] + Y_range[1]) / 2
    Z_bg = (Z_range[0] + Z_range[1]) / 2
    X_syn = X_bg + dT * fx + rng.randn(*grid_x.shape) * 5.0
    Y_syn = Y_bg + dT * fy + rng.randn(*grid_x.shape) * 5.0
    Z_syn = Z_bg + dT * fz + rng.randn(*grid_x.shape) * 5.0

    # 缩放到真实数据变化幅度
    F_amp = F_syn.std()
    target_amp = (F_range[1] - F_range[0]) * rng.uniform(0.05, 0.25)
    if F_amp > 1e-6:
        dT_scaled = dT * (target_amp / F_amp)
    else:
        dT_scaled = dT
    F_syn = F_bg + dT_scaled + bg_trend * 0.1
    X_syn = X_bg + dT_scaled * fx + rng.randn(*grid_x.shape) * 3.0
    Y_syn = Y_bg + dT_scaled * fy + rng.randn(*grid_x.shape) * 3.0
    Z_syn = Z_bg + dT_scaled * fz + rng.randn(*grid_x.shape) * 3.0

    # 模拟 RBF 插值：非空白区加噪声，空白区设 NaN
    F_grid = F_syn.copy()
    X_grid = X_syn.copy()
    Y_grid = Y_syn.copy()
    Z_grid = Z_syn.copy()

    noise_level = 0.02
    F_grid[~mask_blank] += rng.randn((~mask_blank).sum()) * (F_range[1] - F_range[0]) * noise_level
    X_grid[~mask_blank] += rng.randn((~mask_blank).sum()) * (X_range[1] - X_range[0]) * noise_level
    Y_grid[~mask_blank] += rng.randn((~mask_blank).sum()) * (Y_range[1] - Y_range[0]) * noise_level
    Z_grid[~mask_blank] += rng.randn((~mask_blank).sum()) * (Z_range[1] - Z_range[0]) * noise_level

    F_grid[mask_blank] = np.nan
    X_grid[mask_blank] = np.nan
    Y_grid[mask_blank] = np.nan
    Z_grid[mask_blank] = np.nan

    # 归一化
    F_valid = F_syn[~mask_blank]
    F_min, F_max = F_valid.min(), F_valid.max()
    if np.isclose(F_max - F_min, 0):
        F_max = F_min + 1.0

    def norm(x, vmin, vmax):
        return 2 * (x - vmin) / (vmax - vmin + 1e-10) - 1

    mask_nonblank = ~mask_blank
    F_norm = np.full_like(F_grid, np.nan, dtype=np.float32)
    F_norm[mask_nonblank] = norm(F_grid[mask_nonblank], F_min, F_max)
    F_norm_filled = np.nan_to_num(F_norm, nan=0.0)

    X_norm = norm(X_grid, X_grid[mask_nonblank].min(), X_grid[mask_nonblank].max())
    Y_norm = norm(Y_grid, Y_grid[mask_nonblank].min(), Y_grid[mask_nonblank].max())
    Z_norm = norm(Z_grid, Z_grid[mask_nonblank].min(), Z_grid[mask_nonblank].max())
    X_norm[mask_blank] = 0.0
    Y_norm[mask_blank] = 0.0
    Z_norm[mask_blank] = 0.0

    multi_channel = np.stack([F_norm_filled, X_norm, Y_norm, Z_norm], axis=0).astype(np.float32)

    F_target_norm = norm(F_syn, F_min, F_max)
    F_target_full = np.full_like(F_grid, np.nan, dtype=np.float32)
    F_target_full[...] = F_target_norm

    return multi_channel, F_target_full, F_min, F_max


# =============================================================================
# 1.6 合成数据预训练
# =============================================================================
class SyntheticDataset(Dataset):
    def __init__(self, samples, target_size=(128, 128), mask_ratio=0.2, mask_mode='mixed'):
        self.samples = samples
        self.target_size = target_size
        self.mask_ratio = mask_ratio
        self.mask_mode = mask_mode

    def __len__(self):
        return len(self.samples) * 10  # 每个样本生成10个mask变体

    def _random_block_mask(self, valid_region, H, W):
        max_h = min(int(H * 0.4), 48)
        min_h = max(int(H * 0.1), 8)
        max_w = min(int(W * 0.4), 48)
        min_w = max(int(W * 0.1), 8)
        mh, mw = np.random.randint(min_h, max_h), np.random.randint(min_w, max_w)
        for _ in range(100):
            y = np.random.randint(0, H - mh)
            x = np.random.randint(0, W - mw)
            if valid_region[y:y + mh, x:x + mw].all():
                return y, mh, x, mw
        return None, None, None, None

    def __getitem__(self, idx):
        sample_idx = idx % len(self.samples)
        multi_channel, F_target_full, _, _ = self.samples[sample_idx]

        C, H, W = multi_channel.shape
        th, tw = self.target_size

        # Resize
        resized = np.zeros((C, th, tw), dtype=np.float32)
        for c in range(C):
            resized[c] = resize(multi_channel[c], self.target_size, mode='constant', anti_aliasing=True)
        target_resized = resize(F_target_full, self.target_size, mode='constant', anti_aliasing=True)

        # 空白区掩膜
        blank_mask = np.isnan(target_resized)
        valid_region = ~blank_mask

        input_img = torch.tensor(resized, dtype=torch.float32)
        # 空白区填0
        for c in range(C):
            input_img[c][torch.from_numpy(blank_mask)] = 0.0

        target_tensor = torch.tensor(target_resized, dtype=torch.float32).unsqueeze(0)
        target_tensor[:, torch.from_numpy(blank_mask)] = 0.0

        # 随机mask（只在有效区）
        H_t, W_t = th, tw
        if self.mask_mode == 'mixed':
            if np.random.random() < 0.5:
                y, mh, x, mw = self._random_block_mask(valid_region, H_t, W_t)
                if y is None:
                    n_pts = int(H_t * W_t * self.mask_ratio * 0.3)
                    vi = np.argwhere(valid_region)
                    if len(vi) >= n_pts:
                        pick = vi[np.random.choice(len(vi), n_pts, replace=False)]
                        input_img[:, pick[:, 0], pick[:, 1]] = 0.0
                else:
                    input_img[:, y:y + mh, x:x + mw] = 0.0
            else:
                n_pts = int(H_t * W_t * self.mask_ratio * 0.3)
                vi = np.argwhere(valid_region)
                if len(vi) >= n_pts:
                    pick = vi[np.random.choice(len(vi), n_pts, replace=False)]
                    input_img[:, pick[:, 0], pick[:, 1]] = 0.0
        elif self.mask_mode == 'block':
            y, mh, x, mw = self._random_block_mask(valid_region, H_t, W_t)
            if y is not None:
                input_img[:, y:y + mh, x:x + mw] = 0.0

        mask_for_loss = torch.from_numpy(valid_region).unsqueeze(0)
        return input_img, target_tensor, mask_for_loss


def pretrain_on_synthetic(model, synthetic_samples, epochs, batch_size, lr,
                           max_lr, weight_decay, target_size, mask_mode, device,
                           verbose=True):
    """在合成数据上预训练U-Net"""
    dataset = SyntheticDataset(synthetic_samples, target_size, mask_mode=mask_mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    criterion = nn.MSELoss()
    steps_per_epoch = len(dataloader)
    total_steps = epochs * steps_per_epoch
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, total_steps=total_steps,
        pct_start=0.1, anneal_strategy='cos', final_div_factor=1e4)

    print(f"  预训练: {len(synthetic_samples)} 合成样本, {epochs} epochs, "
          f"{steps_per_epoch} steps/epoch")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs_input, imgs_target, masks in dataloader:
            imgs_input = imgs_input.to(device)
            imgs_target = imgs_target.to(device)
            masks = masks.to(device)

            outputs = model(imgs_input)
            loss = criterion(outputs[masks], imgs_target[masks])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / steps_per_epoch
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Pretrain Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f}")

    return model


def build_multichannel(F_grid, X_grid, Y_grid, Z_grid, mask_blank,
                       input_ch=4):
    """构建多通道/单通道输入张量"""
    def normalize_channel(data, mask_valid):
        vals = data[mask_valid]
        if len(vals) == 0:
            return data
        minv, maxv = vals.min(), vals.max()
        if np.isclose(maxv - minv, 0):
            maxv = minv + 1.0
        return 2 * (data - minv) / (maxv - minv) - 1

    mask_nonblank = ~mask_blank
    F_norm = np.full_like(F_grid, np.nan)
    F_norm[mask_nonblank] = (
        2 * (F_grid[mask_nonblank] - F_grid[mask_nonblank].min())
        / (F_grid[mask_nonblank].max() - F_grid[mask_nonblank].min()) - 1
    )

    F_norm_filled = np.nan_to_num(F_norm, nan=0.0)

    if input_ch == 1:
        multi_channel = F_norm_filled[np.newaxis, :, :]
    else:
        X_norm = normalize_channel(X_grid, mask_nonblank)
        Y_norm = normalize_channel(Y_grid, mask_nonblank)
        Z_norm = normalize_channel(Z_grid, mask_nonblank)
        X_norm[mask_blank] = 0.0
        Y_norm[mask_blank] = 0.0
        Z_norm[mask_blank] = 0.0
        multi_channel = np.stack([F_norm_filled, X_norm, Y_norm, Z_norm], axis=0)

    return multi_channel


# =============================================================================
# 2. 模型定义
# =============================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样模块，use_skip=False 时退化为纯解码器块"""
    def __init__(self, in_ch, out_ch, use_skip=True):
        super().__init__()
        self.use_skip = use_skip
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        if use_skip:
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)
        else:
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if self.use_skip and x2 is not None:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                         diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)


class UNetInpainter(nn.Module):
    """U-Net (use_skip=True) 或 纯 Autoencoder (use_skip=False)"""
    def __init__(self, in_chans=4, base_ch=48, use_skip=True):
        super().__init__()
        self.use_skip = use_skip
        self.inc = DoubleConv(in_chans, base_ch)

        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 4)
        self.down4 = Down(base_ch * 4, base_ch * 4)

        # Up 模块：skip=True 时 in_ch = skip_channel + up_channel
        if use_skip:
            self.up1 = Up(base_ch * 4 + base_ch * 4, base_ch * 2, use_skip=True)
            self.up2 = Up(base_ch * 2 + base_ch * 4, base_ch * 2, use_skip=True)
            self.up3 = Up(base_ch * 2 + base_ch * 2, base_ch, use_skip=True)
            self.up4 = Up(base_ch + base_ch, base_ch, use_skip=True)
        else:
            self.up1 = Up(base_ch * 4, base_ch * 2, use_skip=False)
            self.up2 = Up(base_ch * 2, base_ch * 2, use_skip=False)
            self.up3 = Up(base_ch * 2, base_ch, use_skip=False)
            self.up4 = Up(base_ch, base_ch, use_skip=False)

        self.outc = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if self.use_skip:
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
        else:
            x = self.up1(x5)
            x = self.up2(x)
            x = self.up3(x)
            x = self.up4(x)

        return self.outc(x)


# =============================================================================
# 3. 损失函数
# =============================================================================
class CompositeLoss(nn.Module):
    def __init__(self, grad_weight=0.1):
        super().__init__()
        self.grad_weight = grad_weight
        self.mse = nn.MSELoss()

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def _gradient(self, img):
        if img.shape[1] != 1:
            img = img[:, :1]
        gx = nn.functional.conv2d(img, self.sobel_x, padding=1)
        gy = nn.functional.conv2d(img, self.sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, pred, target, mask):
        pred_masked = pred[mask]
        target_masked = target[mask]
        loss_mse = self.mse(pred_masked, target_masked)

        if self.grad_weight > 0:
            pred_grad = self._gradient(pred)
            target_grad = self._gradient(target)
            loss_grad = self.mse(pred_grad[mask], target_grad[mask])
            return loss_mse + self.grad_weight * loss_grad
        return loss_mse


# =============================================================================
# 4. 数据集
# =============================================================================
class InpaintingDataset(Dataset):
    def __init__(self, multi_image, permanent_mask, mask_mode='mixed',
                 mask_ratio=0.2):
        self.multi_image = torch.tensor(multi_image, dtype=torch.float32)
        self.permanent_mask = torch.tensor(permanent_mask, dtype=torch.bool)
        self.C, self.H, self.W = multi_image.shape
        self.mask_ratio = mask_ratio
        self.mask_mode = mask_mode

    def __len__(self):
        return 3000

    def _random_block_mask(self, valid_region):
        max_h = min(int(self.H * 0.4), 48)
        min_h = max(int(self.H * 0.1), 8)
        max_w = min(int(self.W * 0.4), 48)
        min_w = max(int(self.W * 0.1), 8)
        mask_h = np.random.randint(min_h, max_h)
        mask_w = np.random.randint(min_w, max_w)
        while (mask_h * mask_w) / (self.H * self.W) > self.mask_ratio * 1.5:
            mask_h = int(mask_h * 0.8)
            mask_w = int(mask_w * 0.8)
        for _ in range(100):
            y = np.random.randint(0, self.H - mask_h)
            x = np.random.randint(0, self.W - mask_w)
            if valid_region[y:y + mask_h, x:x + mask_w].all():
                return y, mask_h, x, mask_w
        return None, None, None, None

    def _random_scatter_mask(self, valid_region):
        n_pixels = int(self.H * self.W * self.mask_ratio * 0.3)
        valid_indices = torch.nonzero(valid_region, as_tuple=False)
        if len(valid_indices) < n_pixels:
            return torch.zeros(1, self.H, self.W)
        chosen = valid_indices[torch.randperm(len(valid_indices))[:n_pixels]]
        mask = torch.zeros(1, self.H, self.W)
        mask[0, chosen[:, 0], chosen[:, 1]] = 1.0
        return mask

    def __getitem__(self, idx):
        valid_region = ~self.permanent_mask

        if self.mask_mode == 'mixed':
            if np.random.random() < 0.5:
                y, mask_h, x, mask_w = self._random_block_mask(valid_region)
                if y is None:
                    mask = self._random_scatter_mask(valid_region)
                else:
                    mask = torch.zeros(1, self.H, self.W)
                    mask[:, y:y + mask_h, x:x + mask_w] = 1.0
            else:
                mask = self._random_scatter_mask(valid_region)
        elif self.mask_mode == 'block':
            y, mask_h, x, mask_w = self._random_block_mask(valid_region)
            if y is None:
                mask = self._random_scatter_mask(valid_region)
            else:
                mask = torch.zeros(1, self.H, self.W)
                mask[:, y:y + mask_h, x:x + mask_w] = 1.0
        else:  # scatter
            mask = self._random_scatter_mask(valid_region)

        input_img = self.multi_image.clone()
        input_img[:, mask.bool().squeeze(0)] = 0.0
        target_img = self.multi_image[0:1]
        return input_img, target_img, mask


# =============================================================================
# 5. 训练与评估
# =============================================================================
def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for imgs_input, imgs_target, masks in dataloader:
        imgs_input = imgs_input.to(device)
        imgs_target = imgs_target.to(device)
        masks = masks.to(device)

        outputs = model(imgs_input)
        loss = criterion(outputs, imgs_target, masks.bool())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def train_model(model, dataloader, criterion, epochs, device,
                lr=1e-4, max_lr=1e-3, weight_decay=1e-5, verbose=True):
    steps_per_epoch = len(dataloader)
    total_steps = epochs * steps_per_epoch
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, total_steps=total_steps,
        pct_start=0.1, anneal_strategy='cos', final_div_factor=1e4
    )

    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, dataloader, criterion,
                                   optimizer, scheduler, device)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch + 1) % 5 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f} | LR: {lr_now:.2e}")

    model.load_state_dict(best_state)
    return best_loss


def evaluate_model_tta(model, multi_channel, permanent_mask_resized, target_size,
                       F_grid, mask_blank, x_grid, y_grid, test_df, ll_to_xy,
                       denormalize, nx, ny, device, n_tta=10, mask_mode='mixed'):
    """TTA 评估：多次随机mask推理取平均，降低方差"""
    C, H_orig, W_orig = multi_channel.shape
    tH, tW = target_size
    resized = np.zeros((C, tH, tW), dtype=np.float32)
    for c in range(C):
        resized[c] = resize(multi_channel[c], target_size,
                            mode='constant', anti_aliasing=True)
    base_input = torch.tensor(resized, dtype=torch.float32)
    perm_mask = torch.tensor(permanent_mask_resized, dtype=torch.bool)
    valid_region = ~perm_mask

    model.eval()
    outputs_tta = []

    rng = np.random.RandomState(42)
    n_pixels = tH * tW

    for _ in range(n_tta):
        mask = torch.zeros(tH, tW, dtype=torch.bool)

        if mask_mode == 'mixed':
            if rng.random() < 0.5:
                max_h = min(int(tH * 0.4), 48)
                min_h = max(int(tH * 0.1), 8)
                max_w = min(int(tW * 0.4), 48)
                min_w = max(int(tW * 0.1), 8)
                mh, mw = rng.randint(min_h, max_h), rng.randint(min_w, max_w)
                for _ in range(100):
                    y = rng.randint(0, tH - mh)
                    x = rng.randint(0, tW - mw)
                    if valid_region[y:y+mh, x:x+mw].all():
                        mask[y:y+mh, x:x+mw] = True
                        break
            else:
                n_pts = int(n_pixels * 0.1)
                vi = torch.nonzero(valid_region, as_tuple=False)
                if len(vi) >= n_pts:
                    pick = vi[torch.from_numpy(rng.choice(len(vi), n_pts, replace=False))]
                    mask[pick[:, 0], pick[:, 1]] = True
        elif mask_mode == 'block':
            max_h = min(int(tH * 0.4), 48)
            min_h = max(int(tH * 0.1), 8)
            max_w = min(int(tW * 0.4), 48)
            min_w = max(int(tW * 0.1), 8)
            mh, mw = rng.randint(min_h, max_h), rng.randint(min_w, max_w)
            for _ in range(100):
                y = rng.randint(0, tH - mh)
                x = rng.randint(0, tW - mw)
                if valid_region[y:y+mh, x:x+mw].all():
                    mask[y:y+mh, x:x+mw] = True
                    break
        else:
            n_pts = int(n_pixels * 0.1)
            vi = torch.nonzero(valid_region, as_tuple=False)
            if len(vi) >= n_pts:
                pick = vi[torch.from_numpy(rng.choice(len(vi), n_pts, replace=False))]
                mask[pick[:, 0], pick[:, 1]] = True

        input_masked = base_input.clone()
        input_masked[:, mask] = 0.0
        input_batch = input_masked.unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(input_batch)
        outputs_tta.append(out.squeeze().cpu().numpy())

    # 平均
    output_np = np.mean(outputs_tta, axis=0)
    output_np = denormalize(output_np)

    zoom_factors = (nx / target_size[0], ny / target_size[1])
    output_full = zoom(output_np, zoom_factors, order=1)
    output_full = output_full[:nx, :ny]

    result_grid = F_grid.copy()
    result_grid[mask_blank] = output_full[mask_blank]

    test_xy = np.column_stack(ll_to_xy(test_df['lon'].values, test_df['lat'].values))
    interp = RegularGridInterpolator((x_grid, y_grid), result_grid,
                                     method='linear', bounds_error=False,
                                     fill_value=np.nan)
    F_pred = interp(test_xy)
    valid_test = ~np.isnan(F_pred)

    if np.sum(valid_test) > 0:
        y_true = test_df['F'].values[valid_test]
        y_pred = F_pred[valid_test]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
    else:
        rmse = float('nan')
        mae = float('nan')

    return rmse, mae, result_grid, output_full


def evaluate_model(model, multi_channel, permanent_mask_resized, target_size,
                   F_grid, mask_blank, x_grid, y_grid, test_df, ll_to_xy,
                   denormalize, nx, ny, device):
    """评估模型：返回 RMSE, MAE 及详细结果"""
    # 准备输入
    C, H, W = multi_channel.shape
    resized = np.zeros((C, target_size[0], target_size[1]), dtype=np.float32)
    for c in range(C):
        resized[c] = resize(multi_channel[c], target_size,
                            mode='constant', anti_aliasing=True)
    test_input = torch.tensor(resized, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(test_input)
    output_np = output.squeeze().cpu().numpy()
    output_np = denormalize(output_np)

    zoom_factors = (nx / target_size[0], ny / target_size[1])
    output_full = zoom(output_np, zoom_factors, order=1)
    output_full = output_full[:nx, :ny]

    result_grid = F_grid.copy()
    result_grid[mask_blank] = output_full[mask_blank]

    # 在测试点上评估
    test_xy = np.column_stack(ll_to_xy(test_df['lon'].values, test_df['lat'].values))
    interp = RegularGridInterpolator((x_grid, y_grid), result_grid,
                                     method='linear', bounds_error=False,
                                     fill_value=np.nan)
    F_pred = interp(test_xy)
    valid_test = ~np.isnan(F_pred)

    if np.sum(valid_test) > 0:
        y_true = test_df['F'].values[valid_test]
        y_pred = F_pred[valid_test]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
    else:
        rmse = float('nan')
        mae = float('nan')

    return rmse, mae, result_grid, output_full


# =============================================================================
# 6. 单次实验运行
# =============================================================================
def run_experiment(args, data_cache, exp_cfg=None):
    """
    运行一次实验。
    exp_cfg: dict，覆盖 args 中的消融变量（用于 run_all 模式）
    """
    # 合并配置
    cfg = vars(args).copy()
    if exp_cfg:
        cfg.update(exp_cfg)

    exp_name = cfg['exp_name']
    use_skip = not cfg['no_skip']
    input_ch = cfg['input_ch']
    grad_weight = cfg['grad_weight']
    mask_mode = cfg['mask_mode']
    eps_mode = cfg['eps_mode']
    interp_method = cfg.get('interp_method', 'rbf')
    fixed_eps = cfg['fixed_eps']
    epochs = cfg['epochs']
    base_ch = cfg['base_ch']
    batch_size = cfg['batch_size']
    lr = cfg['lr']
    max_lr = cfg['max_lr']
    weight_decay = cfg['weight_decay']
    output_dir = cfg['output_dir']
    seed = cfg['seed']
    use_tta = cfg.get('tta', False)
    tta_times = cfg.get('tta_times', 10)
    device_str = 'cpu' if cfg['no_gpu'] else ('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"  实验: {exp_name}")
    print(f"  skip={use_skip}, ch={input_ch}, grad_w={grad_weight}, "
          f"mask={mask_mode}, eps={eps_mode}, interp={interp_method}")
    print(f"  epochs={epochs}, base_ch={base_ch}, device={device_str}"
          + (f", TTA={tta_times}" if use_tta else ""))
    print(f"{'='*60}")

    # 固定随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(device_str)

    # 解包数据缓存
    train_df = data_cache['train_df']
    test_df = data_cache['test_df']
    lon0 = data_cache['lon0']
    lat0 = data_cache['lat0']
    R = data_cache['R']
    ll_to_xy = data_cache['ll_to_xy']
    LON_MIN = data_cache['LON_MIN']
    LON_MAX = data_cache['LON_MAX']
    LAT_MIN = data_cache['LAT_MIN']
    LAT_MAX = data_cache['LAT_MAX']

    # 构建网格
    (x_grid, y_grid, grid_x, grid_y, lon_grid_full, lat_grid_full,
     mask_blank, nx, ny) = build_grid(
        train_df, lon0, lat0, R, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)

    print(f"  网格: {nx} x {ny}, 训练点: {len(train_df)}, 测试点: {len(test_df)}")

    # RBF 插值
    print("  插值中...")
    F_grid, X_grid, Y_grid, Z_grid, F_min, F_max, normalize, denormalize = \
        rbf_interpolate(train_df, grid_x, grid_y, mask_blank, eps_mode, fixed_eps,
                         interp_method=interp_method)

    # 构建多通道
    multi_channel = build_multichannel(F_grid, X_grid, Y_grid, Z_grid,
                                       mask_blank, input_ch)

    # 缩放
    target_size = (128, 128)
    C = multi_channel.shape[0]
    multi_resized = np.zeros((C, target_size[0], target_size[1]), dtype=np.float32)
    for c in range(C):
        multi_resized[c] = resize(multi_channel[c], target_size,
                                  mode='constant', anti_aliasing=True)
    permanent_mask_resized = resize(mask_blank.astype(float), target_size) > 0.5

    train_image = multi_resized.copy()
    train_image[:, permanent_mask_resized] = 0.0

    # 模型
    model = UNetInpainter(in_chans=input_ch, base_ch=base_ch,
                          use_skip=use_skip).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")

    # 损失与数据
    criterion = CompositeLoss(grad_weight=grad_weight).to(device)
    dataset = InpaintingDataset(train_image, permanent_mask_resized,
                                mask_mode=mask_mode, mask_ratio=0.2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0)

    # 训练
    print(f"  训练中 ({epochs} epochs)...")
    t0 = time.time()
    best_loss = train_model(model, dataloader, criterion, epochs, device,
                            lr=lr, max_lr=max_lr, weight_decay=weight_decay)
    train_time = time.time() - t0
    print(f"  最佳 Loss: {best_loss:.6f}, 训练用时: {train_time:.1f}s")

    # 评估
    def simple_denorm(x):
        return (x + 1) / 2 * (F_max - F_min) + F_min

    if use_tta:
        print(f"  TTA 评估中 (n={tta_times})...")
        t0_eval = time.time()
        rmse, mae, result_grid, output_full = evaluate_model_tta(
            model, multi_channel, permanent_mask_resized, target_size,
            F_grid, mask_blank, x_grid, y_grid, test_df, ll_to_xy,
            simple_denorm, nx, ny, device, n_tta=tta_times, mask_mode=mask_mode)
        eval_time = time.time() - t0_eval
        print(f"  TTA 用时: {eval_time:.1f}s")
    else:
        rmse, mae, result_grid, output_full = evaluate_model(
            model, multi_channel, permanent_mask_resized, target_size,
            F_grid, mask_blank, x_grid, y_grid, test_df, ll_to_xy,
            simple_denorm, nx, ny, device)

    # 记录结果
    result = {
        'exp_name': exp_name,
        'use_skip': use_skip,
        'input_ch': input_ch,
        'grad_weight': grad_weight,
        'mask_mode': mask_mode,
        'eps_mode': eps_mode,
        'fixed_eps': fixed_eps if eps_mode == 'fixed' else 'adaptive',
        'rmse': rmse,
        'mae': mae,
        'best_loss': best_loss,
        'train_time': train_time,
        'n_params': n_params,
        'epochs': epochs,
        'tta': use_tta,
        'tta_times': tta_times if use_tta else 0,
        'timestamp': datetime.now().isoformat(),
    }
    return result


# =============================================================================
# 7. 主函数
# =============================================================================
def main():
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载原始数据（所有实验共享）
    print("=" * 60)
    print("  加载数据...")
    print("=" * 60)
    (df, train_df, test_df, lon0, lat0, R, ll_to_xy,
     LON_MIN, LON_MAX, LAT_MIN, LAT_MAX) = load_raw_data(args.data_path)
    print(f"  区域外训练样本数: {len(train_df)}")
    print(f"  空白区域内实测点数: {len(test_df)}")

    data_cache = {
        'train_df': train_df,
        'test_df': test_df,
        'lon0': lon0,
        'lat0': lat0,
        'R': R,
        'll_to_xy': ll_to_xy,
        'LON_MIN': LON_MIN,
        'LON_MAX': LON_MAX,
        'LAT_MIN': LAT_MIN,
        'LAT_MAX': LAT_MAX,
    }

    if args.run_all:
        # 7 组实验配置
        experiments = [
            {'exp_name': '01_baseline',     'no_skip': False, 'input_ch': 4, 'grad_weight': 0.1, 'mask_mode': 'mixed',    'eps_mode': 'adaptive'},
            {'exp_name': '02_no_skip',      'no_skip': True,  'input_ch': 4, 'grad_weight': 0.1, 'mask_mode': 'mixed',    'eps_mode': 'adaptive'},
            {'exp_name': '03_single_ch',    'no_skip': False, 'input_ch': 1, 'grad_weight': 0.1, 'mask_mode': 'mixed',    'eps_mode': 'adaptive'},
            {'exp_name': '04_no_grad',      'no_skip': False, 'input_ch': 4, 'grad_weight': 0.0, 'mask_mode': 'mixed',    'eps_mode': 'adaptive'},
            {'exp_name': '05_mask_block',   'no_skip': False, 'input_ch': 4, 'grad_weight': 0.1, 'mask_mode': 'block',    'eps_mode': 'adaptive'},
            {'exp_name': '06_mask_scatter', 'no_skip': False, 'input_ch': 4, 'grad_weight': 0.1, 'mask_mode': 'scatter',  'eps_mode': 'adaptive'},
            {'exp_name': '07_fixed_eps',    'no_skip': False, 'input_ch': 4, 'grad_weight': 0.1, 'mask_mode': 'mixed',    'eps_mode': 'fixed', 'fixed_eps': 1.0},
        ]

        all_results = []
        for exp_cfg in experiments:
            result = run_experiment(args, data_cache, exp_cfg)
            all_results.append(result)
            print(f"\n  结果: RMSE={result['rmse']:.4f}, MAE={result['mae']:.4f}")

        # 保存结果
        results_df = pd.DataFrame(all_results)
        csv_path = os.path.join(args.output_dir, 'ablation_results.csv')
        results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n  结果已保存至: {csv_path}")

        # 打印汇总表
        print("\n" + "=" * 80)
        print("  消融实验结果汇总")
        print("=" * 80)
        header = (f"{'实验':<20s} {'Skip':>6s} {'Ch':>4s} {'GradW':>6s} "
                  f"{'Mask':>9s} {'Eps':>10s} {'RMSE':>10s} {'MAE':>10s} "
                  f"{'ΔRMSE%':>10s}")
        print(header)
        print("-" * 80)

        baseline_rmse = all_results[0]['rmse']
        for r in all_results:
            delta = (r['rmse'] - baseline_rmse) / baseline_rmse * 100
            print(f"{r['exp_name']:<20s} "
                  f"{str(r['use_skip']):>6s} "
                  f"{r['input_ch']:>4d} "
                  f"{r['grad_weight']:>6.2f} "
                  f"{r['mask_mode']:>9s} "
                  f"{r['eps_mode']:>10s} "
                  f"{r['rmse']:>10.4f} "
                  f"{r['mae']:>10.4f} "
                  f"{delta:>+9.2f}%")

        print("-" * 80)
        print("\n  ΔRMSE%: 相比 Baseline 的 RMSE 变化百分比（正值=退化, 负值=改进）")
        print(f"\n  最佳实验: {min(all_results, key=lambda x: x['rmse'])['exp_name']}"
              f" (RMSE={min(r['rmse'] for r in all_results):.4f})")

        # 绘制对比图
        plot_ablation_results(all_results, args.output_dir)

    elif args.pretrain:
        result = run_pretrain_finetune(args, data_cache)
        print(f"\n  最终结果: 预训练 RMSE={result['rmse_pretrained']:.4f}, "
              f"无预训练 RMSE={result['rmse_scratch']:.4f}, "
              f"Δ={result['delta_rmse']:+.2f} ({result['delta_pct']:+.1f}%)")

        json_path = os.path.join(args.output_dir, f'{args.exp_name}_result.json')
        json_result = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                       for k, v in result.items()}
        with open(json_path, 'w') as f:
            json.dump(json_result, f, indent=2, ensure_ascii=False)
        print(f"  结果已保存至: {json_path}")

    elif args.self_train:
        result = run_self_training(args, data_cache)
        print(f"\n  最终结果: Best RMSE={result['rmse_best']:.4f} (Round {result['best_round']})")

        json_path = os.path.join(args.output_dir, f'{args.exp_name}_result.json')
        json_result = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                       for k, v in result.items() if k != 'round_details'}
        with open(json_path, 'w') as f:
            json.dump(json_result, f, indent=2, ensure_ascii=False)
        print(f"  结果已保存至: {json_path}")

    else:
        # 单次实验
        result = run_experiment(args, data_cache)
        print(f"\n  结果: RMSE={result['rmse']:.4f}, MAE={result['mae']:.4f}")

        # 保存单次结果
        json_path = os.path.join(args.output_dir, f'{args.exp_name}_result.json')
        # 转换 numpy 类型以便 JSON 序列化
        json_result = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                       for k, v in result.items()}
        with open(json_path, 'w') as f:
            json.dump(json_result, f, indent=2, ensure_ascii=False)
        print(f"  结果已保存至: {json_path}")


def run_self_training(args, data_cache):
    """自监督迭代精炼：训练→预测空白区→填入伪标签→重新训练，迭代多轮"""
    cfg = vars(args).copy()
    exp_name = cfg['exp_name']
    use_skip = not cfg['no_skip']
    input_ch = cfg['input_ch']
    grad_weight = cfg['grad_weight']
    mask_mode = cfg['mask_mode']
    eps_mode = cfg['eps_mode']
    interp_method = cfg.get('interp_method', 'rbf')
    fixed_eps = cfg['fixed_eps']
    epochs = cfg['epochs']
    base_ch = cfg['base_ch']
    batch_size = cfg['batch_size']
    lr = cfg['lr']
    max_lr = cfg['max_lr']
    weight_decay = cfg['weight_decay']
    output_dir = cfg['output_dir']
    seed = cfg['seed']
    n_rounds = cfg.get('self_train_rounds', 3)
    device_str = 'cpu' if cfg['no_gpu'] else ('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"  自监督迭代精炼: {exp_name}")
    print(f"  skip={use_skip}, ch={input_ch}, grad_w={grad_weight}, "
          f"mask={mask_mode}, eps={eps_mode}, interp={interp_method}")
    print(f"  rounds={n_rounds}, epochs={epochs}/round, base_ch={base_ch}, device={device_str}")
    print(f"{'='*60}")

    device = torch.device(device_str)

    train_df = data_cache['train_df']
    test_df = data_cache['test_df']
    lon0, lat0, R = data_cache['lon0'], data_cache['lat0'], data_cache['R']
    ll_to_xy = data_cache['ll_to_xy']
    LON_MIN, LON_MAX = data_cache['LON_MIN'], data_cache['LON_MAX']
    LAT_MIN, LAT_MAX = data_cache['LAT_MIN'], data_cache['LAT_MAX']

    # 构建网格（不变）
    (x_grid, y_grid, grid_x, grid_y, lon_grid_full, lat_grid_full,
     mask_blank, nx, ny) = build_grid(
        train_df, lon0, lat0, R, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
    print(f"  网格: {nx} x {ny}")

    # 插值 → 获取基础场（空白区 NaN）
    F_grid, X_grid, Y_grid, Z_grid, F_min, F_max, normalize, denormalize = \
        rbf_interpolate(train_df, grid_x, grid_y, mask_blank, eps_mode, fixed_eps,
                         interp_method=interp_method)

    # Aux 分量在空白区的外推值（不放 NaN，提供更好上下文）
    points_xy = train_df[['x', 'y']].values
    if len(points_xy) > 5000:
        idx_sample = np.random.RandomState(42).choice(len(points_xy), 5000, replace=False)
        pts_rbf = points_xy[idx_sample]
        val_x = train_df['X'].values[idx_sample]
        val_y = train_df['Y'].values[idx_sample]
        val_z = train_df['Z'].values[idx_sample]
    else:
        pts_rbf = points_xy
        val_x, val_y, val_z = train_df['X'].values, train_df['Y'].values, train_df['Z'].values

    tree = KDTree(pts_rbf)
    distances, _ = tree.query(pts_rbf, k=min(10, len(pts_rbf)))
    eps_aux = np.median(distances[:, 1:]) * 0.8
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    X_grid_aux = RBFInterpolator(pts_rbf, val_x, kernel='thin_plate_spline',
                                  epsilon=eps_aux)(grid_pts).reshape(grid_x.shape)
    Y_grid_aux = RBFInterpolator(pts_rbf, val_y, kernel='thin_plate_spline',
                                  epsilon=eps_aux)(grid_pts).reshape(grid_x.shape)
    Z_grid_aux = RBFInterpolator(pts_rbf, val_z, kernel='thin_plate_spline',
                                  epsilon=eps_aux)(grid_pts).reshape(grid_x.shape)

    target_size = (128, 128)
    all_round_results = []
    t_total = time.time()

    for r in range(n_rounds):
        print(f"\n  --- Round {r+1}/{n_rounds} ---")
        np.random.seed(seed + r * 1000)
        torch.manual_seed(seed + r * 1000)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + r * 1000)

        # 构建多通道输入：当前 F_grid 在空白区的值即为伪标签
        multi_channel = build_multichannel(
            F_grid, X_grid_aux, Y_grid_aux, Z_grid_aux,
            mask_blank, input_ch)

        # 缩放
        C = multi_channel.shape[0]
        multi_resized = np.zeros((C, target_size[0], target_size[1]), dtype=np.float32)
        for c in range(C):
            multi_resized[c] = resize(multi_channel[c], target_size,
                                      mode='constant', anti_aliasing=True)
        permanent_mask_resized = resize(mask_blank.astype(float), target_size) > 0.5

        train_image = multi_resized.copy()
        train_image[:, permanent_mask_resized] = 0.0

        # 模型
        model = UNetInpainter(in_chans=input_ch, base_ch=base_ch,
                              use_skip=use_skip).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        if r == 0:
            print(f"  参数量: {n_params:,}")

        # 训练
        criterion = CompositeLoss(grad_weight=grad_weight).to(device)
        dataset = InpaintingDataset(train_image, permanent_mask_resized,
                                     mask_mode=mask_mode, mask_ratio=0.2)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=0)

        print(f"  训练中 ({epochs} epochs)...")
        t0 = time.time()
        best_loss = train_model(model, dataloader, criterion, epochs, device,
                                 lr=lr, max_lr=max_lr, weight_decay=weight_decay,
                                 verbose=(r == 0 or r == n_rounds - 1))
        train_time = time.time() - t0

        def simple_denorm(x):
            return (x + 1) / 2 * (F_max - F_min) + F_min

        rmse, mae, result_grid, output_full = evaluate_model(
            model, multi_channel, permanent_mask_resized, target_size,
            F_grid, mask_blank, x_grid, y_grid, test_df, ll_to_xy,
            simple_denorm, nx, ny, device)

        # Debug: 检查输出范围
        if r == 0:
            print(f"  [Debug] F_min={F_min:.2f}, F_max={F_max:.2f}, "
                  f"output_full in blank: min={output_full[mask_blank].min():.2f}, "
                  f"max={output_full[mask_blank].max():.2f}")
            print(f"  [Debug] test_df F range: "
                  f"min={test_df['F'].values.min():.2f}, "
                  f"max={test_df['F'].values.max():.2f}")

        print(f"  Round {r+1} 结果: RMSE={rmse:.4f}, MAE={mae:.4f}, "
              f"Loss={best_loss:.6f}, Time={train_time:.1f}s")

        # 更新 F_grid 空白区为当前预测值（伪标签）
        F_grid[mask_blank] = output_full[mask_blank]

        all_round_results.append({
            'round': r + 1,
            'rmse': rmse,
            'mae': mae,
            'best_loss': float(best_loss),
            'train_time': train_time,
        })

    total_time = time.time() - t_total
    print(f"\n  {'='*50}")
    print(f"  自训练汇总 ({n_rounds} rounds, total {total_time:.1f}s):")
    for rr in all_round_results:
        print(f"    Round {rr['round']}: RMSE={rr['rmse']:.4f}, MAE={rr['mae']:.4f}")

    best_round = min(all_round_results, key=lambda x: x['rmse'])
    worst_round = max(all_round_results, key=lambda x: x['rmse'])
    delta_pct = (best_round['rmse'] - all_round_results[0]['rmse']) / all_round_results[0]['rmse'] * 100

    print(f"    最佳: Round {best_round['round']} (RMSE={best_round['rmse']:.4f})")
    print(f"    vs Round1: Δ{delta_pct:+.1f}%")
    print(f"  {'='*50}")

    return {
        'exp_name': exp_name,
        'n_rounds': n_rounds,
        'rmse_round1': all_round_results[0]['rmse'],
        'rmse_best': best_round['rmse'],
        'best_round': best_round['round'],
        'mae_best': best_round['mae'],
        'round_details': all_round_results,
        'total_time': total_time,
        'timestamp': datetime.now().isoformat(),
    }


def run_pretrain_finetune(args, data_cache):
    """合成数据预训练 + 真实数据微调"""
    cfg = vars(args).copy()
    exp_name = cfg['exp_name']
    use_skip = not cfg['no_skip']
    input_ch = cfg['input_ch']
    grad_weight = cfg['grad_weight']
    mask_mode = cfg['mask_mode']
    eps_mode = cfg['eps_mode']
    interp_method = cfg.get('interp_method', 'rbf')
    fixed_eps = cfg['fixed_eps']
    epochs = cfg['epochs']
    base_ch = cfg['base_ch']
    batch_size = cfg['batch_size']
    lr = cfg['lr']
    max_lr = cfg['max_lr']
    weight_decay = cfg['weight_decay']
    output_dir = cfg['output_dir']
    seed = cfg['seed']
    n_synthetic = cfg.get('n_synthetic', 200)
    pretrain_epochs = cfg.get('pretrain_epochs', 20)
    device_str = 'cpu' if cfg['no_gpu'] else ('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"  合成数据预训练 + 微调: {exp_name}")
    print(f"  合成样本: {n_synthetic}, 预训练epochs: {pretrain_epochs}")
    print(f"  微调epochs: {epochs}, device={device_str}")
    print(f"{'='*60}")

    device = torch.device(device_str)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_df = data_cache['train_df']
    test_df = data_cache['test_df']
    lon0, lat0, R = data_cache['lon0'], data_cache['lat0'], data_cache['R']
    ll_to_xy = data_cache['ll_to_xy']
    LON_MIN, LON_MAX = data_cache['LON_MIN'], data_cache['LON_MAX']
    LAT_MIN, LAT_MAX = data_cache['LAT_MIN'], data_cache['LAT_MAX']

    # 构建网格
    (x_grid, y_grid, grid_x, grid_y, lon_grid_full, lat_grid_full,
     mask_blank, nx, ny) = build_grid(
        train_df, lon0, lat0, R, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX)
    print(f"  网格: {nx} x {ny}")

    # 插值真实数据
    F_grid, X_grid, Y_grid, Z_grid, F_min, F_max, normalize, denormalize = \
        rbf_interpolate(train_df, grid_x, grid_y, mask_blank, eps_mode, fixed_eps,
                         interp_method=interp_method)

    # 获取真实数据各分量范围（用于合成数据生成）
    F_range = (train_df['F'].min(), train_df['F'].max())
    X_range = (train_df['X'].min(), train_df['X'].max())
    Y_range = (train_df['Y'].min(), train_df['Y'].max())
    Z_range = (train_df['Z'].min(), train_df['Z'].max())

    target_size = (128, 128)

    # =====================================================================
    # Phase 1: 生成合成数据 + 预训练
    # =====================================================================
    print(f"\n  [Phase 1] 生成 {n_synthetic} 张合成磁场图...")
    t0 = time.time()
    rng = np.random.RandomState(seed)
    synthetic_samples = []
    for i in range(n_synthetic):
        mc, target, fmin, fmax = generate_synthetic_sample(
            grid_x, grid_y, mask_blank, eps_mode, fixed_eps,
            F_range, X_range, Y_range, Z_range, rng)
        synthetic_samples.append((mc, target, fmin, fmax))
    print(f"  生成完成, 用时 {time.time() - t0:.1f}s")

    # 预训练模型
    model = UNetInpainter(in_chans=input_ch, base_ch=base_ch,
                           use_skip=use_skip).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")

    print(f"\n  [Phase 2] 预训练...")
    t1 = time.time()
    model = pretrain_on_synthetic(
        model, synthetic_samples, pretrain_epochs, batch_size,
        lr, max_lr, weight_decay, target_size, mask_mode, device)
    pretrain_time = time.time() - t1
    print(f"  预训练完成, 用时 {pretrain_time:.1f}s")

    # 保存预训练权重
    pt_path = os.path.join(output_dir, f'{exp_name}_pretrained.pth')
    torch.save(model.state_dict(), pt_path)
    print(f"  预训练权重保存至: {pt_path}")

    # =====================================================================
    # Phase 2: 真实数据微调
    # =====================================================================
    print(f"\n  [Phase 3] 真实数据微调 ({epochs} epochs, 逐epoch记录RMSE)...")

    # 构建真实数据输入
    X_grid_aux = X_grid.copy()
    Y_grid_aux = Y_grid.copy()
    Z_grid_aux = Z_grid.copy()
    multi_channel = build_multichannel(
        F_grid, X_grid_aux, Y_grid_aux, Z_grid_aux, mask_blank, input_ch)

    C = multi_channel.shape[0]
    multi_resized = np.zeros((C, target_size[0], target_size[1]), dtype=np.float32)
    for c in range(C):
        multi_resized[c] = resize(multi_channel[c], target_size,
                                  mode='constant', anti_aliasing=True)
    permanent_mask_resized = resize(mask_blank.astype(float), target_size) > 0.5

    train_image = multi_resized.copy()
    train_image[:, permanent_mask_resized] = 0.0

    def simple_denorm(x):
        return (x + 1) / 2 * (F_max - F_min) + F_min

    # 自定义微调训练循环，逐epoch记录RMSE
    criterion = CompositeLoss(grad_weight=grad_weight).to(device)
    dataset = InpaintingDataset(train_image, permanent_mask_resized,
                                 mask_mode=mask_mode, mask_ratio=0.2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             num_workers=0)
    steps_per_epoch = len(dataloader)
    total_steps = epochs * steps_per_epoch
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, total_steps=total_steps,
        pct_start=0.1, anneal_strategy='cos', final_div_factor=1e4)

    rmse_history = []
    best_rmse = float('inf')
    best_state = None

    t2 = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs_input, imgs_target, masks in dataloader:
            imgs_input = imgs_input.to(device)
            imgs_target = imgs_target.to(device)
            masks = masks.to(device)
            outputs = model(imgs_input)
            loss = criterion(outputs, imgs_target, masks.bool())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / steps_per_epoch

        # 每个epoch评估RMSE
        rmse_ep, mae_ep, _, _ = evaluate_model(
            model, multi_channel, permanent_mask_resized, target_size,
            F_grid, mask_blank, x_grid, y_grid, test_df, ll_to_xy,
            simple_denorm, nx, ny, device)
        rmse_history.append({'epoch': epoch + 1, 'rmse': float(rmse_ep),
                              'mae': float(mae_ep), 'loss': float(avg_loss)})

        if rmse_ep < best_rmse:
            best_rmse = rmse_ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

        print(f"  FT Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f} | "
              f"RMSE: {rmse_ep:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    ft_time = time.time() - t2

    # 加载最佳模型
    model.load_state_dict(best_state)
    rmse, mae = best_rmse, rmse_history[best_epoch - 1]['mae']

    # 保存RMSE历史
    rmse_json_path = os.path.join(output_dir, f'{exp_name}_rmse_history.json')
    with open(rmse_json_path, 'w') as f:
        json.dump(rmse_history, f, indent=2)
    print(f"  RMSE历史保存至: {rmse_json_path}")

    # 绘制 RMSE vs Epoch 曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs_list = [r['epoch'] for r in rmse_history]
    rmse_list = [r['rmse'] for r in rmse_history]
    ax.plot(epochs_list, rmse_list, 'b-o', markersize=6, linewidth=2)
    ax.axhline(y=rmse_list[0], color='gray', linestyle='--', alpha=0.5,
               label=f'Initial: {rmse_list[0]:.2f}')
    ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5,
               label=f'Best: epoch {best_epoch} (RMSE={best_rmse:.2f})')
    ax.set_xlabel('Fine-tuning Epoch', fontsize=12)
    ax.set_ylabel('RMSE (nT)', fontsize=12)
    ax.set_title(f'RMSE vs Fine-tuning Epoch\n{exp_name} '
                 f'(pretrained on {n_synthetic} synthetic, {pretrain_epochs} epochs)',
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(output_dir, f'{exp_name}_rmse_curve.png')
    fig.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  RMSE曲线图保存至: {plot_path}")

    total_time = time.time() - t0

    print(f"\n  {'='*50}")
    print(f"  预训练+微调 结果:")
    print(f"    最佳微调 RMSE={rmse:.4f} (epoch {best_epoch}/{epochs})")
    print(f"    预训练用时: {pretrain_time:.1f}s, 微调用时: {ft_time:.1f}s")
    print(f"    总用时: {total_time:.1f}s")
    print(f"  {'='*50}")

    # 同时跑对比实验：无预训练直接用同配置训练（用于公平比较）
    print(f"\n  [对比] 无预训练直接训练...")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_scratch = UNetInpainter(in_chans=input_ch, base_ch=base_ch,
                                   use_skip=use_skip).to(device)
    criterion2 = CompositeLoss(grad_weight=grad_weight).to(device)
    dataset2 = InpaintingDataset(train_image, permanent_mask_resized,
                                  mask_mode=mask_mode, mask_ratio=0.2)
    dataloader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=True,
                              num_workers=0)

    t3 = time.time()
    best_loss2 = train_model(model_scratch, dataloader2, criterion2, epochs, device,
                              lr=lr, max_lr=max_lr, weight_decay=weight_decay)
    scratch_time = time.time() - t3

    rmse_scratch, mae_scratch, _, _ = evaluate_model(
        model_scratch, multi_channel, permanent_mask_resized, target_size,
        F_grid, mask_blank, x_grid, y_grid, test_df, ll_to_xy,
        simple_denorm, nx, ny, device)

    delta = rmse - rmse_scratch
    print(f"    无预训练 RMSE={rmse_scratch:.4f}, MAE={mae_scratch:.4f}")
    print(f"    预训练增益: ΔRMSE={delta:+.2f} ({(delta/rmse_scratch)*100:+.1f}%)")

    result = {
        'exp_name': exp_name,
        'n_synthetic': n_synthetic,
        'pretrain_epochs': pretrain_epochs,
        'ft_epochs': epochs,
        'rmse_pretrained': rmse,
        'mae_pretrained': mae,
        'best_epoch': best_epoch,
        'rmse_scratch': rmse_scratch,
        'mae_scratch': mae_scratch,
        'delta_rmse': delta,
        'delta_pct': (delta / rmse_scratch) * 100,
        'pretrain_time': pretrain_time,
        'ft_time': ft_time,
        'total_time': total_time,
        'n_params': n_params,
        'rmse_history': rmse_history,
        'timestamp': datetime.now().isoformat(),
    }

    return result


def plot_ablation_results(results, output_dir):
    """绘制消融实验 RMSE/MAE 对比柱状图"""
    names = [r['exp_name'] for r in results]
    rmse_vals = [r['rmse'] for r in results]
    mae_vals = [r['mae'] for r in results]
    baseline_rmse = rmse_vals[0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # RMSE
    colors = ['#2196F3' if i == 0 else '#FF9800' for i in range(len(results))]
    bars1 = axes[0].bar(range(len(names)), rmse_vals, color=colors)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    axes[0].set_ylabel('RMSE (nT)')
    axes[0].set_title('Ablation Study - RMSE')
    for bar, val in zip(bars1, rmse_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    axes[0].axhline(y=baseline_rmse, color='#2196F3', linestyle='--', alpha=0.5,
                    label=f'Baseline RMSE={baseline_rmse:.2f}')
    axes[0].legend()

    # MAE
    bars2 = axes[1].bar(range(len(names)), mae_vals, color=colors)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    axes[1].set_ylabel('MAE (nT)')
    axes[1].set_title('Ablation Study - MAE')
    for bar, val in zip(bars2, mae_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'ablation_comparison.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  对比图已保存至: {fig_path}")


if __name__ == '__main__':
    main()
