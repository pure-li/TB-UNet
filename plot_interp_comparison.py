#!/usr/bin/env python
"""对比 RBF cubic / TPS(最小曲率) 插值结果 + U-Net 填补后的地磁图"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from scipy.spatial import KDTree
from scipy.ndimage import zoom, gaussian_filter1d
from skimage.transform import resize
import torch
import json
import os
import warnings
warnings.filterwarnings('ignore')

# --- 1. 加载数据 ---
threshold = 0.0005
df = pd.read_excel(r'F:\PINN实验\venv1\PINN数据3.xlsx', sheet_name='Sheet1')
if df.shape[1] > 7:
    df = df.iloc[:, :7]
df.columns = ['lon', 'lat', 'h', 'X', 'Y', 'Z', 'F']
df['X'] = df['X'].abs()
df['Z'] = df['Z'].abs()
df['diff_lon'] = df['lon'].diff().abs()
df['line_id'] = (df['diff_lon'] > threshold).cumsum() + 1
df_sampled = df.groupby('line_id', group_keys=False).apply(lambda g: g.iloc[::50])
df = df_sampled.reset_index(drop=True)
df['diff_lon'] = df['lon'].diff().abs()
df['line_id'] = (df['diff_lon'] > threshold).cumsum() + 1

lon0, lat0, R_map = df['lon'].mean(), df['lat'].mean(), 6371

def ll_to_xy(lon, lat):
    x = (lon - lon0) * np.pi / 180 * R_map * np.cos(np.radians(lat0))
    y = (lat - lat0) * np.pi / 180 * R_map
    return x, y

df['x'], df['y'] = ll_to_xy(df['lon'].values, df['lat'].values)

LON_MIN, LON_MAX = 113.0085, 113.0155
LAT_MIN, LAT_MAX = 34.5480, 34.5604
mask_inside = ((df['lon'] >= LON_MIN) & (df['lon'] <= LON_MAX) &
               (df['lat'] >= LAT_MIN) & (df['lat'] <= LAT_MAX))
train_df = df[~mask_inside].copy()
test_df = df[mask_inside].copy()

# --- 2. 构建网格（限制在所有测线的纬度公共重叠区）---
line_ymin = train_df.groupby('line_id')['y'].min()
line_ymax = train_df.groupby('line_id')['y'].max()

y_min_raw = line_ymin.max()   # 所有测线南端中最北的
y_max_raw = line_ymax.min()   # 所有测线北端中最南的
x_min_raw = train_df['x'].min()
x_max_raw = train_df['x'].max()
print(f"全区域 x: [{x_min_raw:.2f}, {x_max_raw:.2f}], 公共重叠 y: [{y_min_raw:.2f}, {y_max_raw:.2f}]")

padding = 0.05
x_min, x_max = x_min_raw - padding, x_max_raw + padding
y_min, y_max = y_min_raw - padding, y_max_raw + padding
x_grid = np.arange(x_min, x_max, 0.02)
y_grid = np.arange(y_min, y_max, 0.02)
nx, ny = len(x_grid), len(y_grid)
print(f"网格: {nx} x {ny}")

grid_x, grid_y = np.meshgrid(x_grid, y_grid, indexing='ij')
lon_grid = lon0 + grid_x / (R_map * np.cos(np.radians(lat0))) * (180 / np.pi)
lat_grid = lat0 + grid_y / R_map * (180 / np.pi)
mask_blank = ((lon_grid >= LON_MIN) & (lon_grid <= LON_MAX) &
              (lat_grid >= LAT_MIN) & (lat_grid <= LAT_MAX))

points_xy = train_df[['x', 'y']].values
values_F = train_df['F'].values
grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

# --- 3. RBF cubic 插值 ---
tree = KDTree(points_xy)
distances, _ = tree.query(points_xy, k=min(10, len(points_xy)))
eps_rbf = np.median(distances[:, 1:]) * 0.8
print(f"RBF epsilon: {eps_rbf:.4f}")
rbf_F = RBFInterpolator(points_xy, values_F, kernel='cubic', epsilon=eps_rbf)
F_grid_rbf = rbf_F(grid_points).reshape(grid_x.shape)
F_grid_rbf_display = F_grid_rbf.copy()
F_grid_rbf[mask_blank] = np.nan

# --- 4. TPS (最小曲率) 插值 ---
tps_F = RBFInterpolator(points_xy, values_F, kernel='thin_plate_spline')
F_grid_tps = tps_F(grid_points).reshape(grid_x.shape)
F_grid_tps_display = F_grid_tps.copy()
F_grid_tps[mask_blank] = np.nan

# --- 各向异性平滑（消除测线间条带）---
sigma_strike = 8.0   # 垂直测线方向（X轴），加强平滑
sigma_dip = 1.0      # 沿测线方向（Y轴）
F_grid_rbf_display = gaussian_filter1d(F_grid_rbf_display, sigma_strike, axis=0)
F_grid_rbf_display = gaussian_filter1d(F_grid_rbf_display, sigma_dip, axis=1)
F_grid_tps_display = gaussian_filter1d(F_grid_tps_display, sigma_strike, axis=0)
F_grid_tps_display = gaussian_filter1d(F_grid_tps_display, sigma_dip, axis=1)
print(f"各向异性平滑完成 (sigma_strike={sigma_strike}, sigma_dip={sigma_dip})")

print(f"RBF range: {F_grid_rbf[~mask_blank].min():.1f} ~ {F_grid_rbf[~mask_blank].max():.1f}")
print(f"TPS range: {F_grid_tps[~mask_blank].min():.1f} ~ {F_grid_tps[~mask_blank].max():.1f}")
print(f"Diff in known region: max={np.abs(F_grid_rbf[~mask_blank] - F_grid_tps[~mask_blank]).max():.2f}")

# --- 5. 绘图 ---
fig, axes = plt.subplots(2, 3, figsize=(22, 13))

xi_train = train_df['x'].values
yi_train = train_df['y'].values
xi_test = test_df['x'].values
yi_test = test_df['y'].values

# 空白区边框
bx = ll_to_xy(np.array([LON_MIN, LON_MAX, LON_MAX, LON_MIN, LON_MIN]),
              np.array([LAT_MIN, LAT_MIN, LAT_MAX, LAT_MAX, LAT_MIN]))[0]
by = ll_to_xy(np.array([LON_MIN, LON_MAX, LON_MAX, LON_MIN, LON_MIN]),
              np.array([LAT_MIN, LAT_MIN, LAT_MAX, LAT_MAX, LAT_MIN]))[1]

# 直接用 F 值，统一配色范围
vmin = min(F_grid_rbf_display[~mask_blank].min(), F_grid_tps_display[~mask_blank].min())
vmax = max(F_grid_rbf_display[~mask_blank].max(), F_grid_tps_display[~mask_blank].max())

# (a) RBF 插值结果
im0 = axes[0, 0].pcolormesh(grid_x, grid_y, F_grid_rbf_display, cmap='jet', shading='auto',
                             vmin=vmin, vmax=vmax)
axes[0, 0].plot(bx, by, 'k-', linewidth=2, label='Blank region')
axes[0, 0].scatter(xi_train, yi_train, c='black', s=2, alpha=0.5)
axes[0, 0].set_title(f'RBF Cubic Interpolation\n(ε={eps_rbf:.4f})', fontsize=11)
axes[0, 0].set_xlabel('X (km)'); axes[0, 0].set_ylabel('Y (km)')
plt.colorbar(im0, ax=axes[0, 0], label='F (nT)')

# (b) TPS (最小曲率) 插值结果
im1 = axes[0, 1].pcolormesh(grid_x, grid_y, F_grid_tps_display, cmap='jet', shading='auto',
                             vmin=vmin, vmax=vmax)
axes[0, 1].plot(bx, by, 'k-', linewidth=2, label='Blank region')
axes[0, 1].scatter(xi_train, yi_train, c='black', s=2, alpha=0.5)
axes[0, 1].set_title(f'TPS / Minimum Curvature\n($\\nabla^4 F=0$)', fontsize=11)
axes[0, 1].set_xlabel('X (km)'); axes[0, 1].set_ylabel('Y (km)')
plt.colorbar(im1, ax=axes[0, 1], label='F (nT)')

# (c) RBF - TPS 差异
diff = F_grid_rbf_display - F_grid_tps_display
vmax_diff = np.abs(diff).max()
im2 = axes[0, 2].pcolormesh(grid_x, grid_y, diff, cmap='jet', shading='auto',
                             vmin=-vmax_diff, vmax=vmax_diff)
axes[0, 2].plot(bx, by, 'k-', linewidth=2)
axes[0, 2].scatter(xi_train, yi_train, c='black', s=2, alpha=0.5)
axes[0, 2].set_title(f'RBF − TPS Difference\n(Max |Δ|={vmax_diff:.1f} nT)', fontsize=11)
axes[0, 2].set_xlabel('X (km)'); axes[0, 2].set_ylabel('Y (km)')
plt.colorbar(im2, ax=axes[0, 2], label='ΔF (nT)')

# (d) 放大空白区 - RBF
zoom_x = (bx.min() - 0.05, bx.max() + 0.05)
zoom_y = (by.min() - 0.05, by.max() + 0.05)
im3 = axes[1, 0].pcolormesh(grid_x, grid_y, F_grid_rbf_display, cmap='jet', shading='auto',
                             vmin=vmin, vmax=vmax)
axes[1, 0].plot(bx, by, 'k-', linewidth=2.5)
axes[1, 0].scatter(xi_train, yi_train, c='black', s=8, alpha=0.7)
axes[1, 0].scatter(xi_test, yi_test, c='lime', s=25, marker='s', edgecolors='black',
                   linewidth=0.5, label=f'Test pts ({len(test_df)})')
axes[1, 0].set_xlim(zoom_x); axes[1, 0].set_ylim(zoom_y)
axes[1, 0].set_title('RBF - Zoom to Blank Region', fontsize=11)
axes[1, 0].legend(fontsize=8, loc='lower right')
plt.colorbar(im3, ax=axes[1, 0], label='F (nT)')

# (e) 放大空白区 - TPS
im4 = axes[1, 1].pcolormesh(grid_x, grid_y, F_grid_tps_display, cmap='jet', shading='auto',
                             vmin=vmin, vmax=vmax)
axes[1, 1].plot(bx, by, 'k-', linewidth=2.5)
axes[1, 1].scatter(xi_train, yi_train, c='black', s=8, alpha=0.7)
axes[1, 1].scatter(xi_test, yi_test, c='lime', s=25, marker='s', edgecolors='black',
                   linewidth=0.5, label=f'Test pts ({len(test_df)})')
axes[1, 1].set_xlim(zoom_x); axes[1, 1].set_ylim(zoom_y)
axes[1, 1].set_title('TPS - Zoom to Blank Region', fontsize=11)
axes[1, 1].legend(fontsize=8, loc='lower right')
plt.colorbar(im4, ax=axes[1, 1], label='F (nT)')

# (f) 空白区差异放大
im5 = axes[1, 2].pcolormesh(grid_x, grid_y, diff, cmap='jet', shading='auto',
                             vmin=-vmax_diff, vmax=vmax_diff)
axes[1, 2].plot(bx, by, 'k-', linewidth=2.5)
axes[1, 2].scatter(xi_test, yi_test, c='lime', s=25, marker='s', edgecolors='black', linewidth=0.5)
axes[1, 2].set_xlim(zoom_x); axes[1, 2].set_ylim(zoom_y)
axes[1, 2].set_title(f'RBF−TPS Difference (Zoom)\n(Blank: {diff[mask_blank].mean():.1f}±{diff[mask_blank].std():.1f} nT)', fontsize=11)
plt.colorbar(im5, ax=axes[1, 2], label='ΔF (nT)')

plt.suptitle('Interpolation Method Comparison: RBF Cubic vs TPS (Minimum Curvature)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
out_path = r'F:\PINN实验\venv\U-net\ablation_results\interp_comparison.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Plot saved to: {out_path}")
