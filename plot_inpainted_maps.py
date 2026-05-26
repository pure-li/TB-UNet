#!/usr/bin/env python
"""对比 RBF / TPS 插值基准图 + U-Net 填补后的地磁图"""
import sys
import importlib.util
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
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import warnings
warnings.filterwarnings('ignore')

# 从 U-net.py 导入关键类和函数（文件名有连字符，需用 importlib）
unet_path = r'F:\PINN实验\venv\U-net\U-net.py'
spec = importlib.util.spec_from_file_location("unet_module", unet_path)
unet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unet)

UNetInpainter = unet.UNetInpainter
CompositeLoss = unet.CompositeLoss
InpaintingDataset = unet.InpaintingDataset
rbf_interpolate = unet.rbf_interpolate
build_multichannel = unet.build_multichannel
evaluate_model = unet.evaluate_model
train_model = unet.train_model

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
print(f"训练点: {len(train_df)}, 测试点: {len(test_df)}")

# --- 2. 构建网格（限制在所有测线的纬度公共重叠区）---
line_ymin = train_df.groupby('line_id')['y'].min()
line_ymax = train_df.groupby('line_id')['y'].max()
y_min_raw = line_ymin.max()
y_max_raw = line_ymax.min()
x_min_raw = train_df['x'].min()
x_max_raw = train_df['x'].max()
print(f"全区域 x: [{x_min_raw:.2f}, {x_max_raw:.2f}], 公共重叠 y: [{y_min_raw:.2f}, {y_max_raw:.2f}]")

padding = 0.05
x_min, x_max = x_min_raw - padding, x_max_raw + padding
y_min, y_max = y_min_raw - padding, y_max_raw + padding
x_grid = np.arange(x_min, x_max, 0.02)
y_grid = np.arange(y_min, y_max, 0.02)
nx, ny = len(x_grid), len(y_grid)
grid_x, grid_y = np.meshgrid(x_grid, y_grid, indexing='ij')
lon_grid = lon0 + grid_x / (R_map * np.cos(np.radians(lat0))) * (180 / np.pi)
lat_grid = lat0 + grid_y / R_map * (180 / np.pi)
mask_blank = ((lon_grid >= LON_MIN) & (lon_grid <= LON_MAX) &
              (lat_grid >= LAT_MIN) & (lat_grid <= LAT_MAX))

print(f"网格: {nx} x {ny}, 空白点数: {mask_blank.sum()}")

# --- 3. 训练/评估 U-Net 对 RBF 和 TPS 输入 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

seed = 42
target_size = (128, 128)
input_ch = 4
base_ch = 48
grad_weight = 0.0
epochs = 5
batch_size = 4
lr = 1e-3
max_lr = 1e-3
weight_decay = 1e-6

results = {}
for interp_method in ['rbf', 'tps']:
    label = 'RBF (cubic)' if interp_method == 'rbf' else 'TPS (min-curvature)'
    print(f"\n{'='*50}\n  Training U-Net with {label} input\n{'='*50}")

    # 插值（使用 U-net.py 的标准流程，不额外平滑 → 保持训练精度）
    print(f"  插值中...")
    F_grid, X_grid, Y_grid, Z_grid, F_min, F_max, normalize, denormalize = \
        rbf_interpolate(train_df, grid_x, grid_y, mask_blank,
                        eps_mode='adaptive', interp_method=interp_method)

    # 单独计算平滑后的全网格 → 仅用于可视化（不影响训练）
    from scipy.interpolate import RBFInterpolator as RBFI
    points_xy = train_df[['x', 'y']].values
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    if interp_method == 'rbf':
        tree = KDTree(points_xy)
        distances, _ = tree.query(points_xy, k=min(10, len(points_xy)))
        eps = np.median(distances[:, 1:]) * 0.8
        rbf_F = RBFI(points_xy, train_df['F'].values, kernel='cubic', epsilon=eps)
        F_full = rbf_F(grid_points).reshape(grid_x.shape)
    else:
        rbf_F = RBFI(points_xy, train_df['F'].values, kernel='thin_plate_spline')
        F_full = rbf_F(grid_points).reshape(grid_x.shape)
    # 各向异性平滑（仅用于显示）
    sigma_strike = 8.0
    sigma_dip = 1.0
    F_full = gaussian_filter1d(F_full, sigma_strike, axis=0)
    F_full = gaussian_filter1d(F_full, sigma_dip, axis=1)
    print(f"  平滑显示网格已准备 (sigma_strike={sigma_strike}, sigma_dip={sigma_dip})")

    # 多通道
    multi_channel = build_multichannel(F_grid, X_grid, Y_grid, Z_grid,
                                       mask_blank, input_ch)
    C = multi_channel.shape[0]

    # 缩放
    multi_resized = np.zeros((C, target_size[0], target_size[1]), dtype=np.float32)
    for c in range(C):
        multi_resized[c] = resize(multi_channel[c], target_size,
                                  mode='constant', anti_aliasing=True)
    permanent_mask_resized = resize(mask_blank.astype(float), target_size) > 0.5

    train_image = multi_resized.copy()
    train_image[:, permanent_mask_resized] = 0.0

    # 模型
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = UNetInpainter(in_chans=input_ch, base_ch=base_ch, use_skip=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")

    # 训练
    criterion = CompositeLoss(grad_weight=grad_weight).to(device)
    dataset = InpaintingDataset(train_image, permanent_mask_resized,
                                mask_mode='mixed', mask_ratio=0.2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    _ = train_model(model, dataloader, criterion, epochs, device,
                    lr=lr, max_lr=max_lr, weight_decay=weight_decay)

    # 评估
    def simple_denorm(x):
        return (x + 1) / 2 * (F_max - F_min) + F_min

    rmse, mae, result_grid, output_full = evaluate_model(
        model, multi_channel, permanent_mask_resized, target_size,
        F_grid, mask_blank, x_grid, y_grid, test_df, ll_to_xy,
        simple_denorm, nx, ny, device)

    print(f"  RMSE={rmse:.2f}, MAE={mae:.2f}")

    # U-Net 输出空白区平滑，消除块状/条带
    blank_only = output_full.copy()
    blank_only[~mask_blank] = np.nan
    blank_smoothed = gaussian_filter1d(blank_only, 1.5, axis=0)
    blank_smoothed = gaussian_filter1d(blank_smoothed, 1.5, axis=1)
    output_full[mask_blank] = blank_smoothed[mask_blank]

    # 把结果图的已知区域替换为平滑值，保证与 interp_comparison.png 颜色一致
    result_grid[~mask_blank] = F_full[~mask_blank]

    results[interp_method] = {
        'F_grid': F_full.copy(),        # 平滑后的全网格（用于显示基准图）
        'result_grid': result_grid.copy(),
        'output_full': output_full.copy(),
        'rmse': rmse,
        'mae': mae,
    }

    del model
    torch.cuda.empty_cache()

# --- 4. 绘图 ---
F_rbf = results['rbf']['F_grid']
F_tps = results['tps']['F_grid']
F_rbf_filled = results['rbf']['result_grid']
F_tps_filled = results['tps']['result_grid']

# 基准图显示用：空白区设为 NaN
F_rbf_disp = F_rbf.copy(); F_rbf_disp[mask_blank] = np.nan
F_tps_disp = F_tps.copy(); F_tps_disp[mask_blank] = np.nan

# 直接用 F 值绘图，vmin/vmax 只从平滑显示网格已知区域算（与 interp_comparison 一致）
vmin = min(np.nanmin(F_rbf_disp[~mask_blank]), np.nanmin(F_tps_disp[~mask_blank]))
vmax = max(np.nanmax(F_rbf_disp[~mask_blank]), np.nanmax(F_tps_disp[~mask_blank]))

xi_train = train_df['x'].values
yi_train = train_df['y'].values
xi_test = test_df['x'].values
yi_test = test_df['y'].values

bx = ll_to_xy(np.array([LON_MIN, LON_MAX, LON_MAX, LON_MIN, LON_MIN]),
              np.array([LAT_MIN, LAT_MIN, LAT_MAX, LAT_MAX, LAT_MIN]))[0]
by = ll_to_xy(np.array([LON_MIN, LON_MAX, LON_MAX, LON_MIN, LON_MIN]),
              np.array([LAT_MIN, LAT_MIN, LAT_MAX, LAT_MAX, LAT_MIN]))[1]

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# (a) RBF 插值基准图
im0 = axes[0, 0].pcolormesh(grid_x, grid_y, F_rbf_disp, cmap='jet', shading='auto',
                             vmin=vmin, vmax=vmax)
axes[0, 0].plot(bx, by, 'k-', linewidth=2.5, label='Blank region')
axes[0, 0].scatter(xi_train, yi_train, c='black', s=2, alpha=0.4)
axes[0, 0].set_title('RBF Cubic Interpolation\n(Smoothed for display, blank=NaN)', fontsize=12)
axes[0, 0].set_xlabel('X (km)'); axes[0, 0].set_ylabel('Y (km)')
plt.colorbar(im0, ax=axes[0, 0], label='F (nT)')

# (b) TPS 插值基准图
im1 = axes[0, 1].pcolormesh(grid_x, grid_y, F_tps_disp, cmap='jet', shading='auto',
                             vmin=vmin, vmax=vmax)
axes[0, 1].plot(bx, by, 'k-', linewidth=2.5, label='Blank region')
axes[0, 1].scatter(xi_train, yi_train, c='black', s=2, alpha=0.4)
axes[0, 1].set_title('TPS / Minimum Curvature\n(Smoothed for display, blank=NaN)', fontsize=12)
axes[0, 1].set_xlabel('X (km)'); axes[0, 1].set_ylabel('Y (km)')
plt.colorbar(im1, ax=axes[0, 1], label='F (nT)')

# (c) U-Net + RBF 填补后
im2 = axes[1, 0].pcolormesh(grid_x, grid_y, F_rbf_filled, cmap='jet', shading='auto',
                             vmin=vmin, vmax=vmax)
axes[1, 0].plot(bx, by, 'k-', linewidth=2.5)
axes[1, 0].scatter(xi_train, yi_train, c='black', s=2, alpha=0.4)
axes[1, 0].scatter(xi_test, yi_test, c='lime', s=18, marker='s', edgecolors='black',
                   linewidth=0.5, label=f'Test pts ({len(test_df)})')
axes[1, 0].set_title(f'U-Net + RBF Inpainted\n(RMSE={results["rbf"]["rmse"]:.2f}, MAE={results["rbf"]["mae"]:.2f})', fontsize=12)
axes[1, 0].set_xlabel('X (km)'); axes[1, 0].set_ylabel('Y (km)')
axes[1, 0].legend(fontsize=7, loc='lower right')
plt.colorbar(im2, ax=axes[1, 0], label='F (nT)')

# (d) U-Net + TPS 填补后
im3 = axes[1, 1].pcolormesh(grid_x, grid_y, F_tps_filled, cmap='jet', shading='auto',
                             vmin=vmin, vmax=vmax)
axes[1, 1].plot(bx, by, 'k-', linewidth=2.5)
axes[1, 1].scatter(xi_train, yi_train, c='black', s=2, alpha=0.4)
axes[1, 1].scatter(xi_test, yi_test, c='lime', s=18, marker='s', edgecolors='black',
                   linewidth=0.5, label=f'Test pts ({len(test_df)})')
axes[1, 1].set_title(f'U-Net + TPS Inpainted\n(RMSE={results["tps"]["rmse"]:.2f}, MAE={results["tps"]["mae"]:.2f})', fontsize=12)
axes[1, 1].set_xlabel('X (km)'); axes[1, 1].set_ylabel('Y (km)')
axes[1, 1].legend(fontsize=7, loc='lower right')
plt.colorbar(im3, ax=axes[1, 1], label='F (nT)')

plt.suptitle('Geomagnetic Map: Interpolation Input vs U-Net Inpainted Output',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
out_path = r'F:\PINN实验\venv\U-net\ablation_results\inpainted_comparison.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"\nPlot saved to: {out_path}")

# 空白区特写图
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))
zoom_x = (bx.min() - 0.02, bx.max() + 0.02)
zoom_y = (by.min() - 0.02, by.max() + 0.02)

im4 = axes2[0].pcolormesh(grid_x, grid_y, F_rbf_filled, cmap='jet', shading='auto',
                           vmin=vmin, vmax=vmax)
axes2[0].plot(bx, by, 'k-', linewidth=2.5)
axes2[0].scatter(xi_test, yi_test, c='lime', s=30, marker='s', edgecolors='black',
                 linewidth=0.8, zorder=5)
axes2[0].set_xlim(zoom_x); axes2[0].set_ylim(zoom_y)
axes2[0].set_title(f'U-Net + RBF (Zoom)\nRMSE={results["rbf"]["rmse"]:.2f}, MAE={results["rbf"]["mae"]:.2f}', fontsize=11)
axes2[0].set_xlabel('X (km)'); axes2[0].set_ylabel('Y (km)')
plt.colorbar(im4, ax=axes2[0], label='F (nT)')

im5 = axes2[1].pcolormesh(grid_x, grid_y, F_tps_filled, cmap='jet', shading='auto',
                           vmin=vmin, vmax=vmax)
axes2[1].plot(bx, by, 'k-', linewidth=2.5)
axes2[1].scatter(xi_test, yi_test, c='lime', s=30, marker='s', edgecolors='black',
                 linewidth=0.8, zorder=5)
axes2[1].set_xlim(zoom_x); axes2[1].set_ylim(zoom_y)
axes2[1].set_title(f'U-Net + TPS (Zoom)\nRMSE={results["tps"]["rmse"]:.2f}, MAE={results["tps"]["mae"]:.2f}', fontsize=11)
axes2[1].set_xlabel('X (km)'); axes2[1].set_ylabel('Y (km)')
plt.colorbar(im5, ax=axes2[1], label='F (nT)')

plt.suptitle('Inpainted Blank Region Detail (Test points in green)', fontsize=13, fontweight='bold')
plt.tight_layout()
out_path2 = r'F:\PINN实验\venv\U-net\ablation_results\inpainted_zoom_comparison.png'
plt.savefig(out_path2, dpi=200, bbox_inches='tight')
plt.close()
print(f"Zoom plot saved to: {out_path2}")
