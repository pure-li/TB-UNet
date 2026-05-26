#!/usr/bin/env python
"""Ordinary Kriging 插值 — 双区域基线
========================================
矩形 + 不规则空白区, spherical variogram, 2000 控制点
只预测空白区内格点 (避免内存溢出), 与 RBF / U-Net 对比
"""

import os, time, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pykrige.ok import OrdinaryKriging

# =============================================================================
# 0. 配置
# =============================================================================
DATA_PATH = r'F:\PINN实验\venv\U-net\afghanistan_full\Afghan_mag06A.csv'
OUT_DIR   = r'F:\PINN实验\venv\U-net\Kriging'
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42; np.random.seed(SEED)
N_KRIGING_CTRL = 2000

RECT_LON_MIN, RECT_LON_MAX = 62.0, 63.0
RECT_LAT_MIN, RECT_LAT_MAX = 32.5, 33.5

IRREGULAR_POLYGON = np.array([
    [62.05, 32.65], [62.30, 32.38], [62.70, 32.42], [62.95, 32.75],
    [63.08, 33.05], [62.88, 33.38], [62.48, 33.48], [62.15, 33.22], [61.98, 32.92],
])

print("=" * 60)
print("  Ordinary Kriging (spherical) — 双区域基线")
print(f"  控制点: {N_KRIGING_CTRL}")
print("=" * 60)

# =============================================================================
# 1. 数据加载
# =============================================================================

def load_region_data(poly_vertices=None, rect_bounds=None):
    df_full = pd.read_csv(DATA_PATH)
    df = df_full.iloc[::3].copy().reset_index(drop=True)

    if rect_bounds:
        lon_min, lon_max, lat_min, lat_max = rect_bounds
        mask_inside = ((df['Longitude'] >= lon_min) & (df['Longitude'] <= lon_max) &
                       (df['Latitude'] >= lat_min) & (df['Latitude'] <= lat_max))
    else:
        mask_inside = Path(poly_vertices).contains_points(df[['Longitude', 'Latitude']].values)

    train_df = df[~mask_inside].copy().reset_index(drop=True)
    test_df = df[mask_inside].copy().reset_index(drop=True)

    lon0, lat0 = train_df['Longitude'].mean(), train_df['Latitude'].mean()
    R_map = 6371

    def ll_to_xy(lon, lat):
        x = (lon - lon0) * np.pi / 180 * R_map * np.cos(np.radians(lat0))
        y = (lat - lat0) * np.pi / 180 * R_map
        return x, y

    train_df['x'], train_df['y'] = ll_to_xy(train_df['Longitude'].values, train_df['Latitude'].values)
    test_df['x'], test_df['y'] = ll_to_xy(test_df['Longitude'].values, test_df['Latitude'].values)

    grid_spacing = 1.0
    left_x = train_df.groupby((train_df['Longitude'].diff().abs() > 0.5).cumsum())['x'].min().min()
    right_x = train_df.groupby((train_df['Longitude'].diff().abs() > 0.5).cumsum())['x'].max().max()
    x_min, x_max = left_x - 2., right_x + 2.
    y_min, y_max = train_df['y'].min() - grid_spacing, train_df['y'].max() + grid_spacing
    x_grid = np.arange(x_min, x_max, grid_spacing)
    y_grid = np.arange(y_min, y_max, grid_spacing)
    nx, ny = len(x_grid), len(y_grid)
    grid_x, grid_y = np.meshgrid(x_grid, y_grid, indexing='ij')
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    lon_grid = lon0 + grid_x / (R_map * np.cos(np.radians(lat0))) * (180 / np.pi)
    lat_grid = lat0 + grid_y / R_map * (180 / np.pi)

    if rect_bounds:
        mask_blank = ((lon_grid >= lon_min) & (lon_grid <= lon_max) &
                      (lat_grid >= lat_min) & (lat_grid <= lat_max))
    else:
        mask_blank = Path(poly_vertices).contains_points(
            np.column_stack([lon_grid.ravel(), lat_grid.ravel()])).reshape(grid_x.shape)

    hull = ConvexHull(train_df[['x', 'y']].values)
    hull_xy = hull.points[hull.vertices]
    xc = hull_xy[:, 0].mean()
    for i in range(len(hull_xy)):
        hull_xy[i, 0] += -2. if hull_xy[i, 0] < xc else 2.
    mask_outside = ~Path(hull_xy).contains_points(grid_pts).reshape(grid_x.shape)

    if rect_bounds:
        bx = ll_to_xy(np.array([lon_min, lon_max, lon_max, lon_min, lon_min]),
                      np.array([lat_min, lat_min, lat_max, lat_max, lat_min]))[0]
        by = ll_to_xy(np.array([lon_min, lon_max, lon_max, lon_min, lon_min]),
                      np.array([lat_min, lat_min, lat_max, lat_max, lat_min]))[1]
    else:
        border = np.vstack([poly_vertices, poly_vertices[0]])
        bx, by = ll_to_xy(border[:, 0], border[:, 1])

    return {
        'train_df': train_df, 'test_df': test_df,
        'x_grid': x_grid, 'y_grid': y_grid, 'nx': nx, 'ny': ny,
        'grid_x': grid_x, 'grid_y': grid_y, 'grid_pts': grid_pts,
        'mask_blank': mask_blank, 'mask_outside': mask_outside,
        'bx': bx, 'by': by,
    }

# =============================================================================
# 2. 运行 Kriging (spherical)
# =============================================================================

results = {}
run_start = time.time()

regions = {
    'rect': ('矩形', None, (RECT_LON_MIN, RECT_LON_MAX, RECT_LAT_MIN, RECT_LAT_MAX)),
    'irreg': ('不规则', IRREGULAR_POLYGON, None),
}

for region_key, (rlabel, poly, rect) in regions.items():
    print(f"\n{'='*60}")
    print(f"  区域: {rlabel}空白区")
    print(f"{'='*60}")

    data = load_region_data(poly_vertices=poly, rect_bounds=rect)
    train_df, test_df = data['train_df'], data['test_df']
    mask_blank, mask_outside = data['mask_blank'], data['mask_outside']
    grid_pts = data['grid_pts']
    x_grid, y_grid = data['x_grid'], data['y_grid']

    # 抽取控制点
    n_ctrl = min(N_KRIGING_CTRL, len(train_df))
    idx = np.random.choice(len(train_df), n_ctrl, replace=False)
    ctrl_x = train_df['x'].values[idx]
    ctrl_y = train_df['y'].values[idx]
    ctrl_F = train_df['FinalMag'].values[idx]

    test_xy = np.column_stack([test_df['x'].values, test_df['y'].values])
    test_F = test_df['FinalMag'].values

    print(f"  控制点: {n_ctrl}, 测试点: {len(test_F)}")

    t0 = time.time()

    OK = OrdinaryKriging(ctrl_x, ctrl_y, ctrl_F,
                         variogram_model='spherical',
                         verbose=False, enable_plotting=False)

    # 只预测空白区内的格点 (避免全网格内存溢出)
    blank_idx = np.where(mask_blank.ravel())[0]
    blank_pts = grid_pts[blank_idx]
    print(f"  预测空白区格点: {len(blank_pts):,}")

    z_blank, ss_blank = OK.execute('points', blank_pts[:, 0], blank_pts[:, 1])
    print(f"  Kriging done | {time.time()-t0:.0f}s")

    # 构建完整网格结果: 空白区=Kriging预测, 外部=NaN
    F_grid = np.full(len(grid_pts), np.nan, dtype=np.float32)
    F_grid[blank_idx] = z_blank.astype(np.float32)
    F_grid = F_grid.reshape(data['grid_x'].shape)

    # 平滑
    F_smooth = F_grid.copy()
    F_smooth = gaussian_filter1d(F_smooth, 1.5, axis=0)
    F_smooth = gaussian_filter1d(F_smooth, 1.5, axis=1)

    # 评估: 对测试点做 Kriging 预测
    z_test, ss_test = OK.execute('points', test_xy[:, 0], test_xy[:, 1])
    z_test = z_test.astype(np.float32)
    valid = ~np.isnan(z_test)
    rmse = np.sqrt(mean_squared_error(test_F[valid], z_test[valid])) if np.sum(valid) > 10 else float('nan')
    mae  = mean_absolute_error(test_F[valid], z_test[valid]) if np.sum(valid) > 10 else float('nan')

    print(f"  RMSE={rmse:.2f}, MAE={mae:.2f} | 总耗时: {time.time()-t0:.0f}s")

    results[region_key] = {
        'rmse': float(rmse), 'mae': float(mae),
        'F_grid': F_grid, 'F_smooth': F_smooth,
        'z_test': z_test, 'test_F': test_F,
        'time': time.time()-t0,
    }

# =============================================================================
# 3. 保存 + 对比
# =============================================================================
print(f"\n{'='*60}")
print(f"  结果汇总")
print(f"{'='*60}")

summary = {}
for rk in results:
    summary[rk] = {'rmse': results[rk]['rmse'], 'mae': results[rk]['mae'],
                   'time': results[rk]['time']}
with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# 对比表
print(f"\n  {'方法':<25s} {'矩形 RMSE':>10s} {'不规则 RMSE':>10s}")
print(f"  {'-'*48}")
print(f"  {'Kriging (Ordinary)':<25s} {summary['rect']['rmse']:10.2f} {summary['irreg']['rmse']:10.2f}")
print(f"  {'RBF Cubic':<25s} {'25.00':>10s} {'20.66':>10s}")
print(f"  {'U-Net + Transformer':<25s} {'20.01':>10s} {'18.43':>10s}")

total_time = time.time() - run_start
print(f"\n  总耗时: {total_time/60:.0f} min")
print(f"  全部输出: {OUT_DIR}/")

# =============================================================================
# 4. 绘图
# =============================================================================
print("\n[绘图] ...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save_fig(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(OUT_DIR, f'{name}.{ext}'), dpi=300,
                    bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

for region_key, (rlabel, poly, rect) in regions.items():
    data = load_region_data(poly_vertices=poly, rect_bounds=rect)
    F_smooth = results[region_key]['F_smooth']
    mask_blank, mask_outside = data['mask_blank'], data['mask_outside']
    bx, by = data['bx'], data['by']

    # 将 Kriging 结果拼接到带 NaN 的背景上显示
    F_display = F_smooth.copy()
    F_display[mask_outside] = np.nan

    vmin = np.nanmin(F_display[~mask_outside]) if np.any(~mask_outside & ~np.isnan(F_display)) else -140
    vmax = np.nanmax(F_display[~mask_outside]) if np.any(~mask_outside & ~np.isnan(F_display)) else -70
    if np.isnan(vmin): vmin, vmax = -140, -70

    zx = (bx.min() - 2, bx.max() + 2)
    zy = (by.min() - 2, by.max() + 2)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(data['grid_x'], data['grid_y'], F_display,
                       cmap='jet', shading='auto', vmin=vmin, vmax=vmax)
    ax.plot(bx, by, 'k-', linewidth=2)
    ax.set_xlim(zx); ax.set_ylim(zy)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
    cbar = plt.colorbar(im, ax=ax, label='Mag. Anomaly (nT)')
    cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
    save_fig(fig, f'fig_kriging_{region_key}')

print("  绘图完成!")
print("\n" + "=" * 60)
print("  Kriging 实验完成!")
print("=" * 60)
