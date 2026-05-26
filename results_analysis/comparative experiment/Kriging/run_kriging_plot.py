#!/usr/bin/env python
"""Kriging 填补结果绘图
======================
重新运行 Ordinary Kriging, 生成与 U-Net+Transformer 一致的:
  - fig_result_{rect|irreg}  (jet)
  - fig_residual_{rect|irreg} (RdBu_r, pred - truth)
  - fig_error_{rect|irreg}    (hot, |pred - truth|)
"""

import os, time, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator, RegularGridInterpolator
from scipy.spatial import ConvexHull, KDTree
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pykrige.ok import OrdinaryKriging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# 0. 配置
# =============================================================================
DATA_PATH = r'F:\PINN实验\venv\U-net\afghanistan_full\Afghan_mag06A.csv'
OUT_DIR   = r'F:\PINN实验\venv\U-net\Kriging'
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42; np.random.seed(SEED)
N_KRIGING_CTRL = 2000
N_TRUTH_SAMPLES = 3000

RECT_LON_MIN, RECT_LON_MAX = 62.0, 63.0
RECT_LAT_MIN, RECT_LAT_MAX = 32.5, 33.5

IRREGULAR_POLYGON = np.array([
    [62.05, 32.65], [62.30, 32.38], [62.70, 32.42], [62.95, 32.75],
    [63.08, 33.05], [62.88, 33.38], [62.48, 33.48], [62.15, 33.22], [61.98, 32.92],
])

print("=" * 60)
print("  Kriging 绘图 — 双区域")
print("=" * 60)

# =============================================================================
# 1. 数据加载 (与 U-Net+Transformer 一致)
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
        'train_df': train_df, 'test_df': test_df, 'll_to_xy': ll_to_xy,
        'x_grid': x_grid, 'y_grid': y_grid, 'nx': nx, 'ny': ny,
        'grid_x': grid_x, 'grid_y': grid_y, 'grid_pts': grid_pts,
        'mask_blank': mask_blank, 'mask_outside': mask_outside,
        'bx': bx, 'by': by,
    }


def compute_truth_grid(data):
    """真值网格 (thin_plate_spline, 用于残差计算, 与 U-Net+Transformer 一致)"""
    df_all = pd.read_csv(DATA_PATH).iloc[::3]
    df_truth = df_all.iloc[::max(1, len(df_all)//N_TRUTH_SAMPLES)].copy()
    pts_truth = np.column_stack(data['ll_to_xy'](df_truth['Longitude'].values, df_truth['Latitude'].values))
    val_truth = df_truth['FinalMag'].values
    truth_grid = RBFInterpolator(pts_truth, val_truth, kernel='thin_plate_spline')(
        data['grid_pts']).reshape(data['grid_x'].shape)
    truth_grid = gaussian_filter1d(truth_grid, 2.0, axis=0)
    truth_grid = gaussian_filter1d(truth_grid, 1.0, axis=1)
    truth_grid[data['mask_outside']] = np.nan
    return truth_grid


def save_fig(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(OUT_DIR, f'{name}.{ext}'), dpi=300,
                    bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


# =============================================================================
# 2. 运行 Kriging + 绘图
# =============================================================================

regions = {
    'rect': ('Rectangular', None, (RECT_LON_MIN, RECT_LON_MAX, RECT_LAT_MIN, RECT_LAT_MAX)),
    'irreg': ('Irregular', IRREGULAR_POLYGON, None),
}

results_summary = {}
run_start = time.time()

for region_key, (rlabel, poly, rect) in regions.items():
    print(f"\n[Kriging] {rlabel}...")
    data = load_region_data(poly_vertices=poly, rect_bounds=rect)
    train_df = data['train_df']
    mask_blank, mask_outside = data['mask_blank'], data['mask_outside']
    grid_pts = data['grid_pts']

    # Kriging
    n_ctrl = min(N_KRIGING_CTRL, len(train_df))
    idx = np.random.choice(len(train_df), n_ctrl, replace=False)
    ctrl_x = train_df['x'].values[idx]
    ctrl_y = train_df['y'].values[idx]
    ctrl_F = train_df['FinalMag'].values[idx]

    t0 = time.time()
    OK = OrdinaryKriging(ctrl_x, ctrl_y, ctrl_F,
                         variogram_model='spherical',
                         verbose=False, enable_plotting=False)

    blank_idx = np.where(mask_blank.ravel())[0]
    blank_pts = grid_pts[blank_idx]
    z_blank, _ = OK.execute('points', blank_pts[:, 0], blank_pts[:, 1])
    kriging_time = time.time() - t0

    # 完整网格: 空白区=Kriging预测, 非空白区=NaN
    result_grid = np.full(grid_pts.shape[0], np.nan, dtype=np.float32)
    result_grid[blank_idx] = z_blank.astype(np.float32)
    result_grid = result_grid.reshape(data['grid_x'].shape)

    # 平滑 (gaussian_filter1d 传播 NaN, 先填 0 平滑再恢复)
    temp = np.nan_to_num(result_grid, nan=0.0)
    temp = gaussian_filter1d(temp, 1.5, axis=0)
    temp = gaussian_filter1d(temp, 1.5, axis=1)
    result_grid[mask_blank] = temp[mask_blank]
    result_grid[~mask_blank] = np.nan

    # 评估
    test_xy = np.column_stack([data['test_df']['x'].values, data['test_df']['y'].values])
    test_F = data['test_df']['FinalMag'].values
    z_test, _ = OK.execute('points', test_xy[:, 0], test_xy[:, 1])
    z_test = z_test.astype(np.float32)
    valid = ~np.isnan(z_test)
    rmse = np.sqrt(mean_squared_error(test_F[valid], z_test[valid])) if np.sum(valid) > 10 else float('nan')
    mae  = mean_absolute_error(test_F[valid], z_test[valid]) if np.sum(valid) > 10 else float('nan')
    print(f"  RMSE={rmse:.2f}, MAE={mae:.2f}, Time={kriging_time:.1f}s")

    # 真值网格
    truth_grid = compute_truth_grid(data)

    # 保存
    np.save(os.path.join(OUT_DIR, f'result_grid_{region_key}.npy'), result_grid)
    np.save(os.path.join(OUT_DIR, f'truth_grid_{region_key}.npy'), truth_grid)
    np.save(os.path.join(OUT_DIR, f'mask_blank_{region_key}.npy'), mask_blank)
    np.save(os.path.join(OUT_DIR, f'mask_outside_{region_key}.npy'), mask_outside)
    np.save(os.path.join(OUT_DIR, f'grid_x_{region_key}.npy'), data['grid_x'])
    np.save(os.path.join(OUT_DIR, f'grid_y_{region_key}.npy'), data['grid_y'])
    np.save(os.path.join(OUT_DIR, f'bx_{region_key}.npy'), data['bx'])
    np.save(os.path.join(OUT_DIR, f'by_{region_key}.npy'), data['by'])

    # =========================================================================
    # 绘图 (与 U-Net+Transformer 配色一致)
    # =========================================================================
    grid_x, grid_y = data['grid_x'], data['grid_y']
    bx, by = data['bx'], data['by']
    zx = (bx.min() - 2, bx.max() + 2)
    zy = (by.min() - 2, by.max() + 2)
    vmin = np.nanmin(truth_grid[~mask_outside])
    vmax = np.nanmax(truth_grid[~mask_outside])

    # 1) Result map (jet)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(grid_x, grid_y, result_grid, cmap='jet', shading='auto', vmin=vmin, vmax=vmax)
    ax.plot(bx, by, 'k-', linewidth=2)
    ax.set_xlim(zx); ax.set_ylim(zy)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
    ax.set_title(f'{rlabel} Blank — Ordinary Kriging (RMSE={rmse:.2f} nT)')
    cbar = plt.colorbar(im, ax=ax, label='Mag. Anomaly (nT)')
    cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
    save_fig(fig, f'fig_result_{region_key}')

    # 2) Residual map (pred - true, RdBu_r)
    residual = result_grid.copy()
    residual[~mask_blank] = np.nan
    residual[mask_blank] = result_grid[mask_blank] - truth_grid[mask_blank]
    res_max = max(abs(np.nanmin(residual[mask_blank])), abs(np.nanmax(residual[mask_blank])), 1.0)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(grid_x, grid_y, residual, cmap='RdBu_r', shading='auto', vmin=-res_max, vmax=res_max)
    ax.plot(bx, by, 'k-', linewidth=2)
    ax.set_xlim(zx); ax.set_ylim(zy)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
    ax.set_title(f'{rlabel} Blank — Residual (Pred - True)\nOrdinary Kriging, RMSE={rmse:.2f} nT')
    cbar = plt.colorbar(im, ax=ax, label='Residual (nT)')
    cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
    save_fig(fig, f'fig_residual_{region_key}')

    # 3) Absolute error map (|pred - true|, hot, 越亮=误差越大)
    abs_error = result_grid.copy()
    abs_error[~mask_blank] = np.nan
    abs_error[mask_blank] = np.abs(result_grid[mask_blank] - truth_grid[mask_blank])
    ae_max = np.nanmax(abs_error[mask_blank])

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(grid_x, grid_y, abs_error, cmap='hot', shading='auto',
                       vmin=0, vmax=ae_max * 0.8)
    ax.plot(bx, by, 'k-', linewidth=2)
    ax.set_xlim(zx); ax.set_ylim(zy)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
    ax.set_title(f'{rlabel} Blank — |Error|\nOrdinary Kriging, RMSE={rmse:.2f} nT')
    cbar = plt.colorbar(im, ax=ax, label='Absolute Error (nT)')
    cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
    save_fig(fig, f'fig_error_{region_key}')

    plt.close('all')
    print(f"  图表已保存: fig_result/residual/error_{region_key}.png|svg")

    results_summary[region_key] = {'rmse': float(rmse), 'mae': float(mae), 'time': kriging_time}

# 保存
results_summary['config'] = {'n_kriging_ctrl': N_KRIGING_CTRL, 'variogram': 'spherical'}
with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
    json.dump(results_summary, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"  Kriging 绘图完成!")
print(f"  Rect  RMSE={results_summary['rect']['rmse']:.2f}, Irreg RMSE={results_summary['irreg']['rmse']:.2f}")
print(f"  总耗时: {(time.time()-run_start):.0f}s")
print(f"  输出: {OUT_DIR}/")
print("=" * 60)
