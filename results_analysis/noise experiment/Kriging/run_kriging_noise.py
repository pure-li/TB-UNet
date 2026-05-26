#!/usr/bin/env python
"""Kriging 噪声鲁棒性实验 — rect & irreg, noise σ = 0/2/5/10 nT
==============================================================
只保存数据(.npy) + results.json, 不画图
"""
import os, time, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.spatial import ConvexHull, KDTree
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pykrige.ok import OrdinaryKriging

DATA_PATH = r'F:\PINN实验\venv\U-net\afghanistan_full\Afghan_mag06A.csv'
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
N_KRIGING_CTRL = 2000
N_TRUTH_CTRL = 3000
NOISE_LEVELS = [0, 2, 5, 10]

RECT = (62.0, 63.0, 32.5, 33.5)
IRREG = np.array([
    [62.05, 32.65], [62.30, 32.38], [62.70, 32.42], [62.95, 32.75],
    [63.08, 33.05], [62.88, 33.38], [62.48, 33.48], [62.15, 33.22], [61.98, 32.92],
])

print("=" * 60)
print("  Kriging 噪声鲁棒性实验")
print(f"  Noise: {NOISE_LEVELS} nT | Rect + Irreg")
print("=" * 60)


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
    df_all = pd.read_csv(DATA_PATH).iloc[::3]
    df_truth = df_all.iloc[::max(1, len(df_all) // N_TRUTH_CTRL)].copy()
    pts = np.column_stack(data['ll_to_xy'](df_truth['Longitude'].values, df_truth['Latitude'].values))
    vals = df_truth['FinalMag'].values
    tg = RBFInterpolator(pts, vals, kernel='thin_plate_spline')(
        data['grid_pts']).reshape(data['grid_x'].shape)
    tg = gaussian_filter1d(tg, 2.0, axis=0)
    tg = gaussian_filter1d(tg, 1.0, axis=1)
    tg[data['mask_outside']] = np.nan
    return tg


def run_kriging_noise(data, truth_grid, sigma, region_key, noise_key):
    """添加 sigma nT 高斯噪声后运行 Kriging"""
    np.random.seed(SEED)
    train_df = data['train_df']
    mask_blank, mask_outside = data['mask_blank'], data['mask_outside']
    grid_pts = data['grid_pts']

    n_ctrl = min(N_KRIGING_CTRL, len(train_df))
    idx = np.random.choice(len(train_df), n_ctrl, replace=False)
    ctrl_x = train_df['x'].values[idx]
    ctrl_y = train_df['y'].values[idx]
    ctrl_F_clean = train_df['FinalMag'].values[idx]

    # Add noise
    noise = np.random.normal(0, sigma, n_ctrl).astype(np.float32)
    ctrl_F = ctrl_F_clean + noise

    t0 = time.time()
    OK = OrdinaryKriging(ctrl_x, ctrl_y, ctrl_F,
                         variogram_model='spherical',
                         verbose=False, enable_plotting=False)

    blank_idx = np.where(mask_blank.ravel())[0]
    blank_pts = grid_pts[blank_idx]
    n_blank = len(blank_pts)
    print(f"    Blank pts: {n_blank:,}, batching Kriging...")

    # Batch to avoid memory error (ctrl_pts × batch must fit in RAM)
    BATCH = 1500
    z_blank_all = np.empty(n_blank, dtype=np.float32)
    for b_start in range(0, n_blank, BATCH):
        b_end = min(b_start + BATCH, n_blank)
        z_chunk, _ = OK.execute('points', blank_pts[b_start:b_end, 0],
                                blank_pts[b_start:b_end, 1])
        z_blank_all[b_start:b_end] = z_chunk.astype(np.float32)
        if b_start % 6000 == 0:
            print(f"      {b_end}/{n_blank}")

    kriging_time = time.time() - t0

    result_grid = np.full(grid_pts.shape[0], np.nan, dtype=np.float32)
    result_grid[blank_idx] = z_blank_all
    result_grid = result_grid.reshape(data['grid_x'].shape)

    temp = np.nan_to_num(result_grid, nan=0.0)
    temp = gaussian_filter1d(temp, 1.5, axis=0)
    temp = gaussian_filter1d(temp, 1.5, axis=1)
    result_grid[mask_blank] = temp[mask_blank]

    # Evaluate on test points (batched)
    test_xy = np.column_stack([data['test_df']['x'].values, data['test_df']['y'].values])
    test_F = data['test_df']['FinalMag'].values
    n_test = len(test_F)
    z_test_all = np.empty(n_test, dtype=np.float32)
    for b_start in range(0, n_test, BATCH):
        b_end = min(b_start + BATCH, n_test)
        z_chunk, _ = OK.execute('points', test_xy[b_start:b_end, 0],
                                test_xy[b_start:b_end, 1])
        z_test_all[b_start:b_end] = z_chunk.astype(np.float32)
    z_test = z_test_all
    valid = ~np.isnan(z_test)
    rmse = np.sqrt(mean_squared_error(test_F[valid], z_test[valid])) if np.sum(valid) > 10 else float('nan')
    mae = mean_absolute_error(test_F[valid], z_test[valid]) if np.sum(valid) > 10 else float('nan')

    # Absolute residuals in blank region
    abs_residual = np.full_like(result_grid, np.nan)
    abs_residual[mask_blank] = np.abs(result_grid[mask_blank] - truth_grid[mask_blank])

    # Save
    tag = f'{region_key}_noise_{sigma}'
    np.save(os.path.join(OUT_DIR, f'result_grid_{tag}.npy'), result_grid)
    np.save(os.path.join(OUT_DIR, f'abs_residual_{tag}.npy'), abs_residual)

    result = {
        'method': 'Kriging', 'region': region_key, 'noise_sigma': sigma,
        'rmse': float(rmse), 'mae': float(mae),
        'time': kriging_time, 'status': 'completed',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(OUT_DIR, f'results_{tag}.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  [{region_key}] σ={sigma}nT → RMSE={rmse:.2f}, MAE={mae:.2f} | {kriging_time:.0f}s")
    return result


# =============================================================================
# Main
# =============================================================================
regions = {
    'rect': ('矩形', None, RECT),
    'irreg': ('不规则', IRREG, None),
}

run_start = time.time()
all_results = {}

for rk, (rlabel, poly, rect) in regions.items():
    print(f"\n{'='*60}")
    print(f"  区域: {rlabel}")
    print(f"{'='*60}")

    data = load_region_data(poly_vertices=poly, rect_bounds=rect)
    truth_grid = compute_truth_grid(data)

    # Save shared data (once per region)
    np.save(os.path.join(OUT_DIR, f'truth_grid_{rk}.npy'), truth_grid)
    np.save(os.path.join(OUT_DIR, f'mask_blank_{rk}.npy'), data['mask_blank'])
    np.save(os.path.join(OUT_DIR, f'mask_outside_{rk}.npy'), data['mask_outside'])
    np.save(os.path.join(OUT_DIR, f'grid_x_{rk}.npy'), data['grid_x'])
    np.save(os.path.join(OUT_DIR, f'grid_y_{rk}.npy'), data['grid_y'])
    np.save(os.path.join(OUT_DIR, f'bx_{rk}.npy'), data['bx'])
    np.save(os.path.join(OUT_DIR, f'by_{rk}.npy'), data['by'])

    for sigma in NOISE_LEVELS:
        res = run_kriging_noise(data, truth_grid, sigma, rk, f'noise_{sigma}')
        all_results[f'{rk}_noise_{sigma}'] = res

# Progress summary
progress = {
    'method': 'Kriging',
    'status': 'completed',
    'total_time_min': round((time.time() - run_start) / 60, 1),
    'results': all_results,
}
with open(os.path.join(OUT_DIR, 'progress.json'), 'w') as f:
    json.dump(progress, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"  Kriging 噪声实验完成! 总耗时: {progress['total_time_min']} min")
print(f"  输出: {OUT_DIR}/")
print(f"{'='*60}")
