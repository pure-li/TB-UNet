#!/usr/bin/env python
"""GAN 噪声鲁棒性实验 — 仅推理，不重训
============================================
噪声加在真实测线数据 FinalMag → RBF → GAN 推理
rect & irreg, noise σ = 2/5/10 nT (σ=0 复用已有)
"""
import os, sys, time, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator, RegularGridInterpolator
from scipy.spatial import KDTree, ConvexHull
from matplotlib.path import Path
from scipy.ndimage import zoom, gaussian_filter1d
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.transform import resize
import torch, torch.nn as nn

# Copy existing GAN results for sigma=0
COMPARATIVE_GAN = r'F:\PINN实验\venv\U-net\comparative experiment\gan_results'
COMPARATIVE_ROOT = r'F:\PINN实验\venv\U-net\comparative experiment'

DATA_PATH = r'F:\PINN实验\venv\U-net\afghanistan_full\Afghan_mag06A.csv'
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
TARGET_SIZE = 128

RECT = (62.0, 63.0, 32.5, 33.5)
IRREG = np.array([
    [62.05, 32.65], [62.30, 32.38], [62.70, 32.42], [62.95, 32.75],
    [63.08, 33.05], [62.88, 33.38], [62.48, 33.48], [62.15, 33.22], [61.98, 32.92],
])

# =============================================================================
# GAN model (same as run_gan_recon.py)
# =============================================================================
def conv_block(in_ch, out_ch, kernel=4, stride=2, padding=1, norm=True, act='lrelu'):
    layers = [nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=not norm)]
    if norm: layers.append(nn.BatchNorm2d(out_ch))
    if act == 'lrelu': layers.append(nn.LeakyReLU(0.2, inplace=True))
    elif act == 'relu': layers.append(nn.ReLU(inplace=True))
    elif act == 'tanh': layers.append(nn.Tanh())
    return nn.Sequential(*layers)

def deconv_block(in_ch, out_ch, kernel=4, stride=2, padding=1, norm=True, act='relu', dropout=0.0):
    layers = [nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding, bias=not norm)]
    if norm: layers.append(nn.BatchNorm2d(out_ch))
    if dropout > 0: layers.append(nn.Dropout(dropout))
    if act == 'relu': layers.append(nn.ReLU(inplace=True))
    elif act == 'lrelu': layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

class Generator(nn.Module):
    def __init__(self, in_ch=1, base_ch=64):
        super().__init__()
        self.enc1 = conv_block(in_ch, base_ch, norm=False)
        self.enc2 = conv_block(base_ch, base_ch*2)
        self.enc3 = conv_block(base_ch*2, base_ch*4)
        self.enc4 = conv_block(base_ch*4, base_ch*8)
        self.enc5 = conv_block(base_ch*8, base_ch*8)
        self.bottleneck = nn.Conv2d(base_ch*8, base_ch*8, kernel_size=1)
        self.dec1 = deconv_block(base_ch*8, base_ch*8, dropout=0.5)
        self.dec2 = deconv_block(base_ch*8*2, base_ch*4, dropout=0.5)
        self.dec3 = deconv_block(base_ch*4*2, base_ch*2)
        self.dec4 = deconv_block(base_ch*2*2, base_ch)
        self.dec5 = deconv_block(base_ch*2, 1, norm=False, act='tanh')

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2)
        e4 = self.enc4(e3); e5 = self.enc5(e4)
        b = self.bottleneck(e5)
        d1 = self.dec1(b); d1 = torch.cat([d1, e4], dim=1)
        d2 = self.dec2(d1); d2 = torch.cat([d2, e3], dim=1)
        d3 = self.dec3(d2); d3 = torch.cat([d3, e2], dim=1)
        d4 = self.dec4(d3); d4 = torch.cat([d4, e1], dim=1)
        return self.dec5(d4)


# =============================================================================
# Data loading
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
    df_all = pd.read_csv(DATA_PATH).iloc[::3]
    n_ctrl = 3000
    df_truth = df_all.iloc[::max(1, len(df_all) // n_ctrl)].copy()
    pts = np.column_stack(data['ll_to_xy'](df_truth['Longitude'].values, df_truth['Latitude'].values))
    vals = df_truth['FinalMag'].values
    tg = RBFInterpolator(pts, vals, kernel='thin_plate_spline')(
        data['grid_pts']).reshape(data['grid_x'].shape)
    tg = gaussian_filter1d(tg, 2.0, axis=0)
    tg = gaussian_filter1d(tg, 1.0, axis=1)
    tg[data['mask_outside']] = np.nan
    return tg


def rbf_with_noise(data, sigma):
    """RBF interpolation with noise added to training point values"""
    np.random.seed(SEED)
    train_df = data['train_df']
    points_xy = train_df[['x', 'y']].values
    values_F_clean = train_df['FinalMag'].values

    # Add noise to measurements
    noise = np.random.normal(0, sigma, len(values_F_clean)).astype(np.float32)
    values_F = values_F_clean + noise

    n_sub = min(3000, len(points_xy))
    idx = np.random.choice(len(points_xy), n_sub, replace=False)
    pts_sub, val_sub = points_xy[idx], values_F[idx]

    tree = KDTree(pts_sub)
    dists, _ = tree.query(pts_sub, k=min(10, n_sub))
    eps = np.median(dists[:, 1:]) * 0.8

    F_grid = RBFInterpolator(pts_sub, val_sub, kernel='cubic', epsilon=eps)(
        data['grid_pts']).reshape(data['grid_x'].shape)

    F_display = F_grid.copy()
    F_display = gaussian_filter1d(F_display, 5.0, axis=0)
    F_display = gaussian_filter1d(F_display, 1.0, axis=1)
    F_display[data['mask_outside']] = np.nan

    F_grid_masked = F_grid.copy()
    F_grid_masked[data['mask_blank'] | data['mask_outside']] = np.nan
    F_min = np.nanmin(F_grid_masked)
    F_max = np.nanmax(F_grid_masked)

    return F_grid, F_display, F_grid_masked, F_min, F_max


def gan_inference_noise(G, data, F_display, F_grid_masked, F_min, F_max):
    """GAN inference with resize pipeline"""
    G.eval()

    def norm(x):
        return (x - F_min) / (F_max - F_min) * 2 - 1

    def denorm(x):
        return (x + 1) / 2 * (F_max - F_min) + F_min

    F_input = F_grid_masked.copy()
    F_input[np.isnan(F_input)] = 0.0
    F_input_n = norm(F_input)
    F_resized = resize(F_input_n.astype(np.float32), (TARGET_SIZE, TARGET_SIZE),
                       mode='constant', anti_aliasing=True)

    with torch.no_grad():
        inp = torch.tensor(F_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        out = G(inp)
    out_np = out.squeeze().cpu().numpy()
    out_np = denorm(out_np)

    nx, ny = data['nx'], data['ny']
    zf = (nx / TARGET_SIZE, ny / TARGET_SIZE)
    out_full = zoom(out_np, zf, order=1)[:nx, :ny]

    bo = out_full.copy()
    bo[~data['mask_blank']] = np.nan
    bo = gaussian_filter1d(bo, 1.5, axis=0)
    bo = gaussian_filter1d(bo, 1.5, axis=1)
    out_full[data['mask_blank']] = bo[data['mask_blank']]

    result_grid = F_display.copy()
    result_grid[data['mask_blank']] = out_full[data['mask_blank']]
    return result_grid


def evaluate(result_grid, data):
    interp = RegularGridInterpolator(
        (data['x_grid'], data['y_grid']), result_grid,
        method='linear', bounds_error=False, fill_value=np.nan)
    test_xy = np.column_stack([data['test_df']['x'].values, data['test_df']['y'].values])
    yt = data['test_df']['FinalMag'].values
    pred = interp(test_xy)
    valid = ~np.isnan(pred)
    if np.sum(valid) > 10:
        rmse = np.sqrt(mean_squared_error(yt[valid], pred[valid]))
        mae = mean_absolute_error(yt[valid], pred[valid])
    else:
        rmse = mae = float('nan')
    return rmse, mae


# =============================================================================
# Copy sigma=0 from comparative experiment
# =============================================================================
def copy_noise0(region_key):
    """Copy existing GAN results for noise=0 baseline"""
    import shutil
    src_files = [
        (f'{COMPARATIVE_GAN}/result_grid_{region_key}.npy', f'result_grid_{region_key}_noise_0.npy'),
        (f'{COMPARATIVE_GAN}/truth_grid_{region_key}.npy', f'truth_grid_{region_key}.npy'),
        (f'{COMPARATIVE_GAN}/mask_blank_{region_key}.npy', f'mask_blank_{region_key}.npy'),
        (f'{COMPARATIVE_GAN}/mask_outside_{region_key}.npy', f'mask_outside_{region_key}.npy'),
        (f'{COMPARATIVE_GAN}/grid_x_{region_key}.npy', f'grid_x_{region_key}.npy'),
        (f'{COMPARATIVE_GAN}/grid_y_{region_key}.npy', f'grid_y_{region_key}.npy'),
        (f'{COMPARATIVE_GAN}/bx_{region_key}.npy', f'bx_{region_key}.npy'),
        (f'{COMPARATIVE_GAN}/by_{region_key}.npy', f'by_{region_key}.npy'),
    ]
    for src, dst in src_files:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(OUT_DIR, dst))
            print(f"  Copied: {os.path.basename(src)} → {dst}")

    # Compute abs_residual for noise=0
    result = np.load(os.path.join(OUT_DIR, f'result_grid_{region_key}_noise_0.npy'))
    truth = np.load(os.path.join(OUT_DIR, f'truth_grid_{region_key}.npy'))
    mask = np.load(os.path.join(OUT_DIR, f'mask_blank_{region_key}.npy'))
    abs_res = np.full_like(result, np.nan)
    abs_res[mask] = np.abs(result[mask] - truth[mask])
    np.save(os.path.join(OUT_DIR, f'abs_residual_{region_key}_noise_0.npy'), abs_res)

    # Get RMSE/MAE from existing results
    with open(f'{COMPARATIVE_ROOT}/comparison_metrics.json') as f:
        cm = json.load(f)
    gan_r = cm[region_key]['GAN']
    result_dict = {
        'method': 'GAN', 'region': region_key, 'noise_sigma': 0,
        'rmse': gan_r['rmse'], 'mae': gan_r['mae'],
        'time': 0, 'status': 'completed',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(OUT_DIR, f'results_{region_key}_noise_0.json'), 'w') as f:
        json.dump(result_dict, f, indent=2)
    print(f"  [{region_key}] σ=0nT (reused) → RMSE={gan_r['rmse']:.2f}, MAE={gan_r['mae']:.2f}")


# =============================================================================
# Main
# =============================================================================
regions = {
    'rect': ('矩形', None, RECT),
    'irreg': ('不规则', IRREG, None),
}

print("=" * 60)
print("  GAN 噪声鲁棒性实验 (仅推理)")
print("=" * 60)
print(f"  Device: {DEVICE}")

# Load GAN model
print("\nLoading GAN model...")
checkpoint = torch.load(os.path.join(COMPARATIVE_GAN, 'gan_model.pth'), map_location=DEVICE)
G = Generator(in_ch=1, base_ch=64).to(DEVICE)
G.load_state_dict(checkpoint['G_state_dict'])
G.eval()
print("  GAN model loaded.")

run_start = time.time()
all_results = {}

for rk, (rlabel, poly, rect) in regions.items():
    print(f"\n{'='*60}")
    print(f"  区域: {rlabel}")
    print(f"{'='*60}")

    # Copy noise=0 baseline
    copy_noise0(rk)
    all_results[f'{rk}_noise_0'] = {'rmse': 0, 'mae': 0, 'noise_sigma': 0, 'status': 'reused'}

    # Load data
    data = load_region_data(poly_vertices=poly, rect_bounds=rect)
    truth_grid = compute_truth_grid(data)

    # Save shared data
    np.save(os.path.join(OUT_DIR, f'truth_grid_{rk}.npy'), truth_grid)
    np.save(os.path.join(OUT_DIR, f'mask_blank_{rk}.npy'), data['mask_blank'])
    np.save(os.path.join(OUT_DIR, f'mask_outside_{rk}.npy'), data['mask_outside'])
    np.save(os.path.join(OUT_DIR, f'grid_x_{rk}.npy'), data['grid_x'])
    np.save(os.path.join(OUT_DIR, f'grid_y_{rk}.npy'), data['grid_y'])
    np.save(os.path.join(OUT_DIR, f'bx_{rk}.npy'), data['bx'])
    np.save(os.path.join(OUT_DIR, f'by_{rk}.npy'), data['by'])

    for sigma in [2, 5, 10]:
        print(f"\n  --- σ={sigma} nT ---")
        t0 = time.time()

        # RBF with noise
        F_grid, F_display, F_grid_masked, F_min, F_max = rbf_with_noise(data, sigma)

        # GAN inference
        result_grid = gan_inference_noise(G, data, F_display, F_grid_masked, F_min, F_max)

        # Evaluate
        rmse, mae = evaluate(result_grid, data)
        elapsed = time.time() - t0

        # Abs residuals
        abs_residual = np.full_like(result_grid, np.nan)
        abs_residual[data['mask_blank']] = np.abs(result_grid[data['mask_blank']] - truth_grid[data['mask_blank']])

        # Save
        tag = f'{rk}_noise_{sigma}'
        np.save(os.path.join(OUT_DIR, f'result_grid_{tag}.npy'), result_grid)
        np.save(os.path.join(OUT_DIR, f'abs_residual_{tag}.npy'), abs_residual)

        result = {
            'method': 'GAN', 'region': rk, 'noise_sigma': sigma,
            'rmse': float(rmse), 'mae': float(mae),
            'time': elapsed, 'status': 'completed',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(os.path.join(OUT_DIR, f'results_{tag}.json'), 'w') as f:
            json.dump(result, f, indent=2)

        all_results[tag] = result
        print(f"  [{rk}] σ={sigma}nT → RMSE={rmse:.2f}, MAE={mae:.2f} | {elapsed:.0f}s")

# Progress
progress = {
    'method': 'GAN',
    'status': 'completed',
    'total_time_min': round((time.time() - run_start) / 60, 1),
    'results': all_results,
}
with open(os.path.join(OUT_DIR, 'progress.json'), 'w') as f:
    json.dump(progress, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"  GAN 噪声实验完成! 总耗时: {progress['total_time_min']} min")
print(f"  输出: {OUT_DIR}/")
print(f"{'='*60}")
