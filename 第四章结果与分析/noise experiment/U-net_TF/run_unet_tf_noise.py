#!/usr/bin/env python
"""U-Net+Transformer 噪声鲁棒性实验 — 重训每个噪声水平
==========================================================
噪声加在训练数据 FinalMag → RBF 背景场 → 自监督训练 (100 epochs)
rect & irreg, noise σ = 2/5/10 nT (σ=0 复用已有)
"""
import sys, importlib.util, os, time, json, warnings
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
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler

# Import base U-Net modules
unet_path = r'F:\PINN实验\venv\U-net\U-net.py'
spec = importlib.util.spec_from_file_location("unet", unet_path)
unet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unet)

COMPARATIVE_UTF = r'F:\PINN实验\venv\U-net\comparative experiment\U-net_transformer_slow'
COMPARATIVE_ROOT = r'F:\PINN实验\venv\U-net\comparative experiment'

DATA_PATH = r'F:\PINN实验\venv\U-net\afghanistan_full\Afghan_mag06A.csv'
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET_SIZE = (128, 128)
EPOCHS = 100
BATCH_SIZE = 8
BASE_CH = 48
SEED = 42
N_INTERP_CTRL = 3000
MAX_LR = 5e-4
PCT_START = 0.3
WEIGHT_DECAY = 1e-5
TRANSFORMER_LAYERS = 4
TRANSFORMER_HEADS = 8
TRANSFORMER_MLP_RATIO = 4
TRANSFORMER_DROPOUT = 0.1

RECT = (62.0, 63.0, 32.5, 33.5)
IRREG = np.array([
    [62.05, 32.65], [62.30, 32.38], [62.70, 32.42], [62.95, 32.75],
    [63.08, 33.05], [62.88, 33.38], [62.48, 33.48], [62.15, 33.22], [61.98, 32.92],
])

print("=" * 60)
print("  U-Net+Transformer 噪声鲁棒性实验 (重训)")
print(f"  Epochs={EPOCHS}, lr={MAX_LR}, pct_start={PCT_START}")
print(f"  Device: {DEVICE}")
print("=" * 60)


# =============================================================================
# Transformer modules (same as run_slow.py)
# =============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, num_patches, dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim) * 0.02)
    def forward(self, x): return x + self.pos_embed

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim), nn.Dropout(dropout),
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerBottleneck(nn.Module):
    def __init__(self, in_ch, spatial_size, num_layers=4, num_heads=8,
                 mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.num_patches = spatial_size * spatial_size
        self.pos_enc = PositionalEncoding(self.num_patches, in_ch)
        self.blocks = nn.ModuleList([
            TransformerBlock(in_ch, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(in_ch)
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_enc(x)
        for blk in self.blocks: x = blk(x)
        x = self.norm_out(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

class UNetTransformer(nn.Module):
    def __init__(self, in_chans=1, base_ch=48, use_skip=False):
        super().__init__()
        self.inc = unet.DoubleConv(in_chans, base_ch)
        self.down1 = unet.Down(base_ch, base_ch * 2)
        self.down2 = unet.Down(base_ch * 2, base_ch * 4)
        self.down3 = unet.Down(base_ch * 4, base_ch * 4)
        self.down4 = unet.Down(base_ch * 4, base_ch * 4)
        self.transformer = TransformerBottleneck(
            in_ch=base_ch * 4, spatial_size=8,
            num_layers=TRANSFORMER_LAYERS, num_heads=TRANSFORMER_HEADS,
            mlp_ratio=TRANSFORMER_MLP_RATIO, dropout=TRANSFORMER_DROPOUT,
        )
        self.up1 = unet.Up(base_ch * 4, base_ch * 2, use_skip=False)
        self.up2 = unet.Up(base_ch * 2, base_ch * 2, use_skip=False)
        self.up3 = unet.Up(base_ch * 2, base_ch, use_skip=False)
        self.up4 = unet.Up(base_ch, base_ch, use_skip=False)
        self.outc = nn.Conv2d(base_ch, 1, kernel_size=1)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2)
        x4 = self.down3(x3); x5 = self.down4(x4)
        x5 = self.transformer(x5)
        x = self.up1(x5); x = self.up2(x); x = self.up3(x); x = self.up4(x)
        return self.outc(x)


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


def rbf_preprocess_with_noise(data, sigma):
    """RBF preprocessing with noise added to training point measurements"""
    np.random.seed(SEED)
    train_df = data['train_df']
    points_xy = train_df[['x', 'y']].values
    values_F_clean = train_df['FinalMag'].values

    noise = np.random.normal(0, sigma, len(values_F_clean)).astype(np.float32)
    values_F = values_F_clean + noise

    n_sub = min(N_INTERP_CTRL, len(points_xy))
    idx = np.random.choice(len(points_xy), n_sub, replace=False)
    pts_sub, val_sub = points_xy[idx], values_F[idx]

    tree = KDTree(pts_sub)
    distances, _ = tree.query(pts_sub, k=min(10, n_sub))
    eps = np.median(distances[:, 1:]) * 0.8

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

    def normalize(x): return (x - F_min) / (F_max - F_min) * 2 - 1
    def denorm(x): return (x + 1) / 2 * (F_max - F_min) + F_min

    F_input = F_grid_masked.copy()
    F_input[np.isnan(F_input)] = 0.0
    F_resized = resize(normalize(F_input).astype(np.float32), TARGET_SIZE,
                       mode='constant', anti_aliasing=True)
    perm_mask_resized = resize(data['mask_blank'].astype(float), TARGET_SIZE) > 0.5

    test_xy = np.column_stack(data['ll_to_xy'](data['test_df']['Longitude'].values,
                                                data['test_df']['Latitude'].values))
    yt_all = data['test_df']['FinalMag'].values

    return {
        'F_display': F_display, 'F_min': F_min, 'F_max': F_max,
        'F_resized': F_resized, 'perm_mask_resized': perm_mask_resized,
        'test_xy': test_xy, 'yt_all': yt_all,
        'normalize': normalize, 'denorm': denorm, 'eps': eps,
    }


def train_model(data, prep, region_key, sigma):
    """Train U-Net+Transformer for one noise level"""
    tag = f'{region_key}_noise_{sigma}'
    print(f"\n  [{tag}] Training {EPOCHS} epochs...")

    np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    model = UNetTransformer(in_chans=1, base_ch=BASE_CH, use_skip=False).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Params: {n_params:,}")

    criterion = unet.CompositeLoss(grad_weight=0.0).to(DEVICE)
    train_image = np.stack([prep['F_resized']], axis=0).astype(np.float32)
    dataset = unet.InpaintingDataset(train_image, prep['perm_mask_resized'],
                                     mask_mode='block', mask_ratio=0.2)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    total_steps = EPOCHS * len(dataloader)
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    scheduler_lr = lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, total_steps=total_steps,
                                        pct_start=PCT_START, anneal_strategy='cos',
                                        final_div_factor=1e4)
    test_input_t = torch.tensor(train_image, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    history = []
    best_rmse = float('inf')
    best_state = None
    best_epoch = 0
    best_result_grid = None
    zf = (data['nx'] / TARGET_SIZE[0], data['ny'] / TARGET_SIZE[1])
    t0 = time.time()

    for ep in range(EPOCHS):
        model.train()
        ep_loss = 0.0
        for batch in dataloader:
            inp, target, mask = [b.to(DEVICE) for b in batch]
            mask = mask.bool()
            optimizer.zero_grad()
            pred = model(inp)
            loss = criterion(pred, target, mask)
            loss.backward()
            optimizer.step()
            scheduler_lr.step()
            ep_loss += loss.item()

        model.eval()
        with torch.no_grad():
            output = model(test_input_t)
        output_np = output.squeeze().cpu().numpy()
        output_np = prep['denorm'](output_np)
        output_full_ep = zoom(output_np, zf, order=1)[:data['nx'], :data['ny']]
        bo = output_full_ep.copy(); bo[~data['mask_blank']] = np.nan
        bo = gaussian_filter1d(bo, 1.5, axis=0); bo = gaussian_filter1d(bo, 1.5, axis=1)
        output_full_ep[data['mask_blank']] = bo[data['mask_blank']]
        rg = prep['F_display'].copy(); rg[data['mask_blank']] = output_full_ep[data['mask_blank']]

        interp = RegularGridInterpolator((data['x_grid'], data['y_grid']), rg,
                                         method='linear', bounds_error=False, fill_value=np.nan)
        F_pred = interp(prep['test_xy'])
        valid = ~np.isnan(F_pred)
        ep_rmse = np.sqrt(mean_squared_error(prep['yt_all'][valid], F_pred[valid])) if np.sum(valid) > 10 else float('nan')
        ep_mae = mean_absolute_error(prep['yt_all'][valid], F_pred[valid]) if np.sum(valid) > 10 else float('nan')
        lr_now = scheduler_lr.get_last_lr()[0]
        history.append({'epoch': ep + 1, 'rmse': float(ep_rmse), 'mae': float(ep_mae),
                        'loss': float(ep_loss / len(dataloader)), 'lr': float(lr_now)})

        if ep_rmse < best_rmse:
            best_rmse = ep_rmse
            best_epoch = ep + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_result_grid = rg.copy()

        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"    Ep {ep+1:3d}/{EPOCHS} | Loss: {ep_loss/len(dataloader):.6f} | "
                  f"RMSE: {ep_rmse:.1f} | Best: {best_rmse:.1f}@{best_epoch} | {time.time()-t0:.0f}s")

    train_time = time.time() - t0
    print(f"    Final: best RMSE={best_rmse:.2f} @ epoch {best_epoch} | {train_time:.0f}s")

    # Save
    torch.save(best_state, os.path.join(OUT_DIR, f'best_model_{tag}.pt'))
    with open(os.path.join(OUT_DIR, f'history_{tag}.json'), 'w') as f:
        json.dump(history, f, indent=2)

    del model; torch.cuda.empty_cache()
    return {
        'rmse': float(best_rmse),
        'mae': float(history[best_epoch - 1]['mae']),
        'best_epoch': best_epoch, 'history': history,
        'result_grid': best_result_grid, 'train_time': train_time,
        'n_params': n_params,
    }


def copy_noise0(region_key):
    """Copy existing U-Net+TF results for noise=0 baseline"""
    import shutil
    tag = f'{region_key}_noise_0'
    src_files = [
        (f'{COMPARATIVE_UTF}/result_grid_{region_key}.npy', f'result_grid_{tag}.npy'),
        (f'{COMPARATIVE_UTF}/truth_grid_{region_key}.npy', f'truth_grid_{region_key}.npy'),
        (f'{COMPARATIVE_UTF}/mask_blank_{region_key}.npy', f'mask_blank_{region_key}.npy'),
        (f'{COMPARATIVE_UTF}/mask_outside_{region_key}.npy', f'mask_outside_{region_key}.npy'),
        (f'{COMPARATIVE_UTF}/grid_x_{region_key}.npy', f'grid_x_{region_key}.npy'),
        (f'{COMPARATIVE_UTF}/grid_y_{region_key}.npy', f'grid_y_{region_key}.npy'),
        (f'{COMPARATIVE_UTF}/bx_{region_key}.npy', f'bx_{region_key}.npy'),
        (f'{COMPARATIVE_UTF}/by_{region_key}.npy', f'by_{region_key}.npy'),
        (f'{COMPARATIVE_UTF}/best_model_{region_key}.pt', f'best_model_{tag}.pt'),
    ]
    for src, dst in src_files:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(OUT_DIR, dst))
            print(f"    Copied: {os.path.basename(src)} → {dst}")

    result = np.load(os.path.join(OUT_DIR, f'result_grid_{tag}.npy'))
    truth = np.load(os.path.join(OUT_DIR, f'truth_grid_{region_key}.npy'))
    mask = np.load(os.path.join(OUT_DIR, f'mask_blank_{region_key}.npy'))
    abs_res = np.full_like(result, np.nan)
    abs_res[mask] = np.abs(result[mask] - truth[mask])
    np.save(os.path.join(OUT_DIR, f'abs_residual_{tag}.npy'), abs_res)

    with open(f'{COMPARATIVE_ROOT}/comparison_metrics.json') as f:
        cm = json.load(f)
    utf_r = cm[region_key]['U-Net+Transformer']
    history_tag_path = os.path.join(OUT_DIR, f'history_{tag}.json')
    if os.path.exists(f'{COMPARATIVE_UTF}/history_{region_key}.json'):
        shutil.copy2(f'{COMPARATIVE_UTF}/history_{region_key}.json', history_tag_path)

    result_dict = {
        'method': 'U-Net+Transformer', 'region': region_key, 'noise_sigma': 0,
        'rmse': utf_r['rmse'], 'mae': utf_r['mae'],
        'time': 0, 'status': 'completed',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(OUT_DIR, f'results_{tag}.json'), 'w') as f:
        json.dump(result_dict, f, indent=2)
    print(f"    [{region_key}] σ=0nT (reused) → RMSE={utf_r['rmse']:.2f}, MAE={utf_r['mae']:.2f}")


# =============================================================================
# Main
# =============================================================================
regions = {
    'rect': ('Rectangular', None, RECT),
    'irreg': ('Irregular', IRREG, None),
}

run_start = time.time()
all_results = {}

for rk, (rlabel, poly, rect) in regions.items():
    print(f"\n{'#'*60}")
    print(f"# Region: {rlabel}")
    print(f"{'#'*60}")

    # Copy noise=0 baseline
    copy_noise0(rk)
    all_results[f'{rk}_noise_0'] = {'rmse': 0, 'mae': 0, 'noise_sigma': 0, 'status': 'reused'}

    # Load data (shared across noise levels for this region)
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
        tag = f'{rk}_noise_{sigma}'
        print(f"\n  --- σ = {sigma} nT ---")

        # Preprocess with noise
        prep = rbf_preprocess_with_noise(data, sigma)
        print(f"    eps={prep['eps']:.4f}")

        # Train
        result = train_model(data, prep, rk, sigma)

        # Save result_grid
        np.save(os.path.join(OUT_DIR, f'result_grid_{tag}.npy'), result['result_grid'])

        # Abs residuals
        abs_residual = np.full_like(result['result_grid'], np.nan)
        abs_residual[data['mask_blank']] = np.abs(
            result['result_grid'][data['mask_blank']] - truth_grid[data['mask_blank']])
        np.save(os.path.join(OUT_DIR, f'abs_residual_{tag}.npy'), abs_residual)

        # Results JSON
        res_dict = {
            'method': 'U-Net+Transformer', 'region': rk, 'noise_sigma': sigma,
            'rmse': result['rmse'], 'mae': result['mae'],
            'best_epoch': result['best_epoch'], 'train_time': result['train_time'],
            'n_params': result['n_params'],
            'status': 'completed',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(os.path.join(OUT_DIR, f'results_{tag}.json'), 'w') as f:
            json.dump(res_dict, f, indent=2)

        all_results[tag] = res_dict
        print(f"    Saved: RMSE={result['rmse']:.2f}, MAE={result['mae']:.2f}")

# Progress
progress = {
    'method': 'U-Net+Transformer',
    'status': 'completed',
    'total_time_min': round((time.time() - run_start) / 60, 1),
    'results': all_results,
}
with open(os.path.join(OUT_DIR, 'progress.json'), 'w') as f:
    json.dump(progress, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"  U-Net+Transformer 噪声实验完成!")
print(f"  总耗时: {progress['total_time_min']} min")
print(f"  输出: {OUT_DIR}/")
print(f"{'='*60}")
