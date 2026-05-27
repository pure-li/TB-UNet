#!/usr/bin/env python
"""U-Net + Transformer slow convergence experiment
=============================================
Parameter adjustments: max_lr=5e-4, pct_start=0.3, epochs=150, wd=1e-5
Goal: more stable, slower convergence to reduce random fluctuations
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

unet_path = r'F:\PINN实验\venv\U-net\U-net.py'
spec = importlib.util.spec_from_file_location("unet", unet_path)
unet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unet)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# Configuration (slow convergence)
# =============================================================================
DATA_PATH = r'F:\PINN实验\venv\U-net\afghanistan_full\Afghan_mag06A.csv'
OUT_DIR   = r'F:\PINN实验\venv\U-net\U-net_transformer_slow'
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET_SIZE = (128, 128)
EPOCHS = 150
BATCH_SIZE = 8
BASE_CH = 48
SEED = 42
N_INTERP_CTRL = 3000

# Slow convergence parameters
MAX_LR = 5e-4          # was 1e-3, halved to 5e-4
PCT_START = 0.3        # was 0.1, warmup extended to 30% epochs
WEIGHT_DECAY = 1e-5    # was 1e-6, increased to 1e-5

USE_SKIP = False
GRAD_WEIGHT = 0.0
MASK_MODE = 'block'

TRANSFORMER_LAYERS = 4
TRANSFORMER_HEADS = 8
TRANSFORMER_MLP_RATIO = 4
TRANSFORMER_DROPOUT = 0.1

RECT_LON_MIN, RECT_LON_MAX = 62.0, 63.0
RECT_LAT_MIN, RECT_LAT_MAX = 32.5, 33.5

IRREGULAR_POLYGON = np.array([
    [62.05, 32.65], [62.30, 32.38], [62.70, 32.42], [62.95, 32.75],
    [63.08, 33.05], [62.88, 33.38], [62.48, 33.48], [62.15, 33.22], [61.98, 32.92],
])

print("=" * 60)
print("  U-Net + Transformer slow convergence experiment")
print(f"  max_lr={MAX_LR}, pct_start={PCT_START}, epochs={EPOCHS}, wd={WEIGHT_DECAY}")
print(f"  Rectangular + irregular blank regions")
print("=" * 60)

# =============================================================================
# Transformer module
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
    return {'train_df': train_df, 'test_df': test_df, 'll_to_xy': ll_to_xy,
            'x_grid': x_grid, 'y_grid': y_grid, 'nx': nx, 'ny': ny,
            'grid_x': grid_x, 'grid_y': grid_y, 'grid_pts': grid_pts,
            'mask_blank': mask_blank, 'mask_outside': mask_outside,
            'bx': bx, 'by': by}

# =============================================================================
# RBF preprocessing
# =============================================================================

def rbf_preprocess(data):
    train_df = data['train_df']
    points_xy = train_df[['x', 'y']].values
    values_F = train_df['FinalMag'].values
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
    F_min, F_max = np.nanmin(F_grid_masked), np.nanmax(F_grid_masked)
    def normalize(x): return (x - F_min) / (F_max - F_min) * 2 - 1
    def denorm(x): return (x + 1) / 2 * (F_max - F_min) + F_min
    F_input = F_grid_masked.copy(); F_input[np.isnan(F_input)] = 0.0
    F_resized = resize(normalize(F_input).astype(np.float32), TARGET_SIZE,
                       mode='constant', anti_aliasing=True)
    perm_mask_resized = resize(data['mask_blank'].astype(float), TARGET_SIZE) > 0.5
    test_xy = np.column_stack(data['ll_to_xy'](data['test_df']['Longitude'].values,
                                                data['test_df']['Latitude'].values))
    yt_all = data['test_df']['FinalMag'].values
    return {'F_display': F_display, 'F_min': F_min, 'F_max': F_max,
            'F_resized': F_resized, 'perm_mask_resized': perm_mask_resized,
            'test_xy': test_xy, 'yt_all': yt_all,
            'normalize': normalize, 'denorm': denorm, 'eps': eps}

# =============================================================================
# Training
# =============================================================================

def train_model(data, prep, region_key, region_label):
    print(f"\n[Train] U-Net+Transformer [{region_label}], epochs={EPOCHS}, lr={MAX_LR}, pct={PCT_START}")

    np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    model = UNetTransformer(in_chans=1, base_ch=BASE_CH, use_skip=USE_SKIP).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    criterion = unet.CompositeLoss(grad_weight=GRAD_WEIGHT).to(DEVICE)
    train_image = np.stack([prep['F_resized']], axis=0).astype(np.float32)
    dataset = unet.InpaintingDataset(train_image, prep['perm_mask_resized'],
                                     mask_mode=MASK_MODE, mask_ratio=0.2)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    steps_total = EPOCHS * len(dataloader)
    total_steps = EPOCHS * len(dataloader)
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    scheduler_lr = lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, total_steps=total_steps,
                                        pct_start=PCT_START, anneal_strategy='cos', final_div_factor=1e4)
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
        ep_mae  = mean_absolute_error(prep['yt_all'][valid], F_pred[valid]) if np.sum(valid) > 10 else float('nan')
        lr_now = scheduler_lr.get_last_lr()[0]
        history.append({'epoch': ep + 1, 'rmse': float(ep_rmse), 'mae': float(ep_mae),
                        'loss': float(ep_loss / len(dataloader)), 'lr': float(lr_now)})

        if ep_rmse < best_rmse:
            best_rmse = ep_rmse
            best_epoch = ep + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_result_grid = rg.copy()

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  Ep {ep+1:3d}/{EPOCHS} | Loss: {ep_loss/len(dataloader):.6f} | RMSE: {ep_rmse:.1f} | Best: {best_rmse:.1f}@{best_epoch} | LR: {lr_now:.2e}")

    train_time = time.time() - t0
    model.load_state_dict(best_state)

    # Save
    torch.save(best_state, os.path.join(OUT_DIR, f'best_model_{region_key}.pt'))
    np.save(os.path.join(OUT_DIR, f'result_grid_{region_key}.npy'), best_result_grid)
    with open(os.path.join(OUT_DIR, f'history_{region_key}.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"  Final: epoch={best_epoch}, RMSE={best_rmse:.2f}, MAE={history[best_epoch-1]['mae']:.2f} | Time: {train_time:.0f}s")

    del model; torch.cuda.empty_cache()
    return {'rmse': float(best_rmse), 'mae': float(history[best_epoch-1]['mae']),
            'best_epoch': best_epoch, 'history': history,
            'result_grid': best_result_grid, 'train_time': train_time, 'n_params': n_params}


# =============================================================================
# Plotting
# =============================================================================

def save_fig(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(OUT_DIR, f'{name}.{ext}'), dpi=300,
                    bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

def plot_figures(data, prep, result, region_key, region_label):
    mask_blank = data['mask_blank']
    mask_outside = data['mask_outside']
    bx, by = data['bx'], data['by']
    grid_x, grid_y = data['grid_x'], data['grid_y']
    result_grid = result['result_grid']
    history = result['history']

    zx = (bx.min() - 2, bx.max() + 2)
    zy = (by.min() - 2, by.max() + 2)

    # Ground truth grid (for residual/error maps)
    n_ctrl = 3000
    df_all = pd.read_csv(DATA_PATH).iloc[::3]
    df_truth = df_all.iloc[::max(1, len(df_all)//n_ctrl)].copy()
    pts_truth = np.column_stack(data['ll_to_xy'](df_truth['Longitude'].values, df_truth['Latitude'].values))
    val_truth = df_truth['FinalMag'].values
    truth_grid = RBFInterpolator(pts_truth, val_truth, kernel='thin_plate_spline')(
        data['grid_pts']).reshape(grid_x.shape)
    truth_grid = gaussian_filter1d(truth_grid, 2.0, axis=0)
    truth_grid = gaussian_filter1d(truth_grid, 1.0, axis=1)
    truth_grid[mask_outside] = np.nan
    vmin = np.nanmin(truth_grid[~mask_outside])
    vmax = np.nanmax(truth_grid[~mask_outside])

    tag = f'{region_key}'

    # Loss
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot([h['epoch'] for h in history], [h['loss'] for h in history], color='#4CAF50', lw=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title(f'{region_label} Slow — Loss (lr={MAX_LR}, pct={PCT_START})')
    ax.grid(True, alpha=0.3)
    save_fig(fig, f'fig_loss_{tag}')

    # RMSE
    fig, ax = plt.subplots(figsize=(10, 5))
    eps = [h['epoch'] for h in history]
    rmses = [h['rmse'] for h in history]
    ax.plot(eps, rmses, color='#4CAF50', lw=2)
    ax.scatter(result['best_epoch'], result['rmse'], color='#4CAF50', s=80, zorder=5, marker='*', edgecolors='black')
    ax.set_xlabel('Epoch'); ax.set_ylabel('RMSE (nT)')
    ax.set_title(f'{region_label} Slow — RMSE (best={result["rmse"]:.2f})')
    ax.grid(True, alpha=0.3)
    save_fig(fig, f'fig_rmse_{tag}')

    # LR schedule
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot([h['epoch'] for h in history], [h['lr'] for h in history], color='#4CAF50', lw=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('LR')
    ax.set_title(f'{region_label} Slow — LR Schedule')
    ax.grid(True, alpha=0.3)
    save_fig(fig, f'fig_lr_{tag}')

    # Result
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(grid_x, grid_y, result_grid, cmap='jet', shading='auto', vmin=vmin, vmax=vmax)
    ax.plot(bx, by, 'k-', linewidth=2)
    ax.set_xlim(zx); ax.set_ylim(zy)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
    ax.set_title(f'{region_label} Slow — Result (RMSE={result["rmse"]:.2f})')
    cbar = plt.colorbar(im, ax=ax, label='Mag. Anomaly (nT)')
    cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
    save_fig(fig, f'fig_result_{tag}')

    # Residual
    residual = result_grid.copy(); residual[~mask_blank] = np.nan
    residual[mask_blank] = result_grid[mask_blank] - truth_grid[mask_blank]
    res_max = max(abs(np.nanmin(residual[mask_blank])), abs(np.nanmax(residual[mask_blank])), 1.0)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(grid_x, grid_y, residual, cmap='RdBu_r', shading='auto', vmin=-res_max, vmax=res_max)
    ax.plot(bx, by, 'k-', linewidth=2)
    ax.set_xlim(zx); ax.set_ylim(zy)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
    ax.set_title(f'{region_label} Slow — Residual (RMSE={result["rmse"]:.2f})')
    cbar = plt.colorbar(im, ax=ax, label='Residual (nT)')
    cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
    save_fig(fig, f'fig_residual_{tag}')

    # Error
    abs_error = result_grid.copy(); abs_error[~mask_blank] = np.nan
    abs_error[mask_blank] = np.abs(result_grid[mask_blank] - truth_grid[mask_blank])
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(grid_x, grid_y, abs_error, cmap='hot', shading='auto',
                       vmin=0, vmax=np.nanmax(abs_error[mask_blank]) * 0.8)
    ax.plot(bx, by, 'k-', linewidth=2)
    ax.set_xlim(zx); ax.set_ylim(zy)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
    ax.set_title(f'{region_label} Slow — |Error| (RMSE={result["rmse"]:.2f})')
    cbar = plt.colorbar(im, ax=ax, label='Absolute Error (nT)')
    cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
    save_fig(fig, f'fig_error_{tag}')

    plt.close('all')


# =============================================================================
# Run
# =============================================================================

regions = {
    'rect':  ('Rectangular', None, (RECT_LON_MIN, RECT_LON_MAX, RECT_LAT_MIN, RECT_LAT_MAX)),
    'irreg': ('Irregular', IRREGULAR_POLYGON, None),
}

run_start = time.time()
summary = {}

for rk, (rlabel, poly, rect) in regions.items():
    print(f"\n{'#'*60}")
    print(f"# Region: {rlabel}")
    print(f"{'#'*60}")

    data = load_region_data(poly_vertices=poly, rect_bounds=rect)
    prep = rbf_preprocess(data)
    print(f"  eps={prep['eps']:.4f}, Train: {len(data['train_df']):,}, Test: {len(data['test_df']):,}")

    # Save shared data
    np.save(os.path.join(OUT_DIR, f'mask_blank_{rk}.npy'), data['mask_blank'])
    np.save(os.path.join(OUT_DIR, f'mask_outside_{rk}.npy'), data['mask_outside'])
    np.save(os.path.join(OUT_DIR, f'grid_x_{rk}.npy'), data['grid_x'])
    np.save(os.path.join(OUT_DIR, f'grid_y_{rk}.npy'), data['grid_y'])
    np.save(os.path.join(OUT_DIR, f'bx_{rk}.npy'), data['bx'])
    np.save(os.path.join(OUT_DIR, f'by_{rk}.npy'), data['by'])

    result = train_model(data, prep, rk, rlabel)
    plot_figures(data, prep, result, rk, rlabel)

# Summary
print(f"\n{'='*60}")
print(f"  Slow convergence experiment summary")
print(f"  max_lr={MAX_LR}, pct_start={PCT_START}, epochs={EPOCHS}")
print(f"{'='*60}")
print(f"  {'Region':<8s} {'RMSE':>8s} {'MAE':>8s} {'Epoch':>8s} {'Time':>8s}")
for rk, (rlabel, _, _) in regions.items():
    h = json.load(open(os.path.join(OUT_DIR, f'history_{rk}.json')))
    best = min(h, key=lambda x: x['rmse'])
    summary[rk] = {'rmse': best['rmse'], 'mae': best['mae'], 'best_epoch': best['epoch']}
    print(f"  {rlabel:<8s} {best['rmse']:8.2f} {best['mae']:8.2f} {best['epoch']:>8d}")

total_time = time.time() - run_start
summary['config'] = {'max_lr': MAX_LR, 'pct_start': PCT_START, 'epochs': EPOCHS,
                     'weight_decay': WEIGHT_DECAY, 'base_ch': BASE_CH, 'batch_size': BATCH_SIZE,
                     'transformer_layers': TRANSFORMER_LAYERS, 'transformer_heads': TRANSFORMER_HEADS}
summary['total_time_min'] = round(total_time / 60, 1)
with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\n  Total time: {(time.time()-run_start)/60:.0f} min")
print("=" * 60)
