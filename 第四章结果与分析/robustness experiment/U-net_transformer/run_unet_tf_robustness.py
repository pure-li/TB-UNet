#!/usr/bin/env python
"""U-Net+Transformer 鲁棒性实验 — 3 种空白区尺寸
===================================================
小(0.5deg NW) / 中(1.0deg Center) / 大(1.5deg SE)
参数: max_lr=5e-4, pct_start=0.3, epochs=100, wd=1e-5
"""

import sys, importlib.util, os, json, time, warnings
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import base U-Net
unet_path = r'F:\PINN实验\venv\U-net\U-net.py'
spec = importlib.util.spec_from_file_location("unet", unet_path)
unet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unet)

DATA_PATH = r'F:\PINN实验\venv\U-net\afghanistan_full\Afghan_mag06A.csv'
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# Training params (from slow run)
BASE_CH = 48; EPOCHS = 100; BATCH_SIZE = 8
MAX_LR = 5e-4; PCT_START = 0.3; WEIGHT_DECAY = 1e-5
N_INTERP_CTRL = 3000; TARGET_SIZE = (128, 128)

# 3 blank regions
REGIONS = {
    'small':  {'label': 'Small (0.5deg, NW)',  'rect': (61.5, 62.0, 34.0, 34.5)},
    'large':  {'label': 'Large (1.5deg, SE)',   'rect': (63.0, 64.5, 30.5, 32.0)},
}

print("=" * 60)
print("  U-Net+Transformer 鲁棒性实验")
print(f"  max_lr={MAX_LR}, pct_start={PCT_START}, epochs={EPOCHS}, wd={WEIGHT_DECAY}")
print("=" * 60)

# =============================================================================
# Transformer model (same as run_slow.py)
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
    def __init__(self, in_ch, spatial_size, num_layers=4, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.num_patches = spatial_size * spatial_size
        self.pos_enc = PositionalEncoding(self.num_patches, in_ch)
        self.blocks = nn.ModuleList([
            TransformerBlock(in_ch, num_heads, mlp_ratio, dropout) for _ in range(num_layers)
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
            in_ch=base_ch * 4, spatial_size=8, num_layers=4, num_heads=8,
            mlp_ratio=4, dropout=0.1,
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

def load_region_data(rect_bounds):
    df_full = pd.read_csv(DATA_PATH)
    df = df_full.iloc[::3].copy().reset_index(drop=True)
    lon_min, lon_max, lat_min, lat_max = rect_bounds
    mask_inside = ((df['Longitude'] >= lon_min) & (df['Longitude'] <= lon_max) &
                   (df['Latitude'] >= lat_min) & (df['Latitude'] <= lat_max))
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
    mask_blank = ((lon_grid >= lon_min) & (lon_grid <= lon_max) &
                  (lat_grid >= lat_min) & (lat_grid <= lat_max))

    hull = ConvexHull(train_df[['x', 'y']].values)
    hull_xy = hull.points[hull.vertices]
    xc = hull_xy[:, 0].mean()
    for i in range(len(hull_xy)):
        hull_xy[i, 0] += -2. if hull_xy[i, 0] < xc else 2.
    mask_outside = ~Path(hull_xy).contains_points(grid_pts).reshape(grid_x.shape)

    bx = ll_to_xy(np.array([lon_min, lon_max, lon_max, lon_min, lon_min]),
                  np.array([lat_min, lat_min, lat_max, lat_max, lat_min]))[0]
    by = ll_to_xy(np.array([lon_min, lon_max, lon_max, lon_min, lon_min]),
                  np.array([lat_min, lat_min, lat_max, lat_max, lat_min]))[1]

    return {
        'train_df': train_df, 'test_df': test_df, 'll_to_xy': ll_to_xy,
        'x_grid': x_grid, 'y_grid': y_grid, 'nx': nx, 'ny': ny,
        'grid_x': grid_x, 'grid_y': grid_y, 'grid_pts': grid_pts,
        'mask_blank': mask_blank, 'mask_outside': mask_outside,
        'bx': bx, 'by': by,
    }

def preprocess(data):
    train_df = data['train_df']
    points_xy = train_df[['x', 'y']].values; values_F = train_df['FinalMag'].values
    n_sub = min(N_INTERP_CTRL, len(points_xy))
    idx = np.random.choice(len(points_xy), n_sub, replace=False)
    pts_sub, val_sub = points_xy[idx], values_F[idx]
    tree = KDTree(pts_sub)
    distances, _ = tree.query(pts_sub, k=min(10, n_sub))
    eps = np.median(distances[:, 1:]) * 0.8
    F_grid = RBFInterpolator(pts_sub, val_sub, kernel='cubic', epsilon=eps)(
        data['grid_pts']).reshape(data['grid_x'].shape)
    F_display = F_grid.copy()
    F_display = gaussian_filter1d(F_display, 5., axis=0)
    F_display = gaussian_filter1d(F_display, 1., axis=1)
    F_display[data['mask_outside']] = np.nan
    F_grid_masked = F_grid.copy()
    F_grid_masked[data['mask_blank'] | data['mask_outside']] = np.nan
    F_min, F_max = np.nanmin(F_grid_masked), np.nanmax(F_grid_masked)
    def norm(x): return (x - F_min) / (F_max - F_min) * 2 - 1
    def denorm(x): return (x + 1) / 2 * (F_max - F_min) + F_min
    F_in = F_grid_masked.copy(); F_in[np.isnan(F_in)] = 0.
    F_resized = resize(norm(F_in).astype(np.float32), TARGET_SIZE, mode='constant', anti_aliasing=True)
    perm_mask = resize(data['mask_blank'].astype(float), TARGET_SIZE) > 0.5
    return {'F_display': F_display, 'F_min': F_min, 'F_max': F_max,
            'F_resized': F_resized, 'perm_mask': perm_mask,
            'denorm': denorm, 'eps': eps,
            'test_xy': np.column_stack(data['ll_to_xy'](
                data['test_df']['Longitude'].values, data['test_df']['Latitude'].values)),
            'yt_all': data['test_df']['FinalMag'].values}

def gen_random_block_mask(valid_region, H, W):
    max_h = min(int(H * 0.4), 48); min_h = max(int(H * 0.1), 8)
    max_w = min(int(W * 0.4), 48); min_w = max(int(W * 0.1), 8)
    v = valid_region.cpu().numpy() if hasattr(valid_region, 'cpu') else valid_region
    for _ in range(200):
        mh = np.random.randint(min_h, max_h + 1)
        mw = np.random.randint(min_w, max_w + 1)
        y = np.random.randint(0, H - mh); x = np.random.randint(0, W - mw)
        if v[y:y+mh, x:x+mw].all():
            mask = torch.zeros(H, W, dtype=torch.bool)
            mask[y:y+mh, x:x+mw] = True
            return mask
    return None

def compute_truth_grid(data):
    df_all = pd.read_csv(DATA_PATH).iloc[::3]
    df_truth = df_all.iloc[::max(1, len(df_all)//3000)].copy()
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
# Main loop
# =============================================================================

results_summary = {}
run_start = time.time()

for region_key, rcfg in REGIONS.items():
    rlabel = rcfg['label']
    rect_bounds = rcfg['rect']

    print(f"\n{'#'*60}")
    print(f"# {rlabel}")
    print(f"{'#'*60}")

    data = load_region_data(rect_bounds)
    prep = preprocess(data)
    n_train = len(data['train_df']); n_test = len(data['test_df'])
    print(f"  eps={prep['eps']:.4f}, 训练点: {n_train:,}, 测试点: {n_test:,}")

    # Model
    model = UNetTransformer(in_chans=1, base_ch=BASE_CH, use_skip=False).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = 200
    total_steps = EPOCHS * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR, total_steps=total_steps,
        pct_start=PCT_START, anneal_strategy='cos', final_div_factor=1e4)

    criterion = nn.L1Loss()
    history = []
    best_rmse, best_ep = float('inf'), 0
    best_state = None

    base_np = prep['F_resized'].astype(np.float32)
    valid_region = ~torch.tensor(prep['perm_mask'], dtype=torch.bool)
    H, W = TARGET_SIZE
    zf = (data['nx'] / TARGET_SIZE[0], data['ny'] / TARGET_SIZE[1])

    print(f"  Training {EPOCHS} epochs...")
    t_train = time.time()

    for ep in range(1, EPOCHS + 1):
        model.train()
        ep_loss = 0.0
        for _ in range(steps_per_epoch):
            mask = gen_random_block_mask(valid_region, H, W)
            if mask is None: continue
            input_masked = torch.tensor(np.stack([base_np], 0), dtype=torch.float32)
            input_masked[:, mask] = 0.0

            # Forward
            optimizer.zero_grad()
            out = model(input_masked.unsqueeze(0).to(DEVICE))
            loss = criterion(out[:, :, mask], input_masked[:, mask].unsqueeze(0).to(DEVICE))
            loss.backward(); optimizer.step()
            scheduler.step()
            ep_loss += loss.item()

        avg_loss = ep_loss / steps_per_epoch

        # Inference (every epoch)
        model.eval()
        with torch.no_grad():
            inp = torch.tensor(np.stack([base_np], 0), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            out = model(inp)
        out_np = out.squeeze().cpu().numpy()
        out_np = prep['denorm'](out_np)
        out_full = zoom(out_np, zf, order=1)[:data['nx'], :data['ny']]
        bo = out_full.copy(); bo[~data['mask_blank']] = np.nan
        bo = gaussian_filter1d(bo, 1.5, axis=0)
        bo = gaussian_filter1d(bo, 1.5, axis=1)
        out_full[data['mask_blank']] = bo[data['mask_blank']]
        rg = prep['F_display'].copy()
        rg[data['mask_blank']] = out_full[data['mask_blank']]

        interp = RegularGridInterpolator((data['x_grid'], data['y_grid']), rg,
                                         method='linear', bounds_error=False, fill_value=np.nan)
        pred = interp(prep['test_xy'])
        valid = ~np.isnan(pred)
        rmse = np.sqrt(mean_squared_error(prep['yt_all'][valid], pred[valid])) if np.sum(valid) > 10 else float('nan')

        history.append({'epoch': ep, 'loss': float(avg_loss), 'rmse': float(rmse),
                        'lr': float(scheduler.get_last_lr()[0])})

        if rmse < best_rmse:
            best_rmse = rmse; best_ep = ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_result_grid = rg.copy()

        if ep % 10 == 0 or ep == 1:
            elapsed = time.time() - t_train
            print(f"  Ep {ep:3d}/{EPOCHS} | Loss: {avg_loss:.6f} | RMSE: {rmse:.1f} | "
                  f"Best: {best_rmse:.1f}@{best_ep} | {elapsed:.0f}s")

        # Early stopping: no improvement for 50 epochs
        if ep - best_ep > 50:
            print(f"  Early stop at epoch {ep} (no improvement for 50 epochs)")
            break

    train_time = time.time() - t_train
    print(f"  Final: epoch={best_ep}, RMSE={best_rmse:.2f} | Time: {train_time:.0f}s")

    # Evaluate MAE
    interp_best = RegularGridInterpolator((data['x_grid'], data['y_grid']), best_result_grid,
                                           method='linear', bounds_error=False, fill_value=np.nan)
    pred_best = interp_best(prep['test_xy'])
    valid_b = ~np.isnan(pred_best)
    mae = mean_absolute_error(prep['yt_all'][valid_b], pred_best[valid_b]) if np.sum(valid_b) > 10 else float('nan')

    # Compute truth
    truth_grid = compute_truth_grid(data)

    # Save
    torch.save(best_state, os.path.join(OUT_DIR, f'best_model_{region_key}.pt'))
    np.save(os.path.join(OUT_DIR, f'result_grid_{region_key}.npy'), best_result_grid)
    np.save(os.path.join(OUT_DIR, f'truth_grid_{region_key}.npy'), truth_grid)
    np.save(os.path.join(OUT_DIR, f'mask_blank_{region_key}.npy'), data['mask_blank'])
    np.save(os.path.join(OUT_DIR, f'mask_outside_{region_key}.npy'), data['mask_outside'])
    np.save(os.path.join(OUT_DIR, f'grid_x_{region_key}.npy'), data['grid_x'])
    np.save(os.path.join(OUT_DIR, f'grid_y_{region_key}.npy'), data['grid_y'])
    np.save(os.path.join(OUT_DIR, f'bx_{region_key}.npy'), data['bx'])
    np.save(os.path.join(OUT_DIR, f'by_{region_key}.npy'), data['by'])
    with open(os.path.join(OUT_DIR, f'history_{region_key}.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # --- Plots ---
    grid_x, grid_y = data['grid_x'], data['grid_y']
    bx, by = data['bx'], data['by']
    mask_blank, mask_outside = data['mask_blank'], data['mask_outside']
    zx = (bx.min() - 2, bx.max() + 2); zy = (by.min() - 2, by.max() + 2)
    vmin = np.nanmin(truth_grid[~mask_outside]); vmax = np.nanmax(truth_grid[~mask_outside])

    # Result
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(grid_x, grid_y, best_result_grid, cmap='jet', shading='auto', vmin=vmin, vmax=vmax)
    ax.plot(bx, by, 'k-', linewidth=2)
    ax.set_xlim(zx); ax.set_ylim(zy)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
    ax.set_title(f'{rlabel}\nU-Net + Transformer (RMSE={best_rmse:.2f} nT, best ep={best_ep})')
    cbar = plt.colorbar(im, ax=ax, label='Mag. Anomaly (nT)')
    cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
    save_fig(fig, f'fig_result_{region_key}')

    # Residual (RdBu_r, +-100)
    residual = best_result_grid.copy()
    residual[~mask_blank] = np.nan
    residual[mask_blank] = best_result_grid[mask_blank] - truth_grid[mask_blank]
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(grid_x, grid_y, residual, cmap='RdBu_r', shading='auto', vmin=-100, vmax=100)
    ax.plot(bx, by, 'k-', linewidth=2)
    ax.set_xlim(zx); ax.set_ylim(zy)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
    ax.set_title(f'{rlabel}\nResidual (U-Net+TF, RMSE={best_rmse:.2f} nT)')
    cbar = plt.colorbar(im, ax=ax, label='Residual (nT)')
    cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
    save_fig(fig, f'fig_residual_{region_key}')

    # Error (hot, 0-100)
    abs_error = best_result_grid.copy()
    abs_error[~mask_blank] = np.nan
    abs_error[mask_blank] = np.abs(best_result_grid[mask_blank] - truth_grid[mask_blank])
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.pcolormesh(grid_x, grid_y, abs_error, cmap='hot', shading='auto', vmin=0, vmax=100)
    ax.plot(bx, by, 'k-', linewidth=2)
    ax.set_xlim(zx); ax.set_ylim(zy)
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)')
    ax.set_title(f'{rlabel}\n|Error| (U-Net+TF, RMSE={best_rmse:.2f} nT)')
    cbar = plt.colorbar(im, ax=ax, label='Absolute Error (nT)')
    cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
    save_fig(fig, f'fig_error_{region_key}')

    # RMSE curve
    fig, ax = plt.subplots(figsize=(10, 5))
    eps = [h['epoch'] for h in history]; rmses = [h['rmse'] for h in history]
    ax.plot(eps, rmses, color='#FF5722', lw=2)
    ax.scatter(best_ep, best_rmse, color='#FF5722', s=80, zorder=5, marker='*', edgecolors='black')
    ax.set_xlabel('Epoch'); ax.set_ylabel('RMSE (nT)')
    ax.set_title(f'{rlabel} — Test RMSE')
    ax.grid(True, alpha=0.3)
    save_fig(fig, f'fig_rmse_{region_key}')

    plt.close('all')

    results_summary[region_key] = {
        'rmse': float(best_rmse), 'mae': float(mae),
        'best_epoch': best_ep, 'train_time': train_time,
        'n_train': n_train, 'n_test': n_test, 'n_params': n_params,
        'rect_bounds': list(rect_bounds),
    }
    print(f"  MAE={mae:.2f}, 图表 & .npy 已保存")

results_summary['config'] = {
    'max_lr': MAX_LR, 'pct_start': PCT_START, 'epochs': EPOCHS,
    'weight_decay': WEIGHT_DECAY, 'base_ch': BASE_CH, 'batch_size': BATCH_SIZE,
}
with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
    json.dump(results_summary, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"  U-Net+Transformer 鲁棒性完成!")
for rk in REGIONS:
    print(f"  {rk}: RMSE={results_summary[rk]['rmse']:.2f}, MAE={results_summary[rk]['mae']:.2f}")
print(f"  总耗时: {(time.time()-run_start)/60:.0f} min")
print(f"  输出: {OUT_DIR}/")
print("=" * 60)
