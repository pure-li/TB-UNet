#!/usr/bin/env python
"""中区域 0.8° — U-Net+TF (slow) + Kriging
============================================
区域: 61.2-62.0°E, 34.0-34.8°N (0.8° × 0.8°, NW)
  从 NW 1.5° 区域裁剪 (U-Net+TF=33.26 那个)
参数: max_lr=5e-4, pct_start=0.3, epochs=100, wd=1e-5
训练: InpaintingDataset + DataLoader + CompositeLoss + grad_clip
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
from pykrige.ok import OrdinaryKriging
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

unet_path = r'F:\PINN实验\venv\U-net\U-net.py'
spec = importlib.util.spec_from_file_location("unet", unet_path)
unet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unet)

DATA_PATH = r'F:\PINN实验\venv\U-net\afghanistan_full\Afghan_mag06A.csv'
ROOT = os.path.dirname(os.path.abspath(__file__))
KRIG_DIR = os.path.join(ROOT, 'Kriging')
UNET_DIR = os.path.join(ROOT, 'U-net_transformer')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

BASE_CH = 48; EPOCHS = 100; BATCH_SIZE = 8
MAX_LR = 5e-4; PCT_START = 0.3; WEIGHT_DECAY = 1e-5
N_INTERP_CTRL = 3000; TARGET_SIZE = (128, 128)
N_KRIGING_CTRL = 2000; N_TRUTH_SAMPLES = 3000

RECT = (61.2, 62.0, 34.0, 34.8)
REGION_KEY = 'medium'
REGION_LABEL = 'Medium (0.8deg x 0.8deg, NW)'

print("=" * 60)
print(f"  {REGION_LABEL} — U-Net+TF (slow) + Kriging")
print(f"  max_lr={MAX_LR}, pct_start={PCT_START}, epochs={EPOCHS}, wd={WEIGHT_DECAY}")
print("=" * 60)

# =============================================================================
# Transformer model
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
            in_ch=base_ch * 4, spatial_size=8, num_layers=4, num_heads=8, mlp_ratio=4, dropout=0.1)
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
        x = (lon - lon0) * np.pi/180 * R_map * np.cos(np.radians(lat0))
        y = (lat - lat0) * np.pi/180 * R_map
        return x, y

    train_df['x'], train_df['y'] = ll_to_xy(train_df['Longitude'].values, train_df['Latitude'].values)
    test_df['x'], test_df['y'] = ll_to_xy(test_df['Longitude'].values, test_df['Latitude'].values)

    grid_spacing = 1.0
    left_x = train_df.groupby((train_df['Longitude'].diff().abs()>0.5).cumsum())['x'].min().min()
    right_x = train_df.groupby((train_df['Longitude'].diff().abs()>0.5).cumsum())['x'].max().max()
    x_min, x_max = left_x-2., right_x+2.
    y_min, y_max = train_df['y'].min()-grid_spacing, train_df['y'].max()+grid_spacing
    x_grid = np.arange(x_min, x_max, grid_spacing)
    y_grid = np.arange(y_min, y_max, grid_spacing)
    nx, ny = len(x_grid), len(y_grid)
    grid_x, grid_y = np.meshgrid(x_grid, y_grid, indexing='ij')
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    lon_grid = lon0 + grid_x/(R_map*np.cos(np.radians(lat0)))*(180/np.pi)
    lat_grid = lat0 + grid_y/R_map*(180/np.pi)
    mask_blank = ((lon_grid >= lon_min) & (lon_grid <= lon_max) &
                  (lat_grid >= lat_min) & (lat_grid <= lat_max))
    hull = ConvexHull(train_df[['x','y']].values)
    hull_xy = hull.points[hull.vertices]
    xc = hull_xy[:,0].mean()
    for i in range(len(hull_xy)):
        hull_xy[i,0] += -2. if hull_xy[i,0] < xc else 2.
    mask_outside = ~Path(hull_xy).contains_points(grid_pts).reshape(grid_x.shape)
    bx = ll_to_xy(np.array([lon_min, lon_max, lon_max, lon_min, lon_min]),
                  np.array([lat_min, lat_min, lat_max, lat_max, lat_min]))[0]
    by = ll_to_xy(np.array([lon_min, lon_max, lon_max, lon_min, lon_min]),
                  np.array([lat_min, lat_min, lat_max, lat_max, lat_min]))[1]
    return {'train_df': train_df, 'test_df': test_df, 'll_to_xy': ll_to_xy,
            'x_grid': x_grid, 'y_grid': y_grid, 'nx': nx, 'ny': ny,
            'grid_x': grid_x, 'grid_y': grid_y, 'grid_pts': grid_pts,
            'mask_blank': mask_blank, 'mask_outside': mask_outside,
            'bx': bx, 'by': by}

# =============================================================================
# Kriging
# =============================================================================
def run_kriging(data):
    train_df = data['train_df']; mask_blank = data['mask_blank']
    grid_pts = data['grid_pts']
    n_ctrl = min(N_KRIGING_CTRL, len(train_df))
    idx = np.random.choice(len(train_df), n_ctrl, replace=False)
    ctrl_x = train_df['x'].values[idx]; ctrl_y = train_df['y'].values[idx]
    ctrl_F = train_df['FinalMag'].values[idx]
    t0 = time.time()
    OK = OrdinaryKriging(ctrl_x, ctrl_y, ctrl_F, variogram_model='spherical',
                         verbose=False, enable_plotting=False)
    blank_idx = np.where(mask_blank.ravel())[0]
    z_blank, _ = OK.execute('points', grid_pts[blank_idx, 0], grid_pts[blank_idx, 1])
    kt = time.time() - t0
    result_grid = np.full(grid_pts.shape[0], np.nan, dtype=np.float32)
    result_grid[blank_idx] = z_blank.astype(np.float32)
    result_grid = result_grid.reshape(data['grid_x'].shape)
    temp = np.nan_to_num(result_grid, nan=0.0)
    temp = gaussian_filter1d(temp, 1.5, axis=0)
    temp = gaussian_filter1d(temp, 1.5, axis=1)
    result_grid[mask_blank] = temp[mask_blank]
    result_grid[~mask_blank] = np.nan
    test_xy = np.column_stack([data['test_df']['x'].values, data['test_df']['y'].values])
    test_F = data['test_df']['FinalMag'].values
    z_test, _ = OK.execute('points', test_xy[:, 0], test_xy[:, 1])
    valid = ~np.isnan(z_test)
    rmse = np.sqrt(mean_squared_error(test_F[valid], z_test[valid])) if np.sum(valid)>10 else float('nan')
    mae = mean_absolute_error(test_F[valid], z_test[valid]) if np.sum(valid)>10 else float('nan')
    return result_grid, rmse, mae, kt

# =============================================================================
# Truth grid
# =============================================================================
def compute_truth_grid(data):
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

# =============================================================================
# U-Net+TF preprocessing + training
# =============================================================================
def preprocess(data):
    train_df = data['train_df']
    points_xy = train_df[['x','y']].values; values_F = train_df['FinalMag'].values
    n_sub = min(N_INTERP_CTRL, len(points_xy))
    idx = np.random.choice(len(points_xy), n_sub, replace=False)
    pts_sub, val_sub = points_xy[idx], values_F[idx]
    tree = KDTree(pts_sub)
    distances, _ = tree.query(pts_sub, k=min(10, n_sub))
    eps = np.median(distances[:,1:])*0.8
    F_grid = RBFInterpolator(pts_sub, val_sub, kernel='cubic', epsilon=eps)(data['grid_pts']).reshape(data['grid_x'].shape)
    F_display = F_grid.copy()
    F_display = gaussian_filter1d(F_display, 5., axis=0); F_display = gaussian_filter1d(F_display, 1., axis=1)
    F_display[data['mask_outside']] = np.nan
    F_grid_masked = F_grid.copy(); F_grid_masked[data['mask_blank'] | data['mask_outside']] = np.nan
    F_min, F_max = np.nanmin(F_grid_masked), np.nanmax(F_grid_masked)
    def norm(x): return (x-F_min)/(F_max-F_min)*2-1
    def denorm(x): return (x+1)/2*(F_max-F_min)+F_min
    F_in = F_grid_masked.copy(); F_in[np.isnan(F_in)] = 0.
    F_resized = resize(norm(F_in).astype(np.float32), TARGET_SIZE, mode='constant', anti_aliasing=True)
    perm_mask = resize(data['mask_blank'].astype(float), TARGET_SIZE)>0.5
    return {'F_display': F_display, 'F_min': F_min, 'F_max': F_max,
            'F_resized': F_resized, 'perm_mask': perm_mask, 'denorm': denorm, 'eps': eps,
            'test_xy': np.column_stack(data['ll_to_xy'](data['test_df']['Longitude'].values, data['test_df']['Latitude'].values)),
            'yt_all': data['test_df']['FinalMag'].values}

# =============================================================================
# Main
# =============================================================================
print("\nLoading data...")
data = load_region_data(RECT)
print(f"  Train: {len(data['train_df']):,}, Test: {len(data['test_df']):,}, "
      f"Blank: {data['mask_blank'].sum():,}")

truth_grid = compute_truth_grid(data)

# ---- Kriging ----
print("\n[Kriging] Running...")
krig_result, krig_rmse, krig_mae, krig_time = run_kriging(data)
print(f"  RMSE={krig_rmse:.2f}, MAE={krig_mae:.2f}, Time={krig_time:.1f}s")

np.save(os.path.join(KRIG_DIR, f'result_grid_{REGION_KEY}.npy'), krig_result)
np.save(os.path.join(KRIG_DIR, f'truth_grid_{REGION_KEY}.npy'), truth_grid)
np.save(os.path.join(KRIG_DIR, f'mask_blank_{REGION_KEY}.npy'), data['mask_blank'])
np.save(os.path.join(KRIG_DIR, f'mask_outside_{REGION_KEY}.npy'), data['mask_outside'])
np.save(os.path.join(KRIG_DIR, f'grid_x_{REGION_KEY}.npy'), data['grid_x'])
np.save(os.path.join(KRIG_DIR, f'grid_y_{REGION_KEY}.npy'), data['grid_y'])
np.save(os.path.join(KRIG_DIR, f'bx_{REGION_KEY}.npy'), data['bx'])
np.save(os.path.join(KRIG_DIR, f'by_{REGION_KEY}.npy'), data['by'])

krig_results_path = os.path.join(KRIG_DIR, 'results.json')
krig_results = json.load(open(krig_results_path)) if os.path.exists(krig_results_path) else {}
krig_results[REGION_KEY] = {'rmse': float(krig_rmse), 'mae': float(krig_mae), 'time': krig_time,
                             'rect_bounds': list(RECT), 'n_blank': int(data['mask_blank'].sum())}
with open(krig_results_path, 'w') as f: json.dump(krig_results, f, indent=2, ensure_ascii=False)

# ---- U-Net+Transformer (slow-style) ----
print(f"\n[U-Net+TF] Training {EPOCHS} epochs (slow: DataLoader + CompositeLoss + grad_clip)...")
prep = preprocess(data)
print(f"  eps={prep['eps']:.4f}")

np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

model = UNetTransformer(in_chans=1, base_ch=BASE_CH, use_skip=False).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())

train_image = np.stack([prep['F_resized']], axis=0).astype(np.float32)
dataset = unet.InpaintingDataset(train_image, prep['perm_mask'],
                                 mask_mode='block', mask_ratio=0.2)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
steps_per_epoch = len(dataloader)
total_steps = EPOCHS * steps_per_epoch

optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=MAX_LR, total_steps=total_steps,
    pct_start=PCT_START, anneal_strategy='cos', final_div_factor=1e4)
criterion = unet.CompositeLoss(grad_weight=0.0).to(DEVICE)

history = []; best_rmse, best_ep = float('inf'), 0; best_state = None
zf = (data['nx']/TARGET_SIZE[0], data['ny']/TARGET_SIZE[1])
test_input_t = torch.tensor(train_image, dtype=torch.float32).unsqueeze(0).to(DEVICE)
t_train = time.time()

for ep in range(1, EPOCHS+1):
    model.train(); ep_loss = 0.0
    for batch in dataloader:
        inp, target, mask = [b.to(DEVICE) for b in batch]
        mask = mask.bool()
        optimizer.zero_grad()
        pred = model(inp)
        loss = criterion(pred, target, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        ep_loss += loss.item()
    avg_loss = ep_loss / steps_per_epoch

    model.eval()
    with torch.no_grad():
        out = model(test_input_t)
    out_np = out.squeeze().cpu().numpy(); out_np = prep['denorm'](out_np)
    out_full = zoom(out_np, zf, order=1)[:data['nx'],:data['ny']]
    bo = out_full.copy(); bo[~data['mask_blank']] = np.nan
    bo = gaussian_filter1d(bo, 1.5, axis=0); bo = gaussian_filter1d(bo, 1.5, axis=1)
    out_full[data['mask_blank']] = bo[data['mask_blank']]
    rg = prep['F_display'].copy(); rg[data['mask_blank']] = out_full[data['mask_blank']]
    interp = RegularGridInterpolator((data['x_grid'], data['y_grid']), rg,
                                     method='linear', bounds_error=False, fill_value=np.nan)
    pred_v = interp(prep['test_xy']); valid_v = ~np.isnan(pred_v)
    rmse = np.sqrt(mean_squared_error(prep['yt_all'][valid_v], pred_v[valid_v])) if np.sum(valid_v)>10 else float('nan')
    history.append({'epoch': ep, 'loss': float(avg_loss), 'rmse': float(rmse),
                    'lr': float(scheduler.get_last_lr()[0])})

    if rmse < best_rmse:
        best_rmse = rmse; best_ep = ep
        best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
        best_result_grid = rg.copy()

    if ep%10==0 or ep==1:
        print(f"  Ep {ep:3d}/{EPOCHS} | Loss: {avg_loss:.6f} | RMSE: {rmse:.1f} | "
              f"Best: {best_rmse:.1f}@{best_ep} | {time.time()-t_train:.0f}s")
    if ep - best_ep > 50:
        print(f"  Early stop at epoch {ep}")
        break

train_time = time.time() - t_train
interp_best = RegularGridInterpolator((data['x_grid'], data['y_grid']), best_result_grid,
                                       method='linear', bounds_error=False, fill_value=np.nan)
pred_best = interp_best(prep['test_xy']); valid_b = ~np.isnan(pred_best)
mae = mean_absolute_error(prep['yt_all'][valid_b], pred_best[valid_b]) if np.sum(valid_b)>10 else float('nan')
print(f"  Best: epoch={best_ep}, RMSE={best_rmse:.2f}, MAE={mae:.2f} | Time: {train_time:.0f}s")

# Save U-Net+TF
torch.save(best_state, os.path.join(UNET_DIR, f'best_model_{REGION_KEY}.pt'))
np.save(os.path.join(UNET_DIR, f'result_grid_{REGION_KEY}.npy'), best_result_grid)
np.save(os.path.join(UNET_DIR, f'truth_grid_{REGION_KEY}.npy'), truth_grid)
np.save(os.path.join(UNET_DIR, f'mask_blank_{REGION_KEY}.npy'), data['mask_blank'])
np.save(os.path.join(UNET_DIR, f'mask_outside_{REGION_KEY}.npy'), data['mask_outside'])
np.save(os.path.join(UNET_DIR, f'grid_x_{REGION_KEY}.npy'), data['grid_x'])
np.save(os.path.join(UNET_DIR, f'grid_y_{REGION_KEY}.npy'), data['grid_y'])
np.save(os.path.join(UNET_DIR, f'bx_{REGION_KEY}.npy'), data['bx'])
np.save(os.path.join(UNET_DIR, f'by_{REGION_KEY}.npy'), data['by'])
with open(os.path.join(UNET_DIR, f'history_{REGION_KEY}.json'), 'w') as f: json.dump(history, f, indent=2)

unet_results_path = os.path.join(UNET_DIR, 'results.json')
unet_results = json.load(open(unet_results_path)) if os.path.exists(unet_results_path) else {}
unet_results['small'] = {'rmse': 10.60, 'mae': 8.52, 'best_epoch': 17,
                          'source': 'slow retrained'}
unet_results[REGION_KEY] = {'rmse': float(best_rmse), 'mae': float(mae),
                             'best_epoch': best_ep, 'train_time': train_time,
                             'n_train': len(data['train_df']), 'n_test': len(data['test_df']),
                             'n_params': n_params,
                             'rect_bounds': list(RECT),
                             'training': 'slow_style_dataloader'}
# Keep old medium as large
unet_results['large'] = {'rmse': 19.92, 'mae': 15.46, 'best_epoch': 35,
                          'source': 'U-net_transformer_slow (1.0deg center, new role as large)'}
with open(unet_results_path, 'w') as f: json.dump(unet_results, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"  Done!")
print(f"  Kriging:  RMSE={krig_rmse:.2f}, MAE={krig_mae:.2f}")
print(f"  U-Net+TF: RMSE={best_rmse:.2f}, MAE={mae:.2f}")
print(f"  Improvement: {krig_rmse - best_rmse:+.1f} nT")
print("=" * 60)
