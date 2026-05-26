#!/usr/bin/env python
"""消融实验 — Pure U-Net (无跳跃连接, 无Transformer)
=====================================================
架构: Encoder-DoubleConv_bottleneck-Decoder (纯自编码器)
区域: 矩形(62-63E,32.5-33.5N) + 不规则多边形
训练: slow-style DataLoader + CompositeLoss + grad_clip, 200 epochs, block mask
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
from torch.utils.data import DataLoader

unet_path = r'F:\PINN实验\venv\U-net\U-net.py'
spec = importlib.util.spec_from_file_location("unet", unet_path)
unet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unet)

DATA_PATH = r'F:\PINN实验\venv\U-net\afghanistan_full\Afghan_mag06A.csv'
ROOT = os.path.dirname(os.path.abspath(__file__))
ABL_ROOT = os.path.dirname(ROOT)
OUT_DIR = ROOT
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

BASE_CH = 48; EPOCHS = 200; BATCH_SIZE = 8
MAX_LR = 5e-4; PCT_START = 0.3; WEIGHT_DECAY = 1e-5
N_INTERP_CTRL = 3000; TARGET_SIZE = (128, 128)

RECT = (62.0, 63.0, 32.5, 33.5)
IRREG = np.array([
    [62.05, 32.65], [62.30, 32.38], [62.70, 32.42], [62.95, 32.75],
    [63.08, 33.05], [62.88, 33.38], [62.48, 33.48], [62.15, 33.22], [61.98, 32.92],
])

print("=" * 60)
print("  Ablation: Pure U-Net (no skip, no transformer)")
print(f"  epochs={EPOCHS}, max_lr={MAX_LR}, pct_start={PCT_START}, block mask")
print("=" * 60)

# =============================================================================
# Pure U-Net: Encoder-Bottleneck(Conv)-Decoder
# =============================================================================
class PureUNet(nn.Module):
    def __init__(self, in_chans=1, base_ch=48):
        super().__init__()
        self.inc = unet.DoubleConv(in_chans, base_ch)
        self.down1 = unet.Down(base_ch, base_ch * 2)
        self.down2 = unet.Down(base_ch * 2, base_ch * 4)
        self.down3 = unet.Down(base_ch * 4, base_ch * 4)
        self.down4 = unet.Down(base_ch * 4, base_ch * 4)
        self.bottleneck = unet.DoubleConv(base_ch * 4, base_ch * 4)
        self.up1 = unet.Up(base_ch * 4, base_ch * 2, use_skip=False)
        self.up2 = unet.Up(base_ch * 2, base_ch * 2, use_skip=False)
        self.up3 = unet.Up(base_ch * 2, base_ch, use_skip=False)
        self.up4 = unet.Up(base_ch, base_ch, use_skip=False)
        self.outc = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.bottleneck(x5)
        x = self.up1(x5)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.outc(x)

# =============================================================================
# Data loading (supports both rect and irregular polygon)
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
    if rect_bounds:
        mask_blank = ((lon_grid >= lon_min) & (lon_grid <= lon_max) &
                      (lat_grid >= lat_min) & (lat_grid <= lat_max))
    else:
        mask_blank = Path(poly_vertices).contains_points(
            np.column_stack([lon_grid.ravel(), lat_grid.ravel()])).reshape(grid_x.shape)
    hull = ConvexHull(train_df[['x','y']].values)
    hull_xy = hull.points[hull.vertices]
    xc = hull_xy[:,0].mean()
    for i in range(len(hull_xy)):
        hull_xy[i,0] += -2. if hull_xy[i,0] < xc else 2.
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
    test_xy = np.column_stack(data['ll_to_xy'](data['test_df']['Longitude'].values, data['test_df']['Latitude'].values))
    return {'F_display': F_display, 'F_min': F_min, 'F_max': F_max,
            'F_resized': F_resized, 'perm_mask': perm_mask, 'denorm': denorm, 'eps': eps,
            'test_xy': test_xy, 'yt_all': data['test_df']['FinalMag'].values}

# =============================================================================
# Training
# =============================================================================
def train_model(data, prep, region_key):
    print(f"\n[Pure U-Net] Training {region_key}, epochs={EPOCHS}")
    np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    model = PureUNet(in_chans=1, base_ch=BASE_CH).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

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

    del model; torch.cuda.empty_cache()
    return {'rmse': float(best_rmse), 'mae': float(mae), 'best_epoch': best_ep,
            'train_time': train_time, 'n_params': n_params,
            'result_grid': best_result_grid, 'history': history, 'best_state': best_state}

# =============================================================================
# Main
# =============================================================================
regions = {
    'rect':  ('矩形', None, RECT),
    'irreg': ('不规则', IRREG, None),
}

summary = {}
for rk, (rlabel, poly, rect) in regions.items():
    print(f"\n{'#'*60}")
    print(f"# 区域: {rlabel} ({rk})")
    print(f"{'#'*60}")

    data = load_region_data(poly_vertices=poly, rect_bounds=rect)
    print(f"  Train: {len(data['train_df']):,}, Test: {len(data['test_df']):,}, "
          f"Blank: {data['mask_blank'].sum():,}")

    prep = preprocess(data)
    print(f"  eps={prep['eps']:.4f}")

    # Save shared data for later plotting
    np.save(os.path.join(OUT_DIR, f'mask_blank_{rk}.npy'), data['mask_blank'])
    np.save(os.path.join(OUT_DIR, f'mask_outside_{rk}.npy'), data['mask_outside'])
    np.save(os.path.join(OUT_DIR, f'grid_x_{rk}.npy'), data['grid_x'])
    np.save(os.path.join(OUT_DIR, f'grid_y_{rk}.npy'), data['grid_y'])
    np.save(os.path.join(OUT_DIR, f'bx_{rk}.npy'), data['bx'])
    np.save(os.path.join(OUT_DIR, f'by_{rk}.npy'), data['by'])

    result = train_model(data, prep, rk)

    # Save model results
    torch.save(result['best_state'], os.path.join(OUT_DIR, f'best_model_{rk}.pt'))
    np.save(os.path.join(OUT_DIR, f'result_grid_{rk}.npy'), result['result_grid'])
    with open(os.path.join(OUT_DIR, f'history_{rk}.json'), 'w') as f:
        json.dump(result['history'], f, indent=2)

    summary[rk] = {'rmse': result['rmse'], 'mae': result['mae'],
                   'best_epoch': result['best_epoch'], 'train_time': result['train_time']}
    print(f"  Done! RMSE={result['rmse']:.2f}, MAE={result['mae']:.2f}")

summary['config'] = {'model': 'Pure U-Net (no skip, no transformer)',
                     'max_lr': MAX_LR, 'pct_start': PCT_START, 'epochs': EPOCHS,
                     'weight_decay': WEIGHT_DECAY, 'base_ch': BASE_CH, 'batch_size': BATCH_SIZE,
                     'mask_mode': 'block', 'n_params': result['n_params']}
with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"  Pure U-Net 完成!")
print(f"  rect  RMSE={summary['rect']['rmse']:.2f}, best_epoch={summary['rect']['best_epoch']}")
print(f"  irreg RMSE={summary['irreg']['rmse']:.2f}, best_epoch={summary['irreg']['best_epoch']}")
print(f"  输出: {OUT_DIR}/")
print("=" * 60)
