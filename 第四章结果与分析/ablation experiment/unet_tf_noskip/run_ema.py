#!/usr/bin/env python
"""消融实验 — U-Net + Transformer (no skip) + EMA
=================================================
改进: EMA权重平滑 + Cosine退火 + 每10轮保存checkpoint
"""

import sys, importlib.util, os, json, time, copy, warnings
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
OUT_DIR = ROOT
CKPT_DIR = os.path.join(OUT_DIR, 'checkpoints_ema')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

BASE_CH = 48; EPOCHS = 100; BATCH_SIZE = 8
MAX_LR = 5e-4; PCT_START = 0.3; WEIGHT_DECAY = 1e-5
EMA_DECAY = 0.999; PATIENCE = 30
N_INTERP_CTRL = 3000; TARGET_SIZE = (128, 128)

RECT = (62.0, 63.0, 32.5, 33.5)
IRREG = np.array([
    [62.05, 32.65], [62.30, 32.38], [62.70, 32.42], [62.95, 32.75],
    [63.08, 33.05], [62.88, 33.38], [62.48, 33.48], [62.15, 33.22], [61.98, 32.92],
])

print("=" * 60)
print("  Ablation: U-Net + Transformer (no skip) + EMA")
print(f"  epochs={EPOCHS}, max_lr={MAX_LR}, pct_start={PCT_START}, ema_decay={EMA_DECAY}, patience={PATIENCE}")
print("=" * 60)

# =============================================================================
# EMA
# =============================================================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v
    def apply_to(self, model):
        model.load_state_dict(self.shadow)
    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

# =============================================================================
# Transformer modules
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
    def __init__(self, in_chans=1, base_ch=48):
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
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.transformer(x5)
        x = self.up1(x5)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
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
# Evaluation helper
# =============================================================================
def evaluate_model(model, test_input_t, zf, data, prep):
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
    if np.sum(valid_v) > 10:
        rmse = np.sqrt(mean_squared_error(prep['yt_all'][valid_v], pred_v[valid_v]))
        mae = mean_absolute_error(prep['yt_all'][valid_v], pred_v[valid_v])
    else:
        rmse, mae = float('nan'), float('nan')
    return rmse, mae, rg

# =============================================================================
# Training
# =============================================================================
def train_model(data, prep, region_key):
    print(f"\n[U-Net + TF + EMA] Training {region_key}, epochs={EPOCHS}")
    np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    model = UNetTransformer(in_chans=1, base_ch=BASE_CH).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    ema = EMA(model, decay=EMA_DECAY)
    print(f"  Params: {n_params:,}")

    train_image = np.stack([prep['F_resized']], axis=0).astype(np.float32)
    dataset = unet.InpaintingDataset(train_image, prep['perm_mask'],
                                     mask_mode='block', mask_ratio=0.2)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    steps_per_epoch = len(dataloader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR, total_steps=total_steps,
        pct_start=PCT_START, anneal_strategy='cos', final_div_factor=1e4)
    criterion = unet.CompositeLoss(grad_weight=0.0).to(DEVICE)

    history = []
    best_raw_rmse = float('inf'); best_raw_ep = 0; best_raw_state = None; best_raw_grid = None
    best_ema_rmse = float('inf'); best_ema_ep = 0; best_ema_state = None; best_ema_grid = None

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
            ema.update(model)
            ep_loss += loss.item()
        avg_loss = ep_loss / steps_per_epoch

        # Evaluate raw model
        raw_rmse, raw_mae, raw_grid = evaluate_model(model, test_input_t, zf, data, prep)

        # Evaluate EMA model
        ema_model = copy.deepcopy(model)
        ema.apply_to(ema_model)
        ema_rmse, ema_mae, ema_grid = evaluate_model(ema_model, test_input_t, zf, data, prep)
        del ema_model

        history.append({'epoch': ep, 'loss': float(avg_loss),
                        'raw_rmse': float(raw_rmse), 'ema_rmse': float(ema_rmse),
                        'lr': float(scheduler.get_last_lr()[0])})

        if raw_rmse < best_raw_rmse:
            best_raw_rmse = raw_rmse; best_raw_ep = ep
            best_raw_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
            best_raw_grid = raw_grid.copy()

        if ema_rmse < best_ema_rmse:
            best_ema_rmse = ema_rmse; best_ema_ep = ep
            best_ema_state = {k: v.cpu().clone() for k,v in ema.state_dict().items()}
            best_ema_grid = ema_grid.copy()

        if ep % 10 == 0 or ep == 1:
            print(f"  Ep {ep:3d}/{EPOCHS} | Loss: {avg_loss:.6f} | "
                  f"Raw: {raw_rmse:.1f} | EMA: {ema_rmse:.1f} | "
                  f"Best Raw: {best_raw_rmse:.1f}@{best_raw_ep} | "
                  f"Best EMA: {best_ema_rmse:.1f}@{best_ema_ep} | "
                  f"{time.time()-t_train:.0f}s")

        # Save checkpoint every 10 epochs
        if ep % 10 == 0:
            ckpt = {
                'epoch': ep,
                'raw_state': {k: v.cpu() for k,v in model.state_dict().items()},
                'ema_state': {k: v.cpu() for k,v in ema.shadow.items()},
                'raw_rmse': float(raw_rmse), 'ema_rmse': float(ema_rmse),
            }
            torch.save(ckpt, os.path.join(CKPT_DIR, f'{region_key}_ep{ep:03d}.pt'))

        if ep - best_raw_ep > PATIENCE:
            print(f"  Early stop at epoch {ep}")
            break

    train_time = time.time() - t_train
    print(f"  Best RAW:  epoch={best_raw_ep}, RMSE={best_raw_rmse:.2f}")
    print(f"  Best EMA:  epoch={best_ema_ep}, RMSE={best_ema_rmse:.2f}")
    print(f"  Time: {train_time:.0f}s")

    del model; torch.cuda.empty_cache()

    # Return EMA best as primary result
    use_ema = best_ema_rmse < best_raw_rmse
    best_rmse = best_ema_rmse if use_ema else best_raw_rmse
    best_grid = best_ema_grid if use_ema else best_raw_grid
    best_state = best_ema_state if use_ema else best_raw_state
    best_ep = best_ema_ep if use_ema else best_raw_ep

    # Compute MAE for best
    interp_best = RegularGridInterpolator((data['x_grid'], data['y_grid']), best_grid,
                                           method='linear', bounds_error=False, fill_value=np.nan)
    pred_best = interp_best(prep['test_xy']); valid_b = ~np.isnan(pred_best)
    mae = mean_absolute_error(prep['yt_all'][valid_b], pred_best[valid_b]) if np.sum(valid_b)>10 else float('nan')

    return {'rmse': float(best_rmse), 'mae': float(mae), 'best_epoch': best_ep,
            'train_time': train_time, 'n_params': n_params,
            'result_grid': best_grid, 'history': history, 'best_state': best_state,
            'raw_rmse': float(best_raw_rmse), 'raw_mae': float(raw_mae),
            'ema_rmse': float(best_ema_rmse), 'use_ema': use_ema}

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

    np.save(os.path.join(OUT_DIR, f'mask_blank_{rk}.npy'), data['mask_blank'])
    np.save(os.path.join(OUT_DIR, f'mask_outside_{rk}.npy'), data['mask_outside'])
    np.save(os.path.join(OUT_DIR, f'grid_x_{rk}.npy'), data['grid_x'])
    np.save(os.path.join(OUT_DIR, f'grid_y_{rk}.npy'), data['grid_y'])
    np.save(os.path.join(OUT_DIR, f'bx_{rk}.npy'), data['bx'])
    np.save(os.path.join(OUT_DIR, f'by_{rk}.npy'), data['by'])

    result = train_model(data, prep, rk)

    torch.save(result['best_state'], os.path.join(OUT_DIR, f'best_model_ema_{rk}.pt'))
    np.save(os.path.join(OUT_DIR, f'result_grid_ema_{rk}.npy'), result['result_grid'])
    with open(os.path.join(OUT_DIR, f'history_ema_{rk}.json'), 'w') as f:
        json.dump(result['history'], f, indent=2)

    summary[rk] = {'rmse': result['rmse'], 'mae': result['mae'],
                   'best_epoch': result['best_epoch'], 'train_time': result['train_time'],
                   'raw_rmse': result['raw_rmse'], 'ema_rmse': result['ema_rmse'],
                   'use_ema': bool(result['use_ema'])}
    print(f"  Done! RMSE={result['rmse']:.2f}, MAE={result['mae']:.2f} (use_ema={result['use_ema']})")

summary['config'] = {'model': 'U-Net + Transformer (no skip) + EMA',
                     'max_lr': MAX_LR, 'epochs': EPOCHS, 'ema_decay': EMA_DECAY,
                     'weight_decay': WEIGHT_DECAY, 'base_ch': BASE_CH, 'batch_size': BATCH_SIZE,
                     'mask_mode': 'block', 'patience': PATIENCE,
                     'transformer_layers': 4, 'transformer_heads': 8}
with open(os.path.join(OUT_DIR, 'results_ema.json'), 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"  U-Net + TF + EMA 完成!")
for rk in ['rect', 'irreg']:
    print(f"  {rk:5s} RMSE={summary[rk]['rmse']:.2f}, MAE={summary[rk]['mae']:.2f} "
          f"(raw={summary[rk]['raw_rmse']:.2f}, ema={summary[rk]['ema_rmse']:.2f})")
print(f"  输出: {OUT_DIR}/")
print(f"  Checkpoints: {CKPT_DIR}/")
print("=" * 60)
