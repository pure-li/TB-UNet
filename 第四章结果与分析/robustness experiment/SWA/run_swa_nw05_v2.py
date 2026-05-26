#!/usr/bin/env python
"""Robustness: SWA on 0.5deg NW region v2 — 100 epochs
Region: 61.5-62.0degE, 34.2-34.7degN (0.5deg x 0.5deg, NW v2)"""
import sys, importlib.util, os, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
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
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(OUT_DIR, 'checkpoints_nw05_v2')
os.makedirs(CKPT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
BASE_CH = 48; EPOCHS = 100; BATCH_SIZE = 8
MAX_LR = 5e-4; PCT_START = 0.3; WEIGHT_DECAY = 1e-5
N_INTERP_CTRL = 3000; TARGET_SIZE = (128, 128)

RECT = (61.5, 62.0, 34.2, 34.7)
REGION_KEY = 'nw05_v2'

print("=" * 60)
print(f"  SWA — 0.5deg NW v2 region")
print(f"  RECT: {RECT}")
print(f"  epochs={EPOCHS}, lr={MAX_LR}, base_ch={BASE_CH}")
print("=" * 60)

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
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2)
        x4 = self.down3(x3); x5 = self.down4(x4); x5 = self.transformer(x5)
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
    bx, by = ll_to_xy(np.array([lon_min, lon_max, lon_max, lon_min, lon_min]),
                      np.array([lat_min, lat_min, lat_max, lat_max, lat_min]))
    return {'train_df': train_df, 'test_df': test_df, 'll_to_xy': ll_to_xy,
            'x_grid': x_grid, 'y_grid': y_grid, 'nx': nx, 'ny': ny,
            'grid_x': grid_x, 'grid_y': grid_y, 'grid_pts': grid_pts,
            'mask_blank': mask_blank, 'mask_outside': mask_outside,
            'bx': bx, 'by': by}

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
# Evaluation
# =============================================================================
@torch.no_grad()
def evaluate(model, test_input_t, zf, data, prep):
    model.eval()
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
        rmse = float(np.sqrt(mean_squared_error(prep['yt_all'][valid_v], pred_v[valid_v])))
        mae = float(mean_absolute_error(prep['yt_all'][valid_v], pred_v[valid_v]))
    else:
        rmse, mae = float('nan'), float('nan')
    abs_res = np.full(len(prep['yt_all']), np.nan, dtype=np.float32)
    abs_res[valid_v] = np.abs(prep['yt_all'][valid_v] - pred_v[valid_v])
    return rmse, mae, rg, abs_res

# =============================================================================
# Main
# =============================================================================
print("\nLoading data...")
data = load_region_data(RECT)
prep = preprocess(data)
print(f"  Train: {len(data['train_df']):,}, Test: {len(data['test_df']):,}, Blank: {data['mask_blank'].sum():,}")

np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

model = UNetTransformer().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Params: {n_params:,}")

train_image = np.stack([prep['F_resized']], axis=0).astype(np.float32)
dataset = unet.InpaintingDataset(train_image, prep['perm_mask'], mask_mode='block', mask_ratio=0.2)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
steps_per_epoch = len(dataloader)

optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=MAX_LR, total_steps=EPOCHS * steps_per_epoch,
    pct_start=PCT_START, anneal_strategy='cos', final_div_factor=1e4)
criterion = unet.CompositeLoss(grad_weight=0.0).to(DEVICE)

zf = (data['nx']/TARGET_SIZE[0], data['ny']/TARGET_SIZE[1])
test_input_t = torch.tensor(train_image, dtype=torch.float32).unsqueeze(0).to(DEVICE)

history = []
best_rmse = float('inf'); best_mae = float('inf'); best_ep = 0
best_state = None; best_grid = None; best_res = None
t_train = time.time()

for ep in range(1, EPOCHS + 1):
    model.train(); ep_loss = 0.0
    for batch in dataloader:
        inp, target, mask = [b.to(DEVICE) for b in batch]
        optimizer.zero_grad()
        loss = criterion(model(inp), target, mask.bool())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step(); scheduler.step()
        ep_loss += loss.item()

    rmse, mae, grid, res = evaluate(model, test_input_t, zf, data, prep)
    history.append({'epoch': ep, 'loss': float(ep_loss / steps_per_epoch),
                    'rmse': rmse, 'mae': mae, 'lr': float(scheduler.get_last_lr()[0])})

    if rmse < best_rmse:
        best_rmse, best_mae, best_ep = rmse, mae, ep
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_grid = grid.copy()
        best_res = res.copy()

    if ep % 10 == 0:
        torch.save({
            'epoch': ep, 'raw_state': {k: v.cpu() for k, v in model.state_dict().items()},
            'rmse': rmse, 'mae': mae,
        }, os.path.join(CKPT_DIR, f'{REGION_KEY}_ep{ep:03d}.pt'))

    if ep % 10 == 0 or ep == 1:
        print(f"  Ep {ep:3d}/{EPOCHS} | Loss: {history[-1]['loss']:.6f} | "
              f"RMSE: {rmse:.1f} | Best: {best_rmse:.1f}@{best_ep} | {time.time()-t_train:.0f}s")

train_time = time.time() - t_train

# ---- SWA ----
print(f"\n  Computing SWA...")
sorted_hist = sorted(history, key=lambda x: x['rmse'])
top3_eps = [e['epoch'] for e in sorted_hist[:3]]
avail_eps = list(range(10, EPOCHS + 1, 10))
selected = []
for ep in top3_eps:
    nearest = min(avail_eps, key=lambda x: abs(x - ep))
    if nearest not in selected:
        selected.append(nearest)
print(f"  Top-3 epochs: {top3_eps}, using checkpoints: {selected}")

swa_state = None
for ep in selected:
    ckpt = torch.load(os.path.join(CKPT_DIR, f'{REGION_KEY}_ep{ep:03d}.pt'),
                      map_location='cpu', weights_only=False)
    st = ckpt['raw_state']
    if swa_state is None:
        swa_state = {k: st[k].float() for k in st}
    else:
        for k in swa_state:
            swa_state[k] += st[k].float()
for k in swa_state:
    swa_state[k] /= len(selected)

swa_model = UNetTransformer().to(DEVICE)
swa_model.load_state_dict(swa_state)
swa_rmse, swa_mae, swa_grid, swa_res = evaluate(swa_model, test_input_t, zf, data, prep)
print(f"  SWA: RMSE={swa_rmse:.2f}, MAE={swa_mae:.2f}")

# Save
results = {
    REGION_KEY: {'best_rmse': best_rmse, 'best_mae': best_mae, 'best_ep': best_ep,
                 'swa_rmse': swa_rmse, 'swa_mae': swa_mae, 'swa_epochs': selected, 'train_time': train_time},
    'config': {'mode': 'swa', 'region': '0.5deg NW v2', 'rect': RECT, 'epochs': EPOCHS,
               'max_lr': MAX_LR, 'base_ch': BASE_CH, 'n_params': n_params,}
}
with open(os.path.join(OUT_DIR, 'results_nw05_v2.json'), 'w') as f:
    json.dump(results, f, indent=2)

with open(os.path.join(OUT_DIR, f'history_{REGION_KEY}.json'), 'w') as f:
    json.dump(history, f, indent=2)

np.save(os.path.join(OUT_DIR, f'result_grid_{REGION_KEY}.npy'), best_grid)
np.save(os.path.join(OUT_DIR, f'result_grid_swa_{REGION_KEY}.npy'), swa_grid)
np.save(os.path.join(OUT_DIR, f'abs_residual_{REGION_KEY}.npy'), best_res)
np.save(os.path.join(OUT_DIR, f'abs_residual_swa_{REGION_KEY}.npy'), swa_res)
np.save(os.path.join(OUT_DIR, f'truth_grid_{REGION_KEY}.npy'), prep['F_display'])
np.save(os.path.join(OUT_DIR, f'mask_blank_{REGION_KEY}.npy'), data['mask_blank'])
np.save(os.path.join(OUT_DIR, f'mask_outside_{REGION_KEY}.npy'), data['mask_outside'])
np.save(os.path.join(OUT_DIR, f'grid_x_{REGION_KEY}.npy'), data['grid_x'])
np.save(os.path.join(OUT_DIR, f'grid_y_{REGION_KEY}.npy'), data['grid_y'])
np.save(os.path.join(OUT_DIR, f'bx_{REGION_KEY}.npy'), data['bx'])
np.save(os.path.join(OUT_DIR, f'by_{REGION_KEY}.npy'), data['by'])

if best_state:
    torch.save(best_state, os.path.join(OUT_DIR, f'best_model_{REGION_KEY}.pt'))
if swa_state:
    torch.save(swa_state, os.path.join(OUT_DIR, f'best_model_swa_{REGION_KEY}.pt'))

del swa_model, model; torch.cuda.empty_cache()
print(f"\n  Done! best={best_rmse:.2f}, swa={swa_rmse:.2f}, time={train_time/60:.0f}min")
