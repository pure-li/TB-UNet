#!/usr/bin/env python
"""TTA (Test-Time Augmentation) 评估
======================================
加载已训练好的 U-Net+Transformer 模型, 用多次随机 mask 推理取平均
"""

import sys, importlib.util, os, json, warnings
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

unet_path = r'F:\PINN实验\venv\U-net\U-net.py'
spec = importlib.util.spec_from_file_location("unet", unet_path)
unet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unet)

DATA_PATH = r'F:\PINN实验\venv\U-net\afghanistan_full\Afghan_mag06A.csv'
MODEL_DIR = r'F:\PINN实验\venv\U-net\U-net_transformer_final'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET_SIZE = (128, 128)
SEED = 42
N_INTERP_CTRL = 3000
TTA_TIMES = 20  # 随机 mask 次数

# Transformer 模块 (与训练时完全一致)
# Transformer 模块 (与 run_final.py 完全一致)
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
            num_layers=4, num_heads=8, mlp_ratio=4, dropout=0.1,
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

# 数据加载 (复用)
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
        bx, by = ll_to_xy(border[:,0], border[:,1])
    return {'train_df': train_df, 'test_df': test_df, 'll_to_xy': ll_to_xy,
            'x_grid': x_grid, 'y_grid': y_grid, 'nx': nx, 'ny': ny,
            'grid_x': grid_x, 'grid_y': grid_y, 'grid_pts': grid_pts,
            'mask_blank': mask_blank, 'mask_outside': mask_outside,
            'bx': bx, 'by': by}

def preprocess(data):
    train_df = data['train_df']
    points_xy = train_df[['x','y']].values
    values_F = train_df['FinalMag'].values
    n_sub = min(N_INTERP_CTRL, len(points_xy))
    idx = np.random.choice(len(points_xy), n_sub, replace=False)
    pts_sub, val_sub = points_xy[idx], values_F[idx]
    tree = KDTree(pts_sub)
    distances, _ = tree.query(pts_sub, k=min(10, n_sub))
    eps = np.median(distances[:,1:]) * 0.8
    F_grid = RBFInterpolator(pts_sub, val_sub, kernel='cubic', epsilon=eps)(data['grid_pts']).reshape(data['grid_x'].shape)
    F_display = F_grid.copy()
    F_display = gaussian_filter1d(F_display, 5., axis=0)
    F_display = gaussian_filter1d(F_display, 1., axis=1)
    F_display[data['mask_outside']] = np.nan
    F_grid_masked = F_grid.copy()
    F_grid_masked[data['mask_blank'] | data['mask_outside']] = np.nan
    F_min, F_max = np.nanmin(F_grid_masked), np.nanmax(F_grid_masked)
    def norm(x): return (x - F_min)/(F_max - F_min)*2 - 1
    def denorm(x): return (x + 1)/2*(F_max - F_min) + F_min
    F_in = F_grid_masked.copy(); F_in[np.isnan(F_in)] = 0.
    F_resized = resize(norm(F_in).astype(np.float32), TARGET_SIZE, mode='constant', anti_aliasing=True)
    perm_mask = resize(data['mask_blank'].astype(float), TARGET_SIZE) > 0.5
    test_xy = np.column_stack(data['ll_to_xy'](data['test_df']['Longitude'].values,
                                                data['test_df']['Latitude'].values))
    yt_all = data['test_df']['FinalMag'].values
    return {'F_display': F_display, 'F_min': F_min, 'F_max': F_max,
            'F_resized': F_resized, 'perm_mask': perm_mask,
            'test_xy': test_xy, 'yt_all': yt_all, 'denorm': denorm, 'eps': eps}

def gen_random_block_mask(valid_region, H, W):
    """生成随机 block mask (与训练时逻辑一致)"""
    max_h = min(int(H * 0.4), 48)
    min_h = max(int(H * 0.1), 8)
    max_w = min(int(W * 0.4), 48)
    min_w = max(int(W * 0.1), 8)
    v = valid_region.cpu().numpy() if hasattr(valid_region, 'cpu') else valid_region
    for _ in range(200):
        mh = np.random.randint(min_h, max_h + 1)
        mw = np.random.randint(min_w, max_w + 1)
        y = np.random.randint(0, H - mh)
        x = np.random.randint(0, W - mw)
        if v[y:y+mh, x:x+mw].all():
            mask = torch.zeros(H, W, dtype=torch.bool)
            mask[y:y+mh, x:x+mw] = True
            return mask
    return None

print("=" * 60)
print(f"  TTA 评估 (n={TTA_TIMES})")
print(f"  模型目录: {MODEL_DIR}")
print("=" * 60)

REGIONS = {
    'rect': ('矩形', None, (62.0, 63.0, 32.5, 33.5)),
    'irreg': ('不规则', np.array([[62.05,32.65],[62.30,32.38],[62.70,32.42],[62.95,32.75],
                                   [63.08,33.05],[62.88,33.38],[62.48,33.48],[62.15,33.22],[61.98,32.92]]), None),
}

for rk, (rlabel, poly, rect) in REGIONS.items():
    model_path = os.path.join(MODEL_DIR, f'best_model_{rk}.pt')
    if not os.path.exists(model_path):
        print(f"\n  [{rlabel}] 模型不存在, 跳过")
        continue

    print(f"\n{'='*50}\n  [{rlabel}] TTA 评估\n{'='*50}")

    data = load_region_data(poly_vertices=poly, rect_bounds=rect)
    prep = preprocess(data)
    print(f"  eps={prep['eps']:.4f}, 训练点: {len(data['train_df']):,}, 测试点: {len(data['test_df']):,}")

    model = UNetTransformer(in_chans=1, base_ch=48, use_skip=False).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"  模型已加载: {model_path}")

    # ========================
    # 1) 标准推理 (无 TTA, 原始输入)
    # ========================
    base_input = torch.tensor(np.stack([prep['F_resized']], 0).astype(np.float32)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out_std = model(base_input)
    out_std_np = out_std.squeeze().cpu().numpy()
    out_std_np = prep['denorm'](out_std_np)
    zf = (data['nx']/TARGET_SIZE[0], data['ny']/TARGET_SIZE[1])
    out_std_full = zoom(out_std_np, zf, order=1)[:data['nx'], :data['ny']]
    bo = out_std_full.copy(); bo[~data['mask_blank']] = np.nan
    bo = gaussian_filter1d(bo, 1.5, axis=0); bo = gaussian_filter1d(bo, 1.5, axis=1)
    out_std_full[data['mask_blank']] = bo[data['mask_blank']]
    rg_std = prep['F_display'].copy(); rg_std[data['mask_blank']] = out_std_full[data['mask_blank']]
    interp_std = RegularGridInterpolator((data['x_grid'], data['y_grid']), rg_std,
                                         method='linear', bounds_error=False, fill_value=np.nan)
    pred_std = interp_std(prep['test_xy'])
    valid = ~np.isnan(pred_std)
    rmse_std = np.sqrt(mean_squared_error(prep['yt_all'][valid], pred_std[valid])) if np.sum(valid) > 10 else float('nan')
    mae_std = mean_absolute_error(prep['yt_all'][valid], pred_std[valid]) if np.sum(valid) > 10 else float('nan')
    print(f"  标准推理:        RMSE={rmse_std:.2f}, MAE={mae_std:.2f}")

    # ========================
    # 2) TTA 推理 (多次随机 mask + 平均)
    # ========================
    valid_region = ~torch.tensor(prep['perm_mask'], dtype=torch.bool)
    H, W = TARGET_SIZE
    base_np = prep['F_resized'].astype(np.float32)

    outputs_tta = []
    tta_rmses = []
    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    t0.record()

    for i in range(TTA_TIMES):
        mask = gen_random_block_mask(valid_region, H, W)
        if mask is None:
            mask = torch.zeros(H, W, dtype=torch.bool)
        input_masked = torch.tensor(np.stack([base_np], 0), dtype=torch.float32)
        input_masked[:, mask] = 0.0
        input_batch = input_masked.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(input_batch)
        out_np = out.squeeze().cpu().numpy()
        out_np = prep['denorm'](out_np)
        outputs_tta.append(out_np)

        # 每个 TTA 单独评估
        out_full = zoom(out_np, zf, order=1)[:data['nx'], :data['ny']]
        bo = out_full.copy(); bo[~data['mask_blank']] = np.nan
        bo = gaussian_filter1d(bo, 1.5, axis=0); bo = gaussian_filter1d(bo, 1.5, axis=1)
        out_full[data['mask_blank']] = bo[data['mask_blank']]
        rg = prep['F_display'].copy(); rg[data['mask_blank']] = out_full[data['mask_blank']]
        interp = RegularGridInterpolator((data['x_grid'], data['y_grid']), rg,
                                         method='linear', bounds_error=False, fill_value=np.nan)
        pred = interp(prep['test_xy'])
        valid_i = ~np.isnan(pred)
        rmse_i = np.sqrt(mean_squared_error(prep['yt_all'][valid_i], pred[valid_i])) if np.sum(valid_i) > 10 else float('nan')
        tta_rmses.append(rmse_i)

    t1.record()
    torch.cuda.synchronize()
    tta_time = t0.elapsed_time(t1) / 1000

    # TTA 平均
    out_mean = np.mean(outputs_tta, axis=0)
    out_mean_full = zoom(out_mean, zf, order=1)[:data['nx'], :data['ny']]
    bo = out_mean_full.copy(); bo[~data['mask_blank']] = np.nan
    bo = gaussian_filter1d(bo, 1.5, axis=0); bo = gaussian_filter1d(bo, 1.5, axis=1)
    out_mean_full[data['mask_blank']] = bo[data['mask_blank']]
    rg_tta = prep['F_display'].copy(); rg_tta[data['mask_blank']] = out_mean_full[data['mask_blank']]
    interp_tta = RegularGridInterpolator((data['x_grid'], data['y_grid']), rg_tta,
                                         method='linear', bounds_error=False, fill_value=np.nan)
    pred_tta = interp_tta(prep['test_xy'])
    valid_tta = ~np.isnan(pred_tta)
    rmse_tta = np.sqrt(mean_squared_error(prep['yt_all'][valid_tta], pred_tta[valid_tta])) if np.sum(valid_tta) > 10 else float('nan')
    mae_tta = mean_absolute_error(prep['yt_all'][valid_tta], pred_tta[valid_tta]) if np.sum(valid_tta) > 10 else float('nan')

    print(f"  TTA ({TTA_TIMES}x): RMSE={rmse_tta:.2f}, MAE={mae_tta:.2f}")
    print(f"  TTA 单次 RMSE 范围: {min(tta_rmses):.1f} ~ {max(tta_rmses):.1f}, mean={np.mean(tta_rmses):.1f}")
    print(f"  TTA 提升: RMSE {rmse_std - rmse_tta:+.2f} nT ({(rmse_std - rmse_tta)/rmse_std*100:+.1f}%)")
    print(f"  TTA 耗时: {tta_time:.1f}s")

    # 保存 TTA 结果网格
    np.save(os.path.join(MODEL_DIR, f'result_grid_{rk}_tta.npy'), rg_tta)
    print(f"  TTA 结果网格已保存: result_grid_{rk}_tta.npy")

print(f"\n{'='*60}")
print("  TTA 评估完成!")
print("=" * 60)
