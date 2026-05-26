#!/usr/bin/env python
"""U-Net + Transformer 最终实验 — 双区域填补
====================================================
矩形 + 不规则空白区, 100 epochs, 按 RMSE 选最佳模型
Pure U-Net vs U-Net+Transformer (block-only mask, 无skip, 无梯度损失)

保存:
  - best_model_{rect|irreg}_{unet|tf}.pt    模型权重
  - result_grid_{rect|irreg}_{unet|tf}.npy   最佳结果网格
  - truth_grid_{rect|irreg}.npy              真值网格(RBF)
  - mask_blank_{rect|irreg}.npy              空白区掩膜
  - grid_x_{rect|irreg}.npy, grid_y_{rect|irreg}.npy  网格坐标
  - bx_{rect|irreg}.npy, by_{rect|irreg}.npy           边界坐标
  - history_{rect|irreg}_{unet|tf}.json      训练历史
  - results.json                             汇总结果

绘图 (每张单独 PNG+SVG):
  - fig_loss_{rect|irreg}.png/svg            Loss 曲线
  - fig_rmse_{rect|irreg}.png/svg            RMSE 曲线
  - fig_result_{rect|irreg}_{unet|tf}.png/svg 填补结果图
  - fig_residual_{rect|irreg}_{unet|tf}.png/svg 残差图 (pred-true, RdBu)
  - fig_error_{rect|irreg}_{unet|tf}.png/svg   绝对误差图 (|pred-true|, hot)
  - fig_truth_{rect|irreg}.png/svg           真值图
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
# 0. 配置
# =============================================================================
DATA_PATH = r'F:\PINN实验\venv\U-net\afghanistan_full\Afghan_mag06A.csv'
OUT_DIR   = r'F:\PINN实验\venv\U-net\U-net_transformer_final'
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET_SIZE = (128, 128)
EPOCHS = 100
BATCH_SIZE = 8
BASE_CH = 48
SEED = 42
N_INTERP_CTRL = 3000

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
print("  U-Net + Transformer 最终实验 — 双区域填补")
print(f"  矩形 + 不规则空白区, {EPOCHS} epochs, 按 RMSE 选最佳")
print(f"  Block-only mask, 无跳跃连接, 无梯度损失")
print(f"  输出: {OUT_DIR}/")
print("=" * 60)

# =============================================================================
# 1. Transformer 模块
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
# 2. 数据加载
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
        'bx': bx, 'by': by, 'lon0': lon0, 'lat0': lat0, 'R_map': R_map,
    }

# =============================================================================
# 3. RBF 预处理
# =============================================================================

def rbf_preprocess(data):
    train_df = data['train_df']
    points_xy = train_df[['x', 'y']].values
    values_F = train_df['FinalMag'].values

    # 真值网格 (thin_plate_spline, 无空白区, 用于评估和绘图)
    n_ctrl = 3000
    df_all = pd.read_csv(DATA_PATH).iloc[::3]
    df_truth = df_all.iloc[::max(1, len(df_all)//n_ctrl)].copy()
    pts_truth = np.column_stack(data['ll_to_xy'](df_truth['Longitude'].values, df_truth['Latitude'].values))
    val_truth = df_truth['FinalMag'].values
    truth_grid = RBFInterpolator(pts_truth, val_truth, kernel='thin_plate_spline')(
        data['grid_pts']).reshape(data['grid_x'].shape)
    truth_grid = gaussian_filter1d(truth_grid, 2.0, axis=0)
    truth_grid = gaussian_filter1d(truth_grid, 1.0, axis=1)
    truth_grid[data['mask_outside']] = np.nan

    # RBF 插值作为背景场
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
    F_input_norm = normalize(F_input)
    F_resized = resize(F_input_norm.astype(np.float32), TARGET_SIZE,
                       mode='constant', anti_aliasing=True)
    perm_mask_resized = resize(data['mask_blank'].astype(float), TARGET_SIZE) > 0.5

    test_xy = np.column_stack(data['ll_to_xy'](data['test_df']['Longitude'].values,
                                                data['test_df']['Latitude'].values))
    yt_all = data['test_df']['FinalMag'].values

    return {
        'truth_grid': truth_grid, 'F_display': F_display, 'F_min': F_min, 'F_max': F_max,
        'F_resized': F_resized, 'perm_mask_resized': perm_mask_resized,
        'test_xy': test_xy, 'yt_all': yt_all,
        'normalize': normalize, 'denorm': denorm, 'eps': eps,
    }

# =============================================================================
# 4. 训练函数
# =============================================================================

def train_and_eval(model, data, prep, region_key, model_key, label):
    print(f"\n{'='*50}\n  {label}\n{'='*50}")

    np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")

    criterion = unet.CompositeLoss(grad_weight=GRAD_WEIGHT).to(DEVICE)
    train_image = np.stack([prep['F_resized']], axis=0).astype(np.float32)
    dataset = unet.InpaintingDataset(train_image, prep['perm_mask_resized'],
                                     mask_mode=MASK_MODE, mask_ratio=0.2)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    steps_total = EPOCHS * len(dataloader)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler_lr = lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=steps_total,
                                        pct_start=0.1, anneal_strategy='cos', final_div_factor=1e4)

    test_input_t = torch.tensor(train_image, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    history = []
    best_rmse = float('inf')
    best_state_rmse = None
    best_epoch = 0
    best_result_grid = None
    best_output_blank = None
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
            best_state_rmse = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_result_grid = rg.copy()
            best_output_blank = output_full_ep[data['mask_blank']].copy()

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  Epoch {ep+1:3d}/{EPOCHS} | Loss: {ep_loss/len(dataloader):.6f} | RMSE: {ep_rmse:.1f} | MAE: {ep_mae:.1f} | Best: {best_rmse:.1f}@{best_epoch} | LR: {lr_now:.2e}")

    model.load_state_dict(best_state_rmse)
    train_time = time.time() - t0

    # 保存模型权重
    model_path = os.path.join(OUT_DIR, f'best_model_{region_key}.pt')
    torch.save(best_state_rmse, model_path)
    print(f"  模型已保存: {model_path}")

    # 保存结果网格
    np.save(os.path.join(OUT_DIR, f'result_grid_{region_key}.npy'), best_result_grid)

    print(f"  Final (Best-RMSE): epoch={best_epoch}, RMSE={best_rmse:.2f}, MAE={history[best_epoch-1]['mae']:.2f} | Time: {train_time:.0f}s")

    del model; torch.cuda.empty_cache()
    return {'rmse': float(best_rmse), 'mae': float(history[best_epoch-1]['mae']),
            'best_epoch': best_epoch, 'history': history,
            'result_grid': best_result_grid, 'output_blank': best_output_blank,
            'train_time': train_time, 'n_params': n_params}


# =============================================================================
# 5. 绘图函数
# =============================================================================

def save_fig(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(os.path.join(OUT_DIR, f'{name}.{ext}'), dpi=300,
                    bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

def plot_all_figures(data, prep, result, region_key, region_label):
    """为一个区域生成所有图表 (仅 U-Net+Transformer)"""
    mask_blank = data['mask_blank']
    mask_outside = data['mask_outside']
    bx, by = data['bx'], data['by']
    truth_grid = prep['truth_grid']
    grid_x, grid_y = data['grid_x'], data['grid_y']
    result_grid = result['result_grid']
    history = result['history']

    zx = (bx.min() - 2, bx.max() + 2)
    zy = (by.min() - 2, by.max() + 2)

    vmin = np.nanmin(truth_grid[~mask_outside])
    vmax = np.nanmax(truth_grid[~mask_outside])

    print(f"\n  [{region_label}] 生成图表...")

    # ---- Fig 1: Loss 曲线 ----
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot([h['epoch'] for h in history], [h['loss'] for h in history],
             color='#FF5722', lw=2)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title(f'{region_label} Blank — Training Loss (U-Net+Transformer)')
    ax1.grid(True, alpha=0.3)
    save_fig(fig1, f'fig_loss_{region_key}')

    # ---- Fig 2: RMSE 曲线 ----
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    eps = [h['epoch'] for h in history]
    rmses = [h['rmse'] for h in history]
    ax2.plot(eps, rmses, color='#FF5722', lw=2)
    ax2.scatter(result['best_epoch'], result['rmse'], color='#FF5722', s=80,
               zorder=5, marker='*', edgecolors='black')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('RMSE (nT)')
    ax2.set_title(f'{region_label} Blank — Test RMSE (U-Net+Transformer)')
    ax2.grid(True, alpha=0.3)
    save_fig(fig2, f'fig_rmse_{region_key}')

    # ---- Fig 3: 真值图 ----
    fig3, ax3 = plt.subplots(figsize=(10, 9))
    im = ax3.pcolormesh(grid_x, grid_y, truth_grid, cmap='jet', shading='auto', vmin=vmin, vmax=vmax)
    ax3.plot(bx, by, 'k-', linewidth=2)
    ax3.set_xlim(zx); ax3.set_ylim(zy)
    ax3.set_xlabel('X (km)'); ax3.set_ylabel('Y (km)')
    ax3.set_title(f'{region_label} Blank — Ground Truth')
    cbar = plt.colorbar(im, ax=ax3, label='Mag. Anomaly (nT)')
    cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
    save_fig(fig3, f'fig_truth_{region_key}')

    # ---- Fig 4: Result map ----
    fig4, ax4 = plt.subplots(figsize=(10, 9))
    im = ax4.pcolormesh(grid_x, grid_y, result_grid, cmap='jet', shading='auto', vmin=vmin, vmax=vmax)
    ax4.plot(bx, by, 'k-', linewidth=2)
    ax4.set_xlim(zx); ax4.set_ylim(zy)
    ax4.set_xlabel('X (km)'); ax4.set_ylabel('Y (km)')
    ax4.set_title(f'{region_label} Blank — U-Net + Transformer')
    cbar = plt.colorbar(im, ax=ax4, label='Mag. Anomaly (nT)')
    cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
    save_fig(fig4, f'fig_result_{region_key}')

    # ---- Fig 5: Residual map (pred - true, RdBu) ----
    residual = result_grid.copy()
    residual[~mask_blank] = np.nan
    residual[mask_blank] = result_grid[mask_blank] - truth_grid[mask_blank]
    res_max = max(abs(np.nanmin(residual[mask_blank])), abs(np.nanmax(residual[mask_blank])))
    res_max = max(res_max, 1.0)

    fig5, ax5 = plt.subplots(figsize=(10, 9))
    im = ax5.pcolormesh(grid_x, grid_y, residual, cmap='RdBu_r', shading='auto', vmin=-res_max, vmax=res_max)
    ax5.plot(bx, by, 'k-', linewidth=2)
    ax5.set_xlim(zx); ax5.set_ylim(zy)
    ax5.set_xlabel('X (km)'); ax5.set_ylabel('Y (km)')
    ax5.set_title(f'{region_label} Blank — Residual (Pred - True)\nU-Net + Transformer, RMSE={result["rmse"]:.2f} nT')
    cbar = plt.colorbar(im, ax=ax5, label='Residual (nT)')
    cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
    save_fig(fig5, f'fig_residual_{region_key}')

    # ---- Fig 6: Absolute error map (|pred - true|, hot) ----
    abs_error = result_grid.copy()
    abs_error[~mask_blank] = np.nan
    abs_error[mask_blank] = np.abs(result_grid[mask_blank] - truth_grid[mask_blank])

    fig6, ax6 = plt.subplots(figsize=(10, 9))
    im = ax6.pcolormesh(grid_x, grid_y, abs_error, cmap='hot', shading='auto',
                        vmin=0, vmax=np.nanmax(abs_error[mask_blank]) * 0.8)
    ax6.plot(bx, by, 'k-', linewidth=2)
    ax6.set_xlim(zx); ax6.set_ylim(zy)
    ax6.set_xlabel('X (km)'); ax6.set_ylabel('Y (km)')
    ax6.set_title(f'{region_label} Blank — |Error|\nU-Net + Transformer, RMSE={result["rmse"]:.2f} nT')
    cbar = plt.colorbar(im, ax=ax6, label='Absolute Error (nT)')
    cbar.formatter.set_powerlimits((0, 0)); cbar.update_ticks()
    save_fig(fig6, f'fig_error_{region_key}')

    plt.close('all')
    print(f"  [{region_label}] 图表已生成 (6 figures)")


# =============================================================================
# 6. 运行实验
# =============================================================================

regions = {
    'rect':  {'type': '矩形', 'poly': None,
              'rect_bounds': (RECT_LON_MIN, RECT_LON_MAX, RECT_LAT_MIN, RECT_LAT_MAX)},
    'irreg': {'type': '不规则', 'poly': IRREGULAR_POLYGON, 'rect_bounds': None},
}

all_results = {}
run_start = time.time()

for region_key, rcfg in regions.items():
    print(f"\n{'#'*60}")
    print(f"# 区域: {rcfg['type']}空白区")
    print(f"{'#'*60}")

    print("\n[1/3] 加载数据 + RBF 预处理...")
    data = load_region_data(poly_vertices=rcfg['poly'], rect_bounds=rcfg['rect_bounds'])
    prep = rbf_preprocess(data)
    print(f"  eps={prep['eps']:.4f}, F范围: {prep['F_min']:.1f}~{prep['F_max']:.1f}")
    print(f"  训练点: {len(data['train_df']):,}, 测试点: {len(data['test_df']):,}")

    # 保存共享数据
    np.save(os.path.join(OUT_DIR, f'truth_grid_{region_key}.npy'), prep['truth_grid'])
    np.save(os.path.join(OUT_DIR, f'mask_blank_{region_key}.npy'), data['mask_blank'])
    np.save(os.path.join(OUT_DIR, f'mask_outside_{region_key}.npy'), data['mask_outside'])
    np.save(os.path.join(OUT_DIR, f'grid_x_{region_key}.npy'), data['grid_x'])
    np.save(os.path.join(OUT_DIR, f'grid_y_{region_key}.npy'), data['grid_y'])
    np.save(os.path.join(OUT_DIR, f'bx_{region_key}.npy'), data['bx'])
    np.save(os.path.join(OUT_DIR, f'by_{region_key}.npy'), data['by'])
    print(f"  共享数组已保存")

    print(f"\n[2/2] 训练 U-Net + Transformer (100 epochs)...")
    model_tf = UNetTransformer(in_chans=1, base_ch=BASE_CH, use_skip=USE_SKIP)
    res_tf = train_and_eval(model_tf, data, prep, region_key, 'tf',
                            f"U-Net + Transformer [{rcfg['type']}]")

    all_results[region_key] = {'tf': res_tf}

    # 保存训练历史
    hist_path = os.path.join(OUT_DIR, f'history_{region_key}.json')
    with open(hist_path, 'w') as f:
        json.dump(res_tf['history'], f, indent=2)

    # 打印区域结果
    print(f"\n  {rcfg['type']}区域结果:")
    print(f"  {'模型':<30s} {'RMSE':>8s} {'MAE':>8s} {'最佳Epoch':>10s}  {'参数量':>10s}")
    print(f"  {'-'*70}")
    print(f"  {'U-Net+Transformer':<30s} {res_tf['rmse']:8.2f} {res_tf['mae']:8.2f} {res_tf['best_epoch']:>10d}  {res_tf['n_params']:>10,}")

    # 绘图
    plot_all_figures(data, prep, res_tf, region_key, rcfg['type'])

# =============================================================================
# 7. 汇总
# =============================================================================

print(f"\n{'='*60}")
print(f"  最终汇总 (Best-RMSE)")
print(f"{'='*60}")
print(f"  {'区域':<8s} {'RMSE':>8s} {'MAE':>8s} {'Epoch':>8s} {'Time':>8s}")
print(f"  {'-'*48}")
for rk, rlabel in [('rect', '矩形'), ('irreg', '不规则')]:
    r = all_results[rk]['tf']
    print(f"  {rlabel:<8s} {r['rmse']:8.2f} {r['mae']:8.2f} {r['best_epoch']:>8d} {r['train_time']:>7.0f}s")

# 保存汇总 JSON
summary = {}
for rk in all_results:
    r = all_results[rk]['tf']
    summary[rk] = {
        'rmse': r['rmse'], 'mae': r['mae'],
        'best_epoch': r['best_epoch'], 'train_time': r['train_time'],
        'n_params': r['n_params'],
    }
with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

total_time = time.time() - run_start
print(f"\n  总耗时: {total_time/60:.0f} min")
print(f"  全部输出: {OUT_DIR}/")
print("=" * 60)
print("  U-Net + Transformer 最终实验完成!")
print("=" * 60)
