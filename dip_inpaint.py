#!/usr/bin/env python
"""
Deep Image Prior (DIP) 地磁数据 Inpainting
===========================================
核心思路 (Ulyanov et al., 2018):
  固定随机噪声 → U-Net → 完整场图
  只在已知像素上算 Loss，网络架构本身作为正则化先验
  无需 mask 训练、无需数据增强

消融维度:
  1. U-Net vs 纯 Autoencoder (--no_skip)
  2. 输入通道 (--input_ch 4 / 1)
  3. 梯度损失权重 (--grad_weight)
  4. 网络规模 (--base_ch)
  5. 迭代次数 (--epochs)

用法:
  python dip_inpaint.py --exp_name dip_baseline --epochs 2000
  python dip_inpaint.py --run_all --epochs 2000
"""

import argparse
import os
import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.interpolate import RBFInterpolator, RegularGridInterpolator
from scipy.ndimage import zoom, gaussian_filter1d
from scipy.spatial import KDTree
from skimage.transform import resize

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')


# =============================================================================
# 0. 命令行参数
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Deep Image Prior 地磁数据重建')

    parser.add_argument('--exp_name', type=str, default='dip_baseline')
    parser.add_argument('--run_all', action='store_true')

    # 消融变量
    parser.add_argument('--no_skip', action='store_true')
    parser.add_argument('--input_ch', type=int, default=4)
    parser.add_argument('--grad_weight', type=float, default=0.1)
    parser.add_argument('--eps_mode', type=str, default='adaptive',
                        choices=['adaptive', 'fixed'])
    parser.add_argument('--fixed_eps', type=float, default=1.0)

    # DIP 特有
    parser.add_argument('--noise_type', type=str, default='uniform',
                        choices=['uniform', 'normal', 'fixed_grid'],
                        help='随机噪声类型')
    parser.add_argument('--noise_seed', type=int, default=0,
                        help='噪声种子（DIP中噪声固定）')
    parser.add_argument('--input_noise_ch', type=int, default=32,
                        help='输入噪声通道数')

    # 训练
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--base_ch', type=int, default=48)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=30,
                        help='早停 patience（验证 loss 不降则停止）')
    parser.add_argument('--print_every', type=int, default=200)

    # 路径
    parser.add_argument('--data_path', type=str,
                        default=r'F:\PINN实验\venv1\PINN数据3.xlsx')
    parser.add_argument('--output_dir', type=str, default='dip_results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_gpu', action='store_true')

    return parser.parse_args()


# =============================================================================
# 1. 数据加载（与 U-Net 代码保持一致）
# =============================================================================
def load_raw_data(file_path):
    threshold = 0.0005
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    if df.shape[1] > 7:
        df = df.iloc[:, :7]
    df.columns = ['lon', 'lat', 'h', 'X', 'Y', 'Z', 'F']
    df['X'] = df['X'].abs()
    df['Z'] = df['Z'].abs()

    df['diff_lon'] = df['lon'].diff().abs()
    df['line_id'] = (df['diff_lon'] > threshold).cumsum() + 1
    df_sampled = df.groupby('line_id', group_keys=False).apply(lambda g: g.iloc[::50])
    df = df_sampled.reset_index(drop=True)
    df['diff_lon'] = df['lon'].diff().abs()
    df['line_id'] = (df['diff_lon'] > threshold).cumsum() + 1

    lon0 = df['lon'].mean()
    lat0 = df['lat'].mean()
    R = 6371

    def ll_to_xy(lon, lat):
        x = (lon - lon0) * np.pi / 180 * R * np.cos(np.radians(lat0))
        y = (lat - lat0) * np.pi / 180 * R
        return x, y

    df['x'], df['y'] = ll_to_xy(df['lon'].values, df['lat'].values)

    LON_MIN, LON_MAX = 113.0085, 113.0155
    LAT_MIN, LAT_MAX = 34.5480, 34.5604
    mask_inside = ((df['lon'] >= LON_MIN) & (df['lon'] <= LON_MAX) &
                   (df['lat'] >= LAT_MIN) & (df['lat'] <= LAT_MAX))
    train_df = df[~mask_inside].copy()
    test_df = df[mask_inside].copy()

    return df, train_df, test_df, lon0, lat0, R, ll_to_xy, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX


def build_grid(train_df, lon0, lat0, R, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX,
               resolution_km=0.05):
    x_min, x_max = train_df['x'].min(), train_df['x'].max()
    y_min, y_max = train_df['y'].min(), train_df['y'].max()
    expand = -0.1
    x_min -= expand; x_max += expand
    y_min -= expand; y_max += expand

    x_grid = np.arange(x_min, x_max, resolution_km)
    y_grid = np.arange(y_min, y_max, resolution_km)
    nx, ny = len(x_grid), len(y_grid)

    grid_x, grid_y = np.meshgrid(x_grid, y_grid, indexing='ij')
    lon_grid_full = lon0 + grid_x / (R * np.cos(np.radians(lat0))) * (180 / np.pi)
    lat_grid_full = lat0 + grid_y / R * (180 / np.pi)

    mask_blank = ((lon_grid_full >= LON_MIN) & (lon_grid_full <= LON_MAX) &
                  (lat_grid_full >= LAT_MIN) & (lat_grid_full <= LAT_MAX))

    return x_grid, y_grid, grid_x, grid_y, lon_grid_full, lat_grid_full, mask_blank, nx, ny


def rbf_interpolate(train_df, grid_x, grid_y, mask_blank,
                    eps_mode='adaptive', fixed_eps=1.0):
    points_xy = train_df[['x', 'y']].values
    values_F = train_df['F'].values
    values_X = train_df['X'].values
    values_Y = train_df['Y'].values
    values_Z = train_df['Z'].values

    if len(points_xy) > 5000:
        idx_sample = np.random.choice(len(points_xy), 5000, replace=False)
        points_xy_rbf = points_xy[idx_sample]
        values_F_rbf = values_F[idx_sample]
        values_X_rbf = values_X[idx_sample]
        values_Y_rbf = values_Y[idx_sample]
        values_Z_rbf = values_Z[idx_sample]
    else:
        points_xy_rbf = points_xy
        values_F_rbf = values_F
        values_X_rbf = values_X
        values_Y_rbf = values_Y
        values_Z_rbf = values_Z

    if eps_mode == 'adaptive':
        tree = KDTree(points_xy_rbf)
        distances, _ = tree.query(points_xy_rbf, k=min(10, len(points_xy_rbf)))
        median_dist = np.median(distances[:, 1:])
        epsilon = median_dist * 0.8
    else:
        epsilon = fixed_eps
        median_dist = None

    print(f"  RBF epsilon: {epsilon:.4f} (mode={eps_mode})")

    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    rbf_F = RBFInterpolator(points_xy_rbf, values_F_rbf, kernel='cubic', epsilon=epsilon)
    F_grid = rbf_F(grid_points).reshape(grid_x.shape)

    rbf_X = RBFInterpolator(points_xy_rbf, values_X_rbf, kernel='thin_plate_spline', epsilon=epsilon)
    X_grid = rbf_X(grid_points).reshape(grid_x.shape)

    rbf_Y = RBFInterpolator(points_xy_rbf, values_Y_rbf, kernel='thin_plate_spline', epsilon=epsilon)
    Y_grid = rbf_Y(grid_points).reshape(grid_x.shape)

    rbf_Z = RBFInterpolator(points_xy_rbf, values_Z_rbf, kernel='thin_plate_spline', epsilon=epsilon)
    Z_grid = rbf_Z(grid_points).reshape(grid_x.shape)

    sigma_light = 0.15
    X_grid = gaussian_filter1d(X_grid, sigma_light, axis=0)
    X_grid = gaussian_filter1d(X_grid, sigma_light, axis=1)
    Y_grid = gaussian_filter1d(Y_grid, sigma_light, axis=0)
    Y_grid = gaussian_filter1d(Y_grid, sigma_light, axis=1)
    Z_grid = gaussian_filter1d(Z_grid, sigma_light, axis=0)
    Z_grid = gaussian_filter1d(Z_grid, sigma_light, axis=1)

    F_grid[mask_blank] = np.nan
    X_grid[mask_blank] = np.nan
    Y_grid[mask_blank] = np.nan
    Z_grid[mask_blank] = np.nan

    valid_mask = ~np.isnan(F_grid)
    valid_vals = F_grid[valid_mask]
    F_min, F_max = np.min(valid_vals), np.max(valid_vals)
    if np.isclose(F_max - F_min, 0):
        F_max = F_min + 1.0

    return F_grid, X_grid, Y_grid, Z_grid, F_min, F_max


def build_multichannel(F_grid, X_grid, Y_grid, Z_grid, mask_blank, input_ch=4):
    def normalize_channel(data, mask_valid):
        vals = data[mask_valid]
        if len(vals) == 0:
            return data
        minv, maxv = vals.min(), vals.max()
        if np.isclose(maxv - minv, 0):
            maxv = minv + 1.0
        return 2 * (data - minv) / (maxv - minv) - 1

    mask_nonblank = ~mask_blank
    F_norm = np.full_like(F_grid, np.nan)
    F_norm[mask_nonblank] = (
        2 * (F_grid[mask_nonblank] - F_grid[mask_nonblank].min())
        / (F_grid[mask_nonblank].max() - F_grid[mask_nonblank].min()) - 1
    )
    F_norm_filled = np.nan_to_num(F_norm, nan=0.0)

    if input_ch == 1:
        return F_norm_filled[np.newaxis, :, :]

    X_norm = normalize_channel(X_grid, mask_nonblank)
    Y_norm = normalize_channel(Y_grid, mask_nonblank)
    Z_norm = normalize_channel(Z_grid, mask_nonblank)
    X_norm[mask_blank] = 0.0
    Y_norm[mask_blank] = 0.0
    Z_norm[mask_blank] = 0.0
    return np.stack([F_norm_filled, X_norm, Y_norm, Z_norm], axis=0)


# =============================================================================
# 2. U-Net 模型（与原始代码相同）
# =============================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, use_skip=True):
        super().__init__()
        self.use_skip = use_skip
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        if use_skip:
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)
        else:
            self.conv = DoubleConv(in_ch, out_ch)  # 已修复：use_skip=False 时 in_ch → out_ch

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if self.use_skip and x2 is not None:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                         diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)


class UNetInpainter(nn.Module):
    def __init__(self, in_chans=4, base_ch=48, use_skip=True):
        super().__init__()
        self.use_skip = use_skip
        # 注意：DIP 模式下输入是随机噪声，in_chans 改为 noise_channels
        self.inc = DoubleConv(in_chans, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 4)
        self.down4 = Down(base_ch * 4, base_ch * 4)

        if use_skip:
            self.up1 = Up(base_ch * 4 + base_ch * 4, base_ch * 2, use_skip=True)
            self.up2 = Up(base_ch * 2 + base_ch * 4, base_ch * 2, use_skip=True)
            self.up3 = Up(base_ch * 2 + base_ch * 2, base_ch, use_skip=True)
            self.up4 = Up(base_ch + base_ch, base_ch, use_skip=True)
        else:
            self.up1 = Up(base_ch * 4, base_ch * 2, use_skip=False)
            self.up2 = Up(base_ch * 2, base_ch * 2, use_skip=False)
            self.up3 = Up(base_ch * 2, base_ch, use_skip=False)
            self.up4 = Up(base_ch, base_ch, use_skip=False)

        self.outc = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if self.use_skip:
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
        else:
            x = self.up1(x5)
            x = self.up2(x)
            x = self.up3(x)
            x = self.up4(x)

        return self.outc(x)


# =============================================================================
# 3. DIP 训练（核心）- 带早停
# =============================================================================
def dip_train(model, fixed_noise, target_full, known_mask, val_mask, grad_weight,
              epochs, lr, device, patience=0, print_every=200, log_callback=None):
    """
    Deep Image Prior 训练:
      - fixed_noise: 固定随机噪声 [1, C, H, W]
      - target_full: 目标值 [1, 1, H, W]
      - known_mask: 训练区域 (bool)
      - val_mask: 验证区域 (bool)，patience=0 时不用
      - patience=0: 无早停，全量训练
      - patience>0: val_mask 监控早停
    """
    model.train()
    fixed_noise = fixed_noise.to(device)
    target_full = target_full.to(device)
    known_mask = known_mask.to(device)
    val_mask = val_mask.to(device)
    use_early_stop = patience > 0

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32, device=device).view(1, 1, 3, 3)

    history = {'train_loss': [], 'val_loss': []}
    best_train_loss = float('inf')
    best_epoch = 0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(fixed_noise)

        loss_mse = nn.MSELoss()(output[known_mask], target_full[known_mask])

        if grad_weight > 0:
            gx_o = nn.functional.conv2d(output, sobel_x, padding=1)
            gy_o = nn.functional.conv2d(output, sobel_y, padding=1)
            grad_o = torch.sqrt(gx_o ** 2 + gy_o ** 2 + 1e-6)
            gx_t = nn.functional.conv2d(target_full, sobel_x, padding=1)
            gy_t = nn.functional.conv2d(target_full, sobel_y, padding=1)
            grad_t = torch.sqrt(gx_t ** 2 + gy_t ** 2 + 1e-6)
            loss_grad = nn.MSELoss()(grad_o[known_mask], grad_t[known_mask])
            train_loss = loss_mse + grad_weight * loss_grad
        else:
            train_loss = loss_mse

        train_loss.backward()
        optimizer.step()

        history['train_loss'].append(train_loss.item())

        # 早停模式：监控验证集；普通模式：监控训练集保存最优
        if use_early_stop:
            with torch.no_grad():
                val_loss = nn.MSELoss()(model(fixed_noise)[val_mask], target_full[val_mask])
            history['val_loss'].append(val_loss.item())
            monitor = val_loss.item()
        else:
            monitor = train_loss.item()

        if monitor < best_train_loss:
            best_train_loss = monitor
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        elif use_early_stop:
            patience_counter += 1

        if (epoch + 1) % print_every == 0 or epoch == 0:
            msg = (f"  Epoch {epoch+1:5d}/{epochs} | Loss: {train_loss.item():.6f} | Best: {best_train_loss:.6f}")
            if use_early_stop:
                msg += f" | Patience: {patience_counter}/{patience}"
            print(msg)
            if log_callback:
                log_callback(msg)

        if use_early_stop and patience_counter >= patience:
            print(f"  早停 @ Epoch {epoch+1}")
            if log_callback:
                log_callback(f"  早停 @ Epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return best_train_loss, best_epoch + 1, history


# =============================================================================
# 4. 评估
# =============================================================================
def evaluate_dip(model, fixed_noise, target_size, F_grid, mask_blank,
                 x_grid, y_grid, test_df, ll_to_xy, denormalize, nx, ny, device):
    model.eval()
    with torch.no_grad():
        output = model(fixed_noise.to(device))
    output_np = output.squeeze().cpu().numpy()
    output_np = denormalize(output_np)

    zoom_factors = (nx / target_size[0], ny / target_size[1])
    output_full = zoom(output_np, zoom_factors, order=1)
    output_full = output_full[:nx, :ny]

    result_grid = F_grid.copy()
    result_grid[mask_blank] = output_full[mask_blank]

    test_xy = np.column_stack(ll_to_xy(test_df['lon'].values, test_df['lat'].values))
    interp = RegularGridInterpolator((x_grid, y_grid), result_grid,
                                     method='linear', bounds_error=False,
                                     fill_value=np.nan)
    F_pred = interp(test_xy)
    valid_test = ~np.isnan(F_pred)

    if np.sum(valid_test) > 0:
        y_true = test_df['F'].values[valid_test]
        y_pred = F_pred[valid_test]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
    else:
        rmse = float('nan')
        mae = float('nan')

    return rmse, mae, result_grid, output_full


# =============================================================================
# 5. 单次实验
# =============================================================================
def run_experiment(args, data_cache, exp_cfg=None):
    cfg = vars(args).copy()
    if exp_cfg:
        cfg.update(exp_cfg)

    exp_name = cfg['exp_name']
    use_skip = not cfg['no_skip']
    input_ch = cfg['input_ch']
    grad_weight = cfg['grad_weight']
    eps_mode = cfg['eps_mode']
    fixed_eps = cfg['fixed_eps']
    noise_type = cfg['noise_type']
    input_noise_ch = cfg['input_noise_ch']
    epochs = cfg['epochs']
    base_ch = cfg['base_ch']
    lr = cfg['lr']
    print_every = cfg['print_every']
    output_dir = cfg['output_dir']
    seed = cfg['seed']
    device_str = 'cpu' if cfg['no_gpu'] else ('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"  实验: {exp_name}  (Deep Image Prior)")
    print(f"  skip={use_skip}, ch={input_ch}, grad_w={grad_weight}, "
          f"noise_type={noise_type}, noise_ch={input_noise_ch}")
    print(f"  epochs={epochs}, base_ch={base_ch}, lr={lr}, device={device_str}")
    print(f"{'='*60}")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(device_str)
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f'{exp_name}.log')

    def log(msg):
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    # 解包数据缓存
    train_df = data_cache['train_df']
    test_df = data_cache['test_df']

    # 构建网格
    (x_grid, y_grid, grid_x, grid_y, lon_grid_full, lat_grid_full,
     mask_blank, nx, ny) = build_grid(
        train_df, data_cache['lon0'], data_cache['lat0'], data_cache['R'],
        data_cache['LON_MIN'], data_cache['LON_MAX'],
        data_cache['LAT_MIN'], data_cache['LAT_MAX'])

    print(f"  网格: {nx} x {ny}, 训练点: {len(train_df)}, 测试点: {len(test_df)}")
    log(f"  网格: {nx} x {ny}, 训练点: {len(train_df)}, 测试点: {len(test_df)}")

    # RBF 插值
    print("  RBF 插值中...")
    F_grid, X_grid, Y_grid, Z_grid, F_min, F_max = \
        rbf_interpolate(train_df, grid_x, grid_y, mask_blank, eps_mode, fixed_eps)

    def normalize(x):
        return 2 * (x - F_min) / (F_max - F_min) - 1

    def denormalize(x):
        return (x + 1) / 2 * (F_max - F_min) + F_min

    # 构建多通道
    multi_channel = build_multichannel(F_grid, X_grid, Y_grid, Z_grid,
                                       mask_blank, input_ch)

    # 缩放到目标尺寸
    target_size = (128, 128)
    C = multi_channel.shape[0]
    multi_resized = np.zeros((C, target_size[0], target_size[1]), dtype=np.float32)
    for c in range(C):
        multi_resized[c] = resize(multi_channel[c], target_size,
                                  mode='constant', anti_aliasing=True)
    mask_blank_resized = resize(mask_blank.astype(float), target_size) > 0.5

    # DIP 关键：目标值（仅已知区域有值，空白区置0）
    target_full = multi_resized[0:1].copy()  # F 通道
    target_full[:, mask_blank_resized] = 0.0
    known_mask = ~mask_blank_resized

    target_tensor = torch.tensor(target_full, dtype=torch.float32).unsqueeze(0)
    known_mask_full = torch.tensor(known_mask, dtype=torch.bool)

    patience = cfg.get('patience', 0)
    if patience > 0:
        # 划分训练/验证：已知像素中 10% 用于早停监控
        known_indices = torch.nonzero(known_mask_full, as_tuple=False)
        n_known = len(known_indices)
        n_val = max(int(n_known * 0.1), 10)
        val_idx = known_indices[torch.randperm(n_known)[:n_val]]
        known_mask_train = known_mask_full.clone()
        known_mask_train[val_idx[:, 0], val_idx[:, 1]] = False
        val_mask_only = torch.zeros_like(known_mask_full)
        val_mask_only[val_idx[:, 0], val_idx[:, 1]] = True

        known_mask_tensor = known_mask_train.unsqueeze(0).unsqueeze(0)
        val_mask_tensor = val_mask_only.unsqueeze(0).unsqueeze(0)
        print(f"  训练像素: {known_mask_train.sum().item()}, 验证像素: {val_mask_only.sum().item()}")
        log(f"  训练像素: {known_mask_train.sum().item()}, 验证像素: {val_mask_only.sum().item()}")
    else:
        # 全部已知像素用于训练
        known_mask_tensor = known_mask_full.unsqueeze(0).unsqueeze(0)
        val_mask_tensor = known_mask_tensor  # dummy, 不会被用到

    # DIP 关键：固定随机噪声
    torch.manual_seed(cfg.get('noise_seed', 0))
    if noise_type == 'uniform':
        fixed_noise = torch.rand(1, input_noise_ch, target_size[0], target_size[1]) * 2 - 1
    elif noise_type == 'normal':
        fixed_noise = torch.randn(1, input_noise_ch, target_size[0], target_size[1])
    else:  # fixed_grid
        ys = torch.linspace(-1, 1, target_size[0])
        xs = torch.linspace(-1, 1, target_size[1])
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        grid_noise = torch.stack([gx, gy], dim=0)
        fixed_noise = grid_noise.unsqueeze(0).repeat(1, 1, 1, 1)
        if input_noise_ch > 2:
            extra = torch.randn(1, input_noise_ch - 2, target_size[0], target_size[1]) * 0.1
            fixed_noise = torch.cat([fixed_noise, extra], dim=1)

    # 模型：输入是 input_noise_ch 通道噪声，输出是 1 通道 F 值
    model = UNetInpainter(in_chans=input_noise_ch, base_ch=base_ch,
                          use_skip=use_skip).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")
    log(f"  参数量: {n_params:,}")

    # 训练
    print(f"  DIP 训练中 ({epochs} epochs, 无 mask 增强)...")
    t0 = time.time()
    best_val_loss, best_epoch_num, history = dip_train(
        model, fixed_noise, target_tensor, known_mask_tensor, val_mask_tensor,
        grad_weight, epochs, lr, device, patience=cfg.get('patience', 30),
        print_every=print_every, log_callback=log
    )
    train_time = time.time() - t0
    print(f"  早停在 Epoch {best_epoch_num}, best_val_loss={best_val_loss:.6f}")
    log(f"  早停在 Epoch {best_epoch_num}, best_val_loss={best_val_loss:.6f}")

    # 评估
    rmse, mae, result_grid, output_full = evaluate_dip(
        model, fixed_noise, target_size, F_grid, mask_blank,
        x_grid, y_grid, test_df, data_cache['ll_to_xy'],
        denormalize, nx, ny, device)

    result = {
        'exp_name': exp_name,
        'use_skip': use_skip,
        'input_ch': input_ch,
        'grad_weight': grad_weight,
        'noise_type': noise_type,
        'input_noise_ch': input_noise_ch,
        'eps_mode': eps_mode,
        'rmse': rmse, 'mae': mae,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch_num,
        'train_time': train_time,
        'n_params': n_params,
        'epochs': epochs,
        'timestamp': datetime.now().isoformat(),
    }

    print(f"\n  结果: RMSE={rmse:.4f}, MAE={mae:.4f}")
    print(f"  训练用时: {train_time:.1f}s")
    log(f"\n  结果: RMSE={rmse:.4f}, MAE={mae:.4f}")
    log(f"  训练用时: {train_time:.1f}s")

    # 保存 Loss 曲线
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history['train_loss'], linewidth=0.5, alpha=0.8, label='Train')
    ax.plot(history['val_loss'], linewidth=0.5, alpha=0.8, label='Val')
    ax.axvline(x=best_epoch_num, color='red', linestyle='--', alpha=0.5, label=f'Best@{best_epoch_num}')
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'DIP Training - {exp_name}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{exp_name}_loss.png'), dpi=150)
    plt.close()

    return result


# =============================================================================
# 6. 主函数
# =============================================================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据（所有实验共享）
    print("=" * 60)
    print("  加载数据...")
    print("=" * 60)
    (df, train_df, test_df, lon0, lat0, R, ll_to_xy,
     LON_MIN, LON_MAX, LAT_MIN, LAT_MAX) = load_raw_data(args.data_path)
    print(f"  训练点: {len(train_df)}, 测试点: {len(test_df)}")

    data_cache = {
        'train_df': train_df, 'test_df': test_df,
        'lon0': lon0, 'lat0': lat0, 'R': R, 'll_to_xy': ll_to_xy,
        'LON_MIN': LON_MIN, 'LON_MAX': LON_MAX,
        'LAT_MIN': LAT_MIN, 'LAT_MAX': LAT_MAX,
    }

    if args.run_all:
        experiments = [
            {'exp_name': '01_dip_baseline',   'no_skip': False, 'input_ch': 4, 'grad_weight': 0.1, 'noise_type': 'uniform', 'input_noise_ch': 32, 'eps_mode': 'adaptive'},
            {'exp_name': '02_dip_no_grad',    'no_skip': False, 'input_ch': 4, 'grad_weight': 0.0, 'noise_type': 'uniform', 'input_noise_ch': 32, 'eps_mode': 'adaptive'},
            {'exp_name': '03_dip_single_ch',  'no_skip': False, 'input_ch': 1, 'grad_weight': 0.1, 'noise_type': 'uniform', 'input_noise_ch': 32, 'eps_mode': 'adaptive'},
            {'exp_name': '04_dip_no_skip',    'no_skip': True,  'input_ch': 4, 'grad_weight': 0.0, 'noise_type': 'uniform', 'input_noise_ch': 32, 'eps_mode': 'adaptive'},
            {'exp_name': '05_dip_grid_noise', 'no_skip': False, 'input_ch': 4, 'grad_weight': 0.1, 'noise_type': 'fixed_grid', 'input_noise_ch': 32, 'eps_mode': 'adaptive'},
            {'exp_name': '06_dip_noise64',    'no_skip': False, 'input_ch': 4, 'grad_weight': 0.1, 'noise_type': 'uniform', 'input_noise_ch': 64, 'eps_mode': 'adaptive'},
            {'exp_name': '07_dip_wide',       'no_skip': False, 'input_ch': 4, 'grad_weight': 0.1, 'noise_type': 'uniform', 'input_noise_ch': 32, 'eps_mode': 'adaptive', 'base_ch': 64},
        ]
        all_results = []
        for exp_cfg in experiments:
            result = run_experiment(args, data_cache, exp_cfg)
            all_results.append(result)

        results_df = pd.DataFrame(all_results)
        csv_path = os.path.join(args.output_dir, 'dip_ablation_results.csv')
        results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        print("\n" + "=" * 90)
        print("  Deep Image Prior 消融实验结果汇总")
        print("=" * 90)
        hdr = f"{'实验':<22s} {'Skip':>5s} {'GradW':>6s} {'Noise':>9s} {'RMSE':>10s} {'MAE':>10s}"
        print(hdr)
        print("-" * 90)
        baseline_rmse = all_results[0]['rmse']
        for r in all_results:
            delta = (r['rmse'] - baseline_rmse) / baseline_rmse * 100
            print(f"{r['exp_name']:<22s} {str(r['use_skip']):>5s} {r['grad_weight']:>6.2f} "
                  f"{r['noise_type']:>9s} {r['rmse']:>10.4f} {r['mae']:>10.4f}  Δ{delta:+.1f}%")
        print("-" * 90)
        best = min(all_results, key=lambda x: x['rmse'])
        print(f"\n  最佳实验: {best['exp_name']} (RMSE={best['rmse']:.4f})")

    else:
        result = run_experiment(args, data_cache)
        json_path = os.path.join(args.output_dir, f'{args.exp_name}_result.json')
        json_result = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                       for k, v in result.items()}
        with open(json_path, 'w') as f:
            json.dump(json_result, f, indent=2, ensure_ascii=False)
        print(f"  结果已保存至: {json_path}")


if __name__ == '__main__':
    main()
