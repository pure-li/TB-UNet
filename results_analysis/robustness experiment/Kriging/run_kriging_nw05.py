"""Kriging on 0.5° NW region: 61.3-61.8°E, 34.8-35.3°N"""
import os, time, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.spatial import ConvexHull, KDTree
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pykrige.ok import OrdinaryKriging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_PATH = r'F:\PINN实验\venv\U-net\afghanistan_full\Afghan_mag06A.csv'
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42; np.random.seed(SEED)
N_KRIGING_CTRL = 2000; N_TRUTH_SAMPLES = 3000
RECT = (61.3, 61.8, 34.8, 35.3)
REGION_KEY = 'nw05'

print("=" * 60)
print(f"  Kriging — 0.5° NW region: {RECT}")
print("=" * 60)

# ---- Data loading (same as SWA script) ----
df_full = pd.read_csv(DATA_PATH)
df = df_full.iloc[::3].copy().reset_index(drop=True)
lon_min, lon_max, lat_min, lat_max = RECT
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

print(f"  Train: {len(train_df):,}, Test: {len(test_df):,}, Blank: {mask_blank.sum():,}")

# ---- Truth grid ----
df_truth = df_full.iloc[::max(1, len(df_full)//N_TRUTH_SAMPLES)].copy()
pts_truth = np.column_stack(ll_to_xy(df_truth['Longitude'].values, df_truth['Latitude'].values))
val_truth = df_truth['FinalMag'].values
truth_grid = RBFInterpolator(pts_truth, val_truth, kernel='thin_plate_spline')(grid_pts).reshape(grid_x.shape)
truth_grid = gaussian_filter1d(truth_grid, 2.0, axis=0)
truth_grid = gaussian_filter1d(truth_grid, 1.0, axis=1)
truth_grid[mask_outside] = np.nan

# ---- Kriging ----
print("  Running Ordinary Kriging...")
t0 = time.time()
n_sub = min(N_KRIGING_CTRL, len(train_df))
idx = np.random.choice(len(train_df), n_sub, replace=False)
krige_pts = train_df[['x','y']].values[idx]
krige_vals = train_df['FinalMag'].values[idx]

blank_idx = np.where(mask_blank.ravel())[0]
OK = OrdinaryKriging(krige_pts[:,0], krige_pts[:,1], krige_vals,
                     variogram_model='spherical', enable_plotting=False, verbose=False)
z_blank, _ = OK.execute('points', grid_pts[blank_idx,0], grid_pts[blank_idx,1])

result_grid = np.full(grid_pts.shape[0], np.nan, dtype=np.float32)
result_grid[blank_idx] = z_blank.astype(np.float32)
result_grid = result_grid.reshape(grid_x.shape)

# Fill blank region with Kriging, keep RBF elsewhere
rbf_grid = RBFInterpolator(krige_pts, krige_vals, kernel='cubic',
                           epsilon=KDTree(krige_pts).query(krige_pts, k=min(10,n_sub))[0][:,1:].mean()*0.8)(grid_pts).reshape(grid_x.shape)

F_display = rbf_grid.copy()
F_display = gaussian_filter1d(F_display, 5., axis=0); F_display = gaussian_filter1d(F_display, 1., axis=1)
F_display[mask_outside] = np.nan

final_grid = rbf_grid.copy()
temp = result_grid.copy()
temp = gaussian_filter1d(temp, 1.5, axis=0); temp = gaussian_filter1d(temp, 1.5, axis=1)
final_grid[mask_blank] = temp[mask_blank]

krige_time = time.time() - t0
print(f"  Kriging done in {krige_time:.0f}s")

# ---- RMSE on test points ----
from scipy.interpolate import RegularGridInterpolator
interp = RegularGridInterpolator((x_grid, y_grid), final_grid, method='linear', bounds_error=False, fill_value=np.nan)
pred_v = interp(np.column_stack([test_df['x'].values, test_df['y'].values]))
valid = ~np.isnan(pred_v)
rmse = float(np.sqrt(mean_squared_error(test_df['FinalMag'].values[valid], pred_v[valid])))
mae = float(mean_absolute_error(test_df['FinalMag'].values[valid], pred_v[valid]))
print(f"  Kriging RMSE={rmse:.2f}, MAE={mae:.2f}")

# ---- Save ----
np.save(os.path.join(OUT_DIR, f'result_grid_{REGION_KEY}.npy'), final_grid)
np.save(os.path.join(OUT_DIR, f'truth_grid_{REGION_KEY}.npy'), truth_grid)
np.save(os.path.join(OUT_DIR, f'mask_blank_{REGION_KEY}.npy'), mask_blank)
np.save(os.path.join(OUT_DIR, f'mask_outside_{REGION_KEY}.npy'), mask_outside)
np.save(os.path.join(OUT_DIR, f'grid_x_{REGION_KEY}.npy'), grid_x)
np.save(os.path.join(OUT_DIR, f'grid_y_{REGION_KEY}.npy'), grid_y)

with open(os.path.join(OUT_DIR, f'results_{REGION_KEY}.json'), 'w') as f:
    json.dump({'region': '0.5° NW', 'rect': RECT, 'rmse': rmse, 'mae': mae, 'time': krige_time}, f, indent=2)

print(f"  Done! RMSE={rmse:.2f}, time={krige_time:.0f}s")
