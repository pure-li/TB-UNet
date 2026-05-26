"""Three blank regions schematic — single background map with 3 labeled blank areas"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROB = r'F:\PINN实验\venv\U-net\robustness experiment'
FIG_DIR = os.path.join(ROB, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
DATA_PATH = r'F:\PINN实验\venv\U-net\afghanistan_full\Afghan_mag06A.csv'

REGIONS = [
    {'key': '1', 'label': '① NW (0.5deg x 0.5deg)',  'rect': (61.5, 62.0, 34.2, 34.7), 'fc': 'white', 'ec': 'black'},
    {'key': '2', 'label': '② Center (1.0deg x 1.0deg)', 'rect': (62.0, 63.0, 32.5, 33.5), 'fc': 'white', 'ec': 'black'},
    {'key': '3', 'label': '③ SE (1.5deg x 1.5deg)',   'rect': (64.75, 66.25, 30.25, 31.75), 'fc': 'white', 'ec': 'black'},
]

print("Loading full dataset for background interpolation...")
df_full = pd.read_csv(DATA_PATH)
df_bg = df_full.iloc[::3].copy().reset_index(drop=True)
lon0, lat0 = df_bg['Longitude'].mean(), df_bg['Latitude'].mean()
R_map = 6371

def ll_to_xy(lon, lat):
    x = (lon - lon0) * np.pi/180 * R_map * np.cos(np.radians(lat0))
    y = (lat - lat0) * np.pi/180 * R_map
    return x, y

pts_all = np.column_stack(ll_to_xy(df_bg['Longitude'].values, df_bg['Latitude'].values))
vals_all = df_bg['FinalMag'].values

# Grid
grid_spacing = 1.0
x_all, y_all = pts_all[:,0], pts_all[:,1]
buf = 5.0  # small buffer in km
x_min, x_max = x_all.min()-buf, x_all.max()+buf
y_min, y_max = y_all.min()-buf, y_all.max()+buf
x_grid = np.arange(x_min, x_max, grid_spacing)
y_grid = np.arange(y_min, y_max, grid_spacing)
grid_x, grid_y = np.meshgrid(x_grid, y_grid, indexing='ij')
grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
lon_grid = lon0 + grid_x/(R_map*np.cos(np.radians(lat0)))*(180/np.pi)
lat_grid = lat0 + grid_y/R_map*(180/np.pi)

# Convex hull mask
hull = ConvexHull(pts_all)
hull_xy = hull.points[hull.vertices]
xc = hull_xy[:,0].mean()
for i in range(len(hull_xy)):
    hull_xy[i,0] += -2. if hull_xy[i,0] < xc else 2.
mask_outside = ~Path(hull_xy).contains_points(grid_pts).reshape(grid_x.shape)

print(f"  Grid: {grid_x.shape}, outside: {mask_outside.sum():,} / {mask_outside.size:,}")

print("  Computing RBF background...")
n_rbf = min(5000, len(df_bg))
idx = np.random.choice(len(df_bg), n_rbf, replace=False)
rbf = RBFInterpolator(pts_all[idx], vals_all[idx], kernel='thin_plate_spline')
bg_grid = rbf(grid_pts).reshape(grid_x.shape)
bg_grid = gaussian_filter1d(bg_grid, 2.0, axis=0)
bg_grid = gaussian_filter1d(bg_grid, 1.0, axis=1)
bg_grid[mask_outside] = np.nan

print("  Plotting...")
fig, ax = plt.subplots(figsize=(13, 10))

vmin, vmax = -300, 200
im = ax.imshow(bg_grid.T, cmap='jet', origin='lower', vmin=vmin, vmax=vmax,
               extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]], aspect='auto')

# Draw blank region rectangles
for reg in REGIONS:
    lon1, lon2, lat1, lat2 = reg['rect']
    x1, y1 = ll_to_xy(lon1, lat1)
    x2, y2 = ll_to_xy(lon2, lat2)
    w, h = x2 - x1, y2 - y1
    rect = mpatches.Rectangle((x1, y1), w, h, linewidth=2.8,
                               edgecolor=reg['ec'], facecolor=reg['fc'],
                               alpha=0.93, zorder=10)
    ax.add_patch(rect)
    cx, cy = x1 + w/2, y1 + h/2
    ax.text(cx, cy, reg['key'], fontsize=28, fontweight='bold', ha='center', va='center',
            color='black', zorder=11)

legend_handles = [
    mpatches.Patch(facecolor='white', edgecolor='black', linewidth=2,
                   label=f"{r['key']} {r['label'][2:]}")
    for r in REGIONS
]
leg = ax.legend(handles=legend_handles, fontsize=16, loc='upper right',
                framealpha=0.92, edgecolor='gray')
leg.set_zorder(12)

ax.set_xticks([])
ax.set_yticks([])

cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label('nT', fontsize=16)
cbar.ax.tick_params(labelsize=14)

fig.tight_layout()
for ext in ['png', 'svg']:
    dpi = 1000 if ext == 'png' else 800
    fig.savefig(os.path.join(FIG_DIR, f'fig_three_regions.{ext}'), dpi=dpi,
                bbox_inches='tight', pad_inches=0.05)
plt.close(fig)
print(f"Done. Output: {FIG_DIR}/fig_three_regions.png")
