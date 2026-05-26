"""Survey lines map — all data points colored by magnetic anomaly, jet colormap"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROB = r'F:\PINN实验\venv\U-net\robustness experiment'
FIG_DIR = os.path.join(ROB, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
DATA_PATH = r'F:\PINN实验\venv\U-net\afghanistan_full\Afghan_mag06A.csv'

print("Loading data...")
df = pd.read_csv(DATA_PATH)
# Subsample for plotting performance (every 5th point)
df_plot = df.iloc[::2].copy()
print(f"  Plotting {len(df_plot):,} / {len(df):,} points")

fig, ax = plt.subplots(figsize=(14, 10))

sc = ax.scatter(df_plot['Longitude'], df_plot['Latitude'],
                c=df_plot['FinalMag'], cmap='jet', s=1.2, linewidths=0,
                vmin=-300, vmax=200, rasterized=True)

ax.set_xlabel('Longitude (degE)', fontsize=16)
ax.set_ylabel('Latitude (degN)', fontsize=16)
ax.tick_params(labelsize=14)
ax.set_aspect(1 / np.cos(np.radians(df_plot['Latitude'].mean())))

cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label('Magnetic Anomaly (nT)', fontsize=16)
cbar.ax.tick_params(labelsize=14)

ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5)
fig.tight_layout()

for ext in ['png', 'svg']:
    dpi = 1000 if ext == 'png' else 400
    fig.savefig(os.path.join(FIG_DIR, f'fig_survey_lines.{ext}'), dpi=dpi,
                bbox_inches='tight', pad_inches=0.05)
plt.close(fig)
print(f"Done. Output: {FIG_DIR}/fig_survey_lines.png")
