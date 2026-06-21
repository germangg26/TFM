"""
Genera las figuras para el TFM en latex/figures/.
Ejecutar desde la raiz del proyecto con:
    .venv/Scripts/python scripts/gen_figures.py
"""
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path
import pickle

np.random.seed(42)

ROOT     = Path(__file__).parent.parent
PROC     = ROOT / "data" / "processed"
FIGURES  = ROOT / "latex" / "figures"
FIGURES.mkdir(exist_ok=True)

from _fig_style import set_style, AZUL, ROJO, VERDE, NEUTRO  # noqa: E402
set_style()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Serie temporal de clics semanales
# ─────────────────────────────────────────────────────────────────────────────
print("Generando fig_timeseries.pdf ...")
events = pd.read_csv(PROC / 'events.csv')
events['timestamp'] = pd.to_datetime(events['timestamp'])
clicks_weekly = (
    events[events['event_type'] == 'click']
    .set_index('timestamp')
    .resample('W-MON')['id_event']
    .count()
    .reset_index()
    .rename(columns={'timestamp': 'week', 'id_event': 'clicks'})
)

fig, ax = plt.subplots(figsize=(10, 3.8))
ax.fill_between(clicks_weekly['week'], clicks_weekly['clicks'], alpha=0.15, color=AZUL)
ax.plot(clicks_weekly['week'], clicks_weekly['clicks'],
        color=AZUL, lw=1.5, marker='o', ms=2.5)

split = pd.Timestamp('2025-07-28')
ax.axvspan(clicks_weekly['week'].min(), split,
           alpha=0.08, color=NEUTRO, label='Inactive period (<50 clicks/week)')
ax.axvspan(split, clicks_weekly['week'].max(),
           alpha=0.08, color=AZUL, label='Active period')
ax.axvline(split, color=NEUTRO, lw=1, ls='--')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.set_ylabel('Clicks per week')
ax.set_title('Weekly evolution of click volume (Jan 2025 – Jan 2026)')
ax.legend(loc='upper left')
ax.set_xlim(clicks_weekly['week'].min(), clicks_weekly['week'].max())
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
plt.tight_layout()
fig.savefig(FIGURES / 'fig_timeseries.pdf', bbox_inches='tight')
plt.close()
print("  fig_timeseries.pdf OK")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Forecast M4 (período activo + 8 semanas)
# ─────────────────────────────────────────────────────────────────────────────
print("Generando fig_forecast.pdf ...")
fc = pd.read_csv(PROC / 'forecast_clicks.csv')
fc['week'] = pd.to_datetime(fc['week'])
hist = fc[~fc['is_forecast']].copy()
fut  = fc[fc['is_forecast']].copy()

# Clics reales activos
active = clicks_weekly[clicks_weekly['clicks'] >= 50].copy()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(active['week'], active['clicks'],
        color=AZUL, lw=1.8, marker='o', ms=4,
        label='Actual clicks (active period)', zorder=4)
ax.plot(hist['week'], hist['clicks_forecast'],
        color=ROJO, lw=1.5, ls='--', label='Model fit', zorder=3)
ax.fill_between(hist['week'],
                hist['clicks_lower_95'].clip(lower=0),
                hist['clicks_upper_95'],
                alpha=0.1, color=ROJO)
ax.plot(fut['week'], fut['clicks_forecast'],
        color=VERDE, lw=2, marker='D', ms=4.5,
        label='Forecast +8 weeks', zorder=4)
ax.fill_between(fut['week'],
                fut['clicks_lower_95'].clip(lower=0),
                fut['clicks_upper_95'],
                alpha=0.18, color=VERDE, label='90% CI')
cut = hist['week'].max()
ax.axvline(cut, color=NEUTRO, lw=1, ls=':', label='Horizon start')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.set_ylabel('Clicks per week')
ax.set_title('Weekly click forecast — Prophet (active period + 8 weeks)')
ax.legend(loc='upper right', ncol=2)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(max(x,0)):,}'))
plt.tight_layout()
fig.savefig(FIGURES / 'fig_forecast.pdf', bbox_inches='tight')
plt.close()
print("  fig_forecast.pdf OK")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — UMAP de los clusters demográficos M0
# ─────────────────────────────────────────────────────────────────────────────
print("Generando fig_umap_m0.pdf ...")

# Cargar datos
users = pd.read_csv(PROC / 'users.csv')
segs  = pd.read_csv(PROC / 'users_demo_segments.csv')
df = users.merge(segs[['id_user', 'demo_cluster']], on='id_user', how='inner')

# Encoding ordinal (mismo orden que el notebook 05_demo_segmentation)
gender_map   = {'H': 1, 'M': 0}
labor_map    = {'employed': 2, 'unemployed': 1, 'inactive': 0}
civil_map    = {'casado': 3, 'viudo': 2, 'divorciado': 1, 'soltero': 0}
coche_map    = {True: 1, False: 0}
# size_hogar — extraer número
def parse_size_hogar(x):
    if pd.isna(x):
        return np.nan
    x = str(x)
    if '1 ' in x or x.strip().startswith('1'):
        return 1
    elif '2 ' in x:
        return 2
    elif '3 ' in x:
        return 3
    elif '4 ' in x:
        return 4
    return 5
room_map = {'menos_3_hab': 1, '3_a_6_hab': 2, '7_mas_hab': 3,
            'menos_3': 1, '3_a_6': 2, '7_mas': 3}

df['gender_enc']       = df['gender'].map(gender_map)
df['labor_enc']        = df['labor_status'].map(labor_map)
df['civil_enc']        = df['civil_status'].map(civil_map)
df['coche_enc']        = df['tiene_coche'].map(coche_map)
df['hogar_enc']        = df['size_hogar'].apply(parse_size_hogar)
df['room_enc']         = df['num_room'].map(room_map)

feat_cols = ['age', 'gender_enc', 'labor_enc', 'civil_enc',
             'coche_enc', 'hogar_enc', 'room_enc',
             'ipa_class', 'mun_type', 'distance_type']

X_raw = df[feat_cols].fillna(-1).values.astype(float)

# Cargar scaler y encoder
with open(PROC / 'demo_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

import tensorflow as tf
encoder = tf.keras.models.load_model(str(PROC / 'demo_encoder.keras'))

X_sc = scaler.transform(X_raw)
X_ae = encoder.predict(X_sc, verbose=0)

# UMAP sobre muestra
try:
    import umap as umap_mod
    N = min(10000, len(X_ae))
    idx = np.random.choice(len(X_ae), N, replace=False)
    X_sub = X_ae[idx]
    labels_sub = df['demo_cluster'].values[idx]

    reducer = umap_mod.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                            random_state=42, verbose=False)
    emb = reducer.fit_transform(X_sub)

    palette = sns.color_palette('tab10', 10)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for k in range(10):
        mask = labels_sub == k
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   s=3, alpha=0.4, color=palette[k], label=f'C{k}', rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('UMAP projection of the 10 demographic clusters (M0)\n'
                 f'n = {N:,} users, 4D latent → 2D UMAP')
    handles, labels_leg = ax.get_legend_handles_labels()
    ax.legend(handles, labels_leg, title='Cluster', loc='best',
              markerscale=3, fontsize=8, ncol=2,
              handler_map={plt.scatter: plt.scatter})
    plt.tight_layout()
    fig.savefig(FIGURES / 'fig_umap_m0.pdf', bbox_inches='tight',
                dpi=150)
    plt.close()
    print("  fig_umap_m0.pdf OK")
except ImportError:
    print("  UMAP no disponible — omitiendo fig_umap_m0.pdf")


print("\nFiguras generadas en:", FIGURES)
for f in sorted(FIGURES.iterdir()):
    print(f"  {f.name}")
