"""
EDA comparativo (Tarea 5, punto 3): descriptivos y distribuciones de las variables
OBSERVADAS principales ANTES (datos brutos, entrada al balanceo) vs DESPUÉS (muestra
balanceada + limpieza). Variables comparables = las presentes en el raw: edad, género,
sector y la propia respuesta. Las demográficas finas son sintéticas (no existen en el raw).

Genera: latex/figures/fig_eda_antes_despues.pdf  + imprime una tabla descriptiva.
Reproducible:  python scripts/gen_eda_antes_despues.py   (lee data/raw/, ~unos minutos)
"""
import warnings; warnings.filterwarnings("ignore")
import glob
import numpy as np
import pandas as pd
import janitor  # noqa: F401  (clean_names)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW, PROC, FIG = ROOT/"data"/"raw", ROOT/"data"/"processed", ROOT/"latex"/"figures"
from _fig_style import set_style, AZUL, ROJO, GRIS  # noqa: E402
set_style()
REF = pd.Timestamp("2026-01-05")
KEEP = ["email", "fecha_de_nacimiento", "sexo", "sector", "event_type"]

def edad(serie):
    b = pd.to_datetime(serie, errors="coerce")
    return (REF - b).dt.days // 365.25

# ── 1) Cargar RAW (entrada al balanceo): replica la carga de 01_clean ──
frames = []
for f in glob.glob(str(RAW/"*.csv")):
    df = pd.read_csv(f, sep=";", low_memory=False).clean_names()
    # "Antes" = TODOS los datos brutos (coherente con fig_eda_resampling y el embudo);
    # el filtro CPC->CPL es un paso de limpieza posterior, no parte del "antes".
    frames.append(df[[c for c in KEEP if c in df.columns]].copy())
    del df
raw = pd.concat(frames, ignore_index=True)
raw["click"] = (raw["event_type"] == "click").astype(int)
raw["age"] = edad(raw["fecha_de_nacimiento"])
raw["gender"] = raw["sexo"].map({"M": "H", "H": "H", "F": "M"})  # M=Male->H, F=Female->M

# Per-usuario (dedup por email, edad válida 18-120) para edad/género
raw_u = raw.drop_duplicates("email").copy()
raw_u = raw_u[(raw_u["age"] >= 18) & (raw_u["age"] <= 120)]

# ── 2) Cargar DESPUÉS (procesado) ──
users = pd.read_csv(PROC/"users.csv")
ev = pd.read_csv(PROC/"events.csv")
prods = pd.read_csv(PROC/"products.csv")[["id_product", "sector"]]
ev = ev[ev["event_type"].isin(["click", "open"])].merge(prods, on="id_product", how="left")
ev["click"] = (ev["event_type"] == "click").astype(int)

# ── 3) Tabla descriptiva ANTES vs DESPUÉS ──
def resumen(df_u, df_ev, etiqueta):
    return {
        "conjunto": etiqueta,
        "eventos": len(df_ev),
        "clic_%": round(100*df_ev["click"].mean(), 1),
        "usuarios": df_u["email"].nunique() if "email" in df_u else len(df_u),
        "edad_media": round(df_u["age"].mean(), 1),
        "edad_mediana": round(df_u["age"].median(), 1),
        "hombre_%": round(100*(df_u["gender"] == "H").mean(), 1),
    }
tab = pd.DataFrame([
    resumen(raw_u, raw, "ANTES (bruto)"),
    resumen(users.rename(columns={"id_user": "email"}), ev, "DESPUÉS (balanceado)"),
])
print("=== Descriptivos ANTES vs DESPUÉS ===")
print(tab.to_string(index=False))

# Mezcla por sector (per-evento, %)
sec_antes = raw["sector"].str.lower().str.strip().value_counts(normalize=True).mul(100)
sec_desp = ev["sector"].str.lower().str.strip().value_counts(normalize=True).mul(100)
top = sec_antes.head(6).index
print("\nMezcla por sector (% eventos) ANTES vs DESPUÉS (top 6):")
for s in top:
    print(f"  {s:20} antes {sec_antes.get(s,0):5.1f}%   despues {sec_desp.get(s,0):5.1f}%")

# ── 4) Figura comparativa (2x2) ──
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
# (a) Edad
ax[0,0].hist(raw_u["age"], bins=range(18,101,5), density=True, alpha=0.55, color=GRIS, label="Before (raw)")
ax[0,0].hist(users["age"].dropna(), bins=range(18,101,5), density=True, alpha=0.55, color=ROJO, label="After")
ax[0,0].set_title("Age (density, per user)"); ax[0,0].set_xlabel("years"); ax[0,0].legend()
# (b) Género
g_antes = [100*(raw_u["gender"]=="H").mean(), 100*(raw_u["gender"]=="M").mean()]
g_desp  = [100*(users["gender"]=="H").mean(), 100*(users["gender"]=="M").mean()]
xx = np.arange(2); w=0.38
ax[0,1].bar(xx-w/2, g_antes, w, color=GRIS, label="Before"); ax[0,1].bar(xx+w/2, g_desp, w, color=ROJO, label="After")
ax[0,1].set_xticks(xx); ax[0,1].set_xticklabels(["Male","Female"]); ax[0,1].set_ylabel("% of users")
ax[0,1].set_title("Gender (per user)"); ax[0,1].legend()
# (c) Mezcla por sector (top 6)
xs = np.arange(len(top))
ax[1,0].bar(xs-w/2, [sec_antes.get(s,0) for s in top], w, color=GRIS, label="Before")
ax[1,0].bar(xs+w/2, [sec_desp.get(s,0) for s in top], w, color=ROJO, label="After")
ax[1,0].set_xticks(xs); ax[1,0].set_xticklabels(top, rotation=30, ha="right", fontsize=8)
ax[1,0].set_ylabel("% of events"); ax[1,0].set_title("Sector mix (top 6)"); ax[1,0].legend()
# (d) Respuesta
r_antes=[100*(1-raw["click"].mean()), 100*raw["click"].mean()]
r_desp =[100*(1-ev["click"].mean()), 100*ev["click"].mean()]
ax[1,1].bar(xx-w/2, r_antes, w, color=GRIS, label="Before"); ax[1,1].bar(xx+w/2, r_desp, w, color=ROJO, label="After")
ax[1,1].set_xticks(xx); ax[1,1].set_xticklabels(["Open","Click"]); ax[1,1].set_ylabel("% of events")
ax[1,1].set_title("Response variable"); ax[1,1].legend()
fig.suptitle("Observed variables: before (raw) vs after balanced resampling", fontsize=13)
plt.tight_layout(); plt.savefig(FIG/"fig_eda_antes_despues.pdf", bbox_inches="tight"); plt.close()
print("\nfig_eda_antes_despues.pdf generada")
