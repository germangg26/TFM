"""
Genera las figuras del análisis exploratorio (EDA) para el capítulo de Datos,
en latex/figures/. Reproducible desde data/processed/.
Ejecutar desde la raíz:  python scripts/gen_eda_figures.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(42)

ROOT = Path(__file__).parent.parent
PROC = ROOT / "data" / "processed"
FIG = ROOT / "latex" / "figures"
FIG.mkdir(exist_ok=True)

from _fig_style import set_style, AZUL, ROJO, GRIS, NEUTRO  # noqa: E402
set_style()

events = pd.read_csv(PROC / "events.csv")
users = pd.read_csv(PROC / "users.csv")
prods = pd.read_csv(PROC / "products.csv")

# ── 1) Variable objetivo: click vs apertura ──────────────────────────────────
vc = events["event_type"].value_counts()
total = vc.sum()
fig, ax = plt.subplots(figsize=(5.2, 3.6))
barras = ax.bar(["Apertura", "Clic"], [vc.get("open", 0), vc.get("click", 0)],
                color=[GRIS, ROJO], edgecolor="white", width=0.6)
for b, n in zip(barras, [vc.get("open", 0), vc.get("click", 0)]):
    ax.text(b.get_x() + b.get_width() / 2, n + 400,
            f"{n:,}\n({100*n/total:.1f}\\%)", ha="center", fontsize=10)
ax.set_ylabel("Nº de eventos")
ax.set_title("Distribución de la variable objetivo")
ax.set_ylim(0, total * 0.95)
plt.tight_layout(); plt.savefig(FIG / "fig_eda_target.pdf", bbox_inches="tight"); plt.close()
print("fig_eda_target  -> clic:", vc.get("click", 0), "apertura:", vc.get("open", 0))

# ── 2) Distribuciones demográficas (panel 2x3) ───────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(13, 6.8))

# Edad (continua)
axes[0, 0].hist(users["age"].dropna(), bins=range(18, 101, 5),
                color=AZUL, edgecolor="white")
axes[0, 0].set_title("Edad (continua)"); axes[0, 0].set_xlabel("años"); axes[0, 0].set_ylabel("usuarios")

# Edad categorizada (age_cat: mismas bandas que el modelo M2)
def edad_a_grupo(edad):
    if pd.isna(edad):
        return -1
    for i, lim in enumerate([25, 35, 45, 55, 65]):
        if edad < lim:
            return i
    return 5
age_cat = users["age"].apply(edad_a_grupo)
cat_lbl = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
ac = age_cat[age_cat >= 0].value_counts().sort_index()
axes[0, 1].bar([cat_lbl[k] for k in ac.index], ac.values, color=ROJO, edgecolor="white", width=0.7)
axes[0, 1].set_title("Edad categorizada (age_cat, M2)"); axes[0, 1].set_ylabel("usuarios")
axes[0, 1].tick_params(axis="x", labelrotation=30)

# Género
g = users["gender"].value_counts()
axes[0, 2].bar([{"H": "Hombre", "M": "Mujer"}.get(k, k) for k in g.index], g.values,
               color=AZUL, edgecolor="white", width=0.6)
axes[0, 2].set_title("Género"); axes[0, 2].set_ylabel("usuarios")

# Situación laboral
lab = users["labor_status"].value_counts()
lab_lbl = {"employed": "Empleado", "inactive": "Inactivo", "unemployed": "Desempleado"}
axes[1, 0].bar([lab_lbl.get(k, k) for k in lab.index], lab.values,
               color=AZUL, edgecolor="white", width=0.6)
axes[1, 0].set_title("Situación laboral (sintética)"); axes[1, 0].set_ylabel("usuarios")
axes[1, 0].tick_params(axis="x", labelrotation=20)

# Estado civil
civ = users["civil_status"].value_counts()
civ_lbl = {"casado": "Casado", "soltero": "Soltero", "divorciado": "Divorciado", "viudo": "Viudo"}
axes[1, 1].bar([civ_lbl.get(k, k) for k in civ.index], civ.values,
               color=AZUL, edgecolor="white", width=0.6)
axes[1, 1].set_title("Estado civil (sintético)"); axes[1, 1].set_ylabel("usuarios")
axes[1, 1].tick_params(axis="x", labelrotation=20)

# Clase de renta (IPA)
ipa = users["ipa_class"].value_counts().sort_index()
ipa_lbl = {0: "Baja", 1: "Baja-media", 2: "Media", 3: "Media-alta", 4: "Alta"}
axes[1, 2].bar([ipa_lbl.get(k, k) for k in ipa.index], ipa.values,
               color=AZUL, edgecolor="white", width=0.6)
axes[1, 2].set_title("Clase de renta (IPA, geográfica)"); axes[1, 2].set_ylabel("usuarios")
axes[1, 2].tick_params(axis="x", labelrotation=20)

fig.suptitle("Distribuciones demográficas de los usuarios", fontsize=13)
plt.tight_layout(); plt.savefig(FIG / "fig_eda_demographics.pdf", bbox_inches="tight"); plt.close()
print("fig_eda_demographics -> edad media:", round(users["age"].mean(), 1),
      "| age_cat:", dict(zip([cat_lbl[k] for k in ac.index], ac.values)))

# ── 3) CTR (tasa de clic) por sector ─────────────────────────────────────────
em = events.merge(prods[["id_product", "sector"]], on="id_product", how="left")
em["click"] = (em["event_type"] == "click").astype(int)
ctr = em.groupby("sector")["click"].agg(ctr="mean", n="count").sort_values("ctr")
ctr = ctr[ctr["n"] >= 100]  # sectores con volumen suficiente
fig, ax = plt.subplots(figsize=(8, 4.4))
colores = [ROJO if v >= 0.238 else AZUL for v in ctr["ctr"]]
b = ax.barh(ctr.index, ctr["ctr"], color=colores, edgecolor="white")
ax.axvline(0.238, color=NEUTRO, ls="--", lw=1, label="CTR global (0,238)")
for i, (v, n) in enumerate(zip(ctr["ctr"], ctr["n"])):
    ax.text(v + 0.005, i, f"{v:.2f}  (n={n:,})", va="center", fontsize=8)
ax.set_xlabel("Tasa de clic (clics / eventos)")
ax.set_title("Tasa de clic por sector")
ax.set_xlim(0, ctr["ctr"].max() + 0.12); ax.legend(loc="lower right")
plt.tight_layout(); plt.savefig(FIG / "fig_eda_ctr_sector.pdf", bbox_inches="tight"); plt.close()
print("fig_eda_ctr_sector -> mejor:", ctr.index[-1], round(ctr['ctr'].iloc[-1], 3),
      "| peor:", ctr.index[0], round(ctr['ctr'].iloc[0], 3))

# ── 4) Esparsidad: nº de eventos por usuario ─────────────────────────────────
epu = events.groupby("id_user").size()
buckets = epu.clip(upper=5).value_counts().sort_index()
labels = {1: "1", 2: "2", 3: "3", 4: "4", 5: "5+"}
fig, ax = plt.subplots(figsize=(6.2, 3.8))
barras = ax.bar([labels[k] for k in buckets.index], buckets.values,
                color=AZUL, edgecolor="white", width=0.65)
for b2, n in zip(barras, buckets.values):
    ax.text(b2.get_x() + b2.get_width() / 2, n + 200,
            f"{100*n/len(epu):.1f}\\%", ha="center", fontsize=9)
ax.set_xlabel("Nº de eventos del usuario"); ax.set_ylabel("usuarios")
ax.set_title("Actividad por usuario (matriz usuario--producto 98,1\\% vacía)")
plt.tight_layout(); plt.savefig(FIG / "fig_eda_sparsity.pdf", bbox_inches="tight"); plt.close()
print("fig_eda_sparsity -> %1 evento:", round(100*(epu == 1).mean(), 1), "| max:", epu.max())

# ── 5) Catálogo de producto: volumen por categoría + distribución del CPL ────
em2 = events.merge(prods[["id_product", "sector", "product_new", "cpl"]], on="id_product", how="left")
em2["click"] = (em2["event_type"] == "click").astype(int)
cat = em2.groupby("product_new")["click"].agg(ctr="mean", n="count").sort_values("n", ascending=False).head(12)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
# Izquierda: top categorías por volumen, color por CTR
colores = [ROJO if v >= 0.238 else AZUL for v in cat["ctr"]]
axes[0].barh(cat.index[::-1], cat["n"].values[::-1], color=colores[::-1], edgecolor="white")
for i, (n, c) in enumerate(zip(cat["n"].values[::-1], cat["ctr"].values[::-1])):
    axes[0].text(n + 80, i, f"CTR {c:.2f}", va="center", fontsize=7.5)
axes[0].set_xlabel("Nº de eventos"); axes[0].set_title("Top 12 categorías por volumen (color: CTR vs media)")
axes[0].set_xlim(0, cat["n"].max() * 1.18)
# Derecha: distribución del CPL (productos con CPL)
cpl = pd.to_numeric(prods["cpl"], errors="coerce").dropna()
axes[1].hist(cpl, bins=range(0, 34, 2), color=AZUL, edgecolor="white")
axes[1].axvline(cpl.median(), color=ROJO, ls="--", lw=1.2, label=f"mediana = {cpl.median():.0f}\\,\\euro")
axes[1].set_xlabel("CPL (\\euro\\ por lead)"); axes[1].set_ylabel("nº de productos")
axes[1].set_title(f"Distribución del CPL ({cpl.shape[0]}/{len(prods)} productos)"); axes[1].legend()
plt.tight_layout(); plt.savefig(FIG / "fig_eda_products.pdf", bbox_inches="tight"); plt.close()
print("fig_eda_products -> top cat:", cat.index[0], "| CPL mediana:", cpl.median())

# ── 6) CTR por edad: CONTINUA vs CATEGORIZADA (para ver qué se pierde al agrupar) ──
um = events.merge(users[["id_user", "age", "gender"]], on="id_user", how="left")
um["click"] = (um["event_type"] == "click").astype(int)

# (a) CTR por edad CONTINUA, en tramos finos de 5 años
bins = list(range(18, 96, 5))
um["tramo"] = pd.cut(um["age"], bins=bins, right=False)
ctr_cont = um.groupby("tramo", observed=True)["click"].mean()
centros = [b + 2.5 for b in bins[:-1]][:len(ctr_cont)]

# (b) CTR por BANDA age_cat (las 6 bandas del modelo M2)
def _band(a):
    if pd.isna(a):
        return None
    for lim, lbl in [(25, "18-24"), (35, "25-34"), (45, "35-44"), (55, "45-54"), (65, "55-64")]:
        if a < lim:
            return lbl
    return "65+"
um["ageband"] = um["age"].apply(_band)
orden = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
ctr_band = um.groupby("ageband")["click"].mean().reindex(orden)
ctr_gen = um.groupby("gender")["click"].mean()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
# Izquierda: edad continua (línea)
axes[0].plot(centros, ctr_cont.values, "o-", color=AZUL)
axes[0].axhline(0.238, color=NEUTRO, ls="--", lw=1, label="CTR global (0,238)")
axes[0].set_xlabel("Edad (años)"); axes[0].set_ylabel("Tasa de clic")
axes[0].set_title("CTR por edad continua (tramos de 5 años)"); axes[0].legend()
# Centro: edad categorizada (barras)
axes[1].bar(orden, ctr_band.values, color=AZUL, edgecolor="white", width=0.7)
axes[1].axhline(0.238, color=NEUTRO, ls="--", lw=1)
axes[1].set_ylabel("Tasa de clic"); axes[1].set_title("CTR por banda de edad (age\\_cat, M2)")
axes[1].tick_params(axis="x", labelrotation=20)
# Derecha: género
axes[2].bar([{"H": "Hombre", "M": "Mujer"}.get(k, k) for k in ctr_gen.index], ctr_gen.values,
            color=ROJO, edgecolor="white", width=0.55)
axes[2].axhline(0.238, color=NEUTRO, ls="--", lw=1)
axes[2].set_ylabel("Tasa de clic"); axes[2].set_title("CTR por género")
fig.suptitle("Tasa de clic por variables demográficas reales (edad continua vs categorizada)", fontsize=12)
plt.tight_layout(); plt.savefig(FIG / "fig_eda_ctr_demo.pdf", bbox_inches="tight"); plt.close()
print("fig_eda_ctr_demo -> CTR H/M:", round(ctr_gen.get("H", 0), 3), round(ctr_gen.get("M", 0), 3))

print("\nFiguras EDA generadas en", FIG)
