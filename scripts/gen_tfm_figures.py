"""Genera las figuras del TFM (post-auditoría) como PDF en latex/figures/.
Usa los números finales documentados en reports/RESULTADOS_TFM.md."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from _fig_style import set_style, NAVY, AZUL, ROJO, NARANJA, GRIS, NEUTRO

FIG = Path(__file__).resolve().parents[1] / "latex" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
set_style()

# ── 1) M2: fuga vs honesto (ROC-AUC) ──
fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.bar(["StratifiedKFold\n(por evento, con fuga)", "GroupKFold\n(por usuario, honesto)"],
       [0.695, 0.563], color=[GRIS, ROJO], edgecolor="white")
ax.axhline(0.5, color=NEUTRO, ls="--", lw=1, label="azar (0,5)")
for x, v in zip([0, 1], [0.695, 0.563]):
    ax.text(x, v + 0.01, f"{v:.3f}".replace(".", ","), ha="center", fontweight="bold")
ax.set_ylabel("AUC-ROC (OOF)"); ax.set_ylim(0, 0.85); ax.legend()
ax.set_title("M2: efecto de corregir la fuga de datos")
plt.tight_layout(); plt.savefig(FIG / "fig_m2_leakage.pdf", bbox_inches="tight"); plt.close()

# ── 2) M2: comparación de estrategias (PR-AUC con IC) — modelo plano adoptado ──
modelos = ["Baseline\npopularidad", "Solo variables\nde usuario", "Con producto\n(modelo plano)"]
pr = [0.313, 0.322, 0.402]
lo = [0.309, 0.316, 0.395]
hi = [0.319, 0.328, 0.410]
err = [np.array(pr) - np.array(lo), np.array(hi) - np.array(pr)]
fig, ax = plt.subplots(figsize=(6.0, 3.4))
colores = [GRIS, NARANJA, ROJO]
ax.bar(modelos, pr, yerr=err, capsize=5, color=colores, edgecolor="white")
ax.axhline(0.238, color=NEUTRO, ls="--", lw=1, label="azar (prevalencia 0,238)")
for x, v in enumerate(pr):
    ax.text(x, hi[x] + 0.004, f"{v:.3f}".replace(".", ","), ha="center", fontsize=9)
ax.set_ylabel("PR-AUC (IC 95\%)"); ax.set_ylim(0, 0.45); ax.legend()
ax.set_title("M2: PR-AUC por estrategia (GroupKFold por usuario)")
plt.tight_layout(); plt.savefig(FIG / "fig_m2_models.pdf", bbox_inches="tight"); plt.close()

# ── 2b) M2: comparación de algoritmos, plano vs jerárquico (PR-AUC) ──
algos = ["Random\nForest", "LightGBM", "HistGB", "Reg.\nlogística", "Red neuronal\n(MLP)"]
pr_plano = [0.409, 0.402, 0.397, 0.391, 0.378]
pr_jerar = [0.395, 0.376, 0.369, 0.372, 0.352]
x = np.arange(len(algos)); w = 0.38
fig, ax = plt.subplots(figsize=(6.8, 3.6))
ax.bar(x - w/2, pr_plano, w, label="Plano", color=ROJO, edgecolor="white")
ax.bar(x + w/2, pr_jerar, w, label="Jerárquico", color=AZUL, edgecolor="white")
ax.axhline(0.238, color=NEUTRO, ls="--", lw=1, label="azar (prevalencia 0,238)")
ax.set_xticks(x); ax.set_xticklabels(algos, fontsize=8)
ax.set_ylabel("PR-AUC"); ax.set_ylim(0, 0.45); ax.legend(fontsize=8)
ax.set_title("M2: comparación de algoritmos (plano vs jerárquico)")
plt.tight_layout(); plt.savefig(FIG / "fig_m2_algos.pdf", bbox_inches="tight"); plt.close()

# ── 3) M2: importancia de variables (Variante B, top-10) ──
feats = ["ipa_class", "age_cat", "demo_cluster", "size_hogar", "distance_type",
         "mun_type", "civil_status", "prod_cat_code", "labor_status", "num_room"][::-1]
imp = [413, 408, 388, 322, 305, 303, 271, 220, 148, 127][::-1]
fig, ax = plt.subplots(figsize=(6, 3.8))
colores = [ROJO if f == "demo_cluster" else AZUL for f in feats]
ax.barh(feats, imp, color=colores, edgecolor="white")
ax.set_xlabel("Importancia (nº de splits, LightGBM)")
ax.set_title("M2: importancia de variables (modelo con producto)")
plt.tight_layout(); plt.savefig(FIG / "fig_m2_importance.pdf", bbox_inches="tight"); plt.close()

# ── 4) M3: recomendador por cluster vs baselines (segmento SIN fuga temporal) ──
metricas = ["HR@5", "HR@10", "NDCG@10", "MAP@10"]
datos = {"Popularidad": [0.629, 0.806, 0.580, 0.500],
         "Cluster demográfico": [0.629, 0.789, 0.572, 0.496],
         "Recomendador por cluster (M1)": [0.586, 0.774, 0.521, 0.439]}
x = np.arange(len(metricas)); w = 0.26
fig, ax = plt.subplots(figsize=(7, 3.6))
colores = [GRIS, AZUL, NAVY]
for i, (nombre, vals) in enumerate(datos.items()):
    ax.bar(x + (i - 1) * w, vals, w, label=nombre, color=colores[i], edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(metricas); ax.set_ylabel("Score"); ax.set_ylim(0, 1)
ax.legend(fontsize=8); ax.set_title("M3: recomendador por cluster (M1) vs baselines (test, sin fuga)")
plt.tight_layout(); plt.savefig(FIG / "fig_m3_recommender.pdf", bbox_inches="tight"); plt.close()

# ── 5) estabilidad de k ──
ks = list(range(2, 13))
sil = [0.3441, 0.2945, 0.3276, 0.3392, 0.3565, 0.3753, 0.4006, 0.4114, 0.4317, 0.4485, 0.4565]
ari = [0.9995, 0.9967, 0.9968, 0.9583, 0.9941, 0.9785, 0.9571, 0.9681, 0.9840, 0.9978, 0.9931]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
axes[0].plot(ks, sil, "o-", color=AZUL); axes[0].axvline(10, color=ROJO, ls="-", label="k=10 (adoptado)")
axes[0].set_title("Silueta media"); axes[0].set_xlabel("k"); axes[0].legend(fontsize=8)
axes[1].plot(ks, ari, "o-", color=NARANJA); axes[1].axhline(0.9, color=NEUTRO, ls="--", lw=1, label="ARI=0,90")
axes[1].axvline(10, color=ROJO, ls="-", label="k=10 (adoptado)")
axes[1].set_title("Estabilidad (ARI entre re-muestreos)"); axes[1].set_xlabel("k"); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "fig_k_stability.pdf", bbox_inches="tight"); plt.close()

# ── 6) M4: backtest rolling 1-paso (Prophet vs baselines) ──
modelos_m4 = ["Prophet", "Naïve", "Media móvil 3"]
mae_m4 = [749.7, 502.3, 583.7]
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.bar(modelos_m4, mae_m4, color=[ROJO, AZUL, NARANJA], edgecolor="white")
for x, v in enumerate(mae_m4):
    ax.text(x, v + 8, f"{v:.0f}".replace(".", ","), ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("MAE (clics/semana)"); ax.set_ylim(0, 850)
ax.set_title("M4: error 1-paso (backtest rolling) — menor es mejor")
plt.tight_layout(); plt.savefig(FIG / "fig_m4_backtest.pdf", bbox_inches="tight"); plt.close()

# ── 7) ROI: BD vs modelo según la estructura de coste por email (agregado) ──
# Todos los costes son por email enviado: Fijo (1 €), Variable (% del CPL), Mixto (0,50 € + 15% CPL).
escenarios = ["Fijo\n(1 €)", "Var.\n20%", "Var.\n30%", "Var.\n40%", "Mixto"]
roi_bd     = [101,  18, -21, -41,  13]
roi_modelo = [148,  67,  37,  21,  65]
x = np.arange(len(escenarios)); w = 0.36
fig, ax = plt.subplots(figsize=(7.0, 3.4))
ax.bar(x - w/2, roi_bd, w, label="BD (enviar a todos)", color=GRIS, edgecolor="white")
ax.bar(x + w/2, roi_modelo, w, label="Modelo (priorizar)", color=ROJO, edgecolor="white")
for i in range(len(escenarios)):
    ax.text(x[i] - w/2, roi_bd[i] + (5 if roi_bd[i] >= 0 else -14), f"{roi_bd[i]}%", ha="center", fontsize=8)
    ax.text(x[i] + w/2, roi_modelo[i] + 5, f"{roi_modelo[i]}%", ha="center", fontsize=8)
ax.axhline(0, color=NEUTRO, lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(escenarios); ax.set_ylabel("ROI agregado (\\%)")
ax.set_title("ROI según estructura de coste por email: BD vs modelo"); ax.legend()
plt.tight_layout(); plt.savefig(FIG / "fig_roi.pdf", bbox_inches="tight"); plt.close()

print("Figuras generadas en", FIG)
for f in sorted(FIG.glob("*.pdf")):
    print("  ", f.name)
