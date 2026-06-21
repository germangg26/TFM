"""
EDA comparativo: variable respuesta ANTES vs DESPUÉS del remuestreo balanceado.
Genera latex/figures/fig_eda_resampling.pdf.

ANTES (datos brutos): los conteos de clics/aperturas son el nº de filas de los
ficheros raw (los Clicks-*.csv son clics; los Opens-*.csv son aperturas):
    clics    = 31.169 (CPC) + 15.694 (LOW)               = 46.863
    aperturas= 757.360 + 781.453 (CPC) + 440.490 (LOW)    = 1.979.303
  (reproducible: `for f in data/raw/*.csv; do tail -n +2 "$f" | wc -l; done`)
DESPUÉS: se lee data/processed/events.csv (muestreo 1:1 + dedup + filtros).
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent
FIG = ROOT / "latex" / "figures"; FIG.mkdir(exist_ok=True)
from _fig_style import set_style, AZUL, ROJO, GRIS  # noqa: E402
set_style()

# ── ANTES (datos brutos; conteo por event_type, como el pipeline) ──
#   clics=44.997, no-clics (aperturas 1.977.437 + ignorados 3.732)=1.981.169 → total 2.026.166
clics_antes, aperturas_antes = 44_997, 1_981_169
total_antes = clics_antes + aperturas_antes

# ── DESPUÉS (events.csv tras muestreo balanceado + limpieza) ──
ev = pd.read_csv(ROOT / "data" / "processed" / "events.csv")
ev = ev[ev["event_type"].isin(["click", "open"])]
clics_desp = int((ev["event_type"] == "click").sum())
aperturas_desp = int((ev["event_type"] == "open").sum())
total_desp = clics_desp + aperturas_desp

def pct(n, tot):
    return 100 * n / tot

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# Panel izquierdo: ANTES (escala log por el fuerte desbalanceo)
b0 = axes[0].bar(["No click", "Click"], [aperturas_antes, clics_antes],
                 color=[GRIS, ROJO], edgecolor="white", width=0.6)
axes[0].set_yscale("log")
for b, n, t in zip(b0, [aperturas_antes, clics_antes], [aperturas_antes, clics_antes]):
    axes[0].text(b.get_x() + b.get_width()/2, n * 1.15,
                 f"{n:,}\n({pct(t, total_antes):.1f}\\%)".replace(",", "."),
                 ha="center", fontsize=9)
axes[0].set_ylabel("No. of events (log scale)")
axes[0].set_title(f"BEFORE resampling (raw data)\\n{total_antes:,} events · click = {pct(clics_antes, total_antes):.1f}\\%".replace(",", "."))
axes[0].set_ylim(1e4, 1e7)

# Panel derecho: DESPUÉS (escala lineal)
b1 = axes[1].bar(["No click", "Click"], [aperturas_desp, clics_desp],
                 color=[GRIS, ROJO], edgecolor="white", width=0.6)
for b, n in zip(b1, [aperturas_desp, clics_desp]):
    axes[1].text(b.get_x() + b.get_width()/2, n + 600,
                 f"{n:,}\n({pct(n, total_desp):.1f}\\%)".replace(",", "."),
                 ha="center", fontsize=9)
axes[1].set_ylabel("No. of events")
axes[1].set_title(f"AFTER (1:1 sampling + cleaning)\\n{total_desp:,} events · click = {pct(clics_desp, total_desp):.1f}\\%".replace(",", "."))
axes[1].set_ylim(0, total_desp * 0.95)

fig.suptitle("Response variable: effect of balanced sampling on class imbalance", fontsize=12)
plt.tight_layout()
plt.savefig(FIG / "fig_eda_resampling.pdf", bbox_inches="tight")
plt.close()
print("fig_eda_resampling.pdf generada")
print(f"ANTES : {total_antes:,} eventos | clic {pct(clics_antes,total_antes):.2f}%  ({clics_antes:,} clics)")
print(f"DESPUÉS: {total_desp:,} eventos | clic {pct(clics_desp,total_desp):.2f}%  ({clics_desp:,} clics)")
