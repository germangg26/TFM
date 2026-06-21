"""Estilo visual compartido de las figuras del TFM (paleta + tipografía).

Lo importan los tres generadores de figuras (gen_figures, gen_eda_figures,
gen_tfm_figures) para que todas compartan fuente serif (coherente con Palatino
del documento) y una única paleta alineada con el acento navy del PDF.

Uso:
    from _fig_style import set_style, NAVY, AZUL, ROJO, NARANJA, VERDE, GRIS, NEUTRO
    set_style()
"""
import matplotlib.pyplot as plt

# ── Paleta unificada (el navy es el mismo acento de enlaces/cajas del documento) ──
NAVY    = "#002060"   # acento principal / serie destacada
AZUL    = "#5b8aa6"   # serie de datos principal
ROJO    = "#d1495b"   # resaltado / serie clave / supera umbral
NARANJA = "#e8a33d"   # tercera serie
VERDE   = "#4c956c"   # positivo / bandas de intervalo de confianza
GRIS    = "#bdbdbd"   # baselines / barras de referencia (azar)
NEUTRO  = "#666666"   # líneas y ejes auxiliares (sustituye gray/dimgray)


def set_style():
    """Fija la tipografía y rejilla comunes a todas las figuras del TFM."""
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "figure.dpi": 150,
        "axes.grid": True, "grid.alpha": 0.3,
    })
