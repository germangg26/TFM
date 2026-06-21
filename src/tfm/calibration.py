"""
Corrección de probabilidades por prior shift en muestreo balanceado.

Cuando entrenamos un modelo de propensión sobre una muestra reequilibrada, las
probabilidades predichas quedan desplazadas al prior de entrenamiento (rho_train)
y no al prior real de producción (rho_real). Esta función aplica la corrección
exacta basada en log-odds.

IMPORTANTE: aunque el muestreo de partida sea 1:1, el prior de entrenamiento
EFECTIVO depende de cómo se defina la variable objetivo. En este proyecto el
target es click-vs-open (se descartan los 'ignored') y, tras la deduplicación y
los filtros, rho_train ~ 0.238 (NO 0.5). Por eso `correct_prior` exige pasar
rho_train de forma explícita, medido empíricamente sobre la muestra de
entrenamiento (p. ej. `rho_train = float(events["target"].mean())`).

Derivación
----------
Sea p_m = P_modelo(click|x) entrenado con prior rho_train.
Sea rho_real = prior verdadero (tasa de click en producción).

En escala log-odds:
    logit(p_real) = logit(p_m) + log(rho_real/(1-rho_real)) - log(rho_train/(1-rho_train))

Referencia: Dal Pozzolo et al. (2015) "Calibrating Probability with Undersampling
for Unbalanced Classification", IEEE SSCI.
"""

import numpy as np


def correct_prior(p_model: np.ndarray, rho_real: float, rho_train: float) -> np.ndarray:
    """Corrige probabilidades de un modelo entrenado con prior rho_train al prior real rho_real.

    Parameters
    ----------
    p_model  : probabilidades predichas por el modelo (array en [0, 1])
    rho_real : tasa de click real en producción (e.g. 0.02)
    rho_train: prior EFECTIVO de la muestra de entrenamiento, medido empíricamente
               (e.g. 0.2384 en este proyecto; no se asume 0.5). Parámetro obligatorio.

    Returns
    -------
    Probabilidades corregidas al prior real.

    Example
    -------
    >>> p_raw = np.array([0.9, 0.6, 0.3, 0.1])
    >>> correct_prior(p_raw, rho_real=0.02, rho_train=0.5)
    array([0.155..., 0.059..., 0.017..., 0.005...])
    """
    p_model = np.clip(p_model, 1e-9, 1 - 1e-9)
    logit_model = np.log(p_model / (1 - p_model))
    correction = np.log(rho_real / (1 - rho_real)) - np.log(rho_train / (1 - rho_train))
    logit_corrected = logit_model + correction
    return 1.0 / (1.0 + np.exp(-logit_corrected))
