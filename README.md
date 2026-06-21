# TFM — Análisis de Campañas de Email Marketing en España

Proyecto de ML que analiza campañas reales de email marketing en España. El pipeline limpia los datos brutos, construye tablas dimensionales y las enriquece con datos demográficos y geográficos españoles (INE, censo municipal, estadísticas de vehículos) para habilitar segmentación de clientes y modelos predictivos.

**Autor:** German  
**Python:** 3.12  
**Gestor de paquetes:** [uv](https://github.com/astral-sh/uv)

---

## Instalación

```powershell
uv sync          # instala todas las dependencias del uv.lock
jupyter notebook # arranca el entorno de desarrollo
```

---

## Pipeline de notebooks

Ejecutar en orden desde `notebooks/`:

**Pipeline de datos** (01–05b):

| Notebook | Descripción |
|---|---|
| `01_clean.ipynb` | Ingesta y limpieza de datos brutos (LOW/CPC). Muestreo balanceado, deduplica eventos, genera `users.csv`, `products.csv`, `events.csv` |
| `02_geo.ipynb` | Enriquecimiento geográfico. Procesa datos externos (INE, municipios, BCN/MAD por distrito) y guarda 9 tablas procesadas |
| `03_users.ipynb` | Enriquecimiento de usuarios. Cruza con las tablas geográficas y asigna (sintéticamente) estado laboral, civil, coche, tamaño de hogar y habitaciones → `users.csv` enriquecido |
| `04_eda.ipynb` | Análisis exploratorio (sin salidas al pipeline) |
| `05_demo_segmentation.ipynb` | Segmentación demográfica (M0) para arranque en frío: autoencoder + KMeans |
| `05b_k_stability.ipynb` | Diagnóstico de estabilidad del número de clusters *k* (silueta + ARI por re-muestreo) |

**Modelado** (06–11):

| Notebook | Descripción |
|---|---|
| `06_segmentation.ipynb` | Segmentación comportamental (M1) sobre historial de interacciones |
| `07_propensity.ipynb` | Modelo de propensión al clic (M2): LightGBM con variables de usuario y de producto + embeddings del asunto; compara modelo plano frente a jerárquico y adopta el plano, con validación sin fuga y corrección de *prior* |
| `08_recommender.ipynb` | Recomendador por cluster (M3): sirve el ranking de categorías del segmento; evaluación temporal sin fuga frente al baseline de popularidad |
| `09_forecast.ipynb` | Forecast semanal de clics (M4) con Prophet |
| `10_inverse_profile.ipynb` | Modelo inverso producto→perfil |
| `11_roi.ipynb` | Análisis económico de ROI (BD vs modelo, estructuras de coste) |

---

## Layout de datos

```
data/raw/             # CSVs brutos de campaña — NO incluidos (datos privados)
data/processed/       # Salidas del pipeline — NO incluidas (se generan al ejecutar)
data/external/        # Datos de municipios — NO incluidos
data/external_clean/  # Datos oficiales (INE/censos/DGT) — NO incluidos (descargar aparte)
```

> **Nota sobre los datos.** Por privacidad, los datos de campaña no se incluyen en el repositorio.
> Los datos externos de enriquecimiento son oficiales y de acceso público (INE, censos municipales de
> Madrid y Barcelona, DGT); sus fuentes y URLs están documentadas en el apéndice de la memoria. El código
> del pipeline es íntegramente reproducible una vez disponibles los datos en las carpetas indicadas.

---

## Módulos en `src/tfm/`

| Módulo | Contenido |
|---|---|
| `utils.py` | `find_project_root()`, constantes de ruta (`ROOT_PATH`, `DATA_PATH`, …) |
| `geo.py` | `normalize_mun_code()`, `get_prov_code()`, diccionarios de distrito BCN/MAD, mapeos CP→INE |
| `preprocessing.py` | `clean_subject_text()`, `hash_id()`, `CATEGORY_RULES`, `normalize_product()` |

---

## Memoria (LaTeX)

El documento del TFM está en `latex/`. Para compilarlo (requiere una distribución LaTeX con `biber`):

```powershell
cd latex
latexmk -pdf main.tex
```

---

## Linting

```powershell
ruff check .   # lint
ruff format .  # formato
```

---

## Reproducibilidad

Los notebooks se commitean sin outputs gracias a [nbstripout](https://github.com/kynan/nbstripout), instalado como git hook (`uv run nbstripout --install`).  
El fichero `uv.lock` fija las versiones exactas de todas las dependencias.
