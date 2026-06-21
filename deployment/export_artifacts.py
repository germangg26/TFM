"""Exporta los artefactos necesarios para servir el recomendador en la API de ejemplo.

Genera en deployment/api/artifacts/:
  - recommendations.csv   : top-5 precalculado por usuario (servicio "warm", batch)
  - cluster_affinity.csv  : afinidad demo_cluster × categoría (cold-start)
  - popularity.json       : ranking global de categorías (fallback)
  - categories.json       : lista de categorías
  - demo_*.pkl / .keras   : pipeline de asignación de cluster (copiado de 05)

Uso:  uv run python deployment/export_artifacts.py
"""
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
ART = ROOT / "deployment" / "api" / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

# ── datos ──
events = pd.read_csv(PROC / "events.csv")
events = events[events["event_type"].isin(["click", "open"])].copy()
events["target"] = (events["event_type"] == "click").astype(int)
prods = pd.read_csv(PROC / "products.csv")[["id_product", "product_new"]]
demo = pd.read_csv(PROC / "users_demo_segments.csv")[["id_user", "demo_cluster"]]

df = (events.merge(prods, on="id_product", how="left")
      .merge(demo, on="id_user", how="left")
      .dropna(subset=["product_new"]))
ALL_CATS = sorted(df["product_new"].unique())

# ── afinidad por cluster demográfico (cold-start) ──
cm = (df[df["target"] == 1].groupby(["demo_cluster", "product_new"]).size()
      .unstack(fill_value=0).reindex(columns=ALL_CATS, fill_value=0))
clu_aff = cm.div(cm.sum(1).replace(0, 1), axis=0)
clu_aff.to_csv(ART / "cluster_affinity.csv")

# ── popularidad global (fallback) ──
pop = df[df["target"] == 1]["product_new"].value_counts()
popularity = [c for c in pop.index if c in ALL_CATS]
(ART / "popularity.json").write_text(json.dumps(popularity, ensure_ascii=False), encoding="utf-8")
(ART / "categories.json").write_text(json.dumps(ALL_CATS, ensure_ascii=False), encoding="utf-8")

# ── recomendaciones POR CLUSTER (segmento M1) + mapa usuario → cluster ──
# El servicio resuelve user_id → su cluster → top-5 del cluster.
shutil.copy(PROC / "cluster_recommendations.csv", ART / "cluster_recommendations.csv")
seg = pd.read_csv(PROC / "users_segmented.csv")[["id_user", "segment"]]
seg.to_csv(ART / "user_segment.csv", index=False)

# ── recomendaciones precalculadas por usuario (fallback de compatibilidad) ──
shutil.copy(PROC / "recommendations.csv", ART / "recommendations.csv")

# ── pipeline de asignación de cluster (cold-start desde features) ──
for f in ["demo_scaler.pkl", "demo_model.pkl", "demo_ae_used.pkl", "demo_pca_used.pkl", "demo_encoder.keras"]:
    src = PROC / f
    if src.exists():
        shutil.copy(src, ART / f)

print(f"Artefactos exportados a {ART}")
print(f"  cluster_recommendations.csv ({pd.read_csv(ART / 'cluster_recommendations.csv').shape[0]} clusters M1)")
print(f"  user_segment.csv     ({seg.shape[0]:,} usuarios -> cluster)")
print(f"  recommendations.csv  ({pd.read_csv(ART / 'recommendations.csv').shape[0]:,} usuarios, fallback)")
print(f"  cluster_affinity.csv ({clu_aff.shape[0]} clusters × {clu_aff.shape[1]} categorías)")
print(f"  popularity.json      ({len(popularity)} categorías)  | categories.json")
print("  demo_* (asignación de cluster cold-start)")
