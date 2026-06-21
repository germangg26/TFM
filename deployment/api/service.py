"""Lógica de servicio del recomendador (independiente del framework web).

Carga los artefactos exportados por deployment/export_artifacts.py y resuelve:
  - recommend(user_id)        → top-5 para un usuario conocido (lookup precalculado)
  - recommend_coldstart(perfil) → top-5 para un usuario nuevo vía su cluster demográfico

Diseñado para ser testeable sin levantar la API (solo pandas/joblib).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ART_DIR = Path(__file__).resolve().parent / "artifacts"

DEMO_FEATS = ["age_cat", "gender_enc", "labor_status_enc", "civil_status_enc",
              "tiene_coche_enc", "size_hogar_enc", "num_room_enc",
              "ipa_class", "mun_type", "distance_type"]


def _age_to_cat(age: float) -> int:
    for i, top in enumerate([25, 35, 45, 55, 65]):
        if age < top:
            return i
    return 5


# Diccionarios de label-encoding texto -> número (idénticos a los del entrenamiento,
# notebooks 03/05/07). El usuario introduce el texto; aquí se traduce al código.
GENDER_MAP = {"H": 1, "M": 0}
LABOR_MAP = {"employed": 2, "unemployed": 1, "inactive": 0}
CIVIL_MAP = {"soltero": 0, "divorciado": 1, "viudo": 2, "casado": 3}
SIZE_HOGAR_MAP = {"1 persona": 1, "2 personas": 2, "3 personas": 3,
                  "4 personas": 4, "5 o más personas": 5}
NUM_ROOM_MAP = {"Menos de 3": 1, "3 a 6": 2, "7 o más": 3}
IPA_MAP = {"Baja": 0, "Baja-media": 1, "Media": 2, "Media-alta": 3, "Alta": 4}
MUN_TYPE_MAP = {"Aldea": 0, "Pueblo": 1, "Villa": 2, "Ciudad": 3,
                "Gran ciudad": 4, "Metrópoli": 5}
DISTANCE_MAP = {"Muy cercana": 0, "Cercana": 1, "Media distancia": 2,
                "Lejana": 3, "Muy lejana": 4}


def _encode_profile(perfil: dict) -> dict:
    """Convierte el perfil con valores de TEXTO a las variables numéricas que
    espera el modelo (label-encoding). Acepta también números ya codificados por
    compatibilidad (si el valor no está en el diccionario, se usa tal cual)."""
    def enc(mapa, valor, defecto):
        if valor is None:
            return defecto
        return mapa.get(valor, valor)  # si ya viene como número, se respeta
    cluster = perfil.get("demo_cluster")
    return {
        "age_cat": _age_to_cat(perfil.get("age", 45)),
        "gender_enc": enc(GENDER_MAP, perfil.get("gender", "H"), 1),
        "labor_status_enc": enc(LABOR_MAP, perfil.get("labor_status", "employed"), 2),
        "civil_status_enc": enc(CIVIL_MAP, perfil.get("civil_status", "soltero"), 0),
        "tiene_coche_enc": int(perfil.get("tiene_coche", False)),
        "size_hogar_enc": enc(SIZE_HOGAR_MAP, perfil.get("size_hogar", "3 personas"), 3),
        "num_room_enc": enc(NUM_ROOM_MAP, perfil.get("num_room", "3 a 6"), 2),
        "ipa_class": enc(IPA_MAP, perfil.get("ipa_class", "Media"), 2),
        "mun_type": enc(MUN_TYPE_MAP, perfil.get("mun_type", "Metrópoli"), 5),
        "distance_type": enc(DISTANCE_MAP, perfil.get("distance_type", "Cercana"), 1),
        "demo_cluster": int(cluster) if cluster is not None else -1,
    }


class RecommenderService:
    """Servicio de recomendación de categorías de producto (ejemplo de despliegue)."""

    def __init__(self, artifacts_dir: Path | str = ART_DIR):
        self.dir = Path(artifacts_dir)
        rec_cols = [f"rec_{i}" for i in range(1, 6)]

        # Recomendación POR CLUSTER (segmento M1): tabla cluster -> top-5 + mapa usuario -> cluster.
        # La lógica interna resuelve user_id -> su cluster -> top-5 del cluster.
        self.cluster_recs: dict[int, list] = {}
        crec_path = self.dir / "cluster_recommendations.csv"
        if crec_path.exists():
            crec = pd.read_csv(crec_path)
            self.cluster_recs = {int(r.cluster_id): [getattr(r, c) for c in rec_cols]
                                 for r in crec.itertuples(index=False)}
        self._user2seg: dict[str, int] = {}
        u2s_path = self.dir / "user_segment.csv"
        if u2s_path.exists():
            u2s = pd.read_csv(u2s_path)
            self._user2seg = {r.id_user: int(r.segment) for r in u2s.itertuples(index=False)
                              if pd.notna(r.segment)}

        # Compatibilidad: top-5 precalculado por usuario (fallback si faltan las tablas de cluster).
        self._warm: dict[str, list] = {}
        recs_path = self.dir / "recommendations.csv"
        if recs_path.exists():
            recs = pd.read_csv(recs_path)
            self._warm = {row.id_user: [getattr(row, c) for c in rec_cols]
                          for row in recs.itertuples(index=False)}

        self.cluster_aff = pd.read_csv(self.dir / "cluster_affinity.csv", index_col=0)
        self.cluster_aff.index = self.cluster_aff.index.astype(int)
        self.popularity = json.loads((self.dir / "popularity.json").read_text(encoding="utf-8"))
        self._scaler = self._encoder = self._kmeans = None  # carga perezosa (cold-start desde features)

        # Modelo PLANO de propensión (M2 adoptado) + catálogo de productos, exportados por
        # deployment/export_propensity.py. La API puntúa cada producto del catálogo para un
        # usuario/perfil y agrega por sector (one-vs-rest: probabilidades independientes 0–1).
        self.flat_model = self.catalog = self.user_features = None
        meta_path = self.dir / "flat_meta.json"
        if meta_path.exists():
            import joblib
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.user_feats = meta["user_feats"]
            self.all_names = meta["all_names"]           # orden completo de features del modelo
            self.prod_names = [c for c in meta["all_names"] if c not in meta["user_feats"]]
            self.rho_real = meta["rho_real"]
            self.rho_train = meta["rho_train"]
            self.flat_model = joblib.load(self.dir / "flat_model.joblib")
            self.catalog = pd.read_csv(self.dir / "product_catalog.csv")
            self.user_features = pd.read_csv(self.dir / "user_features.csv").set_index("id_user")

    # ── usuario conocido: lookup usuario → su cluster (segmento M1) → top-5 del cluster ──
    def recommend(self, user_id: str, k: int = 5) -> dict:
        seg = self._user2seg.get(user_id)
        if seg is not None and seg in self.cluster_recs:
            return {"user_id": user_id, "source": "cluster_m1", "cluster_id": seg,
                    "recommendations": self.cluster_recs[seg][:k]}
        # Compatibilidad: top-5 por usuario precalculado (si no hay tablas de cluster)
        if user_id in self._warm:
            return {"user_id": user_id, "source": "warm_lookup", "recommendations": self._warm[user_id][:k]}
        return {"user_id": user_id, "source": "fallback_popularity", "recommendations": self.popularity[:k]}

    # ── propensión al clic por sector y producto (modelo PLANO M2) ──
    def _prior_correct(self, p):
        """Corrige las probabilidades del modelo (escala de entrenamiento balanceado) al prior
        real de producción. Es monótona: no altera el orden de productos/sectores."""
        p = np.clip(p, 1e-9, 1 - 1e-9)
        logit = np.log(p / (1 - p))
        corr = np.log(self.rho_real / (1 - self.rho_real)) - np.log(self.rho_train / (1 - self.rho_train))
        return 1.0 / (1.0 + np.exp(-(logit + corr)))

    def _breakdown(self, user_vals: dict, source: str, extra: dict | None = None) -> dict:
        """Puntúa TODOS los productos del catálogo para un usuario dado con el modelo plano y
        agrega por sector. Devuelve listas ordenadas (desc) de probabilidad de clic por producto
        (categoría product_new) y por sector. Son probabilidades one-vs-rest (independientes,
        0–1): cada una responde '¿clicaría si le envío esto?'; NO suman 1."""
        if self.flat_model is None:
            return {"source": "no_disponible", "sectores": [], "productos": []}
        cat = self.catalog
        n = len(cat)
        # Vector de features: columnas de usuario (constantes) + columnas de producto (del catálogo)
        cols = [np.full(n, user_vals[name], dtype=float) if name in user_vals
                else cat[name].values.astype(float) for name in self.all_names]
        X = np.column_stack(cols)
        p_real = self._prior_correct(self.flat_model.predict_proba(X)[:, 1])  # prob realista
        tmp = cat[["product_new", "sector"]].copy()
        tmp["p"] = p_real
        # Sector de cada categoría: el más frecuente (algunas categorías tocan 2 sectores).
        sector_de = tmp.groupby("product_new")["sector"].agg(lambda s: s.mode().iloc[0])
        prod = tmp.groupby("product_new")["p"].mean().sort_values(ascending=False)
        sect = tmp.groupby("sector")["p"].mean().sort_values(ascending=False)
        out = {"source": source,
               "sectores": [{"sector": s, "p_click": round(float(v), 4)} for s, v in sect.items()],
               "productos": [{"producto": pr, "sector": sector_de[pr], "p_click": round(float(v), 4)}
                             for pr, v in prod.items()]}
        if extra:
            out.update(extra)
        return out

    def propensity(self, user_id: str) -> dict:
        """Propensión por sector y producto de un usuario CONOCIDO (modelo plano + su perfil)."""
        if self.flat_model is None or user_id not in self.user_features.index:
            return {"user_id": user_id, "source": "desconocido", "sectores": [], "productos": []}
        fila = self.user_features.loc[user_id]
        user_vals = {c: float(fila[c]) for c in self.user_feats}
        return self._breakdown(user_vals, source="plano_usuario", extra={"user_id": user_id})

    def propensity_coldstart(self, perfil: dict) -> dict:
        """Propensión por sector y producto de un usuario NUEVO (modelo plano + perfil demográfico)."""
        if self.flat_model is None:
            return {"source": "no_disponible", "sectores": [], "productos": []}
        fila = _encode_profile(perfil)
        user_vals = {c: float(fila[c]) for c in self.user_feats}
        return self._breakdown(user_vals, source="plano_coldstart")

    # ── usuario nuevo (cold-start) ──
    def recommend_coldstart(self, perfil: dict, k: int = 5) -> dict:
        cluster = perfil.get("demo_cluster")
        if cluster is None:
            cluster = self._assign_cluster(perfil)
        if cluster is not None and int(cluster) in self.cluster_aff.index:
            ranked = self.cluster_aff.loc[int(cluster)].sort_values(ascending=False)
            return {"demo_cluster": int(cluster), "source": "cold_start_cluster",
                    "recommendations": ranked.index[:k].tolist()}
        return {"demo_cluster": None, "source": "fallback_popularity",
                "recommendations": self.popularity[:k]}

    # ── asignación de cluster desde features demográficas (pipeline de 05) ──
    def _assign_cluster(self, perfil: dict):
        try:
            if self._scaler is None:
                with open(self.dir / "demo_scaler.pkl", "rb") as f:
                    self._scaler = pickle.load(f)
                with open(self.dir / "demo_model.pkl", "rb") as f:
                    self._kmeans = pickle.load(f)
                with open(self.dir / "demo_ae_used.pkl", "rb") as f:
                    self._ae_used = pickle.load(f)
                if self._ae_used:
                    import tensorflow as tf
                    self._encoder = tf.keras.models.load_model(str(self.dir / "demo_encoder.keras"))
            row = _encode_profile(perfil)
            x = np.array([[row[c] for c in DEMO_FEATS]], dtype=float)
            xs = self._scaler.transform(x)
            xe = self._encoder.predict(xs, verbose=0) if self._ae_used else xs
            return int(self._kmeans.predict(xe)[0])
        except Exception:
            return None  # si falta TF/artefactos → fallback a popularidad
