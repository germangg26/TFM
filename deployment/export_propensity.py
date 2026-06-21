"""Exporta el MODELO PLANO de propensión (M2 adoptado) para servirlo en la API de ejemplo.

El modelo plano estima P(click | usuario, producto) con variables de usuario + producto +
embeddings del asunto (PCA 16). Es el modelo adoptado en el TFM (mejor PR-AUC que el jerárquico).

La API lo usa para devolver, dado un usuario/perfil, la probabilidad de clic por PRODUCTO
(las 25 categorías product_new) y agregada por SECTOR (one-vs-rest: cada una es una
probabilidad independiente 0–1, NO suman 1).

Guarda en deployment/api/artifacts/:
  - flat_model.joblib       : LGBMClassifier plano entrenado con todos los datos
  - flat_meta.json          : orden de features, índices categóricos, n_emb, rho_real/rho_train
  - product_catalog.csv     : una fila por id_product con todas sus features (incl. emb_pca_*)
  - user_features.csv       : features de usuario (USER_FEATS) por usuario conocido
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
EXT = ROOT / "data" / "external_clean"
ART = ROOT / "deployment" / "api" / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

N_EMB = 16
RHO_REAL = 0.02
LGBM_PARAMS = dict(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=15,
                   min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                   random_state=42, verbose=-1, n_jobs=-1)

# ── 1 · Carga ────────────────────────────────────────────────────────────────
users = pd.read_csv(PROC / "users.csv", dtype={"cp_num": str})
events = pd.read_csv(PROC / "events.csv")
events = events[events["event_type"].isin(["click", "open"])].copy()
events["target"] = (events["event_type"] == "click").astype(int)
products = pd.read_csv(PROC / "products.csv")
demo = pd.read_csv(PROC / "users_demo_segments.csv")[["id_user", "demo_cluster"]]
tabla_marcas = pd.read_csv(EXT / "tabla_marcas.csv", sep=None, engine="python")
emb_df = pd.read_csv(PROC / "subject_emb_raw.csv")

# ── 2 · Variables de usuario (idéntico a notebooks 03/05/07) ──────────────────
u = users.copy()
u["gender_enc"] = (u["gender"] == "H").astype(int)
u["labor_status_enc"] = u["labor_status"].map({"employed": 2, "unemployed": 1, "inactive": 0}).fillna(0).astype(int)
u["civil_status_enc"] = u["civil_status"].map({"soltero": 0, "divorciado": 1, "viudo": 2, "casado": 3}).fillna(0).astype(int)
u["tiene_coche_enc"] = u["tiene_coche"].astype(int)
u["num_room_enc"] = u["num_room"].map({"menos_3_hab": 1, "3_a_6_hab": 2, "7_mas_hab": 3}).fillna(2).astype(int)
u["size_hogar_enc"] = u["size_hogar"].apply(
    lambda s: int(str(s)[0]) if pd.notna(s) and str(s)[0].isdigit() and str(s)[0] != "0" else 5).clip(1, 5)
u["age_cat"] = u["age"].apply(
    lambda a: -1 if pd.isna(a) else next((i for i, t in enumerate([25, 35, 45, 55, 65]) if a < t), 5)).astype(int)
u = u.merge(demo, on="id_user", how="left")
u["demo_cluster"] = u["demo_cluster"].fillna(-1).astype(int)
USER_FEATS = ["age_cat", "gender_enc", "labor_status_enc", "civil_status_enc", "tiene_coche_enc",
              "size_hogar_enc", "num_room_enc", "ipa_class", "mun_type", "distance_type", "demo_cluster"]

# ── 3 · Variables de producto (idéntico a notebook 07, celdas 10-11) ──────────
posibles_attrs = ["Urgencia", "Racionalidad", "RiesgoPercibido", "CicloDecision",
                  "Implicación", "Necesidad", "SensibilidadPrecio", "CompetenciaAlta", "PrecioMedio"]
attr_cols = [c for c in posibles_attrs if c in tabla_marcas.columns]
texto_a_numero = {"muy baja": 0, "baja": 1, "baja-media": 1, "media": 2, "medio": 2,
                  "media-alta": 3, "alta": 4, "muy alta": 5, "alto": 4, "bajo": 1,
                  "no": 0, "si": 1, "sí": 1, "corto": 0, "medio-largo": 2, "largo": 3}
tm = tabla_marcas.copy()
attr_num_cols = []
for col in attr_cols:
    nueva = col + "_num"
    if tm[col].dtype == object:
        tm[nueva] = tm[col].astype(str).str.strip().str.lower().map(texto_a_numero)
    else:
        tm[nueva] = pd.to_numeric(tm[col], errors="coerce")
    attr_num_cols.append(nueva)
col_sector = [c for c in tabla_marcas.columns if c.lower() == "sector"][0]
tm["sector_norm"] = tm[col_sector].astype(str).str.strip().str.lower()
sector_attrs = tm.groupby("sector_norm")[attr_num_cols].mean().reset_index()

p = products.copy()
p["sector_norm"] = p["sector"].astype(str).str.strip().str.lower()
p["cpl"] = pd.to_numeric(p["cpl"], errors="coerce")
p["prod_cat_code"] = p["product_new"].astype("category").cat.codes
p["sector_code"] = p["sector"].astype("category").cat.codes
p = p.merge(sector_attrs, on="sector_norm", how="left")
PROD_NUM_FEATS = ["cpl"] + attr_num_cols

# ── 4 · Embeddings del asunto: PCA 16 ajustado sobre todos los productos ──────
EMB_COLS = [c for c in emb_df.columns if c.startswith("emb_")]
emb_raw = emb_df.drop_duplicates("id_product").set_index("id_product")[EMB_COLS]
pca = PCA(n_components=N_EMB, random_state=42).fit(emb_raw.values.astype(np.float32))
emb_pca = pd.DataFrame(pca.transform(emb_raw.values.astype(np.float32)),
                       index=emb_raw.index, columns=[f"emb_pca_{i}" for i in range(N_EMB)])
EMB_NAMES = list(emb_pca.columns)

# ── 5 · Dataset de entrenamiento (un registro por evento) ─────────────────────
FLAT_FEATS = USER_FEATS + ["sector_code", "prod_cat_code"] + PROD_NUM_FEATS
CAT_COLS = ["demo_cluster", "sector_code", "prod_cat_code"]
cols_prod = ["id_product", "sector", "product_new", "sector_code", "prod_cat_code"] + PROD_NUM_FEATS

df = events[["id_user", "id_product", "target"]].copy()
df = df.merge(u[["id_user"] + USER_FEATS], on="id_user", how="left")
df = df.merge(p[cols_prod], on="id_product", how="left")
df = df.dropna(subset=["sector"]).reset_index(drop=True)
emb_mat_df = emb_pca.reindex(df["id_product"].values).reset_index(drop=True)

X = np.hstack([df[FLAT_FEATS].values.astype(float), emb_mat_df.values.astype(float)])
ALL_NAMES = FLAT_FEATS + EMB_NAMES
cat_idx = [ALL_NAMES.index(c) for c in CAT_COLS]
y = df["target"].values

flat = lgb.LGBMClassifier(**LGBM_PARAMS)
flat.fit(X, y, categorical_feature=cat_idx)
rho_train = float(y.mean())

# ── 6 · Catálogo de productos (una fila por id_product, con TODAS sus features) ─
catalog = p[["id_product", "product_new", "sector", "sector_code", "prod_cat_code"] + PROD_NUM_FEATS].copy()
catalog = catalog.merge(emb_pca, left_on="id_product", right_index=True, how="left")
catalog.to_csv(ART / "product_catalog.csv", index=False)

# ── 7 · Persistencia ──────────────────────────────────────────────────────────
joblib.dump(flat, ART / "flat_model.joblib")
(ART / "flat_meta.json").write_text(json.dumps({
    "user_feats": USER_FEATS, "flat_feats": FLAT_FEATS, "emb_names": EMB_NAMES,
    "all_names": ALL_NAMES, "cat_cols": CAT_COLS, "n_emb": N_EMB,
    "rho_real": RHO_REAL, "rho_train": rho_train,
}, ensure_ascii=False, indent=2), encoding="utf-8")

uf = u[["id_user"] + USER_FEATS]
uf.to_csv(ART / "user_features.csv", index=False)

print("Exportado modelo PLANO:")
print(f"  flat_model.joblib  (features: {len(ALL_NAMES)} = {len(FLAT_FEATS)} tabulares + {N_EMB} emb)")
print(f"  product_catalog.csv {catalog.shape}  ({catalog['product_new'].nunique()} categorias, {catalog['sector'].nunique()} sectores)")
print(f"  user_features.csv   {uf.shape}")
print(f"  rho_train={rho_train:.4f}  rho_real={RHO_REAL}")
