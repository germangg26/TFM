"""API de EJEMPLO para servir el recomendador (FastAPI).

⚠️ Es una demostración de productización para el TFM, NO un servicio de producción real
(sin auth, rate-limiting, monitorización ni persistencia robusta).

Arrancar:
    uv add fastapi uvicorn          # instalar dependencias de la API
    uv run python deployment/export_artifacts.py   # generar artefactos
    uv run uvicorn deployment.api.main:app --reload

Probar:
    GET  http://127.0.0.1:8000/health
    GET  http://127.0.0.1:8000/recommend/<id_user>
    POST http://127.0.0.1:8000/recommend/coldstart   (body: ColdStartProfile)
    Documentación interactiva: http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI, HTTPException

from .schema import ColdStartProfile, PropensityBreakdown, Recommendation
from .service import RecommenderService

app = FastAPI(title="TFM — Recomendador (ejemplo)", version="0.1.0")
service: RecommenderService | None = None


@app.on_event("startup")
def _load():
    global service
    service = RecommenderService()


@app.get("/health")
def health():
    return {"status": "ok",
            "usuarios": len(service._user2seg),
            "clusters_m1": len(service.cluster_recs),
            "clusters_demo": len(service.cluster_aff)}


@app.get("/recommend/{user_id}", response_model=Recommendation)
def recommend(user_id: str, k: int = 5):
    """Top-k para un usuario conocido: lookup usuario → su cluster (segmento M1) → top-5 del cluster.
    Fallback a popularidad si el usuario no tiene cluster asignado."""
    if service is None:
        raise HTTPException(503, "Servicio no inicializado")
    return service.recommend(user_id, k=k)


@app.post("/recommend/coldstart", response_model=Recommendation)
def recommend_coldstart(perfil: ColdStartProfile, k: int = 5):
    """Top-k para un usuario nuevo: se asigna su cluster demográfico y se sirven sus afinidades."""
    if service is None:
        raise HTTPException(503, "Servicio no inicializado")
    return service.recommend_coldstart(perfil.model_dump(), k=k)


@app.get("/propensity/{user_id}", response_model=PropensityBreakdown)
def propensity(user_id: str):
    """Propensión al clic por SECTOR y por PRODUCTO de un usuario conocido (modelo plano M2).
    Devuelve p_click (cruda, escala balanceada) y p_real (corregida al prior real; por el
    desbalanceo, es la que debe usarse). One-vs-rest (independientes, 0–1; no suman 1)."""
    if service is None:
        raise HTTPException(503, "Servicio no inicializado")
    return service.propensity(user_id)


@app.post("/propensity", response_model=PropensityBreakdown)
def propensity_coldstart(perfil: ColdStartProfile):
    """Propensión al clic por SECTOR y por PRODUCTO de un usuario nuevo (modelo plano M2 + perfil).
    Devuelve p_click (cruda, escala balanceada) y p_real (corregida al prior real; por el
    desbalanceo, es la que debe usarse). One-vs-rest (independientes, 0–1; no suman 1)."""
    if service is None:
        raise HTTPException(503, "Servicio no inicializado")
    return service.propensity_coldstart(perfil.model_dump())
