"""Esquemas de entrada/salida de la API (pydantic).

El usuario introduce las variables categóricas como TEXTO legible (p. ej.
mun_type="Metrópoli"); la API hace internamente el label-encoding al número que
espera el modelo (ver deployment/api/service.py).
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ColdStartProfile(BaseModel):
    """Perfil demográfico de un usuario nuevo (cold-start), con valores en texto."""
    age: float = Field(45, ge=18, le=120, description="Edad en años (18-120)")
    gender: Literal["H", "M"] = "H"
    labor_status: Literal["employed", "unemployed", "inactive"] = "employed"
    civil_status: Literal["soltero", "casado", "divorciado", "viudo"] = "soltero"
    tiene_coche: bool = False
    size_hogar: Literal[
        "1 persona", "2 personas", "3 personas", "4 personas", "5 o más personas"
    ] = "3 personas"
    num_room: Literal["Menos de 3", "3 a 6", "7 o más"] = "3 a 6"
    ipa_class: Literal["Baja", "Baja-media", "Media", "Media-alta", "Alta"] = "Media"
    mun_type: Literal[
        "Aldea", "Pueblo", "Villa", "Ciudad", "Gran ciudad", "Metrópoli"
    ] = "Metrópoli"
    distance_type: Literal[
        "Muy cercana", "Cercana", "Media distancia", "Lejana", "Muy lejana"
    ] = "Cercana"
    demo_cluster: Optional[int] = Field(None, description="Si se conoce el cluster, se usa directamente")


class Recommendation(BaseModel):
    recommendations: list[str]
    source: str
    user_id: Optional[str] = None
    cluster_id: Optional[int] = None      # segmento M1 (usuario conocido)
    demo_cluster: Optional[int] = None    # cluster demográfico (cold-start)


class SectorProb(BaseModel):
    sector: str
    p_click: float


class ProductProb(BaseModel):
    producto: str
    sector: str          # sector al que pertenece la categoría de producto
    p_click: float


class PropensityBreakdown(BaseModel):
    """Propensión al clic por sector y por producto (modelo plano M2).

    Las probabilidades son one-vs-rest: cada una es la probabilidad independiente de que el
    usuario clique si se le envía ese sector/producto. NO suman 1. Están corregidas al prior
    real de producción (≈2 %)."""
    source: str
    sectores: list[SectorProb]
    productos: list[ProductProb]
    user_id: Optional[str] = None
