import hashlib
import re

import pandas as pd


def clean_subject_text(text: str) -> str:
    """Limpia texto de asunto de email: minúsculas, sin emojis, sin acentos."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-záéíóúüñ0-9\s]", "", text)
    for accented, clean in {"á":"a","é":"e","í":"i","ó":"o","ú":"u","ü":"u"}.items():
        text = text.replace(accented, clean)
    return re.sub(r"\s+", " ", text).strip()


def hash_id(x) -> str:
    """SHA-256 de un valor como string. Usado para anonimizar emails y asuntos."""
    return hashlib.sha256(str(x).encode()).hexdigest()


CATEGORY_RULES: dict[str, list[str]] = {
    # Combinados / Multi-Riesgo
    "multi_insurance_bundle": ["coche hogar", "hogar coche", "linea directa"],
    "fiber_mobile_bundle": [
        "fibra movil", "tarifa combinada", "2 lineas",
        "dos servicios incluidos",
    ],
    "smartphone_bundle": ["iphone", "samsung", "xiaomi", "smartphone"],
    # Telecomunicaciones
    "mobile_plan": [
        "promocion movil", "tarifa movil", "datos extra", "300gb",
        "lineas por 1", "movistar", "vodafone",
    ],
    # Energía
    "energy_plan": [
        "luz", "energia", "gas", "kwh", "iberdrola", "naturgy",
        "repsol", "total energies", "tarifalo",
    ],
    # Seguridad y Seguros
    "home_security":     ["alarma", "securitas direct", "prosegur", "robos", "ocupaciones"],
    "funeral_insurance": ["decesos", "defuncion", "ocaso", "santa lucia", "repatriacion"],
    "health_insurance":  ["salud", "sanitas", "dkv", "adeslas", "divina", "asisa", "cobertura sanitaria"],
    "life_insurance":    ["seguros vida", "protege tu futuro", "axa vida"],
    "dental_insurance":  ["dental", "sanitas dental"],
    "pet_insurance":     ["mascota", "santevet", "perro", "gato"],
    "home_insurance":    ["proteccion hogar", "seguro hogar"],
    "car_insurance":     ["seguro de coche", "seguro coche", "allianz", "verti", "hello prima"],
    "insurance":         ["protegido"],
    # Motor
    "car_rental":        ["alquiler de coche", "alquiler coche", "europcar", "rent a car"],
    "car_leasing":       ["revel", "conduce sin tramites", "suscripcion coche"],
    "car_resale":        ["flexicar", "tasacion", "compramos tu coche", "cambia de coche sin perder valor", "precio"],
    "vehicle_purchase":  ["kia", "toyota", "seat", "dacia", "compra tu vehiculo", "dias unicos", "seat flex"],
    # Finanzas y Servicios Legales
    "loan":          ["prestamo", "credito", "solcredito"],
    "legal_service": [
        "legalitas", "gestion de deudas", "deuda infinita", "reclamacion",
        "intereses abusivos", "cuotas infinitas", "recupera tu dinero",
    ],
    # Otros
    "travel":               ["viaje", "crucero", "hotel", "vacaciones", "melia", "costa crucero", "tailandia", "septiembre"],
    "charity":              ["donacion", "gaza", "ong", "hambre", "salvar vidas", "aldeas infantiles", "accion contra el hambre"],
    "eyewear":              ["gafas", "optica", "gafas es", "progresivas"],
    "beauty_products":      ["clarins", "cosmetica", "belleza", "muestras gratis"],
    "subscription_service": ["aquaservice", "dispensador de agua", "cafetera de regalo"],
    "sweepstakes":          ["lottosocial", "participacion en loteria", "sorteo", "promocion de descuento"],
}


def normalize_product(row) -> str:
    """Clasifica un producto en una categoría usando keyword scoring."""
    text = f"{row['subject_clean']} {row['product']} {row['sector']} {row['brand']}".lower()
    scores = {
        cat: sum(1 for kw in kws if kw in text)
        for cat, kws in CATEGORY_RULES.items()
    }
    scores = {k: v for k, v in scores.items() if v > 0}
    return max(scores, key=scores.get) if scores else "other"
