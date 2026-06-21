# API de despliegue — Recomendador (EJEMPLO)

> ⚠️ **Es un ejemplo demostrativo de productización para el TFM, no un servicio de producción.**
> No incluye autenticación, rate-limiting, monitorización ni base de datos. Sirve para ilustrar
> cómo se expondría el recomendador como servicio.

## Arquitectura

```
export_artifacts.py  ──>  artifacts/        (recommendations.csv, cluster_affinity.csv, demo_*, ...)
                              │
                          service.py         RecommenderService  (lógica, sin framework web)
                              │
                          main.py            FastAPI  (capa HTTP fina)
```

Separamos la **lógica** (`service.py`, testeable sin servidor) de la **capa web** (`main.py`).

## Dos modos de servir

| Caso | Endpoint | Cómo |
|---|---|---|
| Usuario conocido (**warm**) | `GET /recommend/{user_id}` | Lookup del top-5 precalculado en batch (`recommendations.csv`) |
| Usuario nuevo (**cold-start**) | `POST /recommend/coldstart` | Asigna el cluster demográfico (pipeline de 05) y sirve la afinidad del cluster |
| Desconocido / sin datos | (fallback) | Popularidad global |
| **Propensión** usuario conocido | `GET /propensity/{user_id}` | Prob. de clic **por sector y producto** (modelo plano M2) |
| **Propensión** usuario nuevo | `POST /propensity` | Prob. de clic **por sector y producto** (modelo plano M2 + perfil) |

> El endpoint `/propensity` requiere ejecutar antes `uv run python deployment/export_propensity.py`
> (exporta el modelo plano M2 y el catálogo de productos). Devuelve, para un usuario o un perfil, la
> probabilidad de clic de los 13 sectores y las 25 categorías de producto (cada producto con su sector),
> corregida al prior real (~2 %). Son probabilidades independientes *one-vs-rest*: no suman 1.

El modo *warm* sigue el patrón realista de recomendadores: las recomendaciones se calculan en un
**job batch** (notebook 08) y la API solo las sirve. El *cold-start* se resuelve online.

## Uso

```bash
# 1) instalar dependencias de la API (solo para este ejemplo)
uv add fastapi uvicorn

# 2) generar los artefactos a partir de los datos procesados
uv run python deployment/export_artifacts.py

# 3) levantar la API
uv run uvicorn deployment.api.main:app --reload
```

## Ejemplos de petición

```bash
curl http://127.0.0.1:8000/health

# usuario conocido
curl http://127.0.0.1:8000/recommend/<id_user>

# usuario nuevo (cold-start) — los campos categóricos van como TEXTO
curl -X POST http://127.0.0.1:8000/recommend/coldstart \
  -H "Content-Type: application/json" \
  -d '{"age": 35, "gender": "H", "tiene_coche": true, "ipa_class": "Media", "mun_type": "Ciudad"}'

# propensión por sector y producto de un perfil
curl -X POST http://127.0.0.1:8000/propensity \
  -H "Content-Type: application/json" \
  -d '{"age": 30, "gender": "H", "tiene_coche": true, "ipa_class": "Media-alta", "mun_type": "Ciudad"}'
```

Respuesta de `/recommend/coldstart`:

```json
{"recommendations": ["car_insurance", "energy_plan", "..."], "source": "cold_start_cluster", "demo_cluster": 4}
```

Respuesta de `/propensity` (extracto):

```json
{"source": "plano_coldstart",
 "sectores":  [{"sector": "viajes", "p_click": 0.043}, "..."],
 "productos": [{"producto": "car_leasing", "sector": "motor", "p_click": 0.046}, "..."]}
```

Documentación interactiva (Swagger) en `http://127.0.0.1:8000/docs`.

**Manejo de casos sin datos:** una entrada con un campo fuera de dominio devuelve `422` (validación
automática de pydantic). Para un usuario no encontrado no se devuelve `404`: `/recommend` cae a la
popularidad global y `/propensity` responde con `source: "desconocido"` y listas vacías (ambos con
código `200`). Es una simplificación de este ejemplo; un servicio real distinguiría el `404`.

## Notas para un despliegue real (fuera del alcance del TFM)
- Autenticación (API key / OAuth), rate-limiting y CORS.
- Servir artefactos desde un store versionado (S3/registry) en vez de ficheros locales.
- Reentreno y exportación de artefactos programados (batch job).
- Monitorización de latencia y *drift*, logging estructurado, tests de carga.
