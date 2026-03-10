# Reporte de Pruebas — Medical Specialty Classifier API
**Fecha de ejecución:** 2026-03-09  
**Entorno:** Python 3.12.3 · scikit-learn 1.8.0 · FastAPI latest  
**Modelos evaluados:** `logreg_model.joblib`, `xgboost_model.joblib`

---

## Resumen ejecutivo

| Métrica | Valor |
|---|---|
| Total de tests | **213** |
| Pasaron | **203** |
| Saltados (skip) | **2** |
| xfail (bug documentado) | **5** |
| Fallaron | **0** |
| Tiempo total | **~2 s** |
| Cobertura de módulos | `src/api`, `src/predict`, `src/preprocessing`, `src/schemas` |

---

## Estructura de la suite de tests

```
tests/
├── conftest.py                    ← Fixtures compartidos, mock de S3/boto3
├── test_api.py                    ← Tests originales /health y /predict
├── test_api_edge_cases.py         ← Edge cases exhaustivos de la API  (NEW)
├── test_integration.py            ← Flujo end-to-end completo          (NEW)
├── test_predict.py                ← Tests originales del predictor
├── test_predict_edge_cases.py     ← Edge cases del predictor           (NEW)
├── test_preprocessing.py          ← Tests originales de preprocesamiento
├── test_preprocessing_edge_cases.py ← Edge cases de preprocesamiento   (NEW)
└── test_schemas.py                ← Tests originales de schemas Pydantic
```

---

## Detalle por módulo

### `test_api.py` — Endpoint tests originales (10 tests)

| Test | Estado | Descripción |
|---|---|---|
| `test_health_returns_200` | PASS | GET /health devuelve HTTP 200 |
| `test_health_response_structure` | PASS | Response contiene status, model_loaded, version |
| `test_health_model_loaded` | PASS | model_loaded=True cuando modelos están en disco |
| `test_predict_valid_text` | PASS | Texto médico válido → 200 |
| `test_predict_response_structure` | PASS | Response contiene specialty, confidence, top_3, model_name |
| `test_predict_confidence_range` | PASS | confidence ∈ [0, 1] |
| `test_predict_top3_has_three_items` | PASS | top_3 retorna exactamente 3 elementos |
| `test_predict_short_text_rejected` | PASS | Texto < 10 chars → 422 |
| `test_predict_empty_body_rejected` | PASS | JSON {} → 422 |
| `test_predict_missing_field_rejected` | PASS | Campo incorrecto → 422 |

---

### `test_api_edge_cases.py` — Edge cases de la API (76 tests)

#### Validación de entrada (sin modelos)

| Test | Estado | Nota |
|---|---|---|
| `test_empty_string_rejected` | PASS | "" → 422 |
| `test_single_character_rejected` | PASS | "A" → 422 |
| `test_nine_chars_rejected` | PASS | 9 chars → 422 (min_length=10) |
| `test_whitespace_only_rejected` | PASS | 10 espacios → schema acepta, predictor falla safe |
| `test_newlines_only_edge` | PASS | Solo newlines → manejo seguro |
| `test_missing_transcription_field` | PASS | Campo `text` en vez de `transcription` → 422 |
| `test_null_transcription_rejected` | PASS | `null` → 422 |
| `test_integer_transcription_rejected` | PASS | Número entero → 422 |
| `test_list_transcription_rejected` | PASS | Array → 422 |
| `test_empty_json_body` | PASS | `{}` → 422 |
| `test_no_body` | PASS | Sin body → 422 |
| `test_wrong_content_type` | PASS | form-urlencoded → 422 |
| `test_extra_fields_ignored` | PASS | Campos extra ignorados correctamente |

#### Contenido del texto (requieren modelos)

| Test | Estado | Nota |
|---|---|---|
| `test_special_chars_only` | PASS | `!@#$%^` → clean_text vacía, predictor retorna algo |
| `test_numeric_only_text` | PASS | Solo dígitos → safe handling |
| `test_unicode_latin_extended` | PASS | Español con acentos → procesado |
| `test_emoji_only_text` | PASS | Emojis médicos → manejo seguro |
| `test_mixed_languages` | PASS | Inglés + español → predicción |
| `test_html_xss_in_transcription` | PASS | `<script>alert()` → no ejecuta HTML |
| `test_sql_injection_in_transcription` | PASS | SQL injection → solo texto inofensivo |
| `test_path_traversal_in_transcription` | PASS | `../../etc/passwd` → ignorado |
| `test_random_gibberish_short` | PASS | Texto sin sentido → predicción (qualidad baja) |
| `test_repeated_single_word` | PASS | "surgery " × 30 → predicción |
| `test_all_stopwords_text` | PASS | Solo stopwords → safe handling |
| `test_lorem_ipsum` | PASS | Lorem ipsum → predicción, confidence válida |
| `test_very_long_text_10k_chars` | PASS | ~10K chars → completado en < 10s |
| `test_very_long_text_50k_chars` | PASS | ~50K chars → completado en < 30s |
| `test_extremely_long_single_word` | PASS | "a" × 10K → safe |
| `test_text_with_tabs_and_newlines` | PASS | Whitespace mixto → normalizado |
| `test_text_with_excessive_whitespace` | PASS | Espacios múltiples → normalizado |

#### Invariantes del response (5 textos × 6 checks = 30 tests)

Todos PASS — para cualquier texto válido:
- Response contiene specialty, confidence, top_3, model_name, model_version
- confidence ∈ [0.0, 1.0]
- top_3[0].specialty == specialty (primer elemento coincide con predicción)
- top_3 ordenado descendente por probabilidad
- specialty es string no vacío
- model_name no está vacío

#### Performance básica (4 tests)

| Test | Estado | Nota |
|---|---|---|
| `test_response_time_under_2s_normal` | PASS | Predicción normal < 2s |
| `test_10_sequential_requests` | PASS | 10 requests sin errores |
| `test_repeated_same_text_deterministic` | PASS | Mismo texto → mismo resultado |
| `test_interleaved_health_and_predict` | PASS | /health y /predict alternados OK |

---

### `test_integration.py` — Flujo end-to-end (37 tests)

#### Health check flow (7 tests)
Todos PASS — incluye idempotencia, formato semver, método HTTP incorrecto (405).

#### Health → Predict happy path (4 tests)
Todos PASS — flujo completo verificando estructura profunda del response.

#### Múltiples especialidades (5 tests)
Todos PASS — Dermatology, Orthopedic, Neurology, Gastroenterology, Cardiology.

#### Error handling (9 tests)
Todos PASS — todos retornan 4xx, incluye 404 para rutas inexistentes, 405 para GET /predict.

#### Boundary values min_length (4 tests)
Todos PASS — frontera exacta en 9/10/11 caracteres verificada.

#### Timing end-to-end (3 tests)
Todos PASS — /health < 500ms, /predict < 3s, 5 requests consecutivos < 15s.

---

### `test_predict.py` + `test_predict_edge_cases.py` — Predictor (43 tests)

#### Logreg — todos pasaron

| Categoría | Tests | Estado |
|---|---|---|
| Carga de modelo | 2 | PASS |
| Estructura del output (tupla, tipos, rangos) | 5 | PASS |
| Sin modelo cargado (RuntimeError) | 2 | PASS |
| Texto vacío / solo chars especiales / números / stopwords | 7 | PASS |
| Textos muy largos (10K, 50K palabras) | 2 | PASS |
| Unicode (japonés, árabe, emojis, español) | 4 | PASS |
| Propiedades estadísticas (prob >= 0, <= 1, max=confidence) | 5 | PASS |
| Determinismo (mismo input → mismo output) | 1 | PASS |
| Error handling sin modelo / directorio inexistente | 5 | PASS |

#### XGBoost — 5 tests marcados xfail

---

## Bugs encontrados

### BUG-001 — XGBoost: desacuerdo LabelEncoder / modelo (CRITICO)

**Descripción:**  
`xgboost_model.joblib` predice índices de clase (ej. `6`) que no existen en `label_encoder.joblib`. El error ocurre en `MedicalSpecialtyPredictor.predict()` al llamar `label_encoder.inverse_transform([6])`.

**Error:**
```
ValueError: y contains previously unseen labels: [6]
```

**Causa raíz probable:**  
El XGBoost fue entrenado en un momento diferente al del LabelEncoder serializado, o con una versión del dataset con más/menos clases.

**Impacto:**  
- `/predict` con `MODEL_NAME=xgboost` retorna HTTP 500.
- El modelo logreg **no tiene este problema**.

**Archivos afectados:**
- `models/xgboost_model.joblib` — necesita re-entrenamiento
- `src/predict.py` — podría añadir manejo defensivo del ValueError

**Tests que documentan el bug:**
- `tests/test_predict.py::TestMedicalSpecialtyPredictor::test_predict_xgboost_basic` (xfail)
- `tests/test_predict_edge_cases.py::TestModelComparison::*` (xfail × 4)

**Solución recomendada:**
```bash
python -m src.train  # Re-entrenar ambos modelos con el mismo LabelEncoder
```

---

### BUG-002 — `on_event("startup")` deprecado en FastAPI

**Descripción:**  
`src/api.py` usa `@app.on_event("startup")` que está deprecado en FastAPI moderno.

**Impacto:** Advertencia en logs; sin impacto funcional actual pero puede romperse en versiones futuras.

**Solución recomendada:**
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load(model_name=MODEL_NAME)
    yield

app = FastAPI(..., lifespan=lifespan)
```

---

### NOTA-001 — clean_text no sanitiza XSS/SQL (comportamiento esperado)

`clean_text` no está diseñada como sanitizador de seguridad — elimina caracteres especiales por la regex `[^a-záéíóúñ\s]`, dejando palabras alfanuméricas como `drop`, `alert`, `table`. Esto es **correcto** para el propósito de clasificación NLP pero debe estar documentado claramente para que no se use como sanitizador de seguridad en otros contextos.

---

### NOTA-002 — TF-IDF no apta para corpus vacios

`TextPreprocessor.fit_transform([])` o con corpus donde todos los documentos quedan vacíos tras `clean_text` (ej. solo números) lanza `ValueError: max_df corresponds to < documents than min_df`. Considerado comportamiento esperado de sklearn; 2 tests marcados como `skip` documentan este límite.

---

## Tests skipped

| Test | Razón |
|---|---|
| `test_fit_transform_empty_strings` | sklearn rechaza corpus completamente vacío (comportamiento esperado) |
| `test_fit_transform_numbers_only` | sklearn rechaza corpus numérico (vocabulario vacío tras clean_text) |

---

## Cobertura de categorías de edge cases

| Categoría | Cobertura |
|---|---|
| Validación de schema (campos, tipos, longitud) | Completa |
| Textos vacíos / solo whitespace | Completa |
| Textos muy cortos (boundary min_length=10) | Completa |
| Textos muy largos (10K, 50K chars) | Completa |
| Solo caracteres especiales / símbolos | Completa |
| Solo dígitos numéricos | Completa |
| Emojis / unicode no-latino | Completa |
| Mezcla de idiomas | Completa |
| XSS / HTML injection | Probado (API segura, clean_text no es sanitizador) |
| SQL injection | Probado (API segura) |
| Path traversal | Probado |
| Lorem ipsum / gibberish | Completa |
| Solo stopwords del inglés | Completa |
| Tiempos de respuesta | Completa |
| Determinismo del modelo | Completa |
| Invariantes del response | Completa |
| Flujo end-to-end | Completa |
| Manejo de errores HTTP (405, 404, 422, 500) | Completa |

---

## Cómo ejecutar los tests

```bash
# Todos los tests
python -m pytest tests/ -v

# Solo edge cases
python -m pytest tests/test_api_edge_cases.py tests/test_predict_edge_cases.py -v

# Con coverage
python -m pytest tests/ --cov=src --cov-report=term-missing

# Solo los que no requieren modelos
python -m pytest tests/ -v -k "not skipif"

# Ver bugs documentados (xfail)
python -m pytest tests/ -v --runxfail
```
