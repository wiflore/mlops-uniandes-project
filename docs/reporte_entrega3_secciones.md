# Secciones para el Reporte Final (Entrega 3)

---

## 1. Cambios Arquitectónicos frente a la Entrega 2

En la Entrega 2, el modelo y la API corrían en entornos locales o instancias estáticas (EC2), y los datos se almacenaban sin una segregación clara entre producción y experimentación. Para esta Entrega 3, la arquitectura evolucionó hacia un modelo **Cloud-Native y Serverless**:
1. **Despliegue Serverless:** Transición de entornos manuales a contenedores Docker gestionados automáticamente por AWS ECS Fargate.
2. **Data Lake Estructurado:** Migración del almacenamiento local de datos a un esquema de zonas (Golden y Analítica) en Amazon S3, unificando el control de versiones (DVC) y el monitoreo de ML.

---

## 2. Justificación del Servicio Cloud: AWS ECS (Fargate)

Evaluamos EKS, EC2 y ECS (Fargate) para desplegar la API. Elegimos **ECS con Fargate** por tres razones clave:

1. **Baja Complejidad Operativa:** Fargate elimina la necesidad de gestionar servidores subyacentes (parches, OS). EKS era desproporcionadamente complejo (gestión de clústeres Kubernetes) para las necesidades actuales del prototipo.
2. **Eficiencia en Costos:** EC2 cobra por hora encendida y EKS tiene un costo base mensual elevado ($73 USD). Fargate sigue un modelo *pay-as-you-go*, cobrando estrictamente por los recursos (CPU/RAM) consumidos durante la ejecución del contenedor.
3. **Escalabilidad y Ecosistema:** Integración nativa sin fricción con servicios de MLOps de AWS (ECR, S3, CloudWatch) mediante IAM Task Roles, facilitando el escalamiento automático en picos de demanda.

---

## 3. Arquitectura de Datos: S3 Data Lake y DVC

Implementamos un bucket unificado en Amazon S3 (`mlops-medical-project-uniandes-2026`) bajo el estándar de *Medallion Architecture*, dividido en dos zonas estrictas:

- **Zona Golden (`/golden`):** Entorno inmutable y de solo lectura para la API. Almacena el dataset curado y los modelos entrenados (`.joblib`). Se controla explícitamente usando **DVC**.
- **Zona Analítica (`/analytics`):** Entorno de escritura. Un middleware en la API intercepta asíncronamente cada request HTTP y guarda la entrada del usuario junto con la predicción del modelo en formato JSON para futuro monitoreo de *Data Drift*.

**Acceso Downstream para el Dashboard (Streaming de Históricos):**
Para desacoplar el frontend del almacenamiento subyacente y proteger las credenciales de AWS, se expone un endpoint dedicado en la API (`GET /dashboard-data`). En lugar de que Frontend consulte S3 directamente, este endpoint funciona como un puente: lee los registros históricos en la Zona Analítica, los consolida y los retorna al Frontend en un formato estructurado y unificado.

El endpoint está protegido por Autenticación Simple. El dashboard debe enviar:
`Headers: { "Authorization": "Bearer <API_SECRET_KEY_DEL_PROYECTO>" }`

> **Nota para Pruebas (Swagger UI o Terminal):**
> Dado que el endpoint requiere un *Bearer Token*, acceder desde la barra de direcciones arrojará un error `401 Unauthorized`. 
> - **Opción A (Visual):** Entre a `http://<IP_API>:8000/docs`, haga clic en "Authorize", pegue su token secreto y testee el `GET /dashboard-data` dando clic a "Try it out".
> - **Opción B (cURL):** Ejecute en su consola: 
>   `curl -s -H "Authorization: Bearer <API_SECRET>" http://<IP_API>:8000/dashboard-data`

**Estructura de la Respuesta (`/dashboard-data`):**
Devuelve un JSON con una lista consolidada `data` que contiene el histórico de predicciones. Cada elemento respeta el esquema `ApiCallLog`:

```json
{
  "data": [
    {
      "request_id": "b57e067c-cb6e-40c6...",
      "timestamp": "2026-03-06T06:35:55.072Z",
      "method": "POST",
      "path": "/predict",
      "request_body": "{\"transcription\": \"Patient presents with chest pain...\"}",
      "response_body": "{\"specialty\":\"Cardiology\",\"confidence\":0.85...}",
      "status_code": 200,
      "response_time_ms": 4.52
    }
  ]
}
```
*Con esta estructura, el frontend puede deserializar fácilmente `request_body` y `response_body` usando `pandas.json_normalize()` para graficar uso de modelos, confianzas promedio y distribuciones de especialidades.*

### Diagrama de Flujo MLOps (Estructura)

La arquitectura sigue el patrón **API Gateway** con almacenamiento desacoplado:

1. **Interacción de los Clientes:**
   - **Usuarios / Médicos:** Envían la transcripción usando el método `POST /predict`. Las peticiones llegan Directamente a la IP Pública Expuesta por AWS.
   - **Dashboard (Streamlit):** Consume el histórico usando el método `GET /dashboard-data`, enviando el token de autenticación a la misma IP.

2. **Capa de Cómputo (AWS ECS Fargate):**
   - El contenedor FastAPI recibe el tráfico gracias a su Interfaz de Red Elástica (ENI) pública asignada por Fargate.
   - Al encender, el contenedor descarga los modelos de Machine Learning pre-entrenados desde la *Zona Golden* de S3.

3. **Capa de Almacenamiento (S3 Data Lake) y DVC:**
   - **Zona Golden (`/golden`):** Contiene los datos crudos (`mtsamples.csv`) versionados por el remoto de **DVC** y los modelos serializados (`.joblib`). Es de **solo lectura** para la API.
   - **Zona Analítica (`/analytics`):** Cada vez que la API hace una predicción, un *middleware* guarda silenciosamente el texto original y la predicción en formato JSON.
   - Cuando el Dashboard pide los datos, la API descarga y unifica estos JSONs de la *Zona Analítica* para enviarlos listos para graficar.

---

## 4. Corrección del Pipeline de Entrenamiento (`train.py`)

### Problema Detectado

Al analizar los logs de producción vía el endpoint `/dashboard-data`, se identificó que las predicciones del modelo desplegado presentaban confianzas anormalmente bajas (~24-33%) y solo clasificaban en **5 especialidades** en lugar de las 10 documentadas en el notebook de modelado (`02_modeling_v1.ipynb`).

**Causa raíz:** El script `train.py` ejecutaba 3 experimentos secuencialmente, cada uno sobrescribiendo los mismos archivos `.joblib` en `models/`:

```python
# ANTES (problemático): cada llamada sobrescribe los mismos archivos
train_models(min_samples=50,  ...)  # Exp 1: 10 clases ✅
train_models(min_samples=100, ...)  # Exp 2: 5 clases  ← sobrescribe
train_models(min_samples=50,  ..., lr_c=0.1)  # Exp 3  ← sobrescribe de nuevo
```

El **Experimento 2** (`min_samples=100`) filtraba a solo 5 especialidades con ≥100 muestras, y el **Experimento 3** cambiaba `C=0.1`, generando un modelo más conservador. Cualquiera de estos dos experimentos dejaba un modelo subóptimo como el de producción.

### Solución Implementada

Se refactorizó `train.py` para aislar los experimentos del modelo de producción:

```python
# DESPUÉS: cada experimento guarda en su propia subcarpeta
train_models(min_samples=50,  ..., output_dir="models/experiments/exp1_baseline")
train_models(min_samples=100, ..., output_dir="models/experiments/exp2_strict")
train_models(min_samples=50,  ..., output_dir="models/experiments/exp3_alt_hp")

# El modelo de producción SIEMPRE se guarda al final en models/ (raíz)
train_production_model()  # min_samples=50, C=1.0 → 10 clases, Acc: 89.8%
```

**Resultado:** El modelo de producción ahora clasifica consistentemente en **10 especialidades** con un Accuracy del **89.8%** y F1 Macro de **0.82**.

---

## 5. Probabilidades Independientes (Sigmoid vs Softmax)

### Problema con Softmax

El enfoque original usaba `predict_proba()` de scikit-learn, que aplica **softmax** a las puntuaciones del modelo. Esto obliga a que las probabilidades de todas las clases sumen exactamente 1.0 (100%). Con 10 clases, esto distorsiona la interpretación:

| Transcripción | Especialidad | Confianza Softmax |
|---|---|---|
| CT scan abdominal con masa renal | Radiology | 16.7% |
| Fractura de tibia + cirugía | Surgery | 14.2% |

Una confianza del 16.7% sugiere que el modelo "no está seguro", cuando en realidad sí identifica correctamente la especialidad: simplemente no puede asignar más a una clase sin quitarle a las demás.

### Solución: Sigmoid sobre `decision_function`

Se cambió el módulo de inferencia (`predict.py`) para usar la función **sigmoid** aplicada a las puntuaciones crudas (`decision_function`) de cada clase de forma **independiente**:

```python
from scipy.special import expit  # sigmoid

raw_scores = model.decision_function(X)[0]        # puntuaciones crudas por clase
independent_probs = expit(raw_scores)              # sigmoid independiente
# → Radiology: 81.4%, Surgery: 65.4% (NO suman 100%)
```

Cada clase ahora responde a la pregunta: *"¿Qué tan probable es que esta transcripción pertenezca a esta especialidad?"* de forma independiente. Esto permite resultados como:

| Transcripción | Top 3 (Sigmoid) |
|---|---|
| CT scan abdominal con masa renal | Radiology **81.4%**, Surgery 65.4%, Urology 62.0% |
| Fractura de tibia + fijación interna | Radiology **73.3%**, Surgery 71.9%, Orthopedic 62.5% |
| Dolor torácico + elevación ST | SOAP/Notes **66.2%**, Radiology 64.5%, Gen. Medicine 62.6% |

### Justificación Técnica

1. **Interpretabilidad clínica:** En contexto médico, una transcripción puede ser relevante para múltiples especialidades simultáneamente (e.g., una fractura requiere tanto Radiología como Cirugía). Las probabilidades independientes reflejan esta realidad.
2. **Consistencia con el rendimiento real:** El modelo tiene un accuracy del 89.8% en test, pero softmax forzaba confianzas artificialmente bajas (~16%). Sigmoid refleja mejor la certeza real del modelo.
3. **Compatibilidad con XGBoost:** Se mantiene `predict_proba()` como fallback para modelos sin `decision_function` (e.g., XGBoost), garantizando compatibilidad retroactiva.
