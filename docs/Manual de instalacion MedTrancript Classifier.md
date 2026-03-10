# 📘 Manual de Instalación — MedTranscript Classifier

**Proyecto:** MLOps UniAndes — Clasificador de Transcripciones Médicas  
**Versión:** 0.1.0  
**Fecha:** Marzo 2026  
**Repositorio:** [github.com/wiflore/mlops-uniandes-project](https://github.com/wiflore/mlops-uniandes-project)

---

## 📋 Tabla de Contenidos

1. [Descripción General](#1-descripción-general)
2. [Prerequisitos](#2-prerequisitos)
3. [Instalación Local (Desarrollo)](#3-instalación-local-desarrollo)
4. [Modelos Pre-entrenados](#4-modelos-pre-entrenados)
5. [Ejecución de la API (Backend FastAPI)](#5-ejecución-de-la-api-backend-fastapi)
6. [Ejecución del Frontend (Streamlit)](#6-ejecución-del-frontend-streamlit)
7. [Despliegue con Docker](#7-despliegue-con-docker)
8. [Despliegue en la Nube](#8-despliegue-en-la-nube)
9. [Ejecución de Tests](#9-ejecución-de-tests)
10. [Variables de Entorno](#10-variables-de-entorno)
11. [Estructura del Proyecto](#11-estructura-del-proyecto)
12. [Solución de Problemas](#12-solución-de-problemas)

---

## 1. Descripción General

**MedTranscript Classifier** es un sistema de clasificación automática de transcripciones médicas por especialidad, utilizando técnicas de Procesamiento de Lenguaje Natural (NLP). El sistema se compone de:

| Componente | Tecnología | Puerto |
|------------|-----------|--------|
| **API Backend** | FastAPI + Uvicorn | `8000` |
| **Frontend** | Streamlit | `8501` |
| **Modelos ML** | Logistic Regression / XGBoost (pre-entrenados) | — |
| **Vectorización** | TF-IDF (scikit-learn) | — |
| **Almacenamiento** | AWS S3 (modelos y logs) | — |

### Arquitectura de Alto Nivel

```
┌─────────────────┐     HTTP POST /predict     ┌─────────────────────┐
│   Frontend       │ ─────────────────────────► │   API FastAPI        │
│   (Streamlit)    │ ◄───────────────────────── │   (Uvicorn)          │
│   Puerto: 8501   │     JSON Response          │   Puerto: 8000       │
└─────────────────┘                             └──────────┬──────────┘
                                                           │
                                                    ┌──────▼──────┐
                                                    │   Modelos   │
                                                    │  (.joblib)  │
                                                    └──────┬──────┘
                                                           │
                                                    ┌──────▼──────┐
                                                    │   AWS S3    │
                                                    │  (Golden)   │
                                                    └─────────────┘
```

---

## 2. Prerequisitos

### Software Requerido

| Software | Versión Mínima | Verificación |
|----------|---------------|--------------|
| **Python** | 3.9+ (recomendado 3.11) | `python --version` |
| **pip** | 21.0+ | `pip --version` |
| **Git** | 2.30+ | `git --version` |
| **Docker** *(opcional)* | 20.10+ | `docker --version` |
| **AWS CLI** *(para datos/modelos S3)* | 2.x | `aws --version` |

### Cuentas y Accesos Necesarios

- **GitHub**: Acceso al repositorio `wiflore/mlops-uniandes-project`
- **AWS** *(para datos y despliegue)*:
  - Credenciales IAM con acceso al bucket S3 `mlops-medical-project-uniandes-2026`
  - Permisos para S3 (lectura/escritura) y ECR (push de imágenes)
- **Railway** *(para despliegue del frontend)*: Cuenta en [railway.app](https://railway.app/)

---

## 3. Instalación Local (Desarrollo)

### 3.1 Clonar el Repositorio

```bash
git clone https://github.com/wiflore/mlops-uniandes-project.git
cd mlops-uniandes-project
```

### 3.2 Crear Entorno Virtual

**Linux / macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3.3 Instalar Dependencias

**Instalación completa** (desarrollo, notebooks, API y frontend):
```bash
pip install -r requirements.txt
```

**Solo dependencias de la API** (imagen ligera para producción):
```bash
pip install -r requirements-api.txt
```

**Instalación como paquete** (con dependencias de desarrollo):
```bash
pip install -e ".[dev]"
```

### 3.4 Descargar Recursos NLTK

Los stopwords de NLTK se descargan automáticamente al importar el módulo de preprocesamiento. Si deseas hacerlo manualmente:

```python
import nltk
nltk.download('stopwords')
```

### 3.5 Verificar Instalación

```bash
python -c "from src.preprocessing import clean_text; print(clean_text('Test patient with chest pain')); print('✅ Instalación exitosa')"
```

---

## 4. Modelos Pre-entrenados

El proyecto incluye modelos **ya entrenados y listos para usar**. No es necesario re-entrenar. Los modelos se encuentran disponibles en dos ubicaciones:

### 4.1 Modelos en el Repositorio Local

Al clonar el repositorio, los modelos están disponibles en el directorio `models/`:

| Archivo | Descripción | Tamaño |
|---------|-------------|--------|
| `logreg_model.joblib` | Modelo Logistic Regression | ~800 KB |
| `xgboost_model.joblib` | Modelo XGBoost | ~2.3 MB |
| `tfidf_vectorizer.joblib` | Vectorizador TF-IDF | ~400 KB |
| `label_encoder.joblib` | Codificador de etiquetas | ~1 KB |

### 4.2 Modelos en S3 (Zona Golden)

En el despliegue con Docker, los modelos se descargan automáticamente desde la zona golden de S3 al arrancar el contenedor:

```
s3://mlops-medical-project-uniandes-2026/golden/models/
├── logreg_model.joblib
├── xgboost_model.joblib
├── tfidf_vectorizer.joblib
└── label_encoder.joblib
```

> **Nota:** El script `entrypoint.sh` del contenedor API se encarga de descargar estos modelos automáticamente al iniciar.

### 4.3 Selección de Modelo

El sistema soporta dos modelos. La selección se hace mediante la variable de entorno `MODEL_NAME`:

| Modelo | Variable | Accuracy | Descripción |
|--------|----------|----------|-------------|
| **Logistic Regression** *(default)* | `MODEL_NAME=logreg` | ~94.0% | Más ligero, usa sigmoid independiente por clase |
| **XGBoost** | `MODEL_NAME=xgboost` | ~93.5% | Más robusto, usa softmax (probabilidades suman 100%) |

### 4.4 Especialidades Clasificadas (10 clases)

Los modelos clasifican transcripciones médicas en las siguientes especialidades:

- Cardiovascular / Pulmonary
- Orthopedic
- Gastroenterology
- Neurology
- Urology
- General Medicine
- Obstetrics / Gynecology
- Radiology
- Nephrology
- Surgery

---

## 5. Ejecución de la API (Backend FastAPI)

### 5.1 Iniciar el Servidor

> **Prerequisito:** Los modelos deben existir en el directorio `models/`. Si clonaste el repositorio, ya están incluidos.

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

> **Nota:** El flag `--reload` es para desarrollo. En producción, omitirlo.

### 5.2 Endpoints Disponibles

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/health` | Verificar estado de la API y modelo |
| `POST` | `/predict` | Clasificar una transcripción médica |
| `GET` | `/dashboard-data` | Datos de analytics (requiere Bearer token) |
| `GET` | `/docs` | Documentación interactiva Swagger UI |
| `GET` | `/redoc` | Documentación alternativa ReDoc |

### 5.3 Ejemplo de Uso

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Predicción:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"transcription": "Patient presents with chest pain and shortness of breath. ECG shows ST elevation in leads II, III, and aVF."}'
```

**Respuesta esperada:**
```json
{
  "specialty": "Cardiovascular / Pulmonary",
  "confidence": 0.92,
  "top_3": [
    {"specialty": "Cardiovascular / Pulmonary", "probability": 0.92},
    {"specialty": "General Medicine", "probability": 0.45},
    {"specialty": "Neurology", "probability": 0.12}
  ],
  "model_name": "logreg",
  "model_version": "0.1.0"
}
```

### 5.4 Selección de Modelo en Ejecución Local

Por defecto se usa Logistic Regression. Para cambiar a XGBoost:
```bash
set MODEL_NAME=xgboost          # Windows
export MODEL_NAME=xgboost       # Linux/Mac
uvicorn src.api:app --port 8000
```

---

## 6. Ejecución del Frontend (Streamlit)

### 6.1 Iniciar la Aplicación

> **Importante:** La API debe estar ejecutándose antes de iniciar el frontend.

```bash
cd frontend
streamlit run app.py --server.port 8501
```

O desde la raíz del proyecto:
```bash
streamlit run frontend/app.py --server.port 8501
```

Acceder en: `http://localhost:8501`

### 6.2 Configurar URL de la API

Si la API si ejecuta en un host diferente:
```bash
set API_URL=http://mi-servidor:8000       # Windows
export API_URL=http://mi-servidor:8000    # Linux/Mac
```

### 6.3 Pestañas del Frontend

| Pestaña | Descripción |
|---------|-------------|
| 🏠 **Home** | Información general, estadísticas del modelo y disclaimer médico |
| 📊 **Clasificar** | Ingresar transcripciones y obtener predicciones en tiempo real |
| 📈 **Analytics** | Dashboard de monitoreo con datos de producción desde S3 |

---

## 7. Despliegue con Docker

### 7.1 Backend API (Docker)

> **Importante:** El contenedor de la API descarga automáticamente los modelos pre-entrenados desde S3 zona golden al arrancar. No se requiere entrenamiento.

**Construir la imagen:**
```bash
docker build -t medtranscript-api .
```

**Ejecutar el contenedor:**
```bash
docker run -d \
  --name medtranscript-api \
  -p 8000:8000 \
  -e AWS_ACCESS_KEY_ID=<tu-access-key> \
  -e AWS_SECRET_ACCESS_KEY=<tu-secret-key> \
  -e AWS_DEFAULT_REGION=us-east-1 \
  -e S3_BUCKET=mlops-medical-project-uniandes-2026 \
  -e MODEL_NAME=logreg \
  -e API_SECRET_KEY=<tu-clave-secreta> \
  medtranscript-api
```

> El contenedor ejecuta `entrypoint.sh` al arrancar, que descarga los modelos pre-entrenados desde S3 zona golden y luego inicia la API automáticamente.

### 7.2 Frontend Streamlit (Docker)

**Construir la imagen** (desde la raíz del proyecto):
```bash
docker build -f frontend/Dockerfile -t medtranscript-frontend .
```

**Ejecutar el contenedor:**
```bash
docker run -d \
  --name medtranscript-frontend \
  -p 8501:8501 \
  -e API_URL=http://host.docker.internal:8000 \
  -e DASHBOARD_TOKEN=<tu-token> \
  -e MODEL_VERSION=v3.0-sigmoid-10classes \
  medtranscript-frontend
```

### 7.3 Docker Compose (Ambos Servicios)

Crear un archivo `docker-compose.yml` en la raíz:

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
      - AWS_DEFAULT_REGION=us-east-1
      - S3_BUCKET=mlops-medical-project-uniandes-2026
      - MODEL_NAME=logreg
      - API_SECRET_KEY=${API_SECRET_KEY}

  frontend:
    build:
      context: .
      dockerfile: frontend/Dockerfile
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
      - DASHBOARD_TOKEN=${DASHBOARD_TOKEN}
      - MODEL_VERSION=v3.0-sigmoid-10classes
    depends_on:
      - api
```

Ejecutar:
```bash
docker-compose up -d
```

---

## 8. Despliegue en la Nube

### 8.1 Frontend en Railway

1. Crear un proyecto en [Railway](https://railway.app/) → **New Project** → **Deploy from GitHub repo**
2. Seleccionar el repositorio `mlops-uniandes-project`
3. Configurar variables de entorno en la pestaña **Variables**:
   - `API_URL`: URL pública del backend (ej: `http://<ip-ec2>:8000`)
   - `DASHBOARD_TOKEN`: Clave secreta para analytics
   - `MODEL_VERSION`: `v3.0-sigmoid-10classes`
4. Verificar en **Settings**:
   - **Root Directory**: `/`
   - **Dockerfile Path**: `frontend/Dockerfile`
5. Railway construirá automáticamente. Ir a **Settings → Networking → Generate Domain** para la URL pública.

### 8.2 Backend API en AWS ECS (Fargate)

> Los modelos pre-entrenados se descargan automáticamente desde S3 al arrancar cada task de ECS.

#### Paso 1: Subir imagen a ECR

```bash
# Autenticarse en ECR
aws ecr get-login-password --region <REGION> | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com

# Crear repositorio (si no existe)
aws ecr create-repository --repository-name mlops-medical-api

# Tagear y subir imagen
docker tag medtranscript-api:latest <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/mlops-medical-api:latest
docker push <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/mlops-medical-api:latest
```

#### Paso 2: Crear Task Definition

Usar el archivo de ejemplo `task-definition.example.json` como base:
- Reemplazar `<AWS_ACCOUNT_ID>` con tu ID de cuenta AWS
- Reemplazar `<AWS_REGION>` con la región (ej: `us-east-1`)
- Reemplazar `<YOUR_S3_BUCKET_NAME>` con el nombre del bucket S3

```bash
# Copiar y editar
cp task-definition.example.json task-definition.json
# Editar task-definition.json con tus valores

# Registrar la task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

#### Paso 3: Crear servicio ECS

```bash
aws ecs create-service \
  --cluster <tu-cluster> \
  --service-name mlops-medical-api \
  --task-definition mlops-medical-api \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[<subnet-id>],securityGroups=[<sg-id>],assignPublicIp=ENABLED}"
```

### 8.3 Estructura de S3

```
s3://mlops-medical-project-uniandes-2026/
├── golden/
│   ├── data/                    ← Dataset versionado con DVC
│   │   └── mtsamples.csv
│   └── models/                  ← Modelos de producción
│       ├── logreg_model.joblib
│       ├── xgboost_model.joblib
│       ├── tfidf_vectorizer.joblib
│       └── label_encoder.joblib
└── analytics/
    └── api-calls/               ← Logs de predicciones (middleware)
        └── 2026/03/10/
            └── <uuid>.json
```

---

## 9. Ejecución de Tests

### 9.1 Con pytest Directamente

```bash
pytest tests/ -v
```

### 9.2 Con Coverage

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### 9.3 Con Tox (Entorno Aislado)

```bash
pip install tox
tox
```

### 9.4 Linting

```bash
tox -e lint
```

O directamente con flake8:
```bash
flake8 src/ tests/ --max-line-length=120 --exclude=__pycache__
```

### 9.5 Suite de Tests

| Archivo | Descripción |
|---------|-------------|
| `tests/test_api.py` | Tests de endpoints de la API |
| `tests/test_api_edge_cases.py` | Casos extremos de la API |
| `tests/test_predict.py` | Tests del módulo de predicción |
| `tests/test_predict_edge_cases.py` | Casos extremos de predicción |
| `tests/test_preprocessing.py` | Tests de preprocesamiento NLP |
| `tests/test_preprocessing_edge_cases.py` | Casos extremos de preprocesamiento |
| `tests/test_schemas.py` | Tests de schemas Pydantic |
| `tests/test_integration.py` | Tests de integración end-to-end |

---

## 10. Variables de Entorno

### Variables de la API (Backend)

| Variable | Valor por Defecto | Descripción |
|----------|-------------------|-------------|
| `MODELS_DIR` | `models` | Directorio donde se encuentran los modelos pre-entrenados |
| `MODEL_NAME` | `logreg` | Modelo a usar (`logreg` o `xgboost`) |
| `S3_BUCKET` | `mlops-medical-project-uniandes-2026` | Bucket S3 para modelos y logs |
| `API_SECRET_KEY` | *(sin valor)* | Clave para proteger el endpoint `/dashboard-data` |
| `S3_LOG_PREFIX` | `analytics/api-calls` | Prefijo S3 para logs del middleware |

### Variables del Frontend (Streamlit)

| Variable | Valor por Defecto | Descripción |
|----------|-------------------|-------------|
| `API_URL` | `http://localhost:8000` | URL del backend FastAPI |
| `DASHBOARD_TOKEN` | *(vacío)* | Token Bearer para `/dashboard-data` |
| `MODEL_VERSION` | `v3.0-sigmoid-10classes` | Versión a mostrar en la UI |

### Variables AWS (Requeridas en Docker)

| Variable | Descripción |
|----------|-------------|
| `AWS_ACCESS_KEY_ID` | Clave de acceso AWS |
| `AWS_SECRET_ACCESS_KEY` | Clave secreta AWS |
| `AWS_DEFAULT_REGION` | Región AWS (ej: `us-east-1`) |

### Archivo `.env` (Recomendado)

Crear un archivo `.env` en la raíz del proyecto:
```env
# AWS
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1

# API
S3_BUCKET=mlops-medical-project-uniandes-2026
MODEL_NAME=logreg
API_SECRET_KEY=mi-clave-secreta

# Frontend
API_URL=http://localhost:8000
DASHBOARD_TOKEN=mi-token-dashboard
MODEL_VERSION=v3.0-sigmoid-10classes
```

> ⚠️ **No subir `.env` al repositorio.** Ya está incluido en `.gitignore`.

---

## 11. Estructura del Proyecto

```
mlops-uniandes-project/
│
├── src/                          # Código fuente principal
│   ├── __init__.py               # Package init (versión 0.1.0)
│   ├── api.py                    # API FastAPI (endpoints)
│   ├── predict.py                # Módulo de inferencia (sigmoid/softmax)
│   ├── preprocessing.py          # Pipeline NLP (TF-IDF, clean_text)
│   ├── schemas.py                # Schemas Pydantic (request/response)
│   ├── train.py                  # Script de entrenamiento + MLflow
│   └── middleware_logging.py     # Middleware para logs en S3
│
├── frontend/                     # Frontend Streamlit
│   ├── app.py                    # Aplicación principal (3 tabs)
│   ├── Dockerfile                # Dockerfile del frontend
│   └── requirements.txt          # Dependencias del frontend
│
├── models/                       # Modelos PRE-ENTRENADOS (.joblib) — listos para usar
│   ├── logreg_model.joblib       # Logistic Regression (~800 KB)
│   ├── xgboost_model.joblib      # XGBoost (~2.3 MB)
│   ├── tfidf_vectorizer.joblib   # Vectorizador TF-IDF (~400 KB)
│   └── label_encoder.joblib      # Codificador de etiquetas (~1 KB)
│
├── data/                         # Datos del proyecto
│   ├── raw/                      # Datasets originales
│   ├── mtsamples.csv.dvc         # Archivo DVC (tracking)
│   └── DATA_DICTIONARY.md        # Diccionario de datos
│
├── tests/                        # Suite de pruebas
│   ├── conftest.py               # Fixtures compartidas
│   ├── test_api.py
│   ├── test_predict.py
│   ├── test_preprocessing.py
│   ├── test_schemas.py
│   ├── test_integration.py
│   └── *_edge_cases.py           # Tests de casos extremos
│
├── notebooks/                    # Jupyter Notebooks
│   ├── 01_EDA_transcripciones_medicas.ipynb
│   └── 02_modeling_v1.ipynb
│
├── docs/                         # Documentación
│   ├── manual_instalacion.md     # ← Este documento
│   └── tutorial_railway.md       # Tutorial despliegue Railway
│
├── Diseños Tablero/              # Estilos CSS del frontend
│   └── styles.css
│
├── Dockerfile                    # Dockerfile del backend API
├── entrypoint.sh                 # Script de arranque del contenedor
├── requirements.txt              # Dependencias completas
├── requirements-api.txt          # Dependencias solo API (producción)
├── setup.py                      # Empaquetado del proyecto
├── tox.ini                       # Configuración Tox (tests/lint)
├── railway.toml                  # Configuración Railway
├── task-definition.example.json  # Plantilla AWS ECS Task Definition
├── CONTRIBUTING.md               # Guía de contribución
└── README.md                     # Descripción general
```
