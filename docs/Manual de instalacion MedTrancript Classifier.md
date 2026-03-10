# 📋 Manual de Instalación y Despliegue — MedTranscript Classifier

> **Proyecto:** MLOps MedTranscript Classifier 
> **Repositorio:** `mlops-uniandes-project`  
> **Versión del modelo:** v3.0-sigmoid-10classes  
> **Fecha:** Marzo 2026

---

## Tabla de Contenido

1. [Requisitos Previos](#1-requisitos-previos)
2. [Instalación Local (Desarrollo)](#2-instalación-local-desarrollo)
3. [Ejecución del Frontend (Streamlit)](#3-ejecución-del-frontend-streamlit)
4. [Ejecución del Backend (API FastAPI)](#4-ejecución-del-backend-api-fastapi)
5. [Configuración de DVC (Data Version Control)](#5-configuración-de-dvc-data-version-control)
6. [Despliegue con Docker (Local)](#6-despliegue-con-docker-local)
7. [Despliegue en AWS ECS Fargate (Backend)](#7-despliegue-en-aws-ecs-fargate-backend)
8. [Despliegue en Railway (Frontend)](#8-despliegue-en-railway-frontend)
9. [Variables de Entorno Completas](#9-variables-de-entorno-completas)

---

## 1. Requisitos Previos

### Software necesario

| Herramienta       | Versión mínima | Propósito                          |
|-------------------|----------------|------------------------------------|
| Python            | 3.11           | Runtime del proyecto               |
| pip               | 23.0+          | Gestión de paquetes Python         |
| Git               | 2.40+          | Control de versiones               |
| Docker            | 24.0+          | Contenedorización                  |
| AWS CLI           | 2.x            | Interacción con servicios AWS      |
| DVC               | 3.0+           | Versionado de datos y modelos      |

### Cuentas y accesos requeridos

- **AWS:** Cuenta con permisos para ECR, ECS, S3 e IAM (o acceso LabRole de AWS Academy).
- **Railway:** Cuenta gratuita o de pago para el despliegue del frontend.
- **GitHub:** Acceso al repositorio `mlops-uniandes-project`.

---

## 2. Instalación Local (Desarrollo)

### 2.1. Clonar el repositorio

```bash
git clone https://github.com/wiflore/mlops-uniandes-project.git
cd mlops-uniandes-project
```

### 2.2. Crear un entorno virtual

```bash
# Crear entorno virtual
python -m venv venv

# Activar (Windows)
venv\Scripts\activate

# Activar (Linux/Mac)
source venv/bin/activate
```

### 2.3. Instalar dependencias completas (desarrollo)

```bash
pip install -r requirements.txt
```

> **Nota:** `requirements.txt` incluye todas las dependencias: ciencia de datos, API, NLP, MLflow, DVC, Jupyter, Streamlit y visualización.

### 2.4. Descargar datos NLTK

```bash
python -c "import nltk; nltk.download('stopwords')"
```

### 2.5. Obtener datos y modelos con DVC

Si el remoto DVC ya está configurado:

```bash
dvc pull
```

Esto descargará el dataset (`data/raw/mtsamples.csv`) y los modelos entrenados (`models/`) desde S3.

---

## 3. Ejecución del Frontend (Streamlit)

### 3.1. Instalar dependencias del frontend

```bash
pip install -r frontend/requirements.txt
```

Las dependencias del frontend son:

| Paquete     | Versión  | Propósito                              |
|-------------|----------|----------------------------------------|
| streamlit   | ≥ 1.30.0 | Framework de la interfaz               |
| requests    | ≥ 2.31.0 | Comunicación HTTP con la API           |
| pandas      | ≥ 2.0.0  | Manipulación de datos tabulares        |
| plotly      | ≥ 5.18.0 | Gráficas interactivas del dashboard    |

### 3.2. Configurar variables de entorno

Crear el archivo `frontend/.env`:

```env
# URL del backend FastAPI
API_URL=http://localhost:8000

# Token de autenticación para el Dashboard Analytics
DASHBOARD_TOKEN=tu_token_seguro

# Versión del modelo (badge en la UI)
MODEL_VERSION=v3.0-sigmoid-10classes
```

### 3.3. Ejecutar el frontend

```bash
streamlit run frontend/app.py
```

El tablero estará disponible en `http://localhost:8501`.

---

## 4. Ejecución del Backend (API FastAPI)

### 4.1. Verificar que los modelos existen

Asegúrate de que los archivos de modelo estén en la carpeta `models/`:

```
models/
├── logreg_model.joblib        # Modelo Logistic Regression
├── tfidf_vectorizer.joblib    # Vectorizador TF-IDF
└── xgboost_model.joblib       # Modelo XGBoost
```

Si no existen, descargarlos con `dvc pull` (ver [sección 5](#5-configuración-de-dvc-data-version-control)).

> **Nota:** Los modelos ya fueron pre-entrenados y están almacenados en el Data Lake de S3. **No es necesario re-entrenar.** Solo se requiere descargarlos.

### 4.2. Ejecutar la API localmente

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

### 4.3. Verificar que la API funciona

```bash
# Health check
curl http://localhost:8000/

# Predicción de prueba
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"transcription": "Patient presents with chest pain and elevated ST segment on ECG"}'
```

La API estará disponible en `http://localhost:8000` y la documentación interactiva en `http://localhost:8000/docs`.

---

## 5. Configuración de DVC (Data Version Control)

DVC gestiona el versionado de datasets y modelos pesados, sincronizándolos con el Data Lake en Amazon S3.

### 5.1. Inicializar DVC

```bash
dvc init
```

### 5.2. Configurar el remoto S3

```bash
dvc remote add -d s3_remote s3://mlops-medical-project-uniandes-2026/golden
```

> Esto define el bucket S3 (zona Golden) como almacenamiento remoto por defecto.

### 5.3. Rastrear archivos pesados

```bash
# Rastrear el dataset
dvc add data/raw/mtsamples.csv

# Rastrear modelos entrenados
dvc add models/xgboost_model.joblib
```

Esto genera archivos `.dvc` que se versionan con Git:

```bash
git add data/raw/mtsamples.csv.dvc models/xgboost_model.joblib.dvc .gitignore
git commit -m "Track dataset and models with DVC"
```

### 5.4. Subir binarios a S3

```bash
dvc push
```

### 5.5. Recuperar binarios en otro entorno

```bash
dvc pull
```

> **Importante:** Requiere credenciales AWS configuradas (`aws configure`) con permisos de lectura al bucket S3.

---

---

## 6. Despliegue con Docker (Local)

### 6.1. Backend (API FastAPI)

**Construir la imagen:**

```bash
docker build -t medical-api .
```

**Ejecutar el contenedor:**

```bash
docker run -d \
  --name medical-api \
  -p 8000:8000 \
  -e S3_BUCKET=mlops-medical-project-uniandes-2026 \
  -e MODELS_DIR=models \
  -e MODEL_NAME=logreg \
  -e AWS_ACCESS_KEY_ID=<TU_ACCESS_KEY> \
  -e AWS_SECRET_ACCESS_KEY=<TU_SECRET_KEY> \
  -e AWS_DEFAULT_REGION=us-east-1 \
  medical-api
```

> **Nota:** El contenedor descarga automáticamente los modelos desde la zona Golden de S3 al arrancar, gracias al script `entrypoint.sh`. **No es necesario re-entrenar modelos.** Los modelos pre-entrenados ya están almacenados en S3.

### 6.2. Frontend (Streamlit)

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
  -e DASHBOARD_TOKEN=tu_token_seguro \
  -e MODEL_VERSION=v3.0-sigmoid-10classes \
  medtranscript-frontend
```

> Si el backend también corre en Docker, usa `http://host.docker.internal:8000` como `API_URL` en Windows/Mac, o la IP del contenedor en Linux.

---

## 7. Despliegue en AWS ECS Fargate (Backend)

### Diagrama de arquitectura

```
┌───────────────┐     POST /predict      ┌──────────────────────┐
│   Streamlit   │ ──────────────────────► │   FastAPI (Fargate)  │
│  (Railway)    │ ◄────────────────────── │   Puerto 8000        │
└───────────────┘     JSON Response       └──────────┬───────────┘
                                                     │
                                          ┌──────────▼───────────┐
                                          │   Amazon S3 Bucket   │
                                          │  ┌─────────────────┐ │
                                          │  │  /golden (R/O)  │ │
                                          │  │  modelos + data  │ │
                                          │  ├─────────────────┤ │
                                          │  │ /analytics (R/W)│ │
                                          │  │ logs predicción  │ │
                                          │  └─────────────────┘ │
                                          └──────────────────────┘
```

### 7.1. Autenticación en AWS

```bash
aws configure
# Ingresar: AWS Access Key ID, Secret Access Key, región (us-east-1)
```

### 7.2. Crear repositorio en ECR

```bash
aws ecr create-repository --repository-name mlops-medical-api --region us-east-1
```

### 7.3. Construir, etiquetar y subir la imagen

```bash
# Autenticarse en ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# Construir la imagen
docker build -t medical-api .

# Etiquetar para ECR
docker tag medical-api:latest <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/mlops-medical-api:latest

# Subir a ECR
docker push <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/mlops-medical-api:latest
```

### 7.4. Crear la Task Definition

Usar el template `task-definition.example.json` del repositorio como referencia:

```json
{
    "family": "mlops-medical-api",
    "taskRoleArn": "arn:aws:iam::<AWS_ACCOUNT_ID>:role/LabRole",
    "executionRoleArn": "arn:aws:iam::<AWS_ACCOUNT_ID>:role/LabRole",
    "networkMode": "awsvpc",
    "cpu": "512",
    "memory": "1024",
    "requiresCompatibilities": ["FARGATE"],
    "containerDefinitions": [
        {
            "name": "mlops-medical-api-container",
            "image": "<AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/mlops-medical-api:latest",
            "portMappings": [
                {
                    "containerPort": 8000,
                    "hostPort": 8000,
                    "protocol": "tcp"
                }
            ],
            "essential": true,
            "environment": [
                { "name": "S3_BUCKET", "value": "mlops-medical-project-uniandes-2026" }
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/mlops-medical-api",
                    "awslogs-region": "<REGION>",
                    "awslogs-stream-prefix": "ecs"
                }
            }
        }
    ]
}
```

Registrar en AWS:

```bash
aws ecs register-task-definition --cli-input-json file://task-definition.example.json
```

### 7.5. Crear Cluster y Servicio ECS

```bash
# Crear cluster
aws ecs create-cluster --cluster-name mlops-medical-cluster

# Crear servicio
aws ecs create-service \
  --cluster mlops-medical-cluster \
  --service-name medical-api-service \
  --task-definition mlops-medical-api \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[<SUBNET_ID>],securityGroups=[<SG_ID>],assignPublicIp=ENABLED}"
```

### 7.6. Configurar Security Group

Asegurar que el Security Group del VPC tenga las siguientes reglas de entrada:

| Protocolo | Puerto | Origen      | Propósito            |
|-----------|--------|-------------|----------------------|
| TCP       | 8000   | 0.0.0.0/0   | API FastAPI          |

### 7.7. Verificar el despliegue

```bash
# Ver tareas en ejecución
aws ecs list-tasks --cluster mlops-medical-cluster

# Obtener la IP pública del task
aws ecs describe-tasks --cluster mlops-medical-cluster --tasks <TASK_ARN>
```

Probar la API con la IP pública:

```bash
curl http://<PUBLIC_IP>:8000/
```

---

## 8. Despliegue en Railway (Frontend)

Railway despliega automáticamente el frontend Streamlit desde GitHub usando el `Dockerfile` dedicado del frontend.

### 8.1. Conectar el repositorio

1. Acceder a [railway.app](https://railway.app) e iniciar sesión con GitHub.
2. Crear un nuevo proyecto → **Deploy from GitHub repo**.
3. Seleccionar el repositorio `mlops-uniandes-project`.

### 8.2. Configuración del build

Railway detectará automáticamente el archivo `railway.toml` que especifica:

```toml
[build]
dockerfilePath = "frontend/Dockerfile"
```

> El build context debe ser la raíz del repo (`/`), y Railway usará `frontend/Dockerfile` para construir la imagen.

### 8.3. Configurar variables de entorno en Railway

En el dashboard de Railway, ir a **Variables** y agregar:

| Variable           | Valor                                        | Descripción                       |
|--------------------|----------------------------------------------|-----------------------------------|
| `API_URL`          | `http://<IP_PUBLICA_FARGATE>:8000`           | URL del backend en AWS            |
| `DASHBOARD_TOKEN`  | `<token_seguro>`                             | Token para acceder al dashboard   |
| `MODEL_VERSION`    | `v3.0-sigmoid-10classes`                     | Versión del modelo (badge UI)     |
| `PORT`             | `8501`                                       | Puerto de Streamlit               |

### 8.4. Desplegar

Railway desplegará automáticamente al hacer push a la rama principal. Se puede forzar un redespliegue manual desde el dashboard.

---

## 9. Variables de Entorno Completas

### Backend (API FastAPI / Docker / AWS ECS)

| Variable                  | Valor por defecto                           | Descripción                                 |
|---------------------------|---------------------------------------------|---------------------------------------------|
| `S3_BUCKET`               | `mlops-medical-project-uniandes-2026`       | Nombre del bucket S3 del Data Lake          |
| `MODELS_DIR`              | `models`                                    | Carpeta local de modelos dentro del contenedor |
| `MODEL_NAME`              | `logreg`                                    | Nombre del modelo a usar (`logreg` o `xgboost`) |
| `AWS_ACCESS_KEY_ID`       | —                                           | Credencial AWS (solo local)                 |
| `AWS_SECRET_ACCESS_KEY`   | —                                           | Credencial AWS (solo local)                 |
| `AWS_DEFAULT_REGION`      | `us-east-1`                                 | Región AWS                                  |

### Frontend (Streamlit / Railway)

| Variable           | Valor por defecto                | Descripción                           |
|--------------------|----------------------------------|---------------------------------------|
| `API_URL`          | `http://localhost:8000`          | URL del backend FastAPI               |
| `DASHBOARD_TOKEN`  | `""`                             | Token para autenticar el dashboard    |
| `MODEL_VERSION`    | `v3.0-sigmoid-10classes`         | Versión del modelo mostrada en la UI  |
| `PORT`             | `8501`                           | Puerto del servidor Streamlit         |


