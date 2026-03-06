"""
API FastAPI para clasificacion de especialidades medicas.
Ejecutar: uvicorn src.api:app --reload
"""
import os
import time
import uuid
from datetime import datetime, timezone
import boto3
import json
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .schemas import PredictionRequest, PredictionResponse, HealthResponse, ApiCallLog
from .predict import MedicalSpecialtyPredictor
from .middleware_logging import S3LoggingMiddleware

app = FastAPI(
    title="Medical Specialty Classifier API",
    description="Clasifica transcripciones medicas por especialidad",
    version="0.1.0"
)

# Middleware: captura cada API call y lo envia a S3 zona analitica
app.add_middleware(S3LoggingMiddleware)

MODELS_DIR = os.environ.get("MODELS_DIR", "models")
MODEL_NAME = os.environ.get("MODEL_NAME", "logreg")
predictor = MedicalSpecialtyPredictor(models_dir=MODELS_DIR)


@app.on_event("startup")
async def startup():
    predictor.load(model_name=MODEL_NAME)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(model_loaded=predictor.model is not None)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        specialty, confidence, top_3 = predictor.predict(request.transcription)
        return PredictionResponse(
            specialty=specialty, confidence=confidence,
            top_3=top_3, model_name=predictor.model_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


security = HTTPBearer()
# La clave secreta se inyecta desde AWS ECS Task Definition
# No hay fallback hardcodeado para no exponerlo en el repo publico
API_SECRET_KEY = os.environ.get("API_SECRET_KEY")

@app.get("/dashboard-data")
async def get_dashboard_data(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Endpoint para exportar los logs de S3 al Dashboard."""
    if not API_SECRET_KEY or credentials.credentials != API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    s3_bucket = os.environ.get("S3_BUCKET", "mlops-medical-project-uniandes-2026")
    s3_client = boto3.client("s3")
    
    try:
        # Listar objetos en el prefijo analytics/api-calls/
        response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix="analytics/api-calls/")
        if "Contents" not in response:
            return {"data": []}
            
        all_logs = []
        for obj in response["Contents"]:
            if obj["Key"].endswith(".json"):
                # Descargar y parsear cada JSON
                file_obj = s3_client.get_object(Bucket=s3_bucket, Key=obj["Key"])
                file_content = file_obj["Body"].read().decode("utf-8")
                all_logs.append(json.loads(file_content))
                
        return {"data": all_logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading from S3: {str(e)}")
