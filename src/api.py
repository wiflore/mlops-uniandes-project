"""
API FastAPI para clasificacion de especialidades medicas.
Ejecutar: uvicorn src.api:app --reload
"""
import os
from fastapi import FastAPI, HTTPException
from .schemas import PredictionRequest, PredictionResponse, HealthResponse
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
