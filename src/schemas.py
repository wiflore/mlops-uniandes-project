"""
Pydantic schemas para la API de clasificacion de especialidades medicas.
"""
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional


class PredictionRequest(BaseModel):
    transcription: str = Field(
        ...,
        min_length=10,
        description="Texto de la transcripcion medica a clasificar",
        json_schema_extra={
            "example": "Patient presents with chest pain and shortness of breath. "
                       "ECG shows ST elevation in leads II, III, and aVF."
        }
    )


class PredictionResponse(BaseModel):
    specialty: str = Field(..., description="Especialidad medica predicha")
    confidence: float = Field(..., ge=0, le=1, description="Confianza de la prediccion")
    top_3: List[dict] = Field(default_factory=list, description="Top 3 especialidades")
    model_name: str = Field(..., description="Nombre del modelo utilizado")
    model_version: str = Field(default="0.1.0", description="Version del modelo")


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool = True
    version: str = "0.1.0"


class ApiCallLog(BaseModel):
    """Schema para logs de API calls almacenados en S3 zona analitica."""
    request_id: str = Field(..., description="UUID unico del request")
    timestamp: datetime = Field(..., description="Momento de la llamada (UTC)")
    method: str = Field(..., description="Metodo HTTP (GET, POST)")
    path: str = Field(..., description="Ruta del endpoint (/predict, /health)")
    request_body: str = Field(default="", description="Cuerpo del request")
    response_body: str = Field(default="", description="Cuerpo de la respuesta (prediccion)")
    status_code: int = Field(..., description="Codigo de respuesta HTTP")
    response_time_ms: float = Field(..., description="Tiempo de respuesta en ms")

