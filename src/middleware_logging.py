"""
Middleware para capturar API calls y almacenarlos en S3 zona analitica.

Cada request a la API genera un archivo JSON en:
  s3://{S3_BUCKET}/analytics/api-calls/{fecha}/{uuid}.json

El schema ApiCallLog (Pydantic) valida cada entrada antes de enviarla.
"""
import os
import time
import uuid
import logging
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError
from starlette.middleware.base import BaseHTTPMiddleware

from .schemas import ApiCallLog

logger = logging.getLogger(__name__)

# Configuracion S3
S3_BUCKET = os.environ.get("S3_BUCKET", "mlops-medical-project-uniandes-2026")
S3_PREFIX = os.environ.get("S3_LOG_PREFIX", "analytics/api-calls")

# Cliente S3 (se inicializa una sola vez al importar el modulo)
s3_client = boto3.client("s3")


class S3LoggingMiddleware(BaseHTTPMiddleware):
    """
    Intercepta cada request/response y guarda un log en S3.
    
    Flujo:
    1. Request llega → captura body y timestamp
    2. Request pasa al endpoint (ej: /predict)
    3. Respuesta generada → captura status_code y tiempo total
    4. Crea un ApiCallLog (Pydantic valida la estructura)
    5. Envia el JSON a S3 zona analitica (async-safe, no bloquea la respuesta)
    """

    async def dispatch(self, request, call_next):
        # Ignorar rutas de documentacion de FastAPI
        if request.url.path in ["/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)

        # 1. Capturar datos del request ANTES de procesarlo
        start_time = time.time()
        body = await request.body()

        # 2. Dejar que el endpoint procese normalmente
        response = await call_next(request)

        # 3. Calcular tiempo de respuesta
        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        # 4. Crear log validado con Pydantic
        try:
            log_entry = ApiCallLog(
                request_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                method=request.method,
                path=str(request.url.path),
                request_body=body.decode("utf-8", errors="ignore"),
                status_code=response.status_code,
                response_time_ms=elapsed_ms,
            )

            # 5. Enviar a S3 (zona analitica)
            today = datetime.now(timezone.utc).strftime("%Y/%m/%d")
            key = f"{S3_PREFIX}/{today}/{log_entry.request_id}.json"

            s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=key,
                Body=log_entry.model_dump_json(),
                ContentType="application/json",
            )
        except ClientError as e:
            # Si S3 falla, logueamos el error pero NO rompemos la respuesta al usuario
            logger.error(f"Error enviando log a S3: {e}")
        except Exception as e:
            logger.error(f"Error inesperado en middleware de logging: {e}")

        return response
