#!/bin/bash
# entrypoint.sh — Script de arranque del contenedor
# 1. Descarga los modelos desde S3 zona golden
# 2. Descarga datos NLTK necesarios
# 3. Arranca el servidor FastAPI

set -e  # Si algún comando falla, detener todo

echo "=== Descargando modelos desde S3 zona golden ==="
aws s3 cp s3://${S3_BUCKET}/golden/models/ ./models/ --recursive
echo "=== Modelos descargados ==="
ls -la ./models/

echo "=== Descargando stopwords NLTK ==="
python -c "import nltk; nltk.download('stopwords', quiet=True)"

echo "=== Iniciando API FastAPI ==="
exec uvicorn src.api:app --host 0.0.0.0 --port 8000
