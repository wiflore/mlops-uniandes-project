# ============================================
# Dockerfile — API de Clasificación Médica
# ============================================
# Imagen ligera de Python 3.11
# Descarga modelos desde S3 al arrancar (no los embebe)
# ============================================

FROM python:3.11-slim

# Evitar prompts interactivos durante instalación
ENV DEBIAN_FRONTEND=noninteractive

# Instalar AWS CLI (necesario para descargar modelos de S3)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl unzip && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip aws/ && \
    apt-get purge -y curl unzip && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar e instalar dependencias primero (para aprovechar cache de Docker)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Descargar stopwords de NLTK durante el build (no en runtime)
RUN python -c "import nltk; nltk.download('stopwords', quiet=True)"

# Copiar código fuente de la API
COPY src/ ./src/

# Copiar script de arranque
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Crear directorio para modelos (se llenarán desde S3 al arrancar)
RUN mkdir -p models

# Variables de entorno por defecto
ENV S3_BUCKET=mlops-medical-project
ENV MODELS_DIR=models
ENV MODEL_NAME=logreg

# Puerto de la API
EXPOSE 8000

# Arranque: descargar modelos de S3 → iniciar API
ENTRYPOINT ["./entrypoint.sh"]
