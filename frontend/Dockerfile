# ============================================
# Dockerfile — Frontend Streamlit
# ============================================
# Build context: raíz del proyecto (mlops-uniandes-project/)
#   docker build -f frontend/Dockerfile -t medtranscript-frontend .
#
# Railway: configurar Root Directory = / (raíz del repo)
#          y Dockerfile Path = frontend/Dockerfile
# ============================================

FROM python:3.11-slim

WORKDIR /app

# Instalar curl para healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Instalar dependencias del frontend
COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar app y estilos (usar sintaxis JSON para manejar espacios/ñ)
COPY frontend/app.py .
COPY ["Diseños Tablero/", "./Diseños Tablero/"]

# Variables de entorno (valores por defecto seguros)
# ⚠️ SECRETOS: configurar en Railway Variables o docker run -e
#   - API_URL: URL del backend FastAPI
#   - DASHBOARD_TOKEN: token para /dashboard-data (NO poner aquí)
ENV API_URL=http://localhost:8000
ENV DASHBOARD_TOKEN=""
ENV MODEL_VERSION=v3.0-sigmoid-10classes

# Puerto de Streamlit
EXPOSE 8501

# Healthcheck (usando el puerto dinámico)
HEALTHCHECK CMD curl --fail http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Arrancar Streamlit en modo headless (usando shell para poder leer $PORT de Railway)
CMD streamlit run app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
