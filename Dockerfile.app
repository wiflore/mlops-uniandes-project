# ============================================
# Dockerfile.app — Frontend Streamlit
# ============================================
FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del frontend
# Nota: Lo ideal es separar requirements-app.txt, por ahora usamos instalación directa
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir streamlit requests pandas plotly wordcloud

# Copiar los recursos del frontend
COPY frontend/ ./frontend/
COPY "Diseños Tablero/" "./Diseños Tablero/"

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
