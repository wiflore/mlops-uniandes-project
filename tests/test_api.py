"""
Tests para la API FastAPI (src.api).
"""
import os
import pytest
from fastapi.testclient import TestClient

# conftest.py ya parchea S3 e importa la app; reutilizamos el mismo objeto.
from src.api import app


MODELS_DIR = os.environ.get("MODELS_DIR", "models")
MODELS_EXIST = os.path.exists(os.path.join(MODELS_DIR, "logreg_model.joblib"))

# 'client' fixture se hereda de conftest.py (scope="session") cuando no se
# redefine localmente; la siguiente fixture local mantiene la compatibilidad.


@pytest.mark.skipif(not MODELS_EXIST, reason="Modelos no serializados. Corra 'python -m src.train' primero.")
class TestHealthEndpoint:
    """Tests para el endpoint /health."""

    def test_health_returns_200(self, client):
        """Verifica que /health devuelve 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """Verifica la estructura del response de /health."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data

    def test_health_model_loaded(self, client):
        """Verifica que el modelo está cargado."""
        response = client.get("/health")
        data = response.json()
        assert data["model_loaded"] is True


@pytest.mark.skipif(not MODELS_EXIST, reason="Modelos no serializados. Corra 'python -m src.train' primero.")
class TestPredictEndpoint:
    """Tests para el endpoint /predict."""

    def test_predict_valid_text(self, client):
        """Verifica que /predict devuelve 200 con texto válido."""
        response = client.post(
            "/predict",
            json={
                "transcription": "Patient presents with chest pain and shortness of breath. "
                                 "ECG shows ST elevation in leads II, III, and aVF. "
                                 "Cardiac catheterization performed."
            },
        )
        assert response.status_code == 200

    def test_predict_response_structure(self, client):
        """Verifica la estructura del response de /predict."""
        response = client.post(
            "/predict",
            json={
                "transcription": "Surgery performed laparoscopic cholecystectomy under general anesthesia. "
                                 "Patient tolerated procedure well."
            },
        )
        data = response.json()
        assert "specialty" in data
        assert "confidence" in data
        assert "top_3" in data
        assert "model_name" in data

    def test_predict_confidence_range(self, client):
        """Verifica que la confianza del response está entre 0 y 1."""
        response = client.post(
            "/predict",
            json={
                "transcription": "Dermatological examination reveals erythematous rash on upper extremities. "
                                 "Biopsy obtained for further evaluation."
            },
        )
        data = response.json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_top3_has_three_items(self, client):
        """Verifica que top_3 tiene exactamente 3 elementos."""
        response = client.post(
            "/predict",
            json={
                "transcription": "Orthopedic evaluation shows fracture of the left femur. "
                                 "Surgical fixation recommended."
            },
        )
        data = response.json()
        assert len(data["top_3"]) == 3

    def test_predict_short_text_rejected(self, client):
        """Verifica que un texto muy corto es rechazado (422)."""
        response = client.post(
            "/predict",
            json={"transcription": "short"},
        )
        assert response.status_code == 422

    def test_predict_empty_body_rejected(self, client):
        """Verifica que un request sin body es rechazado (422)."""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_missing_field_rejected(self, client):
        """Verifica que un request sin el campo requerido es rechazado."""
        response = client.post("/predict", json={"text": "wrong field name"})
        assert response.status_code == 422
