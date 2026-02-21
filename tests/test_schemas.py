"""
Tests para los schemas Pydantic (src.schemas).
"""
import pytest
from pydantic import ValidationError
from src.schemas import PredictionRequest, PredictionResponse, HealthResponse


class TestPredictionRequest:
    """Tests para el schema PredictionRequest."""

    def test_valid_request(self):
        """Verifica que un request válido se crea correctamente."""
        req = PredictionRequest(
            transcription="Patient presents with chest pain and shortness of breath"
        )
        assert req.transcription == "Patient presents with chest pain and shortness of breath"

    def test_min_length_validation(self):
        """Verifica que textos muy cortos son rechazados (min_length=10)."""
        with pytest.raises(ValidationError):
            PredictionRequest(transcription="short")

    def test_empty_transcription_rejected(self):
        """Verifica que una transcripción vacía es rechazada."""
        with pytest.raises(ValidationError):
            PredictionRequest(transcription="")

    def test_missing_transcription_rejected(self):
        """Verifica que un request sin transcripción es rechazado."""
        with pytest.raises(ValidationError):
            PredictionRequest()

    def test_long_transcription_accepted(self):
        """Verifica que textos largos son aceptados."""
        long_text = "Patient history includes " * 100
        req = PredictionRequest(transcription=long_text)
        assert len(req.transcription) > 100


class TestPredictionResponse:
    """Tests para el schema PredictionResponse."""

    def test_valid_response(self):
        """Verifica que un response válido se crea correctamente."""
        resp = PredictionResponse(
            specialty="Surgery",
            confidence=0.85,
            top_3=[
                {"specialty": "Surgery", "probability": 0.85},
                {"specialty": "Orthopedic", "probability": 0.10},
                {"specialty": "Cardiovascular", "probability": 0.05},
            ],
            model_name="logreg",
        )
        assert resp.specialty == "Surgery"
        assert resp.confidence == 0.85
        assert len(resp.top_3) == 3

    def test_confidence_bounds(self):
        """Verifica que la confianza debe estar entre 0 y 1."""
        with pytest.raises(ValidationError):
            PredictionResponse(
                specialty="Surgery", confidence=1.5,
                top_3=[], model_name="logreg"
            )

    def test_default_model_version(self):
        """Verifica que model_version tiene valor por defecto."""
        resp = PredictionResponse(
            specialty="Surgery", confidence=0.9,
            top_3=[], model_name="logreg"
        )
        assert resp.model_version == "0.1.0"


class TestHealthResponse:
    """Tests para el schema HealthResponse."""

    def test_default_values(self):
        """Verifica que los valores por defecto son correctos."""
        health = HealthResponse()
        assert health.status == "healthy"
        assert health.model_loaded is True
        assert health.version == "0.1.0"

    def test_custom_values(self):
        """Verifica que se pueden pasar valores personalizados."""
        health = HealthResponse(status="degraded", model_loaded=False)
        assert health.status == "degraded"
        assert health.model_loaded is False
