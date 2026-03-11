"""
Tests para el módulo de predicción (src.predict).
"""
import os
import pytest
from src.predict import MedicalSpecialtyPredictor


MODELS_DIR = "models"
MODELS_EXIST = os.path.exists(os.path.join(MODELS_DIR, "logreg_model.joblib"))


@pytest.mark.skipif(not MODELS_EXIST, reason="Modelos no serializados. Corra 'python -m src.train' primero.")
class TestMedicalSpecialtyPredictor:
    """Tests para la clase MedicalSpecialtyPredictor."""

    @pytest.fixture
    def predictor(self):
        """Predictor cargado con modelo logreg."""
        p = MedicalSpecialtyPredictor(models_dir=MODELS_DIR)
        p.load(model_name="logreg")
        return p

    def test_load_logreg(self, predictor):
        """Verifica que el modelo logreg se carga correctamente."""
        assert predictor.model is not None
        assert predictor.vectorizer is not None
        assert predictor.label_encoder is not None
        assert predictor.model_name == "logreg"

    def test_predict_returns_tuple(self, predictor):
        """Verifica que predict devuelve una tupla de 3 elementos."""
        text = "Patient presents with chest pain and shortness of breath. ECG shows ST elevation in leads II, III, and aVF."
        result = predictor.predict(text)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_predict_specialty_is_string(self, predictor):
        """Verifica que la especialidad predicha es un string."""
        text = "Surgery performed laparoscopic cholecystectomy under general anesthesia"
        specialty, confidence, top_3 = predictor.predict(text)
        assert isinstance(specialty, str)
        assert len(specialty) > 0

    def test_predict_confidence_range(self, predictor):
        """Verifica que la confianza está entre 0 y 1."""
        text = "Dermatological examination reveals erythematous rash"
        specialty, confidence, top_3 = predictor.predict(text)
        assert 0.0 <= confidence <= 1.0

    def test_predict_top3_structure(self, predictor):
        """Verifica que top_3 tiene la estructura correcta."""
        text = "Orthopedic evaluation shows fracture of the left femur"
        specialty, confidence, top_3 = predictor.predict(text)
        assert isinstance(top_3, list)
        assert len(top_3) == 3
        for item in top_3:
            assert "specialty" in item
            assert "probability" in item
            assert isinstance(item["specialty"], str)
            assert 0.0 <= item["probability"] <= 1.0

    def test_predict_top3_sorted_descending(self, predictor):
        """Verifica que top_3 está ordenado de mayor a menor probabilidad."""
        text = "Neurological assessment indicates mild cognitive impairment"
        specialty, confidence, top_3 = predictor.predict(text)
        probs = [item["probability"] for item in top_3]
        assert probs == sorted(probs, reverse=True)


class TestPredictorWithoutModels:
    """Tests para el predictor sin modelos cargados."""

    def test_predict_without_load_raises_error(self):
        """Verifica que predecir sin cargar modelo lanza RuntimeError."""
        p = MedicalSpecialtyPredictor(models_dir=MODELS_DIR)
        with pytest.raises(RuntimeError, match="Modelo no cargado"):
            p.predict("some text")

    def test_load_nonexistent_model_raises_error(self):
        """Verifica que cargar un modelo inexistente lanza FileNotFoundError."""
        p = MedicalSpecialtyPredictor(models_dir=MODELS_DIR)
        with pytest.raises(FileNotFoundError):
            p.load(model_name="nonexistent")
