"""
Tests para el módulo de preprocesamiento (src.preprocessing).
"""
import pytest
import numpy as np
from src.preprocessing import clean_text, TextPreprocessor


class TestCleanText:
    """Tests para la función clean_text."""

    def test_basic_cleaning(self):
        """Verifica que el texto se convierte a minúsculas y se limpian caracteres especiales."""
        result = clean_text("Patient HAS chest PAIN!!!")
        assert result == result.lower()
        assert "!" not in result

    def test_removes_stopwords(self):
        """Verifica que se eliminan las stopwords del inglés."""
        result = clean_text("the patient is in the hospital")
        assert "the" not in result.split()
        assert "is" not in result.split()
        assert "in" not in result.split()

    def test_removes_short_words(self):
        """Verifica que se eliminan palabras de 2 caracteres o menos."""
        result = clean_text("I am at my dr office to do an MRI scan")
        words = result.split()
        assert all(len(w) > 2 for w in words)

    def test_empty_string(self):
        """Verifica que un string vacío devuelve string vacío."""
        assert clean_text("") == ""

    def test_non_string_input(self):
        """Verifica que un input no-string devuelve string vacío."""
        assert clean_text(None) == ""
        assert clean_text(123) == ""

    def test_whitespace_normalization(self):
        """Verifica que los espacios múltiples se normalizan."""
        result = clean_text("patient    has     multiple     symptoms")
        assert "  " not in result

    def test_medical_text(self):
        """Verifica que un texto médico real se procesa correctamente."""
        text = "Patient presents with chest pain and shortness of breath. ECG shows ST elevation."
        result = clean_text(text)
        assert len(result) > 0
        assert isinstance(result, str)


class TestTextPreprocessor:
    """Tests para la clase TextPreprocessor."""

    @pytest.fixture
    def sample_texts(self):
        """Textos de ejemplo para testing."""
        return [
            "Patient presents with chest pain and shortness of breath",
            "Surgery performed laparoscopic cholecystectomy under general anesthesia",
            "Dermatological examination reveals erythematous rash on extremities",
            "Orthopedic evaluation shows fracture of the left femur",
            "Neurological assessment indicates mild cognitive impairment",
        ]

    @pytest.fixture
    def preprocessor(self):
        """Instancia de TextPreprocessor para testing con min_df=1 para muestras pequeñas."""
        return TextPreprocessor(max_features=100, ngram_range=(1, 1), min_df=1)

    def test_fit_transform_returns_sparse_matrix(self, preprocessor, sample_texts):
        """Verifica que fit_transform devuelve una matriz sparse."""
        X = preprocessor.fit_transform(sample_texts)
        assert X.shape[0] == len(sample_texts)
        assert X.shape[1] > 0
        assert X.shape[1] <= 100

    def test_transform_after_fit(self, preprocessor, sample_texts):
        """Verifica que transform funciona después de fit_transform."""
        preprocessor.fit_transform(sample_texts)
        X_new = preprocessor.transform(["New patient with headache and fever"])
        assert X_new.shape[0] == 1
        assert X_new.shape[1] == preprocessor.vectorizer.max_features or X_new.shape[1] > 0

    def test_save_and_load(self, preprocessor, sample_texts, tmp_path):
        """Verifica que el preprocessor se puede guardar y cargar."""
        preprocessor.fit_transform(sample_texts)

        save_path = str(tmp_path / "test_vectorizer.joblib")
        preprocessor.save(save_path)

        loaded = TextPreprocessor.load(save_path)
        X_original = preprocessor.transform(["test patient surgery"])
        X_loaded = loaded.transform(["test patient surgery"])

        assert X_original.shape == X_loaded.shape
        assert np.allclose(X_original.toarray(), X_loaded.toarray())
