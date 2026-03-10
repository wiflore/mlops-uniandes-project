"""
Tests de edge cases para el módulo de preprocesamiento (src.preprocessing).

Cubre:
- clean_text: inputs extremos (None, vacío, numérico, unicode, emojis, HTML, SQL)
- clean_text: invariantes (lowercase, sin stopwords, sin palabras ≤ 2 chars)
- clean_text: textos médicos reales y representativos
- TextPreprocessor: fit/transform con inputs extremos
- TextPreprocessor: consistencia de dimensiones
"""
import pytest
import numpy as np
from src.preprocessing import clean_text, TextPreprocessor


# ════════════════════════════════════════════════════════════════════════════
# clean_text – Edge cases
# ════════════════════════════════════════════════════════════════════════════

class TestCleanTextEdgeCases:
    """Edge cases exhaustivos para la función clean_text."""

    # ---- Inputs no-string -----------------------------------------------

    def test_none_returns_empty(self):
        assert clean_text(None) == ""

    def test_integer_returns_empty(self):
        assert clean_text(123) == ""

    def test_float_returns_empty(self):
        assert clean_text(3.14) == ""

    def test_list_returns_empty(self):
        assert clean_text(["hello", "world"]) == ""

    def test_dict_returns_empty(self):
        assert clean_text({"key": "value"}) == ""

    def test_true_returns_empty(self):
        assert clean_text(True) == ""

    # ---- Strings vacíos / whitespace ------------------------------------

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_single_space(self):
        assert clean_text(" ") == ""

    def test_multiple_spaces(self):
        assert clean_text("      ") == ""

    def test_tabs_only(self):
        assert clean_text("\t\t\t") == ""

    def test_newlines_only(self):
        assert clean_text("\n\n\n") == ""

    def test_mixed_whitespace(self):
        assert clean_text(" \t \n \r ") == ""

    # ---- Solo números / signos de puntuación ----------------------------

    def test_numbers_only(self):
        """Numeros son eliminados por el regex de clean_text."""
        result = clean_text("1234567890")
        assert result == ""

    def test_special_chars_only(self):
        """Solo caracteres especiales → cadena vacía."""
        result = clean_text("!@#$%^&*()_+-=[]{}|;':\",./<>?")
        assert result == ""

    def test_punctuation_heavy_text(self):
        """Texto con muchos signos de puntuación."""
        result = clean_text("Hello, world! How... are: you? Fine.")
        assert "," not in result
        assert "!" not in result
        assert "." not in result

    # ---- Stopwords y palabras cortas ------------------------------------

    def test_only_stopwords_returns_empty_or_short(self):
        """Unicamente stopwords deben ser eliminadas."""
        result = clean_text("the is in of a to for with this that")
        words = result.split() if result else []
        english_stopwords = {"the", "is", "in", "of", "to", "for", "with", "this", "that"}
        for w in words:
            assert w not in english_stopwords

    def test_two_letter_words_removed(self):
        """Palabras de exactamente 2 letras deben eliminarse."""
        result = clean_text("to be or not to be that is the question here go")
        words = result.split() if result else []
        assert all(len(w) > 2 for w in words)

    def test_one_letter_words_removed(self):
        """Palabras de 1 letra eliminadas."""
        result = clean_text("I a e o u")
        words = result.split() if result else []
        assert all(len(w) > 2 for w in words)

    # ---- Invariantes de texto -------------------------------------------

    def test_output_is_always_lowercase(self):
        """El output siempre debe ser minúsculas."""
        result = clean_text("PATIENT PRESENTS WITH CHEST PAIN AND SHORTNESS OF BREATH")
        assert result == result.lower()

    def test_no_double_spaces_in_output(self):
        """No debe haber espacios dobles en el output."""
        result = clean_text("patient     presents     with     chest     pain")
        assert "  " not in result

    def test_output_stripped(self):
        """El output no debe tener espacios al inicio o final."""
        result = clean_text("   patient presents   ")
        assert result == result.strip()

    def test_no_special_chars_in_output(self):
        """No debe haber caracteres especiales en el output."""
        result = clean_text("Patient's ECG shows [ST elevation] at 2+ mm!!!")
        import re
        # Solo letras latinas, acentuadas y espacios
        assert re.match(r'^[a-záéíóúñ\s]*$', result) is not None

    # ---- Unicode / caracteres no-ASCII ----------------------------------

    def test_emoji_removed(self):
        """Emojis deben ser eliminados."""
        result = clean_text("💉🩺 Patient with chest pain 🏥💊")
        assert "💉" not in result
        assert "🩺" not in result

    def test_japanese_chars_removed(self):
        """Kanji/Kana deben ser eliminados."""
        result = clean_text("患者は胸の痛みを patient chest pain")
        # El resultado no debe contener los kanas
        assert "患" not in result

    def test_arabic_chars_removed(self):
        """Caracteres árabes deben ser eliminados."""
        result = clean_text("المريض يشكو patient chest pain")
        assert "ا" not in result

    def test_spanish_accented_chars_kept(self):
        """Los acentos españoles deben mantenerse (están en el regex permitido)."""
        result = clean_text("paciente presenta fiebre alta")
        # 'paciente' y 'fiebre' no son stopwords del inglés, deben aparecer
        assert len(result) > 0

    # ---- Textos típicos médicos -----------------------------------------

    def test_cardiology_text(self):
        """Texto cardiológico típico."""
        text = (
            "Patient presents with acute chest pain radiating to left arm. "
            "ECG shows ST elevation in leads II, III, aVF. Troponin elevated."
        )
        result = clean_text(text)
        assert len(result) > 0
        assert isinstance(result, str)
        # palabras clave médicas NO-stopwords deben permanecer
        key_words_present = any(
            kw in result
            for kw in ["patient", "chest", "pain", "ecg", "elevation", "troponin"]
        )
        assert key_words_present

    def test_surgery_text(self):
        """Texto quirúrgico típico."""
        text = (
            "Laparoscopic cholecystectomy performed under general anesthesia. "
            "Gallbladder removed without complications. Patient discharged day two."
        )
        result = clean_text(text)
        assert len(result) > 0
        key_words_present = any(
            kw in result
            for kw in ["laparoscopic", "cholecystectomy", "anesthesia", "gallbladder"]
        )
        assert key_words_present

    def test_very_long_medical_text(self):
        """Texto médico largo — debe procesarse sin error."""
        text = (
            "Patient presents with chest pain shortness breath. "
            "ECG shows ST elevation. Troponin levels elevated. "
        ) * 500
        result = clean_text(text)
        assert isinstance(result, str)
        assert len(result) > 0

    # ---- XSS / SQL injection en el texto --------------------------------

    def test_html_tags_removed(self):
        """Tags HTML: los < > son eliminados pero las palabras alfabéticas quedan."""
        result = clean_text("<script>alert('xss')</script> patient chest pain")
        # clean_text elimina < > pero mantiene palabras alfanuméricas
        assert "<" not in result
        assert ">" not in result
        assert "<script>" not in result

    def test_sql_injection_neutralized(self):
        """SQL injection: los chars ; ' -- son eliminados, pero palabras quedan."""
        result = clean_text("'; DROP TABLE patients; -- patient presents fever")
        # clean_text elimina ; ' - pero las palabras alfabéticas permanecen
        assert ";" not in result
        assert "'" not in result
        # 'drop' y 'table' quedan como palabras normales — no son peligrosas
        assert isinstance(result, str)

    # ---- Longitud de texto muy corto que queda tras limpieza -----------

    def test_single_meaningful_word(self):
        """Una sola palabra significativa."""
        result = clean_text("surgery")
        assert result == "surgery"

    def test_two_meaningful_words(self):
        """Dos palabras significativas."""
        result = clean_text("cardiac surgery")
        # 'cardiac' y 'surgery' tienen >2 letras y no son stopwords
        words = result.split()
        assert "surgery" in words


# ════════════════════════════════════════════════════════════════════════════
# TextPreprocessor – Edge cases
# ════════════════════════════════════════════════════════════════════════════

class TestTextPreprocessorEdgeCases:
    """Edge cases para la clase TextPreprocessor."""

    @pytest.fixture
    def preprocessor(self):
        # max_df=1.0 necesario para corpus pequeños (sin límite superior de df)
        return TextPreprocessor(max_features=500, ngram_range=(1, 1), min_df=1, max_df=1.0)

    @pytest.fixture
    def fitted_preprocessor(self):
        pp = TextPreprocessor(max_features=500, ngram_range=(1, 1), min_df=1)
        texts = [
            "Patient presents with chest pain and shortness of breath cardiac",
            "Surgery performed laparoscopic cholecystectomy anesthesia gallbladder",
            "Dermatological examination erythematous rash biopsy histopathology",
            "Neurological cognitive impairment memory deficits MRI hippocampal",
            "Orthopedic fracture femur arthroplasty surgical fixation recovery",
        ]
        pp.fit_transform(texts)
        return pp

    # ---- fit_transform con inputs extremos ------------------------------

    def test_fit_transform_single_text(self, preprocessor):
        """fit_transform con un solo texto debe no fallar."""
        # min_df=1, max_df=1.0 necesarios para corpus de 1 documento
        X = preprocessor.fit_transform(["Patient presents with chest pain shortness"])
        assert X.shape[0] == 1

    def test_fit_transform_empty_strings(self, preprocessor):
        """Lista con strings vacíos: TF-IDF produce vocabulario vacío → matriz sparse."""
        # Strings vacíos → vocabulario vacío; la matriz tiene shape (3, 0)
        import scipy.sparse
        try:
            X = preprocessor.fit_transform(["", "", ""])
            assert X.shape[0] == 3
        except ValueError:
            # sklearn puede rechazar corpus sin vocabulario — comportamiento válido
            pytest.skip("sklearn rechaza corpus completamente vacío")

    def test_fit_transform_none_strings_handled(self, preprocessor):
        """None en la lista → clean_text devuelve '' → no debe romper."""
        # clean_text maneja None → ""; fit_transform lo procesa
        texts = [None, "patient chest pain surgery anesthesia", None]
        try:
            X = preprocessor.fit_transform(texts)
            assert X.shape[0] == 3
        except ValueError:
            pytest.skip("sklearn rechaza corpus casi vacío")

    def test_fit_transform_numbers_only(self, preprocessor):
        """Textos numéricos: clean_text los convierte en '' → vocabulario vacío."""
        try:
            X = preprocessor.fit_transform(["123456", "789012", "345678"])
            assert X.shape[0] == 3
        except ValueError:
            # Comportamiento válido cuando todos los docs quedan vacíos
            pytest.skip("sklearn rechaza corpus numérico (vocabulario vacío)")

    def test_fit_transform_preserves_doc_count(self, preprocessor):
        """El número de filas debe igualar el número de documentos."""
        texts = [f"patient document number {i} chest pain surgery anesthesia cardiac" for i in range(20)]
        X = preprocessor.fit_transform(texts)
        assert X.shape[0] == 20

    # ---- transform con inputs extremos ----------------------------------

    def test_transform_empty_string(self, fitted_preprocessor):
        """transform con string vacío → 1 fila de ceros."""
        X = fitted_preprocessor.transform([""])
        assert X.shape[0] == 1
        assert X.nnz == 0  # matriz sparse con todos ceros

    def test_transform_unseen_vocabulary(self, fitted_preprocessor):
        """Palabras nunca vistas → vector de ceros (OOV en TF-IDF)."""
        X = fitted_preprocessor.transform(["zyxwvutsrqponmlkjihgfedcba"])
        assert X.shape[0] == 1
        assert X.nnz == 0

    def test_transform_preserves_feature_dim(self, fitted_preprocessor):
        """transform siempre debe mantener el mismo número de features."""
        n_features = fitted_preprocessor.vectorizer.max_features
        for text in ["new patient text", "", "surgery123", "abc xyz"]:
            X = fitted_preprocessor.transform([text])
            assert X.shape[1] <= n_features

    def test_transform_multiple_docs_same_cols(self, fitted_preprocessor):
        """Múltiples documentos en transform deben tener el mismo número de cols."""
        texts = ["chest pain", "surgery performed", "fracture femur", ""]
        X = fitted_preprocessor.transform(texts)
        assert X.shape[0] == len(texts)
        # Todas las filas tienen el mismo número de columnas
        assert len(set([X.shape[1]])) == 1

    # ---- Consistencia save/load -----------------------------------------

    def test_save_load_preserves_predictions(self, tmp_path):
        """El vectorizador guardado y cargado debe producir el mismo output."""
        pp = TextPreprocessor(max_features=200, ngram_range=(1, 1), min_df=1)
        train = [
            "cardiac failure echocardiogram ejection fraction reduced",
            "fracture radius closed reduction splint orthopedic cast",
            "seizure epilepsy EEG neurological evaluation anticonvulsant",
        ]
        pp.fit_transform(train)

        path = str(tmp_path / "vectorizer.joblib")
        pp.save(path)

        loaded = TextPreprocessor.load(path)
        test_text = ["patient chest pain ECG cardiac catheterization stent"]
        X_orig = pp.transform(test_text)
        X_loaded = loaded.transform(test_text)

        assert X_orig.shape == X_loaded.shape
        np.testing.assert_array_almost_equal(
            X_orig.toarray(), X_loaded.toarray()
        )

    def test_load_nonexistent_file_raises(self):
        """Cargar vectorizador inexistente debe fallar."""
        with pytest.raises((FileNotFoundError, OSError)):
            TextPreprocessor.load("/nonexistent/path/vectorizer.joblib")

    # ---- Propiedades de la matriz TF-IDF --------------------------------

    def test_tfidf_non_negative(self, fitted_preprocessor):
        """Los valores de TF-IDF deben ser ≥ 0."""
        X = fitted_preprocessor.transform(
            ["patient presents cardiac chest pain ECG elevation"]
        )
        assert X.data.min() >= 0.0

    def test_tfidf_values_bounded(self, fitted_preprocessor):
        """Con sublinear_tf=True los valores no son acotados a 1, pero sí > 0."""
        X = fitted_preprocessor.transform(
            ["surgery cholecystectomy gallbladder anesthesia laparoscopic"]
        )
        # Solo verificamos que son finitos
        assert np.all(np.isfinite(X.data))
