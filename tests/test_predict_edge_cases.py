"""
Tests de edge cases para el módulo de predicción (src.predict).

Cubre:
- Inputs que después de clean_text quedan vacíos (solo stopwords, solo chars especiales)
- Textos extremadamente largos
- Predicción sin haber cargado el modelo
- Modelos inexistentes
- Consistencia entre modelos (logreg vs xgboost)
- Propiedades estadísticas de las probabilidades
"""
import os
import pytest
from src.predict import MedicalSpecialtyPredictor

MODELS_DIR = os.environ.get("MODELS_DIR", "models")
MODELS_EXIST = os.path.exists(os.path.join(MODELS_DIR, "logreg_model.joblib"))
XGB_EXISTS = os.path.exists(os.path.join(MODELS_DIR, "xgboost_model.joblib"))


@pytest.mark.skipif(not MODELS_EXIST, reason="Modelos no disponibles.")
class TestPredictorEdgeCases:
    """Edge cases del predictor con modelo cargado."""

    @pytest.fixture(scope="class")
    def predictor(self):
        p = MedicalSpecialtyPredictor(models_dir=MODELS_DIR)
        p.load(model_name="logreg")
        return p

    # ---- Texto limpiado queda vacío / muy corto -------------------------

    def test_predict_special_chars_only(self, predictor):
        """Texto solo con caracteres especiales — clean_text retorna ''."""
        specialty, confidence, top_3 = predictor.predict("!@#$%^&*()_+-=[]{}|")
        assert isinstance(specialty, str)
        assert 0.0 <= confidence <= 1.0

    def test_predict_numbers_only(self, predictor):
        """Texto solo con números — clean_text elimina todo."""
        result = predictor.predict("1234567890 0987654321 111222333")
        assert len(result) == 3
        specialty, confidence, top_3 = result
        assert isinstance(specialty, str)

    def test_predict_stopwords_only(self, predictor):
        """Texto compuesto únicamente de stopwords del inglés."""
        result = predictor.predict("the is in of a to for with this that")
        specialty, confidence, top_3 = result
        assert isinstance(specialty, str)
        assert 0.0 <= confidence <= 1.0

    def test_predict_empty_string(self, predictor):
        """String vacío — clean_text retorna ''; el vectorizer transforma cadena vacía."""
        result = predictor.predict("")
        specialty, confidence, top_3 = result
        assert isinstance(specialty, str)

    def test_predict_single_whitespace(self, predictor):
        """Solo un espacio."""
        result = predictor.predict(" ")
        assert len(result) == 3

    def test_predict_newlines_only(self, predictor):
        """Solo newlines."""
        result = predictor.predict("\n\n\n\n")
        assert len(result) == 3

    def test_predict_tab_chars_only(self, predictor):
        """Solo tabulaciones."""
        result = predictor.predict("\t\t\t\t\t")
        assert len(result) == 3

    # ---- Textos muy largos ----------------------------------------------

    def test_predict_10k_words(self, predictor):
        """Texto de 10 000 palabras."""
        text = "patient presents chest pain shortness breath ecg elevation " * 1250
        specialty, confidence, top_3 = predictor.predict(text)
        assert isinstance(specialty, str)
        assert 0.0 <= confidence <= 1.0

    def test_predict_single_word_repeated_50k(self, predictor):
        """Una sola palabra repetida 50 000 veces."""
        text = "surgery " * 50_000
        specialty, confidence, top_3 = predictor.predict(text)
        assert isinstance(specialty, str)

    # ---- Caracteres especiales y Unicode --------------------------------

    def test_predict_kana_japanese(self, predictor):
        """Texto en japonés — clean_text elimina katakana/hiragana."""
        result = predictor.predict("患者は胸の痛みを訴えています。ECG検査を実施しました。")
        specialty, confidence, top_3 = result
        assert isinstance(specialty, str)

    def test_predict_arabic_text(self, predictor):
        """Texto en árabe."""
        result = predictor.predict(
            "المريض يشكو من ألم في الصدر وضيق في التنفس وارتفاع ضغط الدم"
        )
        specialty, confidence, top_3 = result
        assert isinstance(specialty, str)

    def test_predict_emoji_text(self, predictor):
        """Texto de emojis médicos."""
        result = predictor.predict("💉💉💉🩺🩺🩺🏥🏥🏥💊💊💊🩻🩻🩻🫀🫀")
        specialty, confidence, top_3 = result
        assert len(result) == 3

    def test_predict_mixed_english_spanish(self, predictor):
        """Mezcla inglés/español típica de transcripciones reales."""
        result = predictor.predict(
            "El paciente presents with chest pain y fiebre alta. "
            "Se realizó ECG y análisis de sangre. Diagnóstico: angina."
        )
        specialty, confidence, top_3 = result
        assert isinstance(specialty, str)
        assert 0.0 <= confidence <= 1.0

    # ---- Propiedades estadísticas del output ----------------------------

    def test_top3_probabilities_non_negative(self, predictor):
        """Todas las probabilidades de top_3 deben ser ≥ 0."""
        _, _, top_3 = predictor.predict(
            "Patient presents with chest pain and shortness of breath. "
            "ECG shows ST elevation myocardial infarction diagnosis made."
        )
        for item in top_3:
            assert item["probability"] >= 0.0

    def test_top3_probabilities_at_most_1(self, predictor):
        """Todas las probabilidades de top_3 deben ser ≤ 1."""
        _, _, top_3 = predictor.predict(
            "Laparoscopic cholecystectomy performed under general anesthesia. "
            "Patient tolerated procedure well no complications noted."
        )
        for item in top_3:
            assert item["probability"] <= 1.0

    def test_top3_length_always_3_or_fewer(self, predictor):
        """top_3 nunca debe tener más de 3 elementos."""
        _, _, top_3 = predictor.predict(
            "Neurological examination reveals mild cognitive impairment "
            "memory deficits MRI hippocampal atrophy moderate severity"
        )
        assert len(top_3) <= 3

    def test_max_prob_matches_confidence(self, predictor):
        """El máximo de probabilidades de top_3 debe ≈ confidence."""
        text = (
            "Orthopedic evaluation closed fracture left femur arthroplasty "
            "surgical fixation recommended imaging completed diagnosis"
        )
        specialty, confidence, top_3 = predictor.predict(text)
        max_prob = max(item["probability"] for item in top_3)
        assert abs(max_prob - confidence) < 1e-6, (
            f"confidence={confidence} != max_prob={max_prob}"
        )

    def test_first_top3_specialty_matches_prediction(self, predictor):
        """El primer elemento de top_3 debe ser la misma predicción."""
        text = (
            "Dermatological examination erythematous scaly plaques "
            "bilateral upper extremities biopsy histopathology pending"
        )
        specialty, _, top_3 = predictor.predict(text)
        assert top_3[0]["specialty"] == specialty

    # ---- Determinismo ---------------------------------------------------

    def test_predict_is_deterministic(self, predictor):
        """El mismo texto debe producir siempre el mismo resultado."""
        text = (
            "Patient presents with acute myocardial infarction chest pain "
            "diaphoresis dyspnea emergency PCI troponin elevated ST elevation"
        )
        results = [predictor.predict(text) for _ in range(5)]
        specialties = [r[0] for r in results]
        confidences = [r[1] for r in results]

        assert len(set(specialties)) == 1, "Especialidad no determinista"
        assert len(set(confidences)) == 1, "Confianza no determinista"


# ════════════════════════════════════════════════════════════════════════════
# Tests sin modelo cargado
# ════════════════════════════════════════════════════════════════════════════

class TestPredictorWithoutModel:
    """Tests que NO necesitan modelos en disco."""

    def test_predict_without_load_raises_runtime_error(self):
        """Predecir sin load() debe lanzar RuntimeError."""
        p = MedicalSpecialtyPredictor(models_dir=MODELS_DIR)
        with pytest.raises(RuntimeError, match="Modelo no cargado"):
            p.predict("some reasonable text here")

    def test_predict_empty_without_load(self):
        """Predecir texto vacío sin load() también lanza RuntimeError."""
        p = MedicalSpecialtyPredictor(models_dir=MODELS_DIR)
        with pytest.raises(RuntimeError):
            p.predict("")

    def test_load_nonexistent_model_raises(self):
        """Cargar modelo inexistente debe lanzar FileNotFoundError."""
        p = MedicalSpecialtyPredictor(models_dir=MODELS_DIR)
        with pytest.raises(FileNotFoundError):
            p.load(model_name="does_not_exist_model_xyz")

    def test_load_nonexistent_directory(self):
        """Cargar desde directorio inexistente debe fallar."""
        p = MedicalSpecialtyPredictor(models_dir="/nonexistent/path/models")
        with pytest.raises((FileNotFoundError, OSError)):
            p.load(model_name="logreg")

    def test_model_attributes_none_before_load(self):
        """Atributos del predictor deben ser None antes de load()."""
        p = MedicalSpecialtyPredictor(models_dir=MODELS_DIR)
        assert p.model is None
        assert p.vectorizer is None
        assert p.label_encoder is None
        assert p.model_name == ""


# ════════════════════════════════════════════════════════════════════════════
# Tests comparativos entre modelos
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not (MODELS_EXIST and XGB_EXISTS),
    reason="Ambos modelos (logreg + xgboost) necesarios."
)
class TestModelComparison:
    """Compara comportamiento entre logreg y xgboost."""

    REFERENCE_TEXTS = [
        (
            "Cardiac catheterization performed left anterior descending artery "
            "stent placement successful no complications post procedure"
        ),
        (
            "Total knee replacement arthroplasty orthopedic surgery rehabilitation "
            "physical therapy initiated recovery expected six weeks post op"
        ),
    ]

    @pytest.fixture(scope="class")
    def logreg(self):
        p = MedicalSpecialtyPredictor(models_dir=MODELS_DIR)
        p.load("logreg")
        return p

    @pytest.fixture(scope="class")
    def xgboost(self):
        p = MedicalSpecialtyPredictor(models_dir=MODELS_DIR)
        p.load("xgboost")
        return p

    @pytest.mark.xfail(
        reason="BUG CONOCIDO: xgboost_model.joblib predice índice de label (ej. 6) "
               "fuera del rango del LabelEncoder serializado. Desacuerdo de versiones "
               "entre artefactos. Requiere re-entrenamiento sincronizado.",
        strict=True,
    )
    @pytest.mark.parametrize("text", REFERENCE_TEXTS)
    def test_both_models_return_valid_output(self, logreg, xgboost, text):
        """Ambos modelos deben retornar outputs válidos para el mismo texto."""
        for predictor in (logreg, xgboost):
            specialty, confidence, top_3 = predictor.predict(text)
            assert isinstance(specialty, str) and len(specialty) > 0
            assert 0.0 <= confidence <= 1.0
            assert len(top_3) <= 3

    @pytest.mark.xfail(
        reason="BUG CONOCIDO: misma causa que test_both_models_return_valid_output.",
        strict=True,
    )
    @pytest.mark.parametrize("text", REFERENCE_TEXTS)
    def test_both_models_output_structure(self, logreg, xgboost, text):
        """Ambos modelos deben devolver la misma estructura de output."""
        for predictor in (logreg, xgboost):
            _, _, top_3 = predictor.predict(text)
            for item in top_3:
                assert "specialty" in item
                assert "probability" in item
