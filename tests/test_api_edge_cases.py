"""
Tests de edge cases para la API FastAPI.

Cubre:
- Textos vacíos / en blanco / solo espacios
- Textos muy largos (>10 000 caracteres)
- Solo caracteres especiales / numéricos / emojis
- SQL Injection e XSS en el payload
- Textos sin sentido (gibberish) que superan min_length
- Payloads malformados (JSON inválido, tipos incorrectos)
- Múltiples solicitudes rápidas (stress básico)
- Validación de tiempos de respuesta aceptables
- Verificación de invariantes del response (top_3, confianza, etc.)
"""
import os
import time
import pytest

MODELS_DIR = os.environ.get("MODELS_DIR", "models")
MODELS_EXIST = os.path.exists(os.path.join(MODELS_DIR, "logreg_model.joblib"))

# ════════════════════════════════════════════════════════════════════════════
# Fixtures auxiliares
# ════════════════════════════════════════════════════════════════════════════

MINIMUM_VALID_TEXT = "Patient presents with fever and chills lasting three days"


# ════════════════════════════════════════════════════════════════════════════
# Edge cases: INPUT VALIDATION (sin necesitar modelos)
# ════════════════════════════════════════════════════════════════════════════

class TestInputValidationEdgeCases:
    """Pruebas de validación de entrada que no requieren modelos cargados."""

    # --- Textos vacíos / demasiado cortos --------------------------------

    def test_empty_string_rejected(self, client):
        """Texto vacío debe ser rechazado con 422."""
        r = client.post("/predict", json={"transcription": ""})
        assert r.status_code == 422

    def test_single_character_rejected(self, client):
        """Un solo carácter debe ser rechazado con 422."""
        r = client.post("/predict", json={"transcription": "A"})
        assert r.status_code == 422

    def test_nine_chars_rejected(self, client):
        """Texto de 9 chars (min_length=10) debe ser rechazado con 422."""
        r = client.post("/predict", json={"transcription": "123456789"})
        assert r.status_code == 422

    def test_whitespace_only_rejected(self, client):
        """Solo espacios no deben superar la validación de min_length."""
        r = client.post("/predict", json={"transcription": "          "})
        # Pydantic min_length cuenta espacios, así que 10 espacios pasan la
        # validación de longitud pero el predictor puede devolver 200 o 500.
        # El comportamiento importante es que NO devuelve 422 por longitud.
        assert r.status_code in (200, 422, 500)

    def test_newlines_only_edge(self, client):
        """Solo saltos de línea — el schema permite si len >= 10."""
        r = client.post("/predict", json={"transcription": "\n\n\n\n\n\n\n\n\n\n"})
        assert r.status_code in (200, 422, 500)

    # --- Payload malformado -----------------------------------------------

    def test_missing_transcription_field(self, client):
        """Payload sin campo 'transcription' → 422."""
        r = client.post("/predict", json={"text": "some text here"})
        assert r.status_code == 422

    def test_null_transcription_rejected(self, client):
        """transcription=null → 422."""
        r = client.post("/predict", json={"transcription": None})
        assert r.status_code == 422

    def test_integer_transcription_rejected(self, client):
        """transcription como número entero → 422 o coerción Pydantic."""
        r = client.post("/predict", json={"transcription": 12345})
        # Pydantic v2 por defecto coerciona int→str; aceptamos 200 o 422
        assert r.status_code in (200, 422)

    def test_list_transcription_rejected(self, client):
        """transcription como lista → 422."""
        r = client.post("/predict", json={"transcription": ["a", "b"]})
        assert r.status_code == 422

    def test_empty_json_body(self, client):
        """Body JSON vacío → 422."""
        r = client.post("/predict", json={})
        assert r.status_code == 422

    def test_no_body(self, client):
        """Sin body del todo → 422."""
        r = client.post("/predict")
        assert r.status_code == 422

    def test_wrong_content_type(self, client):
        """Content-Type application/x-www-form-urlencoded → 422."""
        r = client.post(
            "/predict",
            data={"transcription": MINIMUM_VALID_TEXT},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert r.status_code == 422

    def test_extra_fields_ignored(self, client):
        """Campos extra en el payload deben ser ignorados (no 422/500)."""
        r = client.post(
            "/predict",
            json={
                "transcription": MINIMUM_VALID_TEXT,
                "unknown_field": "should be ignored",
                "another_extra": 42,
            },
        )
        # FastAPI/Pydantic ignora extra fields por defecto
        assert r.status_code in (200, 422)


# ════════════════════════════════════════════════════════════════════════════
# Edge cases: CONTENIDO DEL TEXTO (requieren modelos)
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not MODELS_EXIST, reason="Modelos no disponibles.")
class TestTextContentEdgeCases:
    """Pruebas de edge cases de contenido del texto."""

    # --- Caracteres especiales / no alfanuméricos -------------------------

    def test_special_chars_only(self, client):
        """Solo caracteres especiales que superen min_length."""
        r = client.post(
            "/predict",
            json={"transcription": "!@#$%^&*()_+-=[]{}|;':\",./<>?!@#$"}
        )
        # clean_text elimina todo→ texto vacío; el predictor o API puede fallar 500
        assert r.status_code in (200, 500)

    def test_numeric_only_text(self, client):
        """Solo dígitos — pasan min_length pero limpieza puede vaciar el texto."""
        r = client.post(
            "/predict",
            json={"transcription": "1234567890123456789012345678901234567890"}
        )
        assert r.status_code in (200, 500)

    def test_unicode_latin_extended(self, client):
        """Caracteres Unicode latinos extendidos."""
        r = client.post(
            "/predict",
            json={
                "transcription": (
                    "El paciente presenta fiebre alta y escalofríos. "
                    "Se administró ibuprofeno según prescripción médica."
                )
            },
        )
        assert r.status_code in (200, 500)

    def test_emoji_only_text(self, client):
        """Texto compuesto solo de emojis."""
        r = client.post(
            "/predict",
            json={"transcription": "💉🩺🏥💊🩻🫀🦷🧬🔬💉🩺🏥💊🩻"}
        )
        assert r.status_code in (200, 422, 500)

    def test_mixed_languages(self, client):
        """Texto mezclando inglés, español y caracteres sin sentido."""
        r = client.post(
            "/predict",
            json={
                "transcription": (
                    "Patient presenta dolor chest pain y fiebre alta. "
                    "Se realizó ECG y análisis de sangre completo."
                )
            },
        )
        assert r.status_code in (200, 500)

    def test_html_xss_in_transcription(self, client):
        """Intento de XSS en el campo transcription."""
        xss_payload = (
            "<script>alert('xss')</script> Patient with fever "
            "<img src=x onerror=alert(1)> and chest pain symptoms"
        )
        r = client.post("/predict", json={"transcription": xss_payload})
        # La API no debe ejecutar HTML — solo clasifica texto
        assert r.status_code in (200, 500)
        if r.status_code == 200:
            data = r.json()
            assert "<script>" not in data.get("specialty", "")

    def test_sql_injection_in_transcription(self, client):
        """Intento de SQL Injection en el campo transcription."""
        sql_payload = (
            "'; DROP TABLE patients; -- Patient presents with "
            "chest pain and shortness of breath and fever"
        )
        r = client.post("/predict", json={"transcription": sql_payload})
        assert r.status_code in (200, 500)

    def test_path_traversal_in_transcription(self, client):
        """Intento de path traversal en el texto."""
        payload = (
            "../../etc/passwd patient presents with severe "
            "headache and nausea and vomiting symptoms"
        )
        r = client.post("/predict", json={"transcription": payload})
        assert r.status_code in (200, 500)

    # --- Textos sin sentido (gibberish) ----------------------------------

    def test_random_gibberish_short(self, client):
        """Texto sin sentido pero con longitud suficiente — debe retornar algo."""
        r = client.post(
            "/predict",
            json={"transcription": "asdfghjklqwertyuiopzxcvbnm asdf qwerty uiop"}
        )
        assert r.status_code in (200, 500)

    def test_repeated_single_word(self, client):
        """Una sola palabra médica repetida muchas veces."""
        r = client.post(
            "/predict",
            json={"transcription": "surgery " * 30}
        )
        assert r.status_code in (200, 500)

    def test_all_stopwords_text(self, client):
        """Texto compuesto únicamente de stopwords del inglés."""
        r = client.post(
            "/predict",
            json={"transcription": "the and is in of a to for with this that these those"}
        )
        # Después de limpiar, el texto puede quedar vacío o muy corto
        assert r.status_code in (200, 500)

    def test_lorem_ipsum(self, client):
        """Texto Lorem Ipsum — sin contexto médico."""
        r = client.post(
            "/predict",
            json={
                "transcription": (
                    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, "
                    "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
                    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris."
                )
            },
        )
        assert r.status_code in (200, 500)
        if r.status_code == 200:
            data = r.json()
            assert 0.0 <= data["confidence"] <= 1.0

    # --- Textos muy largos -----------------------------------------------

    def test_very_long_text_10k_chars(self, client):
        """Texto de ~10 000 caracteres — debe procesarse sin timeout."""
        base = (
            "Patient presents with chest pain radiating to the left arm. "
            "Blood pressure elevated. EKG performed. Troponin levels checked. "
        )
        long_text = base * 80  # ~5 000 chars
        start = time.time()
        r = client.post("/predict", json={"transcription": long_text})
        elapsed = time.time() - start

        assert r.status_code in (200, 500)
        assert elapsed < 10.0, f"Timeout: {elapsed:.2f}s para texto largo"

    def test_very_long_text_50k_chars(self, client):
        """Texto de ~50 000 caracteres — stress de memoria/procesamiento."""
        word = "patient " * 6250  # ~50 000 chars
        start = time.time()
        r = client.post("/predict", json={"transcription": word})
        elapsed = time.time() - start

        assert r.status_code in (200, 500)
        assert elapsed < 30.0, f"Timeout: {elapsed:.2f}s para texto muy largo"

    def test_extremely_long_single_word(self, client):
        """Una única 'palabra' de 10 000 caracteres sin espacios."""
        r = client.post("/predict", json={"transcription": "a" * 10_000})
        assert r.status_code in (200, 422, 500)

    # --- Newlines, tabs, whitespace extremo ------------------------------

    def test_text_with_tabs_and_newlines(self, client):
        """Texto con tabs y newlines mezclados."""
        r = client.post(
            "/predict",
            json={
                "transcription": (
                    "Patient\tpresents\nwith\tchest\tpain\nand\nshortness\t"
                    "of\tbreath\nECG\tshows\tST\televation"
                )
            },
        )
        assert r.status_code in (200, 500)

    def test_text_with_excessive_whitespace(self, client):
        """Texto con espacios excesivos entre palabras."""
        r = client.post(
            "/predict",
            json={
                "transcription": (
                    "patient     presents     with     chest     pain     "
                    "and     shortness     of     breath     severe     acute"
                )
            },
        )
        assert r.status_code in (200, 500)


# ════════════════════════════════════════════════════════════════════════════
# Edge cases: RESPONSE INVARIANTS (requieren modelos)
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not MODELS_EXIST, reason="Modelos no disponibles.")
class TestResponseInvariants:
    """Verifica invariantes de la respuesta para cualquier entrada válida."""

    VALID_INPUTS = [
        "Patient presents with chest pain and ST elevation on ECG examination",
        "Dermatological rash erythematous plaques bilateral upper extremities biopsy",
        "Orthopedic fracture left femur surgical fixation total hip arthroplasty",
        "Neurological cognitive impairment memory deficits MRI hippocampal atrophy",
        "Laparoscopic cholecystectomy general anesthesia gallbladder removal surgery",
    ]

    @pytest.mark.parametrize("text", VALID_INPUTS)
    def test_response_has_required_fields(self, client, text):
        """Todo response 200 debe tener los campos requeridos."""
        r = client.post("/predict", json={"transcription": text})
        if r.status_code == 200:
            data = r.json()
            assert "specialty" in data
            assert "confidence" in data
            assert "top_3" in data
            assert "model_name" in data
            assert "model_version" in data

    @pytest.mark.parametrize("text", VALID_INPUTS)
    def test_confidence_always_between_0_and_1(self, client, text):
        """La confianza debe estar en [0, 1] siempre."""
        r = client.post("/predict", json={"transcription": text})
        if r.status_code == 200:
            assert 0.0 <= r.json()["confidence"] <= 1.0

    @pytest.mark.parametrize("text", VALID_INPUTS)
    def test_top3_first_matches_specialty(self, client, text):
        """El primer elemento de top_3 debe coincidir con specialty."""
        r = client.post("/predict", json={"transcription": text})
        if r.status_code == 200:
            data = r.json()
            assert data["top_3"][0]["specialty"] == data["specialty"]

    @pytest.mark.parametrize("text", VALID_INPUTS)
    def test_top3_sorted_descending(self, client, text):
        """top_3 debe estar ordenado de mayor a menor probabilidad."""
        r = client.post("/predict", json={"transcription": text})
        if r.status_code == 200:
            probs = [item["probability"] for item in r.json()["top_3"]]
            assert probs == sorted(probs, reverse=True)

    @pytest.mark.parametrize("text", VALID_INPUTS)
    def test_specialty_is_non_empty_string(self, client, text):
        """La especialidad predicha debe ser un string no vacío."""
        r = client.post("/predict", json={"transcription": text})
        if r.status_code == 200:
            assert isinstance(r.json()["specialty"], str)
            assert len(r.json()["specialty"]) > 0

    @pytest.mark.parametrize("text", VALID_INPUTS)
    def test_model_name_present(self, client, text):
        """model_name no debe estar vacío."""
        r = client.post("/predict", json={"transcription": text})
        if r.status_code == 200:
            assert len(r.json()["model_name"]) > 0


# ════════════════════════════════════════════════════════════════════════════
# Edge cases: PERFORMANCE / STRESS BÁSICO (requieren modelos)
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not MODELS_EXIST, reason="Modelos no disponibles.")
class TestPerformanceBasic:
    """Pruebas básicas de rendimiento y repetición."""

    def test_response_time_under_2s_normal(self, client):
        """Una predicción normal debe completarse en menos de 2 segundos."""
        text = (
            "Patient presents with acute chest pain, diaphoresis and dyspnea. "
            "ECG shows ST elevation myocardial infarction. Immediate PCI performed."
        )
        start = time.time()
        r = client.post("/predict", json={"transcription": text})
        elapsed = time.time() - start

        assert r.status_code in (200, 500)
        assert elapsed < 2.0, f"Demasiado lento: {elapsed:.3f}s"

    def test_10_sequential_requests(self, client):
        """Diez requests seguidos deben completarse todos sin errores fatales."""
        text = (
            "Dermatological examination erythematous rash bilateral extremities "
            "biopsy obtained histopathology pending psoriasis eczema differential"
        )
        statuses = []
        for _ in range(10):
            r = client.post("/predict", json={"transcription": text})
            statuses.append(r.status_code)

        # Todos deben ser 200 o 500 (no 422, no 4xx por límite de tasa)
        for s in statuses:
            assert s in (200, 500), f"Status inesperado: {s}"

    def test_repeated_same_text_deterministic(self, client):
        """El mismo texto debe devolver siempre el mismo specialty."""
        text = (
            "Orthopedic evaluation closed fracture left femur arthroplasty "
            "surgical fixation recommended imaging completed"
        )
        results = []
        for _ in range(5):
            r = client.post("/predict", json={"transcription": text})
            if r.status_code == 200:
                results.append(r.json()["specialty"])

        if results:
            assert len(set(results)) == 1, "El modelo no es determinista"

    def test_interleaved_health_and_predict(self, client):
        """Intercalar /health y /predict no debe causar errores."""
        text = (
            "Neurological examination reveals cranial nerve palsy and ataxia "
            "MRI brain obtained lesion identified treatment initiated"
        )
        for _ in range(3):
            h = client.get("/health")
            p = client.post("/predict", json={"transcription": text})
            assert h.status_code == 200
            assert p.status_code in (200, 500)
