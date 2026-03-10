"""
Tests de integración end-to-end.

Simula el flujo completo tal como lo usaría un frontend o cliente externo:
1. Verificar salud de la API
2. Enviar transcripción → recibir predicción
3. Validar la respuesta completa end-to-end
4. Flujos de error (inputs inválidos → manejo correcto)

Todos los tests usan el TestClient de FastAPI que levanta la app completa
(incluyendo middleware, startup event, etc.).
"""
import os
import time
import pytest

MODELS_DIR = os.environ.get("MODELS_DIR", "models")
MODELS_EXIST = os.path.exists(os.path.join(MODELS_DIR, "logreg_model.joblib"))


# ════════════════════════════════════════════════════════════════════════════
# Flujo 1 – Health Check siempre disponible
# ════════════════════════════════════════════════════════════════════════════

class TestHealthCheckFlow:
    """El endpoint /health debe estar disponible incondicionalmente."""

    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_content_type_json(self, client):
        r = client.get("/health")
        assert "application/json" in r.headers.get("content-type", "")

    def test_health_response_schema(self, client):
        data = client.get("/health").json()
        assert set(data.keys()) >= {"status", "model_loaded", "version"}

    def test_health_status_is_healthy(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"

    def test_health_version_format(self, client):
        """version debe tener formato semver X.Y.Z."""
        version = client.get("/health").json()["version"]
        parts = version.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_health_idempotent(self, client):
        """Llamar /health dos veces debe devolver exactamente el mismo resultado."""
        r1 = client.get("/health").json()
        r2 = client.get("/health").json()
        assert r1 == r2

    def test_health_not_post(self, client):
        """POST a /health debe devolver 405 Method Not Allowed."""
        r = client.post("/health")
        assert r.status_code == 405


# ════════════════════════════════════════════════════════════════════════════
# Flujo 2 – Health → Predict (happy path)
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not MODELS_EXIST, reason="Modelos no disponibles.")
class TestHealthThenPredictFlow:
    """Flujo completo: verificar salud y luego predecir."""

    def test_health_then_predict_cardiology(self, client):
        """1. /health OK  2. /predict con texto de cardiología → 200."""
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True

        predict = client.post(
            "/predict",
            json={
                "transcription": (
                    "Patient presents with acute chest pain radiating to left arm. "
                    "ECG confirms ST elevation myocardial infarction. "
                    "Emergency PCI performed with stent placement successful."
                )
            },
        )
        assert predict.status_code == 200
        data = predict.json()
        assert isinstance(data["specialty"], str)
        assert 0.0 <= data["confidence"] <= 1.0

    def test_health_then_predict_surgery(self, client):
        """Flujo con texto quirúrgico."""
        assert client.get("/health").status_code == 200
        r = client.post(
            "/predict",
            json={
                "transcription": (
                    "Laparoscopic cholecystectomy performed under general anesthesia "
                    "for symptomatic cholelithiasis. Patient tolerated procedure well. "
                    "No intraoperative complications. Discharged on day two."
                )
            },
        )
        assert r.status_code == 200

    def test_full_response_structure(self, client):
        """Verifica la estructura completa de la respuesta de /predict."""
        r = client.post(
            "/predict",
            json={
                "transcription": (
                    "Neurological examination reveals cognitive impairment. "
                    "MRI brain shows hippocampal atrophy bilateral. "
                    "Diagnosis early Alzheimer disease started medication."
                )
            },
        )
        assert r.status_code == 200
        data = r.json()

        # Campos obligatorios
        assert "specialty" in data
        assert "confidence" in data
        assert "top_3" in data
        assert "model_name" in data
        assert "model_version" in data

        # Tipos
        assert isinstance(data["specialty"], str)
        assert isinstance(data["confidence"], float)
        assert isinstance(data["top_3"], list)
        assert isinstance(data["model_name"], str)
        assert isinstance(data["model_version"], str)

        # Semver de model_version
        parts = data["model_version"].split(".")
        assert len(parts) == 3

    def test_top3_structure_deep(self, client):
        """Verifica la estructura interna de cada elemento de top_3."""
        r = client.post(
            "/predict",
            json={
                "transcription": (
                    "Orthopedic evaluation displaced fracture femoral neck. "
                    "Total hip arthroplasty performed rehabilitation initiated."
                )
            },
        )
        assert r.status_code == 200
        for item in r.json()["top_3"]:
            assert "specialty" in item
            assert "probability" in item
            assert isinstance(item["specialty"], str)
            assert isinstance(item["probability"], float)
            assert 0.0 <= item["probability"] <= 1.0


# ════════════════════════════════════════════════════════════════════════════
# Flujo 3 – Múltiples especialidades médicas
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not MODELS_EXIST, reason="Modelos no disponibles.")
class TestMultipleSpecialtyPredictions:
    """Predice múltiples especialidades para verificar cobertura del modelo."""

    SPECIALTY_TEXTS = [
        (
            "Dermatology",
            "Erythematous papular rash involving face trunk extremities. "
            "Biopsy shows changes consistent psoriasis phototherapy initiated.",
        ),
        (
            "Orthopedic",
            "X-ray confirms comminuted fracture distal radius Colles fracture. "
            "Closed reduction performed short arm cast applied six weeks.",
        ),
        (
            "Neurology",
            "Tremor resting pill-rolling bilateral hand. Bradykinesia rigidity cogwheel. "
            "Diagnosis Parkinson disease levodopa initiated physical therapy.",
        ),
        (
            "Gastroenterology",
            "Upper endoscopy revealed gastric ulcer antrum Helicobacter pylori positive. "
            "Triple therapy initiated proton pump inhibitor amoxicillin clarithromycin.",
        ),
        (
            "Cardiology",
            "Echocardiogram ejection fraction 35 percent dilated cardiomyopathy. "
            "ACE inhibitor beta blocker diuretic started heart failure management.",
        ),
    ]

    @pytest.mark.parametrize("expected_domain,text", SPECIALTY_TEXTS)
    def test_predict_returns_valid_specialty(self, client, expected_domain, text):
        """Cada texto debe producir una predicción válida (no validamos la correcta)."""
        r = client.post("/predict", json={"transcription": text})
        assert r.status_code == 200
        data = r.json()
        assert len(data["specialty"]) > 0
        assert data["confidence"] >= 0.0


# ════════════════════════════════════════════════════════════════════════════
# Flujo 4 – Error handling end-to-end
# ════════════════════════════════════════════════════════════════════════════

class TestErrorHandlingFlow:
    """Verifica que los errores de entrada se manejan de forma adecuada."""

    ERROR_CASES = [
        ("transcription vacía", {"transcription": ""}),
        ("sin campo transcription", {"text": "some text"}),
        ("body vacío", {}),
        ("transcription=null", {"transcription": None}),
        ("transcription muy corta", {"transcription": "short"}),
    ]

    @pytest.mark.parametrize("case_name,payload", ERROR_CASES)
    def test_invalid_input_returns_4xx(self, client, case_name, payload):
        """Inputs inválidos deben retornar 4xx, nunca 200 ni 5xx de servidor."""
        r = client.post("/predict", json=payload)
        assert 400 <= r.status_code < 500, (
            f"caso '{case_name}': se esperaba 4xx, se obtuvo {r.status_code}"
        )

    def test_error_response_has_detail_field(self, client):
        """Las respuestas de error de validación deben incluir 'detail'."""
        r = client.post("/predict", json={"transcription": "x"})
        assert r.status_code == 422
        data = r.json()
        assert "detail" in data

    def test_health_never_fails(self, client):
        """/health no debe fallar independientemente del estado del modelo."""
        r = client.get("/health")
        assert r.status_code == 200

    def test_unknown_endpoint_404(self, client):
        """Un endpoint inexistente debe retornar 404."""
        r = client.get("/nonexistent_route_xyz")
        assert r.status_code == 404

    def test_predict_get_not_allowed(self, client):
        """GET a /predict debe retornar 405."""
        r = client.get("/predict")
        assert r.status_code == 405


# ════════════════════════════════════════════════════════════════════════════
# Flujo 5 – Boundary values para min_length=10
# ════════════════════════════════════════════════════════════════════════════

class TestMinLengthBoundary:
    """Pruebas de frontera alrededor del límite min_length=10."""

    def test_length_9_rejected(self, client):
        """9 caracteres → 422."""
        r = client.post("/predict", json={"transcription": "123456789"})
        assert r.status_code == 422

    def test_length_10_accepted_by_schema(self, client):
        """10 caracteres → pasan schema (200 o 500 según modelo)."""
        r = client.post("/predict", json={"transcription": "1234567890"})
        assert r.status_code in (200, 500)

    def test_length_11_accepted_by_schema(self, client):
        """11 caracteres → pasan schema."""
        r = client.post("/predict", json={"transcription": "12345678901"})
        assert r.status_code in (200, 500)

    def test_exactly_at_boundary_medical(self, client):
        """Exactamente 10 caracteres con contenido médico."""
        r = client.post("/predict", json={"transcription": "chest pain"})
        assert r.status_code in (200, 500)


# ════════════════════════════════════════════════════════════════════════════
# Flujo 6 – Timing end-to-end
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not MODELS_EXIST, reason="Modelos no disponibles.")
class TestEndToEndTiming:
    """Verifica que el flujo completo se ejecuta en tiempos razonables."""

    def test_health_under_500ms(self, client):
        start = time.time()
        client.get("/health")
        assert time.time() - start < 0.5

    def test_predict_under_3s(self, client):
        start = time.time()
        client.post(
            "/predict",
            json={
                "transcription": (
                    "Patient presents chest pain shortness breath ECG "
                    "ST elevation myocardial infarction emergency treatment"
                )
            },
        )
        assert time.time() - start < 3.0

    def test_5_sequential_requests_under_15s(self, client):
        texts = [
            "Cardiac catheterization stent placement successful coronary artery",
            "Laparoscopic appendectomy general anesthesia no complications",
            "MRI brain hippocampal atrophy cognitive impairment dementia",
            "Fracture femoral neck arthroplasty orthopedic surgery rehabilitation",
            "Erythematous rash psoriasis biopsy phototherapy dermatology",
        ]
        start = time.time()
        for text in texts:
            client.post("/predict", json={"transcription": text})
        assert time.time() - start < 15.0
