"""
conftest.py – Fixtures compartidos para toda la suite de tests.

Mockea S3 a nivel sesión para evitar llamadas reales a AWS durante pruebas.
"""
import os
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Patch S3 en el middleware antes de importar la app
# ---------------------------------------------------------------------------
_s3_mock = MagicMock()
_s3_patch = patch("src.middleware_logging.s3_client", _s3_mock)
_s3_patch.start()

# También parchamos boto3.client en api.py (endpoint /dashboard-data)
_boto3_patch = patch("boto3.client", return_value=_s3_mock)
_boto3_patch.start()

from src.api import app  # noqa: E402 – importar DESPUÉS de parchear S3

MODELS_DIR = os.environ.get("MODELS_DIR", "models")
MODELS_EXIST = os.path.exists(os.path.join(MODELS_DIR, "logreg_model.joblib"))


@pytest.fixture(scope="session")
def client():
    """Cliente HTTP de tests con ciclo de vida completo de la app (startup/shutdown)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def models_loaded():
    """Indica si los modelos están disponibles en disco."""
    return MODELS_EXIST


# ---------------------------------------------------------------------------
# Textos reutilizables para tests
# ---------------------------------------------------------------------------
CARDIOLOGY_TEXT = (
    "Patient presents with chest pain and shortness of breath. "
    "ECG shows ST elevation in leads II, III, and aVF. "
    "Troponin levels elevated. Emergency cardiac catheterization performed. "
    "Left anterior descending artery occlusion found and treated with stent placement."
)

SURGERY_TEXT = (
    "Surgery performed laparoscopic cholecystectomy under general anesthesia. "
    "Patient tolerated the procedure well. No complications noted. "
    "Discharged on postoperative day two with appropriate pain management."
)

NEUROLOGY_TEXT = (
    "Neurological assessment indicates mild cognitive impairment with memory deficits. "
    "MRI brain shows hippocampal atrophy consistent with early Alzheimer disease. "
    "Patient started on cholinesterase inhibitor therapy."
)

ORTHOPEDICS_TEXT = (
    "Orthopedic evaluation shows closed fracture of the left femoral neck. "
    "X-ray confirms displaced fracture. Surgical fixation recommended. "
    "Total hip arthroplasty scheduled for next week."
)

DERMATOLOGY_TEXT = (
    "Dermatological examination reveals erythematous scaly plaques on bilateral upper extremities. "
    "Biopsy obtained for histopathological evaluation. "
    "Differential diagnosis includes psoriasis and eczema."
)


@pytest.fixture
def sample_medical_texts():
    """Colección de textos médicos válidos para testing."""
    return {
        "cardiology": CARDIOLOGY_TEXT,
        "surgery": SURGERY_TEXT,
        "neurology": NEUROLOGY_TEXT,
        "orthopedics": ORTHOPEDICS_TEXT,
        "dermatology": DERMATOLOGY_TEXT,
    }
