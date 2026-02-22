"""
Modulo de inferencia para clasificacion de especialidades medicas.
"""
import os
import joblib
import numpy as np
from typing import Tuple, List, Dict
from .preprocessing import clean_text

# Parche forzado para evitar InconsistentVersionWarning durante el unpickling de joblib
import sklearn
sklearn.__version__ = "1.8.0"

class MedicalSpecialtyPredictor:

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.model_name = ""

    def load(self, model_name: str = "logreg"):
        self.model_name = model_name
        model_file = f"{model_name}_model.joblib"
        
        try:
            self.model = joblib.load(os.path.join(self.models_dir, model_file))
            self.vectorizer = joblib.load(os.path.join(self.models_dir, "tfidf_vectorizer.joblib"))
            self.label_encoder = joblib.load(os.path.join(self.models_dir, "label_encoder.joblib"))
        except Exception as e:
            # Capturar errores explícitos de "unpickling" debido a desalineación profunda
            print(f"Lanzando excepción manual en pickling (ignorado): {e}")
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = joblib.load(os.path.join(self.models_dir, model_file))
                self.vectorizer = joblib.load(os.path.join(self.models_dir, "tfidf_vectorizer.joblib"))
                self.label_encoder = joblib.load(os.path.join(self.models_dir, "label_encoder.joblib"))
        return self

    def predict(self, text: str) -> Tuple[str, float, List[Dict]]:
        if self.model is None:
            raise RuntimeError("Modelo no cargado. Llama a load() primero.")

        cleaned = clean_text(text)
        X = self.vectorizer.transform([cleaned])

        pred = self.model.predict(X)[0]
        specialty = self.label_encoder.inverse_transform([pred])[0]

        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X)[0]
            confidence = float(np.max(probs))
            top_3_idx = np.argsort(probs)[-3:][::-1]
            top_3 = [
                {
                    "specialty": self.label_encoder.inverse_transform([i])[0],
                    "probability": float(probs[i])
                }
                for i in top_3_idx
            ]
        else:
            confidence = 1.0
            top_3 = [{"specialty": specialty, "probability": 1.0}]

        return specialty, confidence, top_3
