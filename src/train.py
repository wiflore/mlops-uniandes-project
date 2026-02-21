"""
Script de entrenamiento. Ejecutar: python -m src.train
"""
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
import mlflow
import mlflow.xgboost
import mlflow.sklearn

from .preprocessing import clean_text, TextPreprocessor

DATA_PATH = "data/raw/mtsamples.csv"
MODELS_DIR = "models"
MIN_SAMPLES = 50
MAX_FEATURES = 10_000
RANDOM_STATE = 42

def load_and_prepare_data(data_path: str = DATA_PATH, min_samples: int = 50):
    df = pd.read_csv(data_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df = df.drop_duplicates(subset='transcription', keep='first')
    df = df.dropna(subset=['transcription'])
    df['medical_specialty'] = df['medical_specialty'].str.strip()
    counts = df['medical_specialty'].value_counts()
    valid = counts[counts >= min_samples].index
    df = df[df['medical_specialty'].isin(valid)].reset_index(drop=True)
    return df


def train_models(min_samples=50, xgb_depth=6, lr_c=1.0):
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_and_prepare_data(min_samples=min_samples)
    print(f"\n--- Experimento: min_samples={min_samples}, depth={xgb_depth}, C={lr_c} ---")
    print(f"Dataset: {len(df)} registros, {df['medical_specialty'].nunique()} clases")

    le = LabelEncoder()
    y = le.fit_transform(df['medical_specialty'])
    joblib.dump(le, os.path.join(MODELS_DIR, 'label_encoder.joblib'))

    preprocessor = TextPreprocessor(max_features=MAX_FEATURES, ngram_range=(1, 2))
    X = preprocessor.fit_transform(df['transcription'])
    preprocessor.save(os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib'))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Configuración de MLflow
    # Si la variable MLFLOW_TRACKING_URI no existe, usará ./mlruns localmente.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Medical_Transcriptions_Classification")

    print(f"\nIniciando Tracking de MLflow en: {tracking_uri}")

    # XGBoost
    with mlflow.start_run(run_name=f"XGB_MS{min_samples}_D{xgb_depth}"):
        xgb_params = {
            'n_estimators': 200, 
            'max_depth': xgb_depth, 
            'learning_rate': 0.1,
            'subsample': 0.8, 
            'colsample_bytree': 0.8,
            'objective': 'multi:softprob',
            'random_state': RANDOM_STATE,
            'n_jobs': -1, 
            'verbosity': 0
        }
        mlflow.log_param("min_samples_threshold", min_samples)
        mlflow.log_params(xgb_params)
        
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        print(f"XGBoost - Acc: {acc:.4f}, F1: {f1:.4f}")
        
        mlflow.log_metrics({"accuracy": acc, "f1_macro": f1})
        mlflow.xgboost.log_model(xgb_model, "modelo_xgboost")
        
        joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'xgboost_model.joblib'))

    # LogReg
    with mlflow.start_run(run_name=f"LogReg_MS{min_samples}_C{lr_c}"):
        lr_params = {
            'C': lr_c, 
            'max_iter': 1000, 
            'solver': 'lbfgs',
            'class_weight': 'balanced', 
            'random_state': RANDOM_STATE, 
            'n_jobs': -1
        }
        mlflow.log_param("min_samples_threshold", min_samples)
        mlflow.log_params(lr_params)

        lr_model = LogisticRegression(**lr_params)
        lr_model.fit(X_train, y_train)
        y_pred = lr_model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        print(f"LogReg  - Acc: {acc:.4f}, F1: {f1:.4f}")
        
        mlflow.log_metrics({"accuracy": acc, "f1_macro": f1})
        mlflow.sklearn.log_model(lr_model, "modelo_logreg")

        joblib.dump(lr_model, os.path.join(MODELS_DIR, 'logreg_model.joblib'))

    print(f"\nModelos guardados en {MODELS_DIR}")


if __name__ == "__main__":
    # Experimento 1: Baseline (50 muestras min, logreg C=1.0, xgb depth=6)
    train_models(min_samples=50, xgb_depth=6, lr_c=1.0)
    
    # Experimento 2: Sensibilidad de especialidades (filtro más estricto = menos clases)
    train_models(min_samples=100, xgb_depth=6, lr_c=1.0)
    
    # Experimento 3: Alteración de hiperparámetros de los modelos
    train_models(min_samples=50, xgb_depth=4, lr_c=0.1)
