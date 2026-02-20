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

from .preprocessing import clean_text, TextPreprocessor

DATA_PATH = "data/raw/mtsamples.csv"
MODELS_DIR = "models"
MIN_SAMPLES = 50
MAX_FEATURES = 10_000
RANDOM_STATE = 42


def load_and_prepare_data(data_path: str = DATA_PATH):
    df = pd.read_csv(data_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df = df.drop_duplicates(subset='transcription', keep='first')
    df = df.dropna(subset=['transcription'])
    df['medical_specialty'] = df['medical_specialty'].str.strip()
    counts = df['medical_specialty'].value_counts()
    valid = counts[counts >= MIN_SAMPLES].index
    df = df[df['medical_specialty'].isin(valid)].reset_index(drop=True)
    return df


def train_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_and_prepare_data()
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

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective='multi:softprob', random_state=RANDOM_STATE,
        n_jobs=-1, verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    print(f"XGBoost - Acc: {accuracy_score(y_test, y_pred):.4f}, F1: {f1_score(y_test, y_pred, average='macro'):.4f}")
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'xgboost_model.joblib'))

    # LogReg
    lr_model = LogisticRegression(
        C=1.0, max_iter=1000, solver='lbfgs',
        class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
    )
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    print(f"LogReg  - Acc: {accuracy_score(y_test, y_pred):.4f}, F1: {f1_score(y_test, y_pred, average='macro'):.4f}")
    joblib.dump(lr_model, os.path.join(MODELS_DIR, 'logreg_model.joblib'))

    print(f"\nModelos guardados en {MODELS_DIR}")


if __name__ == "__main__":
    train_models()
