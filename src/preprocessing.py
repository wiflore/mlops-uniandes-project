"""
Pipeline de preprocesamiento NLP para transcripciones medicas.
"""
import re
import joblib
import numpy as np
from typing import Optional

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """Limpia y normaliza texto de transcripciones medicas."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-záéíóúñ\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(words)


class TextPreprocessor:
    """Preprocessor que encapsula TF-IDF vectorizacion."""

    def __init__(
        self,
        max_features: int = 10_000,
        ngram_range: tuple = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True
        )

    def fit_transform(self, texts):
        cleaned = [clean_text(t) for t in texts]
        return self.vectorizer.fit_transform(cleaned)

    def transform(self, texts):
        cleaned = [clean_text(t) for t in texts]
        return self.vectorizer.transform(cleaned)

    def save(self, path: str):
        joblib.dump(self.vectorizer, path)

    @classmethod
    def load(cls, path: str) -> 'TextPreprocessor':
        instance = cls()
        instance.vectorizer = joblib.load(path)
        return instance
