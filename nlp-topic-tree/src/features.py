from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def build_vectorizer(name: str):
    if name == "bow":
        return CountVectorizer(max_features=50_000, stop_words="english")
    if name == "tfidf":
        return TfidfVectorizer(max_features=50_000, stop_words="english")
    raise ValueError(f"Unknown vectorizer: {name}")


def encode_texts(texts: list[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    model = SentenceTransformer(model_name)
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
