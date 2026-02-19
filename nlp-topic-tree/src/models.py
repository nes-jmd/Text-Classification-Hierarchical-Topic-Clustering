from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

from .features import build_vectorizer


def classic_model_pipelines(vectorizer_name: str, seed: int = 42) -> dict[str, Pipeline]:
    return {
        "mnb": Pipeline([
            ("vectorizer", build_vectorizer(vectorizer_name)),
            ("model", MultinomialNB()),
        ]),
        "logreg": Pipeline([
            ("vectorizer", build_vectorizer(vectorizer_name)),
            ("model", LogisticRegression(max_iter=2_000, random_state=seed)),
        ]),
        "linearsvm": Pipeline([
            ("vectorizer", build_vectorizer(vectorizer_name)),
            ("model", LinearSVC(random_state=seed)),
        ]),
        "rf": Pipeline([
            ("vectorizer", build_vectorizer(vectorizer_name)),
            ("model", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)),
        ]),
    }


def embedding_models(seed: int = 42) -> dict[str, Pipeline | object]:
    return {
        "mnb": Pipeline([
            ("scale", MinMaxScaler()),
            ("model", MultinomialNB()),
        ]),
        "logreg": LogisticRegression(max_iter=2_000, random_state=seed),
        "linearsvm": LinearSVC(random_state=seed),
        "rf": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
    }
