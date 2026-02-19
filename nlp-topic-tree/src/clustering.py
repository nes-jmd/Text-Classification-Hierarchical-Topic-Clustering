from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans


def elbow_search(embeddings: np.ndarray, ks=range(2, 10), seed: int = 42):
    inertias = []
    models = {}
    for k in ks:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        km.fit(embeddings)
        inertias.append(float(km.inertia_))
        models[k] = km

    chosen_k = _choose_k_by_distance(np.array(list(ks)), np.array(inertias))
    return {
        "ks": list(ks),
        "inertias": inertias,
        "chosen_k": int(chosen_k),
        "model": models[int(chosen_k)],
    }


def _choose_k_by_distance(ks: np.ndarray, inertias: np.ndarray) -> int:
    x1, y1 = ks[0], inertias[0]
    x2, y2 = ks[-1], inertias[-1]
    distances = []
    for x0, y0 in zip(ks, inertias):
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
        distances.append(numerator / denominator if denominator else 0)
    return int(ks[int(np.argmax(distances))])


def nearest_docs_to_centroid(embeddings: np.ndarray, indices: np.ndarray, centroid: np.ndarray, top_n: int = 8):
    cluster_embeddings = embeddings[indices]
    dists = np.linalg.norm(cluster_embeddings - centroid, axis=1)
    order = np.argsort(dists)[:top_n]
    return indices[order]


def save_elbow(path: Path, data: dict):
    payload = {"ks": data["ks"], "inertias": data["inertias"], "chosen_k": data["chosen_k"]}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
