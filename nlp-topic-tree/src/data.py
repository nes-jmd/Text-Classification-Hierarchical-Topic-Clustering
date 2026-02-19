from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split


@dataclass
class DatasetBundle:
    texts: list[str]
    y: np.ndarray
    target_names: list[str]


@dataclass
class SplitBundle:
    x_train: list[str]
    x_test: list[str]
    y_train: np.ndarray
    y_test: np.ndarray


def load_dataset(n_samples: int = 10_000, seed: int = 42) -> DatasetBundle:
    dataset = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    texts = np.array(dataset.data, dtype=object)
    y = np.array(dataset.target)

    if n_samples > len(texts):
        raise ValueError(f"n_samples={n_samples} exceeds dataset size {len(texts)}")

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(texts), size=n_samples, replace=False)
    sampled_texts = texts[idx].tolist()
    sampled_y = y[idx]
    return DatasetBundle(texts=sampled_texts, y=sampled_y, target_names=dataset.target_names)


def stratified_split(texts: list[str], y: np.ndarray, test_size: float = 0.2, seed: int = 42) -> SplitBundle:
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    return SplitBundle(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
