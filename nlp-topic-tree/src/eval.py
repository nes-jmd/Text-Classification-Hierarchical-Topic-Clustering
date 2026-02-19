from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def evaluate_predictions(y_true, y_pred, target_names: list[str], top_n: int = 15):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    confusions = top_confusion_pairs(cm, target_names=target_names, top_n=top_n)
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "top_confusions": confusions,
    }


def top_confusion_pairs(cm: np.ndarray, target_names: list[str], top_n: int = 15):
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append(
                    {
                        "true": target_names[i],
                        "pred": target_names[j],
                        "count": int(cm[i, j]),
                    }
                )
    pairs.sort(key=lambda x: x["count"], reverse=True)
    return pairs[:top_n]


def save_metrics(path: Path, metrics: dict):
    serializable = {
        k: (v.tolist() if hasattr(v, "tolist") else v)
        for k, v in metrics.items()
        if k != "confusion_matrix"
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def save_confusions(path: Path, confusions: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(confusions, f, indent=2)
