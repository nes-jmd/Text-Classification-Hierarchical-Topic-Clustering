from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def plot_confusion_matrix(cm: np.ndarray, path: Path, title: str):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_elbow(ks: list[int], inertias: list[float], chosen_k: int, path: Path):
    plt.figure(figsize=(7, 5))
    plt.plot(ks, inertias, marker="o")
    plt.axvline(chosen_k, linestyle="--", color="red", label=f"chosen_k={chosen_k}")
    plt.title("Elbow Method")
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def write_demo_report(
    path: Path,
    config: dict,
    p1_metrics: dict,
    p2_metrics: dict,
    chosen_k: int,
    top_clusters: list[dict],
    tree_text: str,
):
    p1_rows = [[k, f"{v['accuracy']:.4f}", f"{v['macro_f1']:.4f}"] for k, v in p1_metrics.items()]
    p2_rows = [[k, f"{v['accuracy']:.4f}", f"{v['macro_f1']:.4f}"] for k, v in p2_metrics.items()]

    p1_best = max(p1_metrics.items(), key=lambda kv: kv[1]["macro_f1"])
    p2_best = max(p2_metrics.items(), key=lambda kv: kv[1]["macro_f1"])
    comp = (
        f"Best classic model is **{p1_best[0]}** (Macro-F1={p1_best[1]['macro_f1']:.4f}); "
        f"best embedding model is **{p2_best[0]}** (Macro-F1={p2_best[1]['macro_f1']:.4f})."
    )

    cluster_rows = [[c["cluster_id"], c["label"], c["size"]] for c in top_clusters]

    content = f"""# Demo Report

## Environment + Config
- Seed: {config['seed']}
- n_samples: {config['n_samples']}
- test_size: {config['test_size']}
- vectorizer: {config['vectorizer']}
- sentence-transformer model: {config['st_model']}

## Part 1 — Classic Features
{markdown_table(['Model', 'Accuracy', 'Macro-F1'], p1_rows)}

![Part1 Confusion](confusion_matrix_part1.png)

## Part 2 — Embeddings
{markdown_table(['Model', 'Accuracy', 'Macro-F1'], p2_rows)}

![Part2 Confusion](confusion_matrix_part2.png)

### Comparison
{comp}

## Part 3 — Topic Clustering and Tree
![Elbow](elbow.png)

- Chosen K: **{chosen_k}**

{markdown_table(['Cluster', 'Label', 'Size'], cluster_rows)}

```text
{tree_text}
```
"""
    path.write_text(content, encoding="utf-8")
