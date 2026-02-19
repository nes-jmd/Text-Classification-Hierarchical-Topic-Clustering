#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Part 1: classic text feature models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-samples", type=int, default=10_000, help="Number of sampled documents")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split proportion")
    parser.add_argument("--vectorizer", choices=["bow", "tfidf"], default="tfidf", help="Text vectorizer")
    parser.add_argument("--outputs-dir", default="outputs", help="Directory for output artifacts")
    parser.add_argument("--st-model", default="all-MiniLM-L6-v2", help="Unused; kept for uniform CLI")
    return parser


def run(args):
    from src.config import ensure_outputs_dir
    from src.data import load_dataset, stratified_split
    from src.eval import evaluate_predictions
    from src.models import classic_model_pipelines
    from src.reporting import plot_confusion_matrix

    out_dir = ensure_outputs_dir(args.outputs_dir)
    data = load_dataset(n_samples=args.n_samples, seed=args.seed)
    split = stratified_split(data.texts, data.y, test_size=args.test_size, seed=args.seed)

    models = classic_model_pipelines(args.vectorizer, seed=args.seed)
    metrics, confusions = {}, {}
    best = None

    for name, pipe in models.items():
        pipe.fit(split.x_train, split.y_train)
        pred = pipe.predict(split.x_test)
        ev = evaluate_predictions(split.y_test, pred, data.target_names)
        metrics[name] = {"accuracy": ev["accuracy"], "macro_f1": ev["macro_f1"]}
        confusions[name] = ev["top_confusions"]
        if best is None or ev["macro_f1"] > best[1]["macro_f1"]:
            best = (name, ev)

    (out_dir / "metrics_part1.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "confusions_part1.json").write_text(json.dumps(confusions, indent=2), encoding="utf-8")
    plot_confusion_matrix(best[1]["confusion_matrix"], out_dir / "confusion_matrix_part1.png", f"Part1 best={best[0]}")
    return metrics


if __name__ == "__main__":
    parser = get_parser()
    run(parser.parse_args())
