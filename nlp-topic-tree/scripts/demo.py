#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts import run_part1_classic, run_part2_embeddings, run_part3_topic_tree, run_all
from src.docs_autogen import regenerate_docs


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Narrated full demo run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-samples", type=int, default=10_000, help="Number of sampled documents")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split proportion")
    parser.add_argument("--vectorizer", choices=["bow", "tfidf"], default="tfidf", help="Text vectorizer for part1")
    parser.add_argument("--st-model", default="all-MiniLM-L6-v2", help="SentenceTransformer model for parts 2/3")
    parser.add_argument("--outputs-dir", default="outputs", help="Directory for output artifacts")
    return parser


def narrate(header: str, message: str):
    print(f"\n=== {header} ===")
    print(message)


def run(args):
    project_root = Path(__file__).resolve().parents[1]
    parsers = {
        "run_part1_classic.py": run_part1_classic.get_parser(),
        "run_part2_embeddings.py": run_part2_embeddings.get_parser(),
        "run_part3_topic_tree.py": run_part3_topic_tree.get_parser(),
        "run_all.py": run_all.get_parser(),
        "demo.py": get_parser(),
    }
    narrate("Docs", "Regenerating README.md and ARCHITECTURE.md from parser/config metadata.")
    regenerate_docs(project_root, parsers)

    narrate("Part 1", "Running classic TF-IDF/BoW style baseline models for multi-class classification.")
    p1_metrics = run_part1_classic.run(args)

    narrate("Part 2", "Encoding text with SentenceTransformer and training the same classifier family.")
    p2_metrics = run_part2_embeddings.run(args)

    narrate("Part 3", "Building top-level clusters, then sub-clusters, and producing a 2-level topic tree.")
    part3 = run_part3_topic_tree.run(args)

    narrate("Reporting", "Writing outputs/DEMO_REPORT.md with tables, comparisons, plots, and topic tree excerpt.")
    from src.reporting import write_demo_report

    out_dir = Path(args.outputs_dir)
    write_demo_report(
        out_dir / "DEMO_REPORT.md",
        config={
            "seed": args.seed,
            "n_samples": args.n_samples,
            "test_size": args.test_size,
            "vectorizer": args.vectorizer,
            "st_model": args.st_model,
        },
        p1_metrics=p1_metrics,
        p2_metrics=p2_metrics,
        chosen_k=part3["chosen_k"],
        top_clusters=part3["top_clusters"],
        tree_text=part3["tree_text"],
    )
    narrate("Done", "Demo complete. Open outputs/DEMO_REPORT.md for recording-ready narrative.")


if __name__ == "__main__":
    parser = get_parser()
    run(parser.parse_args())
