#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts import run_part1_classic, run_part2_embeddings, run_part3_topic_tree
from src.docs_autogen import regenerate_docs


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run all three parts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-samples", type=int, default=10_000, help="Number of sampled documents")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split proportion")
    parser.add_argument("--vectorizer", choices=["bow", "tfidf"], default="tfidf", help="Text vectorizer for part1")
    parser.add_argument("--st-model", default="all-MiniLM-L6-v2", help="SentenceTransformer model for parts 2/3")
    parser.add_argument("--outputs-dir", default="outputs", help="Directory for output artifacts")
    return parser


def run(args):
    project_root = Path(__file__).resolve().parents[1]
    from scripts import demo

    parsers = {
        "run_part1_classic.py": run_part1_classic.get_parser(),
        "run_part2_embeddings.py": run_part2_embeddings.get_parser(),
        "run_part3_topic_tree.py": run_part3_topic_tree.get_parser(),
        "run_all.py": get_parser(),
        "demo.py": demo.get_parser(),
    }
    regenerate_docs(project_root, parsers)

    run_part1_classic.run(args)
    run_part2_embeddings.run(args)
    run_part3_topic_tree.run(args)


if __name__ == "__main__":
    parser = get_parser()
    run(parser.parse_args())
