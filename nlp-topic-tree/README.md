# NLP Topic Tree

## Overview
This project builds an end-to-end NLP pipeline on a deterministic 10,000-sample subset of 20 Newsgroups: classic text classifiers, sentence-embedding classifiers, and a two-level topic tree with modular LLM/heuristic labels.

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env
```

## Run Commands
- Part 1: `python scripts/run_part1_classic.py`
- Part 2: `python scripts/run_part2_embeddings.py`
- Part 3: `python scripts/run_part3_topic_tree.py`
- Full run: `python scripts/run_all.py`
- Demo walkthrough (recommended for recording): `python scripts/demo.py`

## CLI Options (source of truth = argparse)

### run_part1_classic.py
- `--seed`: Random seed (default: 42)
- `--n-samples`: Number of sampled documents (default: 10000)
- `--test-size`: Test split proportion (default: 0.2)
- `--vectorizer`: Text vectorizer (default: tfidf)
- `--outputs-dir`: Directory for output artifacts (default: outputs)
- `--st-model`: Unused; kept for uniform CLI (default: all-MiniLM-L6-v2)

### run_part2_embeddings.py
- `--seed`: Random seed (default: 42)
- `--n-samples`: Number of sampled documents (default: 10000)
- `--test-size`: Test split proportion (default: 0.2)
- `--vectorizer`: Unused; kept for uniform CLI (default: tfidf)
- `--st-model`: SentenceTransformer model (default: all-MiniLM-L6-v2)
- `--outputs-dir`: Directory for output artifacts (default: outputs)

### run_part3_topic_tree.py
- `--seed`: Random seed (default: 42)
- `--n-samples`: Number of sampled documents (default: 10000)
- `--test-size`: Unused; present for interface consistency (default: 0.2)
- `--vectorizer`: Unused; kept for uniform CLI (default: tfidf)
- `--st-model`: SentenceTransformer model (default: all-MiniLM-L6-v2)
- `--outputs-dir`: Directory for output artifacts (default: outputs)

### run_all.py
- `--seed`: Random seed (default: 42)
- `--n-samples`: Number of sampled documents (default: 10000)
- `--test-size`: Test split proportion (default: 0.2)
- `--vectorizer`: Text vectorizer for part1 (default: tfidf)
- `--st-model`: SentenceTransformer model for parts 2/3 (default: all-MiniLM-L6-v2)
- `--outputs-dir`: Directory for output artifacts (default: outputs)

### demo.py
- `--seed`: Random seed (default: 42)
- `--n-samples`: Number of sampled documents (default: 10000)
- `--test-size`: Test split proportion (default: 0.2)
- `--vectorizer`: Text vectorizer for part1 (default: tfidf)
- `--st-model`: SentenceTransformer model for parts 2/3 (default: all-MiniLM-L6-v2)
- `--outputs-dir`: Directory for output artifacts (default: outputs)

## Outputs
All outputs are written to `outputs/` (or `--outputs-dir`). Key artifacts include metrics JSON, confusion matrices, elbow analysis, cluster labels, and `DEMO_REPORT.md`.

## LLM Labeling
If `OPENAI_API_KEY` is set, OpenAI labeling is used. Otherwise the pipeline automatically falls back to a heuristic labeler and prints a warning.
