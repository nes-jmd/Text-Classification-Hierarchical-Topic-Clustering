from __future__ import annotations

from pathlib import Path

from .config import DEFAULT_CONFIG


def _parser_options(parser) -> str:
    lines: list[str] = []
    for action in parser._actions:
        if not action.option_strings:
            continue
        if set(action.option_strings) == {"-h", "--help"}:
            continue
        opts = ", ".join(action.option_strings)
        default = "" if action.default is None else f" (default: {action.default})"
        lines.append(f"- `{opts}`: {action.help or ''}{default}")
    return "\n".join(lines)


def regenerate_docs(project_root: Path, parsers: dict[str, object]):
    readme = f"""# NLP Topic Tree

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
"""
    for name, parser in parsers.items():
        readme += f"\n### {name}\n{_parser_options(parser)}\n"

    readme += """
## Outputs
All outputs are written to `outputs/` (or `--outputs-dir`). Key artifacts include metrics JSON, confusion matrices, elbow analysis, cluster labels, and `DEMO_REPORT.md`.

## LLM Labeling
If `OPENAI_API_KEY` is set, OpenAI labeling is used. Otherwise the pipeline automatically falls back to a heuristic labeler and prints a warning.
"""

    arch = f"""# Architecture

## Data Flow
```text
20 Newsgroups -> deterministic 10k sample -> stratified split
  -> Part 1: vectorizer (BoW/TF-IDF) + classifiers -> metrics/confusions/plot
  -> Part 2: SentenceTransformer embeddings + classifiers -> metrics/confusions/plot
  -> Part 3: embeddings -> elbow KMeans -> top labels -> subcluster labels -> topic tree
```

## Module Responsibilities
- `src/config.py`: Defaults and output directory helpers
- `src/data.py`: Dataset loading and deterministic sampling/splitting
- `src/features.py`: Vectorizers and embedding generation
- `src/models.py`: Classifier definitions for both feature families
- `src/eval.py`: Metrics and confusion extraction
- `src/clustering.py`: Elbow search and representative document selection
- `src/labeling.py`: OpenAI + heuristic labeling backends
- `src/topic_tree.py`: Text rendering for the 2-level tree
- `src/reporting.py`: Plotting and markdown report generation
- `src/docs_autogen.py`: Regenerates README and ARCHITECTURE from parser/config defaults

## Scripts
- `run_part1_classic.py`: Runs classic feature model comparison
- `run_part2_embeddings.py`: Runs embedding model comparison
- `run_part3_topic_tree.py`: Runs clustering and hierarchical topic labeling
- `run_all.py`: Regenerates docs and runs parts 1-3 non-narrated
- `demo.py`: Regenerates docs, prints narration, runs full pipeline, writes DEMO_REPORT

## Config Defaults
- seed: {DEFAULT_CONFIG.seed}
- n_samples: {DEFAULT_CONFIG.n_samples}
- test_size: {DEFAULT_CONFIG.test_size}
- vectorizer: {DEFAULT_CONFIG.vectorizer}
- st_model: {DEFAULT_CONFIG.st_model}
- outputs_dir: {DEFAULT_CONFIG.outputs_dir}
"""

    (project_root / "README.md").write_text(readme, encoding="utf-8")
    (project_root / "ARCHITECTURE.md").write_text(arch, encoding="utf-8")
