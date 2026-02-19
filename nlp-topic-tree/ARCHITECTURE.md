# Architecture

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
- seed: 42
- n_samples: 10000
- test_size: 0.2
- vectorizer: tfidf
- st_model: all-MiniLM-L6-v2
- outputs_dir: outputs
