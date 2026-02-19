from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    seed: int = 42
    n_samples: int = 10_000
    test_size: float = 0.2
    vectorizer: str = "tfidf"
    st_model: str = "all-MiniLM-L6-v2"
    outputs_dir: str = "outputs"


DEFAULT_CONFIG = PipelineConfig()


def ensure_outputs_dir(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path
