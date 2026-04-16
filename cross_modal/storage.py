"""Save and load embeddings, metadata, and config files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

# re-export so other modules can import from here
from cross_modal.vector_store import load_jsonl  # noqa: F401


def save_embeddings(embeddings: np.ndarray, path: Path | str) -> None:
    """Save embeddings to .npy file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # create dirs if needed
    np.save(str(path), embeddings)


def load_embeddings(path: Path | str) -> np.ndarray:
    """Load embeddings from .npy file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    return np.load(str(path))


def save_metadata(records: Iterable[Dict[str, Any]], path: Path | str) -> None:
    """Save metadata as a .jsonl file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_run_config(config: Dict[str, Any], path: Path | str) -> None:
    """Save run config as .json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_run_config(path: Path | str) -> Dict[str, Any]:
    """Load run config from .json."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
