"""Persistence helpers for embeddings and metadata.

This module provides a unified interface for the save/load operations
that are used across the pipeline:

  - ``generate_embeddings.py`` writes .npy, .jsonl, and .json files
  - ``vector_store.py`` reads .npy and .jsonl files via ``EmbeddingBundle``
  - ``evaluation.py`` reads .npy and .jsonl files

Rather than duplicating those implementations, this module re-exports
the shared reader (``load_jsonl`` from ``vector_store``) and adds the
corresponding write helpers so callers have a single import point.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

# Re-export the canonical JSONL reader from vector_store
from cross_modal.vector_store import load_jsonl  # noqa: F401


def save_embeddings(embeddings: np.ndarray, path: Path | str) -> None:
    """Save an embedding matrix to a ``.npy`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), embeddings)


def load_embeddings(path: Path | str) -> np.ndarray:
    """Load an embedding matrix from a ``.npy`` file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    return np.load(str(path))


def save_metadata(records: Iterable[Dict[str, Any]], path: Path | str) -> None:
    """Save metadata records as a ``.jsonl`` file (one JSON object per line).

    This is the write counterpart to ``load_jsonl`` from ``vector_store``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_run_config(config: Dict[str, Any], path: Path | str) -> None:
    """Save a run configuration snapshot as a ``.json`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_run_config(path: Path | str) -> Dict[str, Any]:
    """Load a run configuration snapshot from a ``.json`` file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
