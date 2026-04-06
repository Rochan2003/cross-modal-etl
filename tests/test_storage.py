"""Tests for cross_modal.storage persistence helpers."""
from pathlib import Path

import numpy as np
import pytest

from cross_modal.storage import (
    load_embeddings,
    load_jsonl,
    load_run_config,
    save_embeddings,
    save_metadata,
    save_run_config,
)


def test_embeddings_roundtrip(tmp_path: Path) -> None:
    original = np.random.randn(10, 64).astype(np.float32)
    path = tmp_path / "embeddings.npy"
    save_embeddings(original, path)
    loaded = load_embeddings(path)
    np.testing.assert_array_equal(original, loaded)


def test_metadata_roundtrip(tmp_path: Path) -> None:
    """save_metadata writes, load_jsonl (re-exported from vector_store) reads."""
    records = [
        {"id": "img_0", "caption": "a cat", "modality": "image"},
        {"id": "img_1", "caption": "a dog", "modality": "image"},
    ]
    path = tmp_path / "meta.jsonl"
    save_metadata(records, path)
    loaded = load_jsonl(path)
    assert loaded == records


def test_run_config_roundtrip(tmp_path: Path) -> None:
    config = {"model": "clip-vit-large", "batch_size": 32, "device": "cpu"}
    path = tmp_path / "config.json"
    save_run_config(config, path)
    loaded = load_run_config(path)
    assert loaded == config


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c" / "embeddings.npy"
    data = np.zeros((2, 4), dtype=np.float32)
    save_embeddings(data, nested)
    assert nested.is_file()


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_embeddings(tmp_path / "nonexistent.npy")
    with pytest.raises(FileNotFoundError):
        load_run_config(tmp_path / "nonexistent.json")
