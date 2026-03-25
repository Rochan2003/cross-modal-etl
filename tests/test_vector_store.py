import json
from pathlib import Path

import numpy as np
import pytest

from cross_modal.vector_store import (
    FaissIPIndex,
    build_audio_index,
    build_image_index,
    load_jsonl,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_faiss_ip_index_self_match(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    dim = 8
    n = 20
    raw = rng.standard_normal((n, dim)).astype(np.float32)
    meta = [{"id": str(i), "caption": f"cap{i}", "modality": "image"} for i in range(n)]
    index = FaissIPIndex(raw, meta)
    query = raw[3]
    hits = index.search(query, top_k=5)
    assert hits[0]["rank"] == 1
    assert hits[0]["metadata"]["id"] == "3"
    assert hits[0]["score"] > 0.99


def test_top_k_clipped_to_index_size(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    emb = rng.standard_normal((3, 4)).astype(np.float32)
    meta = [{"id": str(i), "caption": "x", "modality": "image"} for i in range(3)]
    index = FaissIPIndex(emb, meta)
    q = emb[0]
    hits = index.search(q, top_k=100)
    assert len(hits) == 3


def test_load_jsonl_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    rows = [{"a": 1}, {"b": "two"}]
    _write_jsonl(p, rows)
    assert load_jsonl(p) == rows


def test_build_indexes_from_disk(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    img_emb = rng.standard_normal((4, 8)).astype(np.float32)
    aud_emb = rng.standard_normal((3, 8)).astype(np.float32)
    np.save(tmp_path / "clip_image_embeddings.npy", img_emb)
    np.save(tmp_path / "clap_audio_embeddings.npy", aud_emb)
    _write_jsonl(
        tmp_path / "image_metadata.jsonl",
        [{"id": str(i), "caption": f"i{i}", "modality": "image"} for i in range(4)],
    )
    _write_jsonl(
        tmp_path / "audio_metadata.jsonl",
        [{"id": str(i), "caption": f"a{i}", "modality": "audio"} for i in range(3)],
    )
    img_idx = build_image_index(tmp_path)
    aud_idx = build_audio_index(tmp_path)
    assert img_idx.size == 4
    assert aud_idx.size == 3


def test_mismatched_metadata_raises() -> None:
    emb = np.zeros((2, 3), dtype=np.float32)
    meta = [{"id": "0"}]
    with pytest.raises(ValueError, match="Metadata length"):
        FaissIPIndex(emb, meta)
