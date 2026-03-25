from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import faiss
import numpy as np


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _l2_normalize_rows(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (matrix / norms).astype(np.float32, copy=False)


def _ensure_float32(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float32)


class FaissIPIndex:
    """Inner-product index for L2-normalized vectors (dot product = cosine similarity)."""

    def __init__(self, embeddings: np.ndarray, metadata: Sequence[Dict[str, Any]]):
        self._metadata = list(metadata)
        vectors = _ensure_float32(embeddings)
        if vectors.ndim != 2:
            raise ValueError(f"Embeddings must be 2D, got shape {vectors.shape}")
        if len(self._metadata) != vectors.shape[0]:
            raise ValueError(
                f"Metadata length ({len(self._metadata)}) != embedding rows ({vectors.shape[0]})"
            )
        vectors = _l2_normalize_rows(vectors)
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        self._index = index
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def size(self) -> int:
        return self._index.ntotal

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        if query_vector.ndim == 1:
            query_vector = query_vector[np.newaxis, :]
        query = _l2_normalize_rows(_ensure_float32(query_vector))
        if query.shape[1] != self._dim:
            raise ValueError(
                f"Query dim {query.shape[1]} does not match index dim {self._dim}"
            )
        k = min(int(top_k), self.size) if self.size else 0
        if k <= 0:
            return []
        scores, indices = self._index.search(query, k)
        row_scores = scores[0]
        row_indices = indices[0]
        results: List[Dict[str, Any]] = []
        for rank, (score, idx) in enumerate(zip(row_scores, row_indices), start=1):
            if idx < 0:
                continue
            meta = dict(self._metadata[idx])
            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "metadata": meta,
                }
            )
        return results


class EmbeddingBundle:
    """Loads paired .npy + .jsonl from a directory."""

    def __init__(
        self,
        embeddings_dir: Path | str,
        embeddings_filename: str,
        metadata_filename: str,
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_filename = embeddings_filename
        self.metadata_filename = metadata_filename

    def load(self) -> tuple[np.ndarray, List[Dict[str, Any]]]:
        emb_path = self.embeddings_dir / self.embeddings_filename
        meta_path = self.embeddings_dir / self.metadata_filename
        if not emb_path.is_file():
            raise FileNotFoundError(f"Missing embeddings file: {emb_path}")
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing metadata file: {meta_path}")
        embeddings = np.load(emb_path)
        metadata = load_jsonl(meta_path)
        return embeddings, metadata


def build_image_index(embeddings_dir: Path | str) -> FaissIPIndex:
    bundle = EmbeddingBundle(
        embeddings_dir,
        "clip_image_embeddings.npy",
        "image_metadata.jsonl",
    )
    embeddings, metadata = bundle.load()
    return FaissIPIndex(embeddings, metadata)


def build_audio_index(embeddings_dir: Path | str) -> FaissIPIndex:
    bundle = EmbeddingBundle(
        embeddings_dir,
        "clap_audio_embeddings.npy",
        "audio_metadata.jsonl",
    )
    embeddings, metadata = bundle.load()
    return FaissIPIndex(embeddings, metadata)
