"""Offline evaluation for cross-modal retrieval (Recall@K, MRR, latency)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from cross_modal.vector_store import FaissIPIndex, load_jsonl, _l2_normalize_rows


def _build_id_to_indices(metadata: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """Map each unique ID to every index where it appears."""
    mapping: Dict[str, List[int]] = {}
    for idx, record in enumerate(metadata):
        record_id = record["id"]
        mapping.setdefault(record_id, []).append(idx)
    return mapping


def recall_at_k(
    query_embeddings: np.ndarray,
    query_metadata: List[Dict[str, Any]],
    gallery_embeddings: np.ndarray,
    gallery_metadata: List[Dict[str, Any]],
    top_k: int = 10,
    sample_size: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute Recall@K and MRR for text-to-media retrieval.

    A query is considered a *hit* when any gallery item sharing the same ``id``
    appears in the top-K results.  This correctly handles the one-to-many
    relationship between images and captions in COCO.

    Parameters
    ----------
    query_embeddings : (N, D) text embeddings used as queries.
    query_metadata : length-N list with at least an ``"id"`` field per entry.
    gallery_embeddings : (M, D) media (image/audio) embeddings to search.
    gallery_metadata : length-M list with at least an ``"id"`` field per entry.
    top_k : number of results to retrieve per query.
    sample_size : if set, randomly sample this many queries (for speed).
    seed : random seed for reproducible sampling.

    Returns
    -------
    dict with keys: recall_at_k, mrr, top_k, sample_size, num_gallery,
                    per_k_recall (dict mapping k -> recall for k in {1, 5, top_k}).
    """
    assert query_embeddings.shape[0] == len(query_metadata)
    assert gallery_embeddings.shape[0] == len(gallery_metadata)

    rng = np.random.RandomState(seed)
    n_queries = query_embeddings.shape[0]

    if sample_size and sample_size < n_queries:
        indices = rng.choice(n_queries, size=sample_size, replace=False)
    else:
        indices = np.arange(n_queries)
        sample_size = n_queries

    # Build the FAISS index over gallery embeddings
    index = FaissIPIndex(gallery_embeddings, gallery_metadata)

    # Map gallery IDs to their indices for ground-truth matching
    gallery_id_to_indices = _build_id_to_indices(gallery_metadata)

    hits_at = {1: 0, 5: 0, top_k: 0}
    reciprocal_ranks: List[float] = []
    latencies: List[float] = []

    for qi in indices:
        query_id = query_metadata[qi]["id"]
        query_vec = query_embeddings[qi]

        t0 = time.perf_counter()
        results = index.search(query_vec, top_k)
        latencies.append(time.perf_counter() - t0)

        # Check which ranks contain a matching ID
        first_hit_rank = None
        for result in results:
            if result["metadata"]["id"] == query_id:
                rank = result["rank"]
                if first_hit_rank is None:
                    first_hit_rank = rank
                break  # only need the first hit for MRR

        for k in hits_at:
            if first_hit_rank is not None and first_hit_rank <= k:
                hits_at[k] += 1

        reciprocal_ranks.append(1.0 / first_hit_rank if first_hit_rank else 0.0)

    n = len(indices)
    per_k_recall = {k: hits / n for k, hits in hits_at.items()}

    return {
        "recall_at_k": per_k_recall[top_k],
        "mrr": float(np.mean(reciprocal_ranks)),
        "top_k": top_k,
        "sample_size": n,
        "num_gallery": gallery_embeddings.shape[0],
        "gallery_dim": gallery_embeddings.shape[1],
        "per_k_recall": per_k_recall,
        "mean_latency_ms": float(np.mean(latencies) * 1000),
    }


def evaluate_modality(
    embeddings_dir: Path | str,
    modality: str,
    top_k: int = 10,
    sample_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Run recall evaluation for a single modality.

    Parameters
    ----------
    embeddings_dir : directory containing .npy and .jsonl files.
    modality : ``"image"`` or ``"audio"``.
    top_k : number of results per query.
    sample_size : optional cap on number of queries.
    """
    embeddings_dir = Path(embeddings_dir)

    if modality == "image":
        query_file = "clip_text_from_image_captions.npy"
        gallery_file = "clip_image_embeddings.npy"
        meta_file = "image_metadata.jsonl"
    elif modality == "audio":
        query_file = "clap_text_from_audio_captions.npy"
        gallery_file = "clap_audio_embeddings.npy"
        meta_file = "audio_metadata.jsonl"
    else:
        raise ValueError(f"Unknown modality: {modality!r}")

    query_embeddings = np.load(embeddings_dir / query_file)
    gallery_embeddings = np.load(embeddings_dir / gallery_file)
    metadata = load_jsonl(embeddings_dir / meta_file)

    result = recall_at_k(
        query_embeddings=query_embeddings,
        query_metadata=metadata,
        gallery_embeddings=gallery_embeddings,
        gallery_metadata=metadata,
        top_k=top_k,
        sample_size=sample_size,
    )
    result["modality"] = modality
    result["embeddings_dir"] = str(embeddings_dir)
    return result


def run_full_evaluation(
    embeddings_dir: Path | str,
    top_k: int = 10,
    sample_size: Optional[int] = 2000,
) -> Dict[str, Any]:
    """Run evaluation for both image and audio modalities and return combined results."""
    embeddings_dir = Path(embeddings_dir)
    results: Dict[str, Any] = {}

    for modality in ("image", "audio"):
        gallery_file = (
            "clip_image_embeddings.npy" if modality == "image"
            else "clap_audio_embeddings.npy"
        )
        if (embeddings_dir / gallery_file).is_file():
            results[f"{modality}_recall"] = evaluate_modality(
                embeddings_dir, modality, top_k=top_k, sample_size=sample_size,
            )
        else:
            results[f"{modality}_recall"] = {"skipped": True, "reason": f"{gallery_file} not found"}

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate cross-modal retrieval quality")
    parser.add_argument("--embeddings-dir", required=True, help="Directory with .npy and .jsonl files")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--sample-size", type=int, default=2000, help="Number of queries to sample (0 = all)")
    parser.add_argument("--output", help="Optional JSON file to write results to")
    args = parser.parse_args()

    sample = args.sample_size if args.sample_size > 0 else None
    results = run_full_evaluation(args.embeddings_dir, top_k=args.top_k, sample_size=sample)

    print(json.dumps(results, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output}")
