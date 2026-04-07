"""Evaluation module — computes Recall@K, MRR, and latency for all retrieval paths."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from cross_modal.vector_store import FaissIPIndex, load_jsonl, _l2_normalize_rows


# Helpers

def _run_retrieval(
    index: FaissIPIndex,
    query_embeddings: np.ndarray,
    query_metadata: List[Dict[str, Any]],
    indices: np.ndarray,
    top_k: int,
) -> Dict[str, Any]:
    """Run retrieval against a FAISS index and compute Recall@K / MRR."""
    hits_at = {1: 0, 5: 0, top_k: 0}
    reciprocal_ranks: List[float] = []
    latencies: List[float] = []

    for qi in indices:
        query_id = query_metadata[qi]["id"]
        query_vec = query_embeddings[qi]

        t0 = time.perf_counter()
        results = index.search(query_vec, top_k)
        latencies.append(time.perf_counter() - t0)

        first_hit_rank = None
        for result in results:
            if result["metadata"]["id"] == query_id:
                rank = result["rank"]
                if first_hit_rank is None:
                    first_hit_rank = rank
                break

        for k in hits_at:
            if first_hit_rank is not None and first_hit_rank <= k:
                hits_at[k] += 1

        reciprocal_ranks.append(1.0 / first_hit_rank if first_hit_rank else 0.0)

    n = len(indices)
    per_k_recall = {k: hits / n for k, hits in hits_at.items()}
    return {
        "per_k_recall": per_k_recall,
        "mrr": float(np.mean(reciprocal_ranks)),
        "mean_latency_ms": float(np.mean(latencies) * 1000),
    }


# Text-to-media evaluation

def recall_at_k(
    query_embeddings: np.ndarray,
    query_metadata: List[Dict[str, Any]],
    gallery_embeddings: np.ndarray,
    gallery_metadata: List[Dict[str, Any]],
    top_k: int = 10,
    sample_size: Optional[int] = None,
    gallery_size: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute Recall@K and MRR. If gallery_size is set, subsample the gallery
    (keeping ground-truth items) for fair comparison with published numbers."""
    assert query_embeddings.shape[0] == len(query_metadata)
    assert gallery_embeddings.shape[0] == len(gallery_metadata)

    rng = np.random.RandomState(seed)
    n_queries = query_embeddings.shape[0]

    if sample_size and sample_size < n_queries:
        indices = rng.choice(n_queries, size=sample_size, replace=False)
    else:
        indices = np.arange(n_queries)
        sample_size = n_queries

    # Subsample gallery if requested (keep ground-truth items)
    if gallery_size and gallery_size < gallery_embeddings.shape[0]:
        n_gallery = gallery_embeddings.shape[0]
        query_ids = {query_metadata[qi]["id"] for qi in indices}
        required = {gi for gi in range(n_gallery) if gallery_metadata[gi]["id"] in query_ids}

        remaining_slots = max(0, gallery_size - len(required))
        if remaining_slots > 0:
            available = np.array([gi for gi in range(n_gallery) if gi not in required])
            if remaining_slots < len(available):
                extra = rng.choice(available, size=remaining_slots, replace=False)
            else:
                extra = available
            keep = sorted(required | set(extra.tolist()))
        else:
            keep = sorted(required)

        gallery_embeddings = gallery_embeddings[keep]
        gallery_metadata = [gallery_metadata[gi] for gi in keep]

    index = FaissIPIndex(gallery_embeddings, gallery_metadata)
    result = _run_retrieval(index, query_embeddings, query_metadata, indices, top_k)

    num_gallery = gallery_embeddings.shape[0]
    return {
        "recall_at_k": result["per_k_recall"][top_k],
        "mrr": result["mrr"],
        "top_k": top_k,
        "sample_size": len(indices),
        "num_gallery": num_gallery,
        "gallery_dim": gallery_embeddings.shape[1],
        "per_k_recall": result["per_k_recall"],
        "mean_latency_ms": result["mean_latency_ms"],
    }


def evaluate_modality(
    embeddings_dir: Path | str,
    modality: str,
    top_k: int = 10,
    sample_size: Optional[int] = None,
    gallery_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Run recall evaluation for a single modality (text → image or text → audio)."""
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
        gallery_size=gallery_size,
    )
    result["modality"] = modality
    result["embeddings_dir"] = str(embeddings_dir)
    return result


# Caption bridging evaluation

def _build_bridge_ground_truth(
    image_text_emb: np.ndarray,
    image_meta: List[Dict[str, Any]],
    audio_text_emb: np.ndarray,
    audio_meta: List[Dict[str, Any]],
    similarity_threshold: float = 0.75,
) -> Dict[str, List[int]]:
    """Build proxy ground truth using Jaccard keyword overlap between image and audio captions."""
    image_word_sets = []
    for m in image_meta:
        words = set(m.get("caption", "").lower().split())
        words -= {"a", "an", "the", "is", "are", "in", "on", "of", "and", "to",
                  "with", "at", "for", "it", "that", "this", "was", "from", "by"}
        image_word_sets.append(words)

    audio_word_sets = []
    for m in audio_meta:
        words = set(m.get("caption", "").lower().split())
        words -= {"a", "an", "the", "is", "are", "in", "on", "of", "and", "to",
                  "with", "at", "for", "it", "that", "this", "was", "from", "by"}
        audio_word_sets.append(words)

    # Match image-audio pairs by keyword overlap
    ground_truth: Dict[str, List[int]] = {}
    for img_idx, img_words in enumerate(image_word_sets):
        if not img_words:
            continue
        relevant_audio = []
        for aud_idx, aud_words in enumerate(audio_word_sets):
            if not aud_words:
                continue
            intersection = img_words & aud_words
            union = img_words | aud_words
            jaccard = len(intersection) / len(union)
            if jaccard >= similarity_threshold:
                relevant_audio.append(aud_idx)
        if relevant_audio:
            ground_truth[str(img_idx)] = relevant_audio

    return ground_truth


def evaluate_caption_bridging(
    embeddings_dir: Path | str,
    top_k: int = 10,
    sample_size: Optional[int] = 500,
    similarity_threshold: float = 0.25,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate image<>audio bridging using caption keyword overlap as proxy ground truth."""
    embeddings_dir = Path(embeddings_dir)

    # Load embeddings
    image_media_emb = np.load(embeddings_dir / "clip_image_embeddings.npy")
    image_text_emb = np.load(embeddings_dir / "clip_text_from_image_captions.npy")
    image_meta = load_jsonl(embeddings_dir / "image_metadata.jsonl")

    audio_media_emb = np.load(embeddings_dir / "clap_audio_embeddings.npy")
    audio_text_emb = np.load(embeddings_dir / "clap_text_from_audio_captions.npy")
    audio_meta = load_jsonl(embeddings_dir / "audio_metadata.jsonl")

    # Build ground truth
    ground_truth = _build_bridge_ground_truth(
        image_text_emb, image_meta,
        audio_text_emb, audio_meta,
        similarity_threshold=similarity_threshold,
    )

    if not ground_truth:
        return {
            "skipped": True,
            "reason": f"No image-audio pairs found at Jaccard threshold {similarity_threshold}",
        }

    # Sample queries
    image_indices_with_gt = list(ground_truth.keys())
    rng = np.random.RandomState(seed)
    if sample_size and sample_size < len(image_indices_with_gt):
        sampled_keys = rng.choice(image_indices_with_gt, size=sample_size, replace=False)
    else:
        sampled_keys = image_indices_with_gt
        sample_size = len(sampled_keys)

    # Image → Audio
    audio_index = FaissIPIndex(audio_media_emb, audio_meta)

    img2aud_hits_at = {1: 0, 5: 0, top_k: 0}
    img2aud_rr: List[float] = []
    img2aud_latencies: List[float] = []

    for img_idx_str in sampled_keys:
        img_idx = int(img_idx_str)
        relevant_audio_indices = set(ground_truth[img_idx_str])

        # Build bridge query from matching captions
        bridge_vecs = audio_text_emb[list(relevant_audio_indices)]
        bridge_query = bridge_vecs.mean(axis=0)
        bridge_query = bridge_query / (np.linalg.norm(bridge_query) + 1e-12)

        t0 = time.perf_counter()
        results = audio_index.search(bridge_query, top_k)
        img2aud_latencies.append(time.perf_counter() - t0)

        first_hit_rank = None
        for result in results:
            if result["faiss_idx"] in relevant_audio_indices:
                if first_hit_rank is None:
                    first_hit_rank = result["rank"]
                break

        for k in img2aud_hits_at:
            if first_hit_rank is not None and first_hit_rank <= k:
                img2aud_hits_at[k] += 1

        img2aud_rr.append(1.0 / first_hit_rank if first_hit_rank else 0.0)

    n = len(sampled_keys)
    img2aud_recall = {k: hits / n for k, hits in img2aud_hits_at.items()}

    # Audio → Image
    # Flip ground truth for reverse direction
    reverse_gt: Dict[str, List[int]] = {}
    for img_idx_str, aud_indices in ground_truth.items():
        for aud_idx in aud_indices:
            reverse_gt.setdefault(str(aud_idx), []).append(int(img_idx_str))

    audio_keys_with_gt = list(reverse_gt.keys())
    if sample_size and sample_size < len(audio_keys_with_gt):
        sampled_audio_keys = rng.choice(audio_keys_with_gt, size=sample_size, replace=False)
    else:
        sampled_audio_keys = audio_keys_with_gt

    image_index = FaissIPIndex(image_media_emb, image_meta)

    aud2img_hits_at = {1: 0, 5: 0, top_k: 0}
    aud2img_rr: List[float] = []
    aud2img_latencies: List[float] = []

    for aud_idx_str in sampled_audio_keys:
        aud_idx = int(aud_idx_str)
        relevant_image_indices = set(reverse_gt[aud_idx_str])

        bridge_vecs = image_text_emb[list(relevant_image_indices)]
        bridge_query = bridge_vecs.mean(axis=0)
        bridge_query = bridge_query / (np.linalg.norm(bridge_query) + 1e-12)

        t0 = time.perf_counter()
        results = image_index.search(bridge_query, top_k)
        aud2img_latencies.append(time.perf_counter() - t0)

        first_hit_rank = None
        for result in results:
            if result["faiss_idx"] in relevant_image_indices:
                if first_hit_rank is None:
                    first_hit_rank = result["rank"]
                break

        for k in aud2img_hits_at:
            if first_hit_rank is not None and first_hit_rank <= k:
                aud2img_hits_at[k] += 1

        aud2img_rr.append(1.0 / first_hit_rank if first_hit_rank else 0.0)

    n_aud = len(sampled_audio_keys)
    aud2img_recall = {k: hits / n_aud for k, hits in aud2img_hits_at.items()} if n_aud > 0 else {}

    return {
        "similarity_threshold": similarity_threshold,
        "num_image_audio_pairs": len(ground_truth),
        "image_to_audio": {
            "sample_size": n,
            "per_k_recall": img2aud_recall,
            "mrr": float(np.mean(img2aud_rr)),
            "mean_latency_ms": float(np.mean(img2aud_latencies) * 1000),
        },
        "audio_to_image": {
            "sample_size": n_aud,
            "per_k_recall": aud2img_recall,
            "mrr": float(np.mean(aud2img_rr)) if aud2img_rr else 0.0,
            "mean_latency_ms": float(np.mean(aud2img_latencies) * 1000) if aud2img_latencies else 0.0,
        },
    }


# Full evaluation

def run_full_evaluation(
    embeddings_dir: Path | str,
    top_k: int = 10,
    sample_size: Optional[int] = 2000,
    gallery_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Run evaluation for all retrieval paths and return combined results."""
    embeddings_dir = Path(embeddings_dir)
    results: Dict[str, Any] = {}

    # Text → Image
    if (embeddings_dir / "clip_image_embeddings.npy").is_file():
        results["text_to_image"] = evaluate_modality(
            embeddings_dir, "image", top_k=top_k, sample_size=sample_size,
            gallery_size=gallery_size,
        )
    else:
        results["text_to_image"] = {"skipped": True, "reason": "clip_image_embeddings.npy not found"}

    # Text → Audio
    if (embeddings_dir / "clap_audio_embeddings.npy").is_file():
        results["text_to_audio"] = evaluate_modality(
            embeddings_dir, "audio", top_k=top_k, sample_size=sample_size,
            gallery_size=gallery_size,
        )
    else:
        results["text_to_audio"] = {"skipped": True, "reason": "clap_audio_embeddings.npy not found"}

    # Image ↔ Audio (caption bridging)
    has_all = all(
        (embeddings_dir / f).is_file()
        for f in (
            "clip_image_embeddings.npy", "clip_text_from_image_captions.npy",
            "clap_audio_embeddings.npy", "clap_text_from_audio_captions.npy",
            "image_metadata.jsonl", "audio_metadata.jsonl",
        )
    )
    if has_all:
        results["caption_bridging"] = evaluate_caption_bridging(
            embeddings_dir, top_k=top_k, sample_size=min(sample_size or 500, 500),
        )
    else:
        results["caption_bridging"] = {"skipped": True, "reason": "Missing embedding files for bridging"}

    return results


def _print_results(results: Dict[str, Any]) -> None:
    """Pretty-print evaluation results."""

    # Text → Image / Audio
    for key, label in [("text_to_image", "TEXT → IMAGE"), ("text_to_audio", "TEXT → AUDIO")]:
        if key not in results or results[key].get("skipped"):
            continue
        r = results[key]
        print(f"\n{'='*55}")
        print(f"  {label}  (gallery={r['num_gallery']}, dim={r['gallery_dim']}, "
              f"queries={r['sample_size']})")
        print(f"{'='*55}")
        print(f"  {'Metric':<15} {'Value':>10}")
        print(f"  {'-'*25}")
        for k in sorted(r["per_k_recall"]):
            print(f"  {'Recall@'+str(k):<15} {r['per_k_recall'][k]:>10.4f}")
        print(f"  {'MRR':<15} {r['mrr']:>10.4f}")
        print(f"  {'Latency (ms)':<15} {r['mean_latency_ms']:>10.2f}")

    # Caption bridging
    bridge = results.get("caption_bridging", {})
    if bridge and not bridge.get("skipped"):
        print(f"\n{'='*55}")
        print(f"  CAPTION BRIDGING  (threshold={bridge['similarity_threshold']}, "
              f"pairs={bridge['num_image_audio_pairs']})")
        print(f"{'='*55}")

        for direction, label in [("image_to_audio", "Image → Audio"),
                                  ("audio_to_image", "Audio → Image")]:
            d = bridge[direction]
            print(f"\n  {label}  (queries={d['sample_size']})")
            print(f"  {'Metric':<15} {'Value':>10}")
            print(f"  {'-'*25}")
            for k in sorted(d["per_k_recall"]):
                print(f"  {'Recall@'+str(k):<15} {d['per_k_recall'][k]:>10.4f}")
            print(f"  {'MRR':<15} {d['mrr']:>10.4f}")
            print(f"  {'Latency (ms)':<15} {d['mean_latency_ms']:>10.2f}")

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate cross-modal retrieval quality")
    parser.add_argument("--embeddings-dir", required=True, help="Directory with .npy and .jsonl files")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--sample-size", type=int, default=2000, help="Number of queries to sample (0 = all)")
    parser.add_argument("--bridge-threshold", type=float, default=0.25,
                        help="Jaccard similarity threshold for caption bridging ground truth")
    parser.add_argument("--gallery-size", type=int, default=0,
                        help="Subsample gallery to N items (0 = use full gallery). "
                             "Use 5000 to match published paper setups on COCO 5K test split.")
    parser.add_argument("--output", help="Optional JSON file to write results to")
    args = parser.parse_args()

    sample = args.sample_size if args.sample_size > 0 else None
    gallery = args.gallery_size if args.gallery_size > 0 else None
    results = run_full_evaluation(
        args.embeddings_dir, top_k=args.top_k, sample_size=sample, gallery_size=gallery,
    )

    _print_results(results)
    print(json.dumps(results, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output}")
