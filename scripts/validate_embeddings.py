"""Quick check that embeddings and metadata are aligned and not corrupted."""
import numpy as np
import json
from pathlib import Path


def validate_embeddings(embedding_path, metadata_path, name):
    print(f"--- {name} ---")

    try:
        embeddings = np.load(embedding_path)
        with open(metadata_path, 'r') as f:
            metadata = [json.loads(line) for line in f]
    except Exception as e:
        print(f"  ERROR loading: {e}")
        return

    num_vectors = embeddings.shape[0]
    dims = embeddings.shape[1]
    num_meta = len(metadata)

    print(f"  Vectors: {num_vectors}, Dims: {dims}, Metadata: {num_meta}")

    if num_vectors != num_meta:
        print(f"  WARNING: count mismatch ({num_vectors} vs {num_meta})")
    else:
        print(f"  Counts match.")

    # check for dead/empty vectors
    norms = np.linalg.norm(embeddings, axis=1)
    zero_count = np.sum(norms == 0)
    if zero_count > 0:
        print(f"  WARNING: {zero_count} zero vectors found")
    print(f"  Avg norm: {np.mean(norms):.4f} (should be ~1.0)\n")


if __name__ == "__main__":
    # change this path to wherever your embeddings are stored
    base = Path("/Volumes/Samsung_T7/dataset/embeddings")

    validate_embeddings(
        base / "clap_audio_embeddings.npy",
        base / "audio_metadata.jsonl",
        "Audio (CLAP, 512-dim)"
    )

    validate_embeddings(
        base / "clip_image_embeddings.npy",
        base / "image_metadata.jsonl",
        "Image (CLIP, 768-dim)"
    )
