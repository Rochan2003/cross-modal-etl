import numpy as np
import json
from pathlib import Path


def validate_embeddings(embedding_path, metadata_path, modality_name):
    print(f"--- Validating {modality_name.upper()} ---")

    # 1. Load the files
    try:
        embeddings = np.load(embedding_path)
        with open(metadata_path, 'r') as f:
            metadata = [json.loads(line) for line in f]
    except Exception as e:
        print(f"ERROR: Could not load files for {modality_name}: {e}")
        return

    # 2. Check Shape Alignment
    num_vectors = embeddings.shape[0]
    dims = embeddings.shape[1]
    num_meta = len(metadata)

    print(f"Vectors found: {num_vectors}")
    print(f"Dimensions: {dims} (Expected 768 for CLIP, 512 for CLAP)")
    print(f"Metadata entries: {num_meta}")

    if num_vectors != num_meta:
        print(f"❌ WARNING: Mismatch! Vectors ({num_vectors}) != Metadata ({num_meta})")
    else:
        print(f"✅ Alignment: Success.")

    # 3. Check for "Dead" Vectors (All Zeros)
    norms = np.linalg.norm(embeddings, axis=1)
    zero_vectors = np.sum(norms == 0)
    if zero_vectors > 0:
        print(f"⚠️ ALERT: Found {zero_vectors} vectors that are completely empty (all zeros).")
    else:
        print(f"✅ Data Quality: No empty vectors found.")

    print(f"Average Vector Norm: {np.mean(norms):.4f} (Should be near 1.0 for normalized embeddings)\n")


# Update these paths to your Samsung T7
base_path = Path("/Volumes/Samsung_T7/dataset/embeddings")

# Validate Audio
validate_embeddings(
    base_path / "clap_audio_embeddings.npy",
    base_path / "audio_metadata.jsonl",
    "Audio (CLAP)"
)

# Validate Images
validate_embeddings(
    base_path / "clip_image_embeddings.npy",
    base_path / "image_metadata.jsonl",
    "Visual (CLIP)"
)