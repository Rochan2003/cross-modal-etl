"""Cross-modal ETL pipeline for semantic audio-visual retrieval.

Modules
-------
ingestion   - Data loaders for MS COCO images and AudioCaps/Clotho audio.
embedding   - CLIP and CLAP encoding engines with L2 normalization.
storage     - Persistence helpers for .npy, .jsonl, and .json files.
vector_store - FAISS inner-product index wrapper.
retrieval   - Cross-modal search orchestrator (text/image/audio -> all modalities).
evaluation  - Recall@K, MRR evaluation with random baseline comparison.
api         - FastAPI endpoints for serving retrieval over HTTP.
"""

from cross_modal.vector_store import FaissIPIndex
from cross_modal.retrieval import CrossModalRetriever


def __getattr__(name: str):
    """Lazy imports for heavy torch-dependent classes."""
    if name == "CLIPEmbeddingEngine":
        from cross_modal.embedding import CLIPEmbeddingEngine
        return CLIPEmbeddingEngine
    if name == "CLAPEmbeddingEngine":
        from cross_modal.embedding import CLAPEmbeddingEngine
        return CLAPEmbeddingEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CrossModalRetriever",
    "CLIPEmbeddingEngine",
    "CLAPEmbeddingEngine",
    "FaissIPIndex",
]
