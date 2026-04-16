"""Cross-modal ETL pipeline package."""

# light imports (no torch dependency)
from cross_modal.vector_store import FaissIPIndex
from cross_modal.retrieval import CrossModalRetriever


def __getattr__(name: str):
    # lazy-load CLIP/CLAP engines so torch isn't imported until needed
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
