from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from cross_modal.vector_store import FaissIPIndex, build_audio_index, build_image_index

DEFAULT_EMBEDDINGS_DIR = "/Volumes/Samsung_T7/dataset/embeddings"


class CrossModalRetriever:
    """
    Text-to-image (CLIP) and text-to-audio (CLAP) retrieval over precomputed embeddings.
    """

    def __init__(
        self,
        embeddings_dir: Path | str,
        clip_model: str = "openai/clip-vit-base-patch32",
        clap_model: str = "laion/clap-htsat-unfused",
        device: str | None = None,
        use_fp16: bool = True,
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.clip_model = clip_model
        self.clap_model = clap_model
        self.device = device
        self.use_fp16 = use_fp16
        self._clip_engine: Optional[Any] = None
        self._clap_engine: Optional[Any] = None
        self._image_index: Optional[FaissIPIndex] = None
        self._audio_index: Optional[FaissIPIndex] = None

    @classmethod
    def from_env(cls) -> CrossModalRetriever:
        path = os.environ.get("EMBEDDINGS_DIR", DEFAULT_EMBEDDINGS_DIR)
        device = os.environ.get("RETRIEVAL_DEVICE")
        return cls(embeddings_dir=path, device=device)

    def load_indexes(self) -> None:
        self._image_index = build_image_index(self.embeddings_dir)
        self._audio_index = build_audio_index(self.embeddings_dir)

    def load_encoders(self) -> None:
        from cross_modal.embedding import CLAPEmbeddingEngine, CLIPEmbeddingEngine

        self._clip_engine = CLIPEmbeddingEngine(
            model_name=self.clip_model,
            device=self.device,
            use_fp16=self.use_fp16,
        )
        self._clap_engine = CLAPEmbeddingEngine(
            model_name=self.clap_model,
            device=self.device,
            use_fp16=self.use_fp16,
        )

    def load_all(self) -> None:
        self.load_indexes()
        self.load_encoders()

    @property
    def image_index(self) -> FaissIPIndex:
        if self._image_index is None:
            raise RuntimeError("Image index not loaded; call load_indexes() or load_all()")
        return self._image_index

    @property
    def audio_index(self) -> FaissIPIndex:
        if self._audio_index is None:
            raise RuntimeError("Audio index not loaded; call load_indexes() or load_all()")
        return self._audio_index

    def encode_query(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        if self._clip_engine is None or self._clap_engine is None:
            raise RuntimeError("Encoders not loaded; call load_encoders() or load_all()")
        clip_vec = self._clip_engine.encode_texts([query])[0]
        clap_vec = self._clap_engine.encode_texts([query])[0]
        return clip_vec, clap_vec

    def search(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        clip_vec, clap_vec = self.encode_query(query)
        k = max(1, int(top_k))
        image_hits = self.image_index.search(clip_vec, k)
        audio_hits = self.audio_index.search(clap_vec, k)
        return {
            "query": query,
            "top_k": k,
            "image_results": image_hits,
            "audio_results": audio_hits,
        }
