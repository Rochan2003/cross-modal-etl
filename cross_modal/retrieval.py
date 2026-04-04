from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from cross_modal.vector_store import (
    FaissIPIndex,
    build_audio_index,
    build_audio_text_index,
    build_image_index,
    build_image_text_index,
)

DEFAULT_EMBEDDINGS_DIR = "/Volumes/Samsung_T7/dataset/embeddings"


class CrossModalRetriever:
    """Handles all cross-modal search: text, image, and audio queries."""

    def __init__(
        self,
        embeddings_dir: Path | str,
        clip_model: str = "openai/clip-vit-large-patch14",
        clap_model: str = "laion/larger_clap_music_and_speech",
        device: str | None = None,
        use_fp16: bool = True,
        use_hnsw: bool = False,
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.clip_model = clip_model
        self.clap_model = clap_model
        self.device = device
        self.use_fp16 = use_fp16
        self.use_hnsw = use_hnsw
        self._clip_engine: Optional[Any] = None
        self._clap_engine: Optional[Any] = None
        self._image_index: Optional[FaissIPIndex] = None
        self._audio_index: Optional[FaissIPIndex] = None
        self._image_text_index: Optional[FaissIPIndex] = None
        self._audio_text_index: Optional[FaissIPIndex] = None

    @classmethod
    def from_env(cls) -> CrossModalRetriever:
        path = os.environ.get("EMBEDDINGS_DIR", DEFAULT_EMBEDDINGS_DIR)
        device = os.environ.get("RETRIEVAL_DEVICE")
        return cls(embeddings_dir=path, device=device)

    def load_indexes(self) -> None:
        self._image_index = build_image_index(self.embeddings_dir, use_hnsw=self.use_hnsw)
        self._audio_index = build_audio_index(self.embeddings_dir, use_hnsw=self.use_hnsw)
        # Text-caption indexes for cross-modal bridging
        try:
            self._image_text_index = build_image_text_index(self.embeddings_dir, use_hnsw=self.use_hnsw)
        except FileNotFoundError:
            print("[retriever] Warning: image text index not found, image→audio bridging disabled")
        try:
            self._audio_text_index = build_audio_text_index(self.embeddings_dir, use_hnsw=self.use_hnsw)
        except FileNotFoundError:
            print("[retriever] Warning: audio text index not found, audio→image bridging disabled")

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

    # Encoding

    def encode_query(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        """Encode text into both CLIP and CLAP spaces."""
        if self._clip_engine is None or self._clap_engine is None:
            raise RuntimeError("Encoders not loaded; call load_encoders() or load_all()")
        clip_vec = self._clip_engine.encode_texts([query])[0]
        clap_vec = self._clap_engine.encode_texts([query])[0]
        return clip_vec, clap_vec

    def encode_image(self, pixel_values) -> np.ndarray:
        """Encode an image tensor into CLIP space."""
        if self._clip_engine is None:
            raise RuntimeError("CLIP encoder not loaded")
        return self._clip_engine.encode_image_tensors(pixel_values)[0]

    def encode_audio(self, waveform) -> np.ndarray:
        """Encode an audio tensor into CLAP space."""
        if self._clap_engine is None:
            raise RuntimeError("CLAP encoder not loaded")
        return self._clap_engine.encode_audio_tensors(waveform)[0]

    # Search

    def search(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """Text → Images + Audio (original search)."""
        clip_vec, clap_vec = self.encode_query(query)
        k = max(1, int(top_k))
        image_hits = self.image_index.search(clip_vec, k)
        audio_hits = self.audio_index.search(clap_vec, k)
        return {
            "query": query,
            "modality": "text",
            "top_k": k,
            "image_results": image_hits,
            "audio_results": audio_hits,
        }

    def search_by_image(self, clip_image_vec: np.ndarray, top_k: int = 10) -> Dict[str, Any]:
        """Image search — find similar images, text captions, and audio via caption bridge."""
        k = max(1, int(top_k))

        image_hits = self.image_index.search(clip_image_vec, k)

        text_results = []
        for hit in image_hits:
            text_results.append({
                "rank": hit["rank"],
                "score": hit["score"],
                "text": hit["metadata"].get("caption", ""),
                "source": f"image {hit['metadata'].get('id', '')}",
            })

        # bridge to audio: take top captions, encode with CLAP, search audio index
        audio_hits = []
        if image_hits and self._clap_engine is not None:
            bridge_captions = [h["metadata"].get("caption", "") for h in image_hits[:3]]
            bridge_captions = [c for c in bridge_captions if c]
            if bridge_captions:
                clap_text_vecs = self._clap_engine.encode_texts(bridge_captions)
                avg_vec = clap_text_vecs.mean(axis=0)
                avg_vec = avg_vec / (np.linalg.norm(avg_vec) + 1e-12)
                audio_hits = self.audio_index.search(avg_vec, k)

        return {
            "modality": "image",
            "top_k": k,
            "image_results": image_hits,
            "text_results": text_results,
            "audio_results": audio_hits,
        }

    def search_by_audio(self, clap_audio_vec: np.ndarray, top_k: int = 10) -> Dict[str, Any]:
        """Audio search — find similar audio, text captions, and images via caption bridge."""
        k = max(1, int(top_k))

        audio_hits = self.audio_index.search(clap_audio_vec, k)

        text_results = []
        for hit in audio_hits:
            text_results.append({
                "rank": hit["rank"],
                "score": hit["score"],
                "text": hit["metadata"].get("caption", ""),
                "source": f"audio {hit['metadata'].get('id', '')}",
            })

        # bridge to images: take top captions, encode with CLIP, search image index
        image_hits = []
        if audio_hits and self._clip_engine is not None:
            bridge_captions = [h["metadata"].get("caption", "") for h in audio_hits[:3]]
            bridge_captions = [c for c in bridge_captions if c]
            if bridge_captions:
                clip_text_vecs = self._clip_engine.encode_texts(bridge_captions)
                avg_vec = clip_text_vecs.mean(axis=0)
                avg_vec = avg_vec / (np.linalg.norm(avg_vec) + 1e-12)
                image_hits = self.image_index.search(avg_vec, k)

        return {
            "modality": "audio",
            "top_k": k,
            "image_results": image_hits,
            "text_results": text_results,
            "audio_results": audio_hits,
        }
