from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import torch
from transformers import AutoProcessor, CLIPModel, CLIPTokenizerFast, ClapModel


def _resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _l2_normalize(embeddings: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return embeddings / torch.clamp(embeddings.norm(dim=-1, keepdim=True), min=eps)


def _to_numpy(embeddings: torch.Tensor) -> np.ndarray:
    return embeddings.detach().cpu().float().numpy()


class CLIPEmbeddingEngine:
    """Wrapper around CLIP for image and text embedding generation."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str | None = None,
        use_fp16: bool = True,
    ):
        self.device = _resolve_device(device)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
        self.use_fp16 = bool(use_fp16 and self.device.type == "cuda")
        if self.use_fp16:
            self.model = self.model.half()

    def encode_image_tensors(self, pixel_values: torch.Tensor) -> np.ndarray:
        with torch.inference_mode():
            pixel_values = pixel_values.to(self.device, non_blocking=True)
            if self.use_fp16:
                pixel_values = pixel_values.half()
            embeddings = self.model.get_image_features(pixel_values=pixel_values)
            embeddings = _l2_normalize(embeddings)
        return _to_numpy(embeddings)

    def encode_texts(self, texts: Iterable[str], max_length: int = 77) -> np.ndarray:
        text_list = [text if text else "" for text in texts]
        tokens = self.tokenizer(
            text_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        with torch.inference_mode():
            embeddings = self.model.get_text_features(**tokens)
            embeddings = _l2_normalize(embeddings)
        return _to_numpy(embeddings)


class CLAPEmbeddingEngine:
    """Wrapper around CLAP for audio and text embedding generation."""

    def __init__(
        self,
        model_name: str = "laion/larger_clap_music_and_speech",
        device: str | None = None,
        use_fp16: bool = True,
        sampling_rate: int = 48000,
    ):
        self.device = _resolve_device(device)
        self.model = ClapModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.sampling_rate = sampling_rate
        self.use_fp16 = bool(use_fp16 and self.device.type == "cuda")
        if self.use_fp16:
            self.model = self.model.half()

    def _prepare_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        prepared = {}
        for key, value in inputs.items():
            tensor = value.to(self.device)
            if self.use_fp16 and tensor.is_floating_point():
                tensor = tensor.half()
            prepared[key] = tensor
        return prepared

    def encode_audio_tensors(self, waveforms: torch.Tensor) -> np.ndarray:
        if waveforms.ndim == 3 and waveforms.shape[1] == 1:
            waveforms = waveforms[:, 0, :]
        audio_batch: List[np.ndarray] = [waveform.detach().cpu().numpy() for waveform in waveforms]
        inputs = self.processor(
            audios=audio_batch,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = self._prepare_inputs(inputs)
        with torch.inference_mode():
            embeddings = self.model.get_audio_features(**inputs)
            embeddings = _l2_normalize(embeddings)
        return _to_numpy(embeddings)

    def encode_texts(self, texts: Iterable[str]) -> np.ndarray:
        text_list = [text if text else "" for text in texts]
        inputs = self.processor(text=text_list, return_tensors="pt", padding=True, truncation=True)
        inputs = self._prepare_inputs(inputs)
        with torch.inference_mode():
            embeddings = self.model.get_text_features(**inputs)
            embeddings = _l2_normalize(embeddings)
        return _to_numpy(embeddings)