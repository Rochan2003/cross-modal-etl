import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch


def _make_clip_engine():
    """Create a CLIPEmbeddingEngine with mocked model to avoid downloading weights."""
    with patch("cross_modal.embedding.CLIPModel") as MockModel, \
         patch("cross_modal.embedding.CLIPTokenizerFast") as MockTokenizer:

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.half.return_value = mock_model

        def fake_image_features(pixel_values):
            batch = pixel_values.shape[0]
            torch.manual_seed(42)
            return torch.randn(batch, 512)

        def fake_text_features(**kwargs):
            batch = kwargs["input_ids"].shape[0]
            torch.manual_seed(99)
            return torch.randn(batch, 512)

        mock_model.get_image_features.side_effect = fake_image_features
        mock_model.get_text_features.side_effect = fake_text_features
        MockModel.from_pretrained.return_value = mock_model

        mock_tok = MagicMock()
        mock_tok.side_effect = lambda texts, **kw: {
            "input_ids": torch.zeros(len(texts), 77, dtype=torch.long),
            "attention_mask": torch.ones(len(texts), 77, dtype=torch.long),
        }
        MockTokenizer.from_pretrained.return_value = mock_tok

        from cross_modal.embedding import CLIPEmbeddingEngine
        engine = CLIPEmbeddingEngine(device="cpu", use_fp16=False)

    return engine


def _make_clap_engine():
    """Create a CLAPEmbeddingEngine with mocked model."""
    with patch("cross_modal.embedding.ClapModel") as MockModel, \
         patch("cross_modal.embedding.AutoProcessor") as MockProcessor:

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.half.return_value = mock_model

        def fake_audio_features(**kwargs):
            batch = kwargs["input_features"].shape[0]
            torch.manual_seed(77)
            return torch.randn(batch, 512)

        def fake_text_features(**kwargs):
            batch = kwargs["input_ids"].shape[0]
            torch.manual_seed(88)
            return torch.randn(batch, 512)

        mock_model.get_audio_features.side_effect = fake_audio_features
        mock_model.get_text_features.side_effect = fake_text_features
        MockModel.from_pretrained.return_value = mock_model

        mock_proc = MagicMock()
        mock_proc.side_effect = lambda **kw: {
            "input_features": torch.randn(len(kw.get("audios", kw.get("text", []))), 1, 1001, 64),
            "input_ids": torch.zeros(len(kw.get("text", [kw.get("audios")])), 77, dtype=torch.long),
            "attention_mask": torch.ones(len(kw.get("text", [kw.get("audios")])), 77, dtype=torch.long),
        }
        MockProcessor.from_pretrained.return_value = mock_proc

        from cross_modal.embedding import CLAPEmbeddingEngine
        engine = CLAPEmbeddingEngine(device="cpu", use_fp16=False)

    return engine


class TestCLIPEmbeddingEngine:

    def test_image_embeddings_shape_and_norm(self):
        engine = _make_clip_engine()
        pixel_values = torch.randn(4, 3, 224, 224)
        embeddings = engine.encode_image_tensors(pixel_values)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (4, 512)
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_text_embeddings_shape_and_norm(self):
        engine = _make_clip_engine()
        texts = ["a dog", "a cat", "a bird"]
        embeddings = engine.encode_texts(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 512)
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_embeddings_are_nondegenerate(self):
        engine = _make_clip_engine()
        embeddings = engine.encode_image_tensors(torch.randn(1, 3, 224, 224))
        assert embeddings.shape == (1, 512)
        assert not np.allclose(embeddings, 0.0)


class TestCLAPEmbeddingEngine:

    def test_audio_embeddings_shape_and_norm(self):
        engine = _make_clap_engine()
        waveforms = torch.randn(3, 1, 480000)
        embeddings = engine.encode_audio_tensors(waveforms)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 512)
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_text_embeddings_shape_and_norm(self):
        engine = _make_clap_engine()
        texts = ["thunder", "rain falling"]
        embeddings = engine.encode_texts(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 512)
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)
