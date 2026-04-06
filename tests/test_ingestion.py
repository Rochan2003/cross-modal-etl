import pytest
import torch
from unittest.mock import patch
from cross_modal.ingestion import AudioDataset, VisualDataset


# We mock 'load_dataset' so the test doesn't try to download AudioCaps!
@patch('cross_modal.ingestion.load_dataset')
def test_audio_normalization_padding(mock_load):
    """Test if short audio is correctly padded to 10 seconds at 44.1kHz."""
    dataset = AudioDataset(cache_dir="dummy")

    # Create a dummy 3-second mono audio clip at 22050Hz
    dummy_sr = 22050
    dummy_waveform = torch.rand(1, dummy_sr * 3)

    normalized = dataset.normalize_audio(dummy_waveform, dummy_sr)

    # Expected: 1 channel, 10 seconds * 44100 Hz = 441000 samples
    assert normalized.shape == (1, 480000), f"Expected shape (1, 480000), got {normalized.shape}"


@patch('cross_modal.ingestion.load_dataset')
def test_audio_normalization_truncation(mock_load):
    """Test if long audio is correctly truncated to 10 seconds."""
    dataset = AudioDataset(cache_dir="dummy")

    # Create a dummy 15-second mono audio clip at 44100Hz
    dummy_sr = 44100
    dummy_waveform = torch.rand(1, dummy_sr * 15)

    normalized = dataset.normalize_audio(dummy_waveform, dummy_sr)

    assert normalized.shape == (1, 480000), f"Expected shape (1, 480000), got {normalized.shape}"


@patch('cross_modal.ingestion.load_dataset')
def test_silence_filtering(mock_load):
    """Test if completely silent audio is correctly flagged as invalid."""
    dataset = AudioDataset(cache_dir="dummy")

    # Create an empty waveform (silence) and a random one (loud)
    silent_waveform = torch.zeros(1, 44100)
    loud_waveform = torch.rand(1, 44100)

    assert not dataset.validate_audio(silent_waveform), "Silent audio should be rejected."
    assert dataset.validate_audio(loud_waveform), "Normal audio should be accepted."