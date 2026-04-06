import logging
import os
import json
import torch
import torchaudio
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset

logger = logging.getLogger(__name__)


class VisualDataset(Dataset):
    """Data loader for MS COCO images, deduplicated by image_id.

    Each item corresponds to one unique image. Captions are joined with ' | '
    so that every image is embedded exactly once.
    """

    def __init__(self, image_dir, annotation_file):
        self.image_dir = image_dir
        self._warned_missing = False

        with open(annotation_file, 'r') as f:
            data = json.load(f)

        # Deduplicate: group captions by image_id, keep one entry per image
        captions_by_id = defaultdict(list)
        for ann in data['annotations']:
            captions_by_id[ann['image_id']].append(ann['caption'])

        self.entries = []
        for image_id, captions in captions_by_id.items():
            self.entries.append({
                "image_id": image_id,
                "caption": captions[0],
                "all_captions": captions,
            })

        # Visual Norm: Resize 224x224 and normalize for CLIP
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711))
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img_id = str(entry['image_id']).zfill(12)
        caption = entry['caption']

        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")

        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            valid = True
        except Exception as e:
            if not self._warned_missing:
                logger.warning(
                    "Failed to load image %s: %s — further warnings suppressed. "
                    "Check that --image-dir points to the folder containing .jpg files "
                    "(e.g. .../train2017, not .../images).",
                    img_path, e,
                )
                self._warned_missing = True
            image = torch.zeros((3, 224, 224))
            valid = False

        return {"image": image, "caption": caption, "id": img_id, "valid": valid}


class AudioDataset(Dataset):
    """Data loader for AudioCaps, reading directly from the Hugging Face cache."""

    def __init__(self, cache_dir, split="train", target_sr=48000, duration_sec=10,
                 dataset_name="d0rj/audiocaps", caption_column="caption"):
        self.target_sr = target_sr
        self.target_length = target_sr * duration_sec

        # Load from Hugging Face, caching to external storage
        self.hf_dataset = load_dataset(
            dataset_name,
            split=split,
            cache_dir=cache_dir,
        )

        # Detect caption column: some datasets use 'text' instead of 'caption'
        if caption_column not in self.hf_dataset.column_names:
            for fallback in ("text", "caption", "sentence"):
                if fallback in self.hf_dataset.column_names:
                    caption_column = fallback
                    break
        self.caption_column = caption_column

    def validate_audio(self, waveform):
        """Filter low-quality/silent clips by checking Root Mean Square energy."""
        rms_energy = torch.sqrt(torch.mean(waveform ** 2))
        return rms_energy > 0.001

    def normalize_audio(self, waveform, sr):
        """Audio Norm: Resample 44.1kHz, Mono, Pad/Truncate 10s."""
        # 1. Convert to Mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 2. Resample to 44.1kHz
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        # 3. Pad or Truncate to exactly 10 seconds
        if waveform.shape[1] > self.target_length:
            waveform = waveform[:, :self.target_length]
        elif waveform.shape[1] < self.target_length:
            pad_amount = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        return waveform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        row = self.hf_dataset[idx]
        caption = row[self.caption_column]

        # Hugging Face automatically decodes the audio into a numpy array for us
        audio_array = row['audio']['array']
        sr = row['audio']['sampling_rate']

        # Convert numpy array to PyTorch tensor
        waveform = torch.tensor(audio_array).float()

        # Ensure it has a channel dimension (1, length)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Validate and Normalize
        if not self.validate_audio(waveform):
            return {"audio": torch.zeros((1, self.target_length)), "caption": caption, "valid": False}

        waveform = self.normalize_audio(waveform, sr)

        return {"audio": waveform, "caption": caption, "valid": True}