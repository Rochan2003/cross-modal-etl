"""Download AudioCaps dataset to external storage.

Clotho doesn't need a separate script — it's downloaded automatically
by generate_embeddings.py when you pass --audio-dataset-name CLAPv2/Clotho.
"""
import os

# change this to wherever you want the dataset cached
ssd_cache_path = "/Volumes/Samsung_T7/dataset/audiocaps"
os.makedirs(ssd_cache_path, exist_ok=True)

# set env vars before importing HF so it caches to the right place
os.environ["HF_HOME"] = ssd_cache_path
os.environ["HF_DATASETS_CACHE"] = ssd_cache_path

from datasets import load_dataset

print(f"Downloading to {ssd_cache_path}...")
dataset = load_dataset("TwinkStart/AudioCaps", cache_dir=ssd_cache_path)
print("Done.")
