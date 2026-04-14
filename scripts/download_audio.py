import os

# 1. Define the exact path on your SSD
# Replace 'YOUR_SSD_NAME' with your actual SSD name!
ssd_cache_path = "/Volumes/Samsung_T7/dataset/audiocaps"
os.makedirs(ssd_cache_path, exist_ok=True)

# 2. CRITICAL: Set environment variables BEFORE importing Hugging Face.
# This forces all temporary downloads and caches to happen on the SSD.
os.environ["HF_HOME"] = ssd_cache_path
os.environ["HF_DATASETS_CACHE"] = ssd_cache_path

# 3. Now we can safely import the library
from datasets import load_dataset

print(f"Routing ALL downloads and temporary files directly to {ssd_cache_path}...")

# 4. Run the download
dataset = load_dataset(
    "TwinkStart/AudioCaps",
    cache_dir=ssd_cache_path
)

print("\nSuccessfully loaded the full dataset entirely on the external drive!")