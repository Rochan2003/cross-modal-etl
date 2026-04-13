"""Export audio clips from HF cache to individual WAV files for demo playback."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm


def load_metadata_captions(embeddings_dir: Path) -> list[str]:
    """Read captions from metadata file."""
    meta_path = embeddings_dir / "audio_metadata.jsonl"
    captions = []
    with open(meta_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                captions.append(json.loads(line)["caption"])
    return captions


def export_sequential(hf_id: str, split: str, cache_dir: str,
                      output_dir: Path, start_pos: int, count: int,
                      valid_captions: set[str] | None = None) -> int:
    """Export WAVs from a HF dataset, optionally filtering by caption set."""
    ds = load_dataset(hf_id, split=split, cache_dir=cache_dir)
    name = hf_id.split("/")[-1]

    # Some datasets use 'text' instead of 'caption'
    caption_col = "caption"
    if caption_col not in ds.column_names:
        for fallback in ("text", "sentence"):
            if fallback in ds.column_names:
                caption_col = fallback
                break

    print(f"  {name} [{split}]: {len(ds)} rows, exporting to positions {start_pos}+")

    pos = start_pos
    written = 0
    skipped = 0
    for i in tqdm(range(len(ds)), desc=f"  {name}"):
        row = ds[i]
        caption = row[caption_col]

        if valid_captions is not None and caption not in valid_captions:
            skipped += 1
            continue

        if written >= count:
            break

        out_path = output_dir / f"{pos}.wav"
        if not out_path.is_file():
            audio = row["audio"]
            waveform = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]
            sf.write(str(out_path), waveform, sr)

        pos += 1
        written += 1

    if skipped:
        print(f"  Skipped {skipped} clips not found in metadata")
    print(f"  Wrote {written} WAV files (positions {start_pos}-{start_pos + written - 1})")
    return written


def main():
    parser = argparse.ArgumentParser(description="Export audio to WAV files")
    parser.add_argument("--output-dir", default="/Volumes/Samsung_T7/dataset/audio_wav")
    parser.add_argument("--audio-cache", default="/Volumes/Samsung_T7/dataset/audiocaps")
    parser.add_argument("--embeddings-dir", default="/Volumes/Samsung_T7/dataset/embeddings")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir = Path(args.embeddings_dir)

    print("Loading metadata...")
    all_captions = load_metadata_captions(embeddings_dir)
    total = len(all_captions)
    print(f"  {total} entries in metadata")

    audiocaps_count = 3985
    clotho_count = total - audiocaps_count

    # Clotho captions for filtering
    clotho_captions = set(all_captions[audiocaps_count:])
    print(f"  AudioCaps: {audiocaps_count}, Clotho: {clotho_count}")

    # AudioCaps
    print("\nExporting AudioCaps...")
    export_sequential(
        "TwinkStart/AudioCaps", "test", args.audio_cache,
        output_dir, start_pos=0, count=audiocaps_count,
    )

    # Clotho
    print("\nExporting Clotho...")
    export_sequential(
        "CLAPv2/Clotho", "train", args.audio_cache,
        output_dir, start_pos=audiocaps_count, count=clotho_count,
        valid_captions=clotho_captions,
    )

    wav_count = len(list(output_dir.glob("*.wav")))
    print(f"\nDone — {wav_count} WAV files in {output_dir}")
    if wav_count != total:
        print(f"WARNING: expected {total}, got {wav_count}")


if __name__ == "__main__":
    main()
