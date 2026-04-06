from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from cross_modal.ingestion import AudioDataset, VisualDataset


def _ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def _concat_or_empty(arrays: Sequence[np.ndarray]) -> np.ndarray:
    if arrays:
        return np.concatenate(arrays, axis=0)
    return np.empty((0, 0), dtype=np.float32)


def _subset_dataset(dataset, limit: int | None):
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def _write_jsonl(path: Path, records: Iterable[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=True) + "\n")


def generate_visual_embeddings(
        dataset: VisualDataset,
        clip_engine: Any,
        batch_size: int,
        num_workers: int,
        skip_invalid: bool,
) -> Dict[str, np.ndarray | List[Dict[str, str]]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    image_vectors: List[np.ndarray] = []
    text_vectors: List[np.ndarray] = []
    metadata: List[Dict[str, str]] = []

    for batch in tqdm(loader, desc="Encoding images (CLIP)"):
        if skip_invalid and "valid" in batch:
            valid = batch["valid"]
            if not isinstance(valid, torch.Tensor):
                valid = torch.tensor(valid, dtype=torch.bool)
            keep_idx = torch.where(valid)[0]
            if keep_idx.numel() == 0:
                continue
            images = batch["image"][keep_idx]
            ids = [batch["id"][i] for i in keep_idx.tolist()]
            captions = [batch["caption"][i] for i in keep_idx.tolist()]
        else:
            images = batch["image"]
            ids = list(batch["id"])
            captions = list(batch["caption"])

        image_vectors.append(clip_engine.encode_image_tensors(images))
        text_vectors.append(clip_engine.encode_texts(captions))
        metadata.extend(
            {"id": str(sample_id), "caption": str(caption), "modality": "image"}
            for sample_id, caption in zip(ids, captions)
        )

    return {
        "image_embeddings": _concat_or_empty(image_vectors),
        "text_embeddings": _concat_or_empty(text_vectors),
        "metadata": metadata,
    }


def generate_audio_embeddings(
        dataset: AudioDataset,
        clap_engine: Any,
        batch_size: int,
        num_workers: int,
        skip_invalid: bool,
) -> Dict[str, np.ndarray | List[Dict[str, str]]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    audio_vectors: List[np.ndarray] = []
    text_vectors: List[np.ndarray] = []
    metadata: List[Dict[str, str]] = []

    # Synthetic ID counter to avoid KeyError: 'youtube_id'
    current_idx = 0

    for batch in tqdm(loader, desc="Encoding audio (CLAP)"):
        batch_len = len(batch["caption"])

        # Generate unique IDs for this batch
        ids = [f"audio_{i}" for i in range(current_idx, current_idx + batch_len)]
        current_idx += batch_len

        if skip_invalid and "valid" in batch:
            valid = batch["valid"]
            if not isinstance(valid, torch.Tensor):
                valid = torch.tensor(valid, dtype=torch.bool)
            keep_idx = torch.where(valid)[0]
            if keep_idx.numel() == 0:
                continue
            audios = batch["audio"][keep_idx]
            # Filter the synthetic IDs and captions based on validity
            ids = [ids[i] for i in keep_idx.tolist()]
            captions = [batch["caption"][i] for i in keep_idx.tolist()]
        else:
            audios = batch["audio"]
            captions = list(batch["caption"])

        audio_vectors.append(clap_engine.encode_audio_tensors(audios))
        text_vectors.append(clap_engine.encode_texts(captions))

        # Consistent metadata schema using "id" to match the image modality
        metadata.extend(
            {"id": str(sample_id), "caption": str(caption), "modality": "audio"}
            for sample_id, caption in zip(ids, captions)
        )

    return {
        "audio_embeddings": _concat_or_empty(audio_vectors),
        "text_embeddings": _concat_or_empty(text_vectors),
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-modal preprocessing and embedding generation")
    parser.add_argument("--output-dir", required=True, help="Directory where embeddings are written")

    parser.add_argument("--image-dir", help="COCO image directory containing *.jpg files")
    parser.add_argument("--coco-annotations", help="COCO captions annotation JSON path")
    parser.add_argument("--image-limit", type=int, default=0, help="Optional max number of images")
    parser.add_argument("--image-size", type=int, default=224, help="Image resize dimensions")

    parser.add_argument("--audio-cache-dir", help="HF cache directory for audio dataset")
    parser.add_argument("--audio-split", default="train", help="Audio dataset split")
    parser.add_argument("--audio-dataset-name", default="d0rj/audiocaps", help="HF dataset id")
    parser.add_argument("--audio-limit", type=int, default=0, help="Optional max number of audio samples")
    parser.add_argument("--audio-target-sr", type=int, default=48000, help="Target SR for audio preprocessing")
    parser.add_argument("--audio-duration-sec", type=int, default=10, help="Fixed audio clip duration")

    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14", help="CLIP model id")
    parser.add_argument("--clap-model", default="laion/larger_clap_music_and_speech", help="CLAP model id")
    parser.add_argument("--device", default=None, help="Computation device, e.g. mps, cuda, or cpu")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for all encoders")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--skip-invalid", action="store_true", default=True,
                        help="Skip corrupted/invalid samples (default: True)")
    parser.add_argument("--no-skip-invalid", dest="skip_invalid", action="store_false",
                        help="Include invalid samples (zero-filled fallback)")
    parser.add_argument("--disable-fp16", action="store_true", help="Disable FP16 inference on GPU")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = _ensure_dir(args.output_dir)

    from cross_modal.embedding import CLAPEmbeddingEngine, CLIPEmbeddingEngine

    use_fp16 = not args.disable_fp16
    clip_engine = CLIPEmbeddingEngine(model_name=args.clip_model, device=args.device, use_fp16=use_fp16)
    clap_engine = CLAPEmbeddingEngine(
        model_name=args.clap_model,
        device=args.device,
        use_fp16=use_fp16,
        sampling_rate=args.audio_target_sr,
    )

    if args.image_dir and args.coco_annotations:
        visual_dataset = VisualDataset(args.image_dir, args.coco_annotations)
        visual_dataset = _subset_dataset(visual_dataset, args.image_limit)
        visual_result = generate_visual_embeddings(
            visual_dataset,
            clip_engine,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            skip_invalid=args.skip_invalid,
        )
        np.save(output_dir / "clip_image_embeddings.npy", visual_result["image_embeddings"])
        np.save(output_dir / "clip_text_from_image_captions.npy", visual_result["text_embeddings"])
        _write_jsonl(output_dir / "image_metadata.jsonl", visual_result["metadata"])

    if args.audio_cache_dir:
        audio_dataset = AudioDataset(
            cache_dir=args.audio_cache_dir,
            split=args.audio_split,
            target_sr=args.audio_target_sr,
            duration_sec=args.audio_duration_sec,
            dataset_name=args.audio_dataset_name,
        )
        audio_dataset = _subset_dataset(audio_dataset, args.audio_limit)
        audio_result = generate_audio_embeddings(
            audio_dataset,
            clap_engine,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            skip_invalid=args.skip_invalid,
        )
        np.save(output_dir / "clap_audio_embeddings.npy", audio_result["audio_embeddings"])
        np.save(output_dir / "clap_text_from_audio_captions.npy", audio_result["text_embeddings"])
        _write_jsonl(output_dir / "audio_metadata.jsonl", audio_result["metadata"])

    run_config = {
        "clip_model": args.clip_model,
        "clap_model": args.clap_model,
        "device": args.device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "audio_target_sr": args.audio_target_sr,
        "audio_duration_sec": args.audio_duration_sec,
        "skip_invalid": args.skip_invalid,
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as file_obj:
        json.dump(run_config, file_obj, indent=2, ensure_ascii=True)

    print(f"\nSUCCESS! Embeddings and metadata written to: {output_dir}")


if __name__ == "__main__":
    main()