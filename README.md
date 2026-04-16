# Cross-Modal ETL Pipeline for Semantic Audio-Visual Retrieval

ETL pipeline that encodes images (MS COCO) and audio (AudioCaps, Clotho) into vector embeddings using CLIP and CLAP, stores them in FAISS indexes, and supports cross-modal search across text, image, and audio.

## Architecture

```
Raw Data                   Embedding Models              Vector Store           Retrieval
─────────                  ────────────────              ────────────           ─────────
MS COCO images ──┐                                    ┌─ FAISS (image) ──┐
                 ├─ ingestion.py ─► embedding.py ─┬──►│                  ├─► retrieval.py ─► API / Demo
AudioCaps/Clotho ┘   (Extract)      (Transform)  │   └─ FAISS (audio) ──┘     (Search)
                                                  │
                                        storage.py (Load)
                                        .npy + .jsonl files
```

**Cross-modal search** works through two mechanisms:
- **Direct encoding**: Text queries are encoded by both CLIP (768-dim) and CLAP (512-dim) to search image and audio indexes directly.
- **Caption bridging**: Image queries retrieve similar images, then their captions are encoded via CLAP to find related audio (and vice versa for audio-to-image).

## Project Structure

```
cross-modal-etl/
├── cross_modal/               # Core Python package
│   ├── __init__.py
│   ├── ingestion.py           # Extract: COCO + AudioCaps/Clotho data loaders
│   ├── embedding.py           # Transform: CLIP (768-d) + CLAP (512-d) encoders
│   ├── storage.py             # Load: save/load .npy, .jsonl, .json files
│   ├── vector_store.py        # FAISS inner-product index wrapper
│   ├── retrieval.py           # Cross-modal search orchestrator
│   ├── evaluation.py          # Recall@K, MRR evaluation
│   └── api.py                 # FastAPI endpoints (/health, /search)
│
├── tests/                     # Unit tests (pytest)
│   ├── test_ingestion.py
│   ├── test_embedding.py
│   ├── test_storage.py
│   ├── test_vector_store.py
│   ├── test_retrieval.py
│   └── test_api.py
│
├── scripts/
│   ├── download_audio.py      # Download AudioCaps from Hugging Face
│   ├── export_audio_wav.py    # Export audio to WAV for demo playback
│   └── validate_embeddings.py # Check embedding shapes and norms
│
├── generate_embeddings.py     # Run the full ETL pipeline
├── demo.py                    # Flask demo (text/image/audio search)
├── pyproject.toml
└── requirements.txt
```

## Modules

| Module | Role | Key Classes / Functions |
|--------|------|------------------------|
| `ingestion.py` | **Extract** — Load raw data | `VisualDataset` (COCO, 224x224), `AudioDataset` (HF, 48kHz, 10s) |
| `embedding.py` | **Transform** — Encode to vectors | `CLIPEmbeddingEngine` (768-dim), `CLAPEmbeddingEngine` (512-dim) |
| `storage.py` | **Load** — Save to disk | `save_embeddings`, `save_metadata`, `save_run_config` |
| `vector_store.py` | Build FAISS indexes | `FaissIPIndex` (exact or HNSW), `EmbeddingBundle` |
| `retrieval.py` | Cross-modal search | `CrossModalRetriever.search()`, `.search_by_image()`, `.search_by_audio()` |
| `evaluation.py` | Measure retrieval quality | `recall_at_k`, `evaluate_modality`, `evaluate_caption_bridging` |
| `api.py` | HTTP API | FastAPI `/health` and `/search` endpoints |

## Quick Start

### 1. Install

```bash
git clone https://github.com/Rochan2003/cross-modal-etl.git
cd cross-modal-etl
pip install -e ".[dev]"
```

### 2. Generate Embeddings

```bash
python generate_embeddings.py \
    --output-dir ./embeddings \
    --image-dir /path/to/coco/train2017 \
    --coco-annotations /path/to/coco/annotations/captions_train2017.json \
    --audio-cache-dir /path/to/audiocaps \
    --device cpu \
    --batch-size 32 \
    --skip-invalid
```

Produces 7 files:
- `clip_image_embeddings.npy` — CLIP image vectors (N x 768)
- `clip_text_from_image_captions.npy` — CLIP text vectors from image captions
- `clap_audio_embeddings.npy` — CLAP audio vectors (M x 512)
- `clap_text_from_audio_captions.npy` — CLAP text vectors from audio captions
- `image_metadata.jsonl` / `audio_metadata.jsonl` — Per-sample metadata
- `run_config.json` — Config snapshot for reproducibility

### 3. Run Evaluation

```bash
python -m cross_modal.evaluation \
    --embeddings-dir ./embeddings \
    --top-k 10 \
    --sample-size 2000 \
    --gallery-size 5000
```

Outputs Recall@1, Recall@5, Recall@10, MRR, and latency for text→image, text→audio, and image↔audio caption bridging. Use `--gallery-size 5000` to subsample the gallery for comparison with published numbers.

### 4. Run Tests

```bash
pytest tests/ -v
```

### 5. Launch Demo

```bash
python demo.py
# Open http://127.0.0.1:7860 in your browser
```

Three search modes:
- **Text query** — type a description to find matching images and audio
- **Image upload** — drag and drop an image to find similar images, text, and audio
- **Audio upload** — upload audio to find similar sounds, text, and images

## Models

| Model | Hugging Face ID | Dimension | Modalities |
|-------|----------------|-----------|------------|
| CLIP ViT-L/14 | `openai/clip-vit-large-patch14` | 768 | Image + Text |
| CLAP (Large) | `laion/larger_clap_music_and_speech` | 512 | Audio + Text |

Both produce L2-normalized embeddings — cosine similarity via FAISS inner-product search.

## Datasets

| Dataset | Modality | Size | Source |
|---------|----------|------|--------|
| MS COCO 2017 | Image + Caption | ~118K images, ~591K captions | [cocodataset.org](https://cocodataset.org) |
| AudioCaps | Audio + Caption | ~3,985 clips | [HuggingFace](https://huggingface.co/datasets/d0rj/audiocaps) |
| Clotho | Audio + Caption | ~3,837 clips | [HuggingFace](https://huggingface.co/datasets/CLAPv2/Clotho) |

## References

1. Radford et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML, 2021.
2. Wu et al. "Large-Scale Contrastive Language-Audio Pretraining." ICASSP, 2023.
3. Johnson et al. "Billion-scale similarity search with GPUs." (FAISS), 2019.
4. Lin et al. "Microsoft COCO: Common Objects in Context." 2014.
5. Kim et al. "AudioCaps: Generating Captions for Audios in The Wild." 2019.
6. Drossos et al. "Clotho: An Audio Captioning Dataset." 2020.
