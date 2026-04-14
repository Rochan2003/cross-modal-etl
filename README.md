# Cross-Modal ETL Pipeline for Semantic Audio-Visual Retrieval

A complete Extract-Transform-Load pipeline that converts raw images (MS COCO) and audio clips (AudioCaps, Clotho) into L2-normalized vector embeddings, stores them in FAISS indexes, and supports cross-modal semantic search across text, image, and audio modalities.

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

**Cross-modal search** is achieved through two mechanisms:
- **Direct encoding**: Text queries are encoded by both CLIP (768-dim) and CLAP (512-dim) to search image and audio indexes directly.
- **Caption bridging**: Image queries retrieve similar images, then their captions are encoded via CLAP to find related audio (and vice versa for audio-to-image search).

## Project Structure

```
cross-modal-etl/
├── cross_modal/               # Core Python package
│   ├── __init__.py            # Package exports
│   ├── ingestion.py           # Extract: COCO + AudioCaps/Clotho data loaders
│   ├── embedding.py           # Transform: CLIP (768-d) + CLAP (512-d) encoders
│   ├── storage.py             # Load: save/load .npy, .jsonl, .json files
│   ├── vector_store.py        # FAISS inner-product index wrapper
│   ├── retrieval.py           # Cross-modal search orchestrator
│   ├── evaluation.py          # Recall@K, MRR with random baseline comparison
│   └── api.py                 # FastAPI endpoints (/health, /search)
│
├── tests/                     # Unit tests (pytest)
│   ├── test_ingestion.py      # Audio normalization, silence filtering
│   ├── test_embedding.py      # CLIP/CLAP shape, L2 norm, non-degeneracy
│   ├── test_storage.py        # Roundtrip persistence for all formats
│   ├── test_vector_store.py   # FAISS index correctness, edge cases
│   ├── test_retrieval.py      # End-to-end retrieval with mock indexes
│   └── test_api.py            # FastAPI endpoint contracts
│
├── scripts/                   # One-off utility scripts
│   ├── download_audio.py      # Download AudioCaps from Hugging Face
│   ├── export_audio_wav.py    # Export audio to WAV files for demo playback
│   └── validate_embeddings.py # Validate generated embeddings (norms, alignment)
│
├── generate_embeddings.py     # Main CLI: run the full ETL pipeline
├── demo.py                    # Interactive Flask demo (text/image/audio search)
├── pyproject.toml             # Package configuration
└── requirements.txt           # Python dependencies
```

## Modules

| Module | Role | Key Classes / Functions |
|--------|------|------------------------|
| `ingestion.py` | **Extract** — Loads raw data into normalized tensors | `VisualDataset` (COCO, 224x224, CLIP normalization), `AudioDataset` (HF datasets, 48kHz, 10s, silence filtering) |
| `embedding.py` | **Transform** — Encodes tensors into vector embeddings | `CLIPEmbeddingEngine` (768-dim), `CLAPEmbeddingEngine` (512-dim) |
| `storage.py` | **Load** — Persists embeddings and metadata to disk | `save_embeddings`, `load_embeddings`, `save_metadata`, `save_run_config` |
| `vector_store.py` | Builds FAISS indexes from saved files | `FaissIPIndex` (exact or HNSW), `EmbeddingBundle`, `build_image_index`, `build_audio_index` |
| `retrieval.py` | Orchestrates cross-modal search | `CrossModalRetriever.search()` (text), `.search_by_image()`, `.search_by_audio()` |
| `evaluation.py` | Measures retrieval quality against a random baseline | `recall_at_k`, `evaluate_modality`, `run_full_evaluation` |
| `api.py` | Serves retrieval over HTTP | FastAPI app with `/health` and `/search` endpoints |

## Quick Start

### 1. Install

```bash
git clone https://github.com/Rochan2003/cross-modal-etl.git
cd cross-modal-etl
pip install -e ".[dev]"
```

### 2. Generate Embeddings (ETL Pipeline)

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

This produces 7 files in the output directory:
- `clip_image_embeddings.npy` — CLIP image vectors (N x 768)
- `clip_text_from_image_captions.npy` — CLIP text vectors from image captions
- `clap_audio_embeddings.npy` — CLAP audio vectors (M x 512)
- `clap_text_from_audio_captions.npy` — CLAP text vectors from audio captions
- `image_metadata.jsonl` / `audio_metadata.jsonl` — Per-sample metadata
- `run_config.json` — Configuration snapshot for reproducibility

### 3. Run Evaluation

```bash
python -m cross_modal.evaluation \
    --embeddings-dir ./embeddings \
    --top-k 10 \
    --sample-size 2000
```

Outputs Recall@1, Recall@5, Recall@10, MRR, and latency for both modalities, compared against a random retrieval baseline.

### 4. Run Tests

```bash
pytest tests/ -v
```

### 5. Launch Interactive Demo

```bash
python demo.py
# Open http://127.0.0.1:5001 in your browser
```

The demo supports three search modes:
- **Text query** — type a natural language description to find matching images and audio
- **Image upload** — drag and drop or upload an image to find similar images, related text, and audio
- **Audio upload** — upload an audio clip to find similar sounds, related text, and images

## Models and Embedding Dimensions

| Model | Hugging Face ID | Output Dimension | Modalities |
|-------|----------------|-----------------|------------|
| CLIP ViT-L/14 | `openai/clip-vit-large-patch14` | **768** | Image + Text |
| CLAP (Large) | `laion/larger_clap_music_and_speech` | **512** | Audio + Text |

Both models produce L2-normalized embeddings, enabling cosine similarity via FAISS inner-product search.

## Datasets

| Dataset | Modality | Size | Source |
|---------|----------|------|--------|
| MS COCO 2017 | Image + Caption | ~118K unique images, ~591K captions | [cocodataset.org](https://cocodataset.org) |
| AudioCaps | Audio + Caption | ~3,985 clips (test split) | [Hugging Face](https://huggingface.co/datasets/d0rj/audiocaps) |
| Clotho | Audio + Caption | ~3,837 clips (train split) | [Hugging Face](https://huggingface.co/datasets/CLAPv2/Clotho) |

## References

1. Radford, A. et al. "Learning Transferable Visual Models From Natural Language Supervision." (CLIP), OpenAI, 2021.
2. Wu, Y. et al. "Large-Scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation." (CLAP), LAION, 2023.
3. Johnson, J. et al. "Billion-scale similarity search with GPUs." (FAISS), Meta AI, 2019.
4. Lin, T.-Y. et al. "Microsoft COCO: Common Objects in Context." 2014.
5. Kim, C. et al. "AudioCaps: Generating Captions for Audios in The Wild." 2019.
6. Drossos, K. et al. "Clotho: An Audio Captioning Dataset." 2020.
