# Appendix — How to Run Everything From Scratch

This guide walks through setting up the entire project on a fresh machine without our external SSD.

## 1. Clone and Install

```bash
git clone https://github.com/Rochan2003/cross-modal-etl.git
cd cross-modal-etl
pip install -e ".[dev]"
```

This installs all dependencies including PyTorch, CLIP/CLAP models (via transformers), FAISS, Flask, FastAPI, and test tools.

## 2. Download Datasets

### MS COCO 2017

Download the training images and caption annotations:

- **Images**: https://cocodataset.org/#download → 2017 Train images (18GB)
- **Annotations**: https://cocodataset.org/#download → 2017 Train/Val annotations

After downloading, you should have:
```
/path/to/coco/
├── train2017/           # ~118K .jpg images
└── annotations/
    └── captions_train2017.json
```

### AudioCaps and Clotho

These are loaded automatically from HuggingFace on first run. The embedding pipeline downloads and caches them. If you want to pre-download:

- **AudioCaps**: https://huggingface.co/datasets/d0rj/audiocaps
- **Clotho**: https://huggingface.co/datasets/CLAPv2/Clotho

## 3. Generate Embeddings (ETL Pipeline)

Run the full pipeline to encode images and audio into vector embeddings:

```bash
# Images only
python generate_embeddings.py \
    --output-dir ./embeddings \
    --image-dir /path/to/coco/train2017 \
    --coco-annotations /path/to/coco/annotations/captions_train2017.json \
    --device cpu \
    --batch-size 32 \
    --skip-invalid

# Audio (AudioCaps)
python generate_embeddings.py \
    --output-dir ./embeddings \
    --audio-cache-dir ./hf_cache \
    --audio-dataset-name d0rj/audiocaps \
    --audio-split train \
    --device cpu \
    --batch-size 32 \
    --skip-invalid

# Audio (Clotho) — append to same output dir
python generate_embeddings.py \
    --output-dir ./embeddings \
    --audio-cache-dir ./hf_cache \
    --audio-dataset-name CLAPv2/Clotho \
    --audio-split train \
    --device cpu \
    --batch-size 32 \
    --skip-invalid
```

Use `--device mps` on Apple Silicon or `--device cuda` on NVIDIA GPUs for faster encoding.

After this, your embeddings directory should contain:
```
embeddings/
├── clip_image_embeddings.npy
├── clip_text_from_image_captions.npy
├── clap_audio_embeddings.npy
├── clap_text_from_audio_captions.npy
├── image_metadata.jsonl
├── audio_metadata.jsonl
└── run_config.json
```

## 4. Run Evaluation

```bash
python -m cross_modal.evaluation \
    --embeddings-dir ./embeddings \
    --top-k 10 \
    --sample-size 2000 \
    --gallery-size 5000
```

This evaluates text→image, text→audio, and image↔audio caption bridging, printing Recall@K, MRR, and latency. The `--gallery-size 5000` flag subsamples the gallery for comparison with published numbers.

## 5. Export Audio WAVs (for Demo)

The demo needs WAV files to play audio results in the browser:

```bash
python scripts/export_audio_wav.py --output-dir ./audio_wav
```

## 6. Launch the Demo

```bash
python demo.py \
    --embeddings-dir ./embeddings \
    --image-dir /path/to/coco/train2017 \
    --audio-wav-dir ./audio_wav
```

Open http://127.0.0.1:7860 in your browser. First search will take 30-60 seconds to load the text encoders.

## 7. Run Tests

```bash
pytest tests/ -v
```

All 22 tests should pass. Tests use mocked models so no GPU or downloaded weights are needed.

## 8. Run the API Server (Optional)

```bash
EMBEDDINGS_DIR=./embeddings uvicorn cross_modal.api:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health` — index sizes and status
- `GET /search?query=dog+playing&top_k=10` — text search
