"""Interactive cross-modal retrieval demo.

Uses Flask (single-threaded) to avoid segfaults from torch/transformers
on macOS Apple Silicon.

Launch:
    python demo.py [--embeddings-dir ...] [--image-dir ...] [--audio-wav-dir ...]

Search modes:
    Text  → Images + Audio
    Image → Audio + Text + Similar Images
    Audio → Images + Text + Similar Audio
"""
from __future__ import annotations

# ---- Environment guards: MUST come before any torch/numpy/transformers import ----
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import io
import tempfile
import threading
import traceback
from pathlib import Path

import numpy as np
import torch
from flask import Flask, Response, jsonify, request, send_file
from PIL import Image
from torchvision import transforms

from cross_modal.retrieval import CrossModalRetriever, DEFAULT_EMBEDDINGS_DIR

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_IMAGE_DIR = "/Volumes/Samsung_T7/dataset/coco/images/train2017"
DEFAULT_AUDIO_WAV_DIR = "/Volumes/Samsung_T7/dataset/audio_wav"

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
retriever: CrossModalRetriever | None = None
image_dir: Path = Path(DEFAULT_IMAGE_DIR)
audio_wav_dir: Path = Path(DEFAULT_AUDIO_WAV_DIR)

_inference_lock = threading.Lock()
_encoders_loaded = False

# CLIP image preprocessing (must match what was used for embeddings)
_clip_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    ),
])


def _ensure_encoders():
    global _encoders_loaded
    if _encoders_loaded:
        return
    with _inference_lock:
        if _encoders_loaded:
            return
        print("[demo] Loading text encoders (lazy, first request)...")
        retriever.load_encoders()
        print("[demo] Encoders ready.")
        _encoders_loaded = True


# ---------------------------------------------------------------------------
# HTML frontend
# ---------------------------------------------------------------------------
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cross-Modal Semantic Retrieval</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0f0f0f; color: #e0e0e0; min-height: 100vh; }
  .header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 2rem; text-align: center; border-bottom: 1px solid #333; }
  .header h1 { font-size: 1.8rem; color: #fff; margin-bottom: 0.3rem; }
  .header p { color: #aaa; font-size: 0.95rem; }

  /* Mode selector */
  .mode-bar { display: flex; gap: 8px; max-width: 800px; margin: 1.2rem auto 0; padding: 0 1rem;
              justify-content: center; }
  .mode-btn { padding: 8px 20px; border: 1px solid #444; border-radius: 20px; background: #1a1a1a;
              color: #aaa; cursor: pointer; font-size: 0.9rem; transition: all 0.2s; }
  .mode-btn.active { background: #5b8def; color: #fff; border-color: #5b8def; }
  .mode-btn:hover { border-color: #5b8def; }

  /* Search areas */
  .search-area { max-width: 800px; margin: 1rem auto; padding: 0 1rem; }
  .search-area.hidden { display: none; }
  .search-bar { display: flex; gap: 10px; }
  .search-bar input[type=text] { flex: 1; padding: 12px 16px; font-size: 1rem; border: 1px solid #444;
                       border-radius: 8px; background: #1a1a1a; color: #fff; outline: none; }
  .search-bar input[type=text]:focus { border-color: #5b8def; }
  .search-bar select { padding: 12px; font-size: 1rem; border: 1px solid #444;
                        border-radius: 8px; background: #1a1a1a; color: #fff; }
  .search-bar button { padding: 12px 24px; font-size: 1rem; border: none; border-radius: 8px;
                        background: #5b8def; color: #fff; cursor: pointer; font-weight: 600; }
  .search-bar button:hover { background: #4a7de0; }
  .search-bar button:disabled { background: #555; cursor: wait; }

  /* Upload zone */
  .upload-zone { border: 2px dashed #444; border-radius: 12px; padding: 2rem; text-align: center;
                 cursor: pointer; transition: border-color 0.2s; margin-bottom: 10px; }
  .upload-zone:hover { border-color: #5b8def; }
  .upload-zone.dragover { border-color: #5b8def; background: #1a1a2e; }
  .upload-zone input[type=file] { display: none; }
  .upload-zone .icon { font-size: 2rem; margin-bottom: 0.5rem; }
  .upload-zone .label { color: #aaa; }
  .upload-zone .filename { color: #5b8def; margin-top: 0.5rem; font-size: 0.9rem; }
  .preview-row { display: flex; gap: 10px; align-items: center; margin-bottom: 10px; }
  .preview-row img { max-height: 80px; border-radius: 6px; }
  .preview-row audio { max-width: 300px; }

  /* Results tabs */
  .tabs { display: flex; gap: 0; max-width: 1200px; margin: 1rem auto 0; padding: 0 1rem; }
  .tab { padding: 10px 24px; cursor: pointer; border: 1px solid #333; border-bottom: none;
         border-radius: 8px 8px 0 0; background: #1a1a1a; color: #aaa; font-weight: 500; }
  .tab.active { background: #222; color: #fff; border-color: #555; }
  .results { max-width: 1200px; margin: 0 auto; padding: 0 1rem 2rem; }
  .results-panel { display: none; background: #222; border: 1px solid #555;
                   border-radius: 0 8px 8px 8px; padding: 1.5rem; min-height: 200px; }
  .results-panel.active { display: block; }
  .status { text-align: center; color: #888; padding: 2rem; font-size: 1rem; }

  /* Image grid */
  .image-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 16px; }
  .image-card { background: #1a1a1a; border-radius: 8px; overflow: hidden; border: 1px solid #333; }
  .image-card img { width: 100%; height: 180px; object-fit: cover; }
  .image-card .info { padding: 10px; }
  .image-card .rank { font-weight: 700; color: #5b8def; }
  .image-card .score { color: #aaa; font-size: 0.85rem; }
  .image-card .caption { color: #ccc; font-size: 0.82rem; margin-top: 4px;
                          display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }

  /* Audio list */
  .audio-list { display: flex; flex-direction: column; gap: 12px; }
  .audio-card { display: flex; align-items: center; gap: 16px; background: #1a1a1a;
                border-radius: 8px; padding: 14px; border: 1px solid #333; }
  .audio-card .rank { font-size: 1.3rem; font-weight: 700; color: #5b8def; min-width: 36px; text-align: center; }
  .audio-card .details { flex: 1; }
  .audio-card .caption { color: #ccc; font-size: 0.9rem; }
  .audio-card .score { color: #888; font-size: 0.82rem; margin-top: 2px; }
  .audio-card audio { flex-shrink: 0; }

  /* Text list */
  .text-list { display: flex; flex-direction: column; gap: 8px; }
  .text-card { background: #1a1a1a; border-radius: 8px; padding: 14px; border: 1px solid #333; }
  .text-card .rank { font-weight: 700; color: #5b8def; }
  .text-card .score { color: #888; font-size: 0.82rem; }
  .text-card .text-content { color: #e0e0e0; margin-top: 6px; font-size: 0.95rem; line-height: 1.4; }
  .text-card .source { color: #666; font-size: 0.8rem; margin-top: 4px; }

  .metrics { max-width: 1200px; margin: 0.5rem auto; padding: 0 1rem;
             display: flex; gap: 1rem; color: #888; font-size: 0.85rem; }
  .loading-banner { display: none; text-align: center; color: #f0ad4e; padding: 1rem;
                     font-size: 0.95rem; background: #2a2a1a; border: 1px solid #554400;
                     border-radius: 8px; max-width: 800px; margin: 0.5rem auto; }
</style>
</head>
<body>
  <div class="header">
    <h1>Cross-Modal Semantic Retrieval</h1>
    <p>Search across <strong>images</strong>, <strong>audio</strong>, and <strong>text</strong> using any modality</p>
  </div>

  <div class="mode-bar">
    <div class="mode-btn active" onclick="setMode('text')">Text Query</div>
    <div class="mode-btn" onclick="setMode('image')">Image Query</div>
    <div class="mode-btn" onclick="setMode('audio')">Audio Query</div>
  </div>

  <!-- TEXT search -->
  <div class="search-area" id="text-search">
    <div class="search-bar">
      <input id="query" type="text" placeholder="e.g. a dog barking, thunderstorm, sunset over ocean..." autofocus />
      <select id="topk">
        <option value="5">Top 5</option>
        <option value="10" selected>Top 10</option>
        <option value="20">Top 20</option>
      </select>
      <button id="textSearchBtn" onclick="doTextSearch()">Search</button>
    </div>
  </div>

  <!-- IMAGE search -->
  <div class="search-area hidden" id="image-search">
    <div class="upload-zone" id="imageDropZone" onclick="document.getElementById('imageFile').click()">
      <input type="file" id="imageFile" accept="image/*" onchange="onImageSelected(this)" />
      <div class="icon">🖼️</div>
      <div class="label">Drop an image here or click to upload</div>
      <div class="filename" id="imageFilename"></div>
    </div>
    <div class="preview-row" id="imagePreviewRow" style="display:none">
      <img id="imagePreview" />
      <select id="topk-img">
        <option value="5">Top 5</option>
        <option value="10" selected>Top 10</option>
        <option value="20">Top 20</option>
      </select>
      <button id="imgSearchBtn" onclick="doImageSearch()">Search with Image</button>
    </div>
  </div>

  <!-- AUDIO search -->
  <div class="search-area hidden" id="audio-search">
    <div class="upload-zone" id="audioDropZone" onclick="document.getElementById('audioFile').click()">
      <input type="file" id="audioFile" accept="audio/*" onchange="onAudioSelected(this)" />
      <div class="icon">🔊</div>
      <div class="label">Drop an audio file here or click to upload</div>
      <div class="filename" id="audioFilename"></div>
    </div>
    <div class="preview-row" id="audioPreviewRow" style="display:none">
      <audio id="audioPreview" controls></audio>
      <select id="topk-aud">
        <option value="5">Top 5</option>
        <option value="10" selected>Top 10</option>
        <option value="20">Top 20</option>
      </select>
      <button id="audSearchBtn" onclick="doAudioSearch()">Search with Audio</button>
    </div>
  </div>

  <div class="loading-banner" id="loadingBanner">
    Loading models for first search... this may take 30-60 seconds.
  </div>
  <div class="metrics" id="metrics"></div>

  <div class="tabs" id="resultTabs">
    <div class="tab active" onclick="switchTab('images')">Images</div>
    <div class="tab" onclick="switchTab('audio')">Audio</div>
    <div class="tab" onclick="switchTab('text')" style="display:none" id="textTab">Text</div>
  </div>
  <div class="results">
    <div class="results-panel active" id="images-panel">
      <div class="status">Choose a search mode and enter a query</div>
    </div>
    <div class="results-panel" id="audio-panel">
      <div class="status">Choose a search mode and enter a query</div>
    </div>
    <div class="results-panel" id="text-panel">
      <div class="status">Text results appear for image/audio queries</div>
    </div>
  </div>

<script>
let firstSearch = true;
let currentMode = 'text';
let selectedImageFile = null;
let selectedAudioFile = null;

function setMode(mode) {
  currentMode = mode;
  document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.mode-btn').forEach(b => {
    if (b.textContent.toLowerCase().includes(mode)) b.classList.add('active');
  });
  document.getElementById('text-search').classList.toggle('hidden', mode !== 'text');
  document.getElementById('image-search').classList.toggle('hidden', mode !== 'image');
  document.getElementById('audio-search').classList.toggle('hidden', mode !== 'audio');
  // Show/hide text tab for non-text queries
  document.getElementById('textTab').style.display = (mode !== 'text') ? 'block' : 'none';
}

function switchTab(name) {
  const tabs = ['images', 'audio', 'text'];
  document.querySelectorAll('.tab').forEach((t, i) => {
    t.classList.toggle('active', tabs[i] === name);
  });
  tabs.forEach(t => {
    document.getElementById(t + '-panel').classList.toggle('active', t === name);
  });
}

// --- File selection ---
function onImageSelected(input) {
  if (input.files.length) {
    selectedImageFile = input.files[0];
    document.getElementById('imageFilename').textContent = selectedImageFile.name;
    document.getElementById('imagePreviewRow').style.display = 'flex';
    const url = URL.createObjectURL(selectedImageFile);
    document.getElementById('imagePreview').src = url;
  }
}
function onAudioSelected(input) {
  if (input.files.length) {
    selectedAudioFile = input.files[0];
    document.getElementById('audioFilename').textContent = selectedAudioFile.name;
    document.getElementById('audioPreviewRow').style.display = 'flex';
    const url = URL.createObjectURL(selectedAudioFile);
    document.getElementById('audioPreview').src = url;
  }
}

// --- Drag & drop ---
['imageDropZone', 'audioDropZone'].forEach(id => {
  const zone = document.getElementById(id);
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
      if (id === 'imageDropZone') {
        document.getElementById('imageFile').files = e.dataTransfer.files;
        onImageSelected(document.getElementById('imageFile'));
      } else {
        document.getElementById('audioFile').files = e.dataTransfer.files;
        onAudioSelected(document.getElementById('audioFile'));
      }
    }
  });
});

document.getElementById('query').addEventListener('keydown', e => {
  if (e.key === 'Enter') doTextSearch();
});

// --- Search functions ---
function showLoading(btn) {
  btn.disabled = true; btn.textContent = 'Searching...';
  if (firstSearch) document.getElementById('loadingBanner').style.display = 'block';
}
function hideLoading(btn, label) {
  btn.disabled = false; btn.textContent = label;
  firstSearch = false;
  document.getElementById('loadingBanner').style.display = 'none';
}

async function doTextSearch() {
  const query = document.getElementById('query').value.trim();
  if (!query) return;
  const topk = document.getElementById('topk').value;
  const btn = document.getElementById('textSearchBtn');
  showLoading(btn);
  const t0 = performance.now();
  try {
    const res = await fetch('/api/search/text?query=' + encodeURIComponent(query) + '&top_k=' + topk);
    if (!res.ok) throw new Error((await res.json()).error || res.statusText);
    const data = await res.json();
    showResults(data, t0);
  } catch (err) {
    document.getElementById('metrics').textContent = 'Error: ' + err.message;
  }
  hideLoading(btn, 'Search');
}

async function doImageSearch() {
  if (!selectedImageFile) return;
  const topk = document.getElementById('topk-img').value;
  const btn = document.getElementById('imgSearchBtn');
  showLoading(btn);
  const t0 = performance.now();
  try {
    const form = new FormData();
    form.append('file', selectedImageFile);
    form.append('top_k', topk);
    const res = await fetch('/api/search/image', { method: 'POST', body: form });
    if (!res.ok) throw new Error((await res.json()).error || res.statusText);
    const data = await res.json();
    showResults(data, t0);
  } catch (err) {
    document.getElementById('metrics').textContent = 'Error: ' + err.message;
  }
  hideLoading(btn, 'Search with Image');
}

async function doAudioSearch() {
  if (!selectedAudioFile) return;
  const topk = document.getElementById('topk-aud').value;
  const btn = document.getElementById('audSearchBtn');
  showLoading(btn);
  const t0 = performance.now();
  try {
    const form = new FormData();
    form.append('file', selectedAudioFile);
    form.append('top_k', topk);
    const res = await fetch('/api/search/audio', { method: 'POST', body: form });
    if (!res.ok) throw new Error((await res.json()).error || res.statusText);
    const data = await res.json();
    showResults(data, t0);
  } catch (err) {
    document.getElementById('metrics').textContent = 'Error: ' + err.message;
  }
  hideLoading(btn, 'Search with Audio');
}

function showResults(data, t0) {
  const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
  const parts = [];
  if (data.image_results && data.image_results.length) parts.push(data.image_results.length + ' images');
  if (data.audio_results && data.audio_results.length) parts.push(data.audio_results.length + ' audio');
  if (data.text_results && data.text_results.length) parts.push(data.text_results.length + ' text');
  document.getElementById('metrics').textContent = parts.join(' + ') + ' in ' + elapsed + 's';

  renderImages(data.image_results || []);
  renderAudio(data.audio_results || []);
  renderText(data.text_results || []);

  // Show text tab if text results exist
  document.getElementById('textTab').style.display = (data.text_results && data.text_results.length) ? 'block' : 'none';
}

function renderImages(results) {
  const panel = document.getElementById('images-panel');
  if (!results.length) { panel.innerHTML = '<div class="status">No image results</div>'; return; }
  let html = '<div class="image-grid">';
  for (const r of results) {
    html += '<div class="image-card">' +
      '<img src="/media/image/' + r.metadata.id + '" loading="lazy" onerror="this.style.display=\'none\'" />' +
      '<div class="info">' +
        '<span class="rank">#' + r.rank + '</span> ' +
        '<span class="score">score: ' + r.score.toFixed(4) + '</span>' +
        '<div class="caption">' + esc(r.metadata.caption || '') + '</div>' +
      '</div></div>';
  }
  panel.innerHTML = html + '</div>';
}

function renderAudio(results) {
  const panel = document.getElementById('audio-panel');
  if (!results.length) { panel.innerHTML = '<div class="status">No audio results</div>'; return; }
  let html = '<div class="audio-list">';
  for (const r of results) {
    html += '<div class="audio-card">' +
      '<div class="rank">#' + r.rank + '</div>' +
      '<div class="details">' +
        '<div class="caption">' + esc(r.metadata.caption || '') + '</div>' +
        '<div class="score">score: ' + r.score.toFixed(4) + '</div>' +
      '</div>' +
      '<audio controls preload="none" src="/media/audio/' + r.faiss_idx + '"></audio>' +
    '</div>';
  }
  panel.innerHTML = html + '</div>';
}

function renderText(results) {
  const panel = document.getElementById('text-panel');
  if (!results.length) { panel.innerHTML = '<div class="status">No text results</div>'; return; }
  let html = '<div class="text-list">';
  for (const r of results) {
    html += '<div class="text-card">' +
      '<span class="rank">#' + r.rank + '</span> ' +
      '<span class="score">score: ' + r.score.toFixed(4) + '</span>' +
      '<div class="text-content">' + esc(r.text || '') + '</div>' +
      '<div class="source">from ' + esc(r.source || '') + '</div>' +
    '</div>';
  }
  panel.innerHTML = html + '</div>';
}

function esc(s) {
  const d = document.createElement('div'); d.textContent = s; return d.innerHTML;
}
</script>
</body>
</html>
"""


def create_app(args: argparse.Namespace) -> Flask:
    global retriever, image_dir, audio_wav_dir

    image_dir = Path(args.image_dir)
    audio_wav_dir = Path(args.audio_wav_dir)

    print("[demo] Loading FAISS indexes...")
    retriever = CrossModalRetriever(
        embeddings_dir=args.embeddings_dir,
        device="cpu",
        use_fp16=False,
    )
    retriever.load_indexes()
    print(f"[demo] Image index: {retriever.image_index.size} vectors")
    print(f"[demo] Audio index: {retriever.audio_index.size} vectors")

    wav_count = len(list(audio_wav_dir.glob("*.wav"))) if audio_wav_dir.is_dir() else 0
    print(f"[demo] Audio WAV files found: {wav_count}")
    print("[demo] Text encoders will load on first search request.")

    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

    @app.route("/")
    def index():
        return Response(HTML_PAGE, mimetype="text/html")

    # --- Text search ---
    @app.route("/api/search/text")
    def search_text():
        query = request.args.get("query", "").strip()
        if not query:
            return jsonify({"error": "query parameter is required"}), 400
        top_k = _parse_topk(request.args.get("top_k", "10"))
        try:
            _ensure_encoders()
            with _inference_lock:
                results = retriever.search(query, top_k=top_k)
            return jsonify(results)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    # --- Image search ---
    @app.route("/api/search/image", methods=["POST"])
    def search_image():
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        top_k = _parse_topk(request.form.get("top_k", "10"))
        try:
            _ensure_encoders()
            # Read and preprocess image
            img = Image.open(io.BytesIO(file.read())).convert("RGB")
            pixel_values = _clip_transform(img).unsqueeze(0)  # (1, 3, 224, 224)
            with _inference_lock:
                clip_vec = retriever.encode_image(pixel_values)
                results = retriever.search_by_image(clip_vec, top_k=top_k)
            return jsonify(results)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    # --- Audio search ---
    @app.route("/api/search/audio", methods=["POST"])
    def search_audio():
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        top_k = _parse_topk(request.form.get("top_k", "10"))
        try:
            _ensure_encoders()
            # Read audio with soundfile or librosa
            import soundfile as sf
            audio_bytes = file.read()
            waveform, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            # Convert to mono if stereo
            if waveform.ndim == 2:
                waveform = waveform.mean(axis=1)
            # Resample to 48kHz if needed
            target_sr = 48000
            if sr != target_sr:
                import librosa
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
            # Pad or trim to 10 seconds
            max_len = target_sr * 10
            if len(waveform) > max_len:
                waveform = waveform[:max_len]
            else:
                waveform = np.pad(waveform, (0, max_len - len(waveform)))
            # Shape: (1, samples) for CLAP
            waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)
            with _inference_lock:
                clap_vec = retriever.encode_audio(waveform_tensor)
                results = retriever.search_by_audio(clap_vec, top_k=top_k)
            return jsonify(results)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    # --- Media serving ---
    @app.route("/media/image/<image_id>")
    def serve_image(image_id):
        padded = image_id.zfill(12)
        path = image_dir / f"{padded}.jpg"
        if not path.is_file():
            return Response("Image not found", status=404)
        return send_file(path, mimetype="image/jpeg")

    @app.route("/media/audio/<int:faiss_idx>")
    def serve_audio(faiss_idx):
        path = audio_wav_dir / f"{faiss_idx}.wav"
        if not path.is_file():
            return Response("Audio not found", status=404)
        return send_file(path, mimetype="audio/wav")

    @app.route("/api/health")
    def health():
        return jsonify({
            "status": "ok",
            "image_index_size": retriever.image_index.size,
            "audio_index_size": retriever.audio_index.size,
            "encoders_loaded": _encoders_loaded,
        })

    return app


def _parse_topk(val: str) -> int:
    try:
        return max(1, min(100, int(val)))
    except ValueError:
        return 10


def main():
    parser = argparse.ArgumentParser(description="Cross-modal retrieval demo")
    parser.add_argument("--embeddings-dir", default=DEFAULT_EMBEDDINGS_DIR)
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--audio-wav-dir", default=DEFAULT_AUDIO_WAV_DIR)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    app = create_app(args)

    print(f"\n[demo] Server running at http://{args.host}:{args.port}")
    print("[demo] Search modes: Text | Image | Audio")
    print("[demo] Press Ctrl+C to stop.\n")

    app.run(
        host=args.host,
        port=args.port,
        debug=False,
        threaded=False,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
