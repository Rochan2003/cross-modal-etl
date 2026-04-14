import json
from pathlib import Path

import numpy as np

from cross_modal.retrieval import CrossModalRetriever


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_cross_modal_retriever_search_without_loading_encoders(tmp_path: Path) -> None:
    # create small fake embeddings and test that search returns correct IDs
    rng = np.random.default_rng(42)
    dim = 8
    img_emb = rng.standard_normal((5, dim)).astype(np.float32)
    aud_emb = rng.standard_normal((4, dim)).astype(np.float32)
    np.save(tmp_path / "clip_image_embeddings.npy", img_emb)
    np.save(tmp_path / "clap_audio_embeddings.npy", aud_emb)
    _write_jsonl(
        tmp_path / "image_metadata.jsonl",
        [{"id": str(i), "caption": f"i{i}", "modality": "image"} for i in range(5)],
    )
    _write_jsonl(
        tmp_path / "audio_metadata.jsonl",
        [{"id": str(i), "caption": f"a{i}", "modality": "audio"} for i in range(4)],
    )

    r = CrossModalRetriever(tmp_path)
    r.load_indexes()
    clip_q = img_emb[2]
    clap_q = aud_emb[1]
    # skip loading real encoders — just inject the vectors directly
    r.encode_query = lambda q: (clip_q, clap_q)  # type: ignore[method-assign]

    out = r.search("ignored", top_k=2)
    assert out["query"] == "ignored"
    assert out["top_k"] == 2
    assert out["image_results"][0]["metadata"]["id"] == "2"
    assert out["audio_results"][0]["metadata"]["id"] == "1"
