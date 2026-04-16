from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from cross_modal.api import create_app


# fake index so we don't need real FAISS data for API tests
class _FakeIndex:
    def __init__(self, n: int) -> None:
        self.size = n


@pytest.fixture
def fake_retriever() -> MagicMock:
    # mock retriever with canned search results
    r = MagicMock()
    r.embeddings_dir = Path("/tmp/fake_embeddings")
    r.image_index = _FakeIndex(42)
    r.audio_index = _FakeIndex(7)
    r.search.return_value = {
        "query": "dog",
        "top_k": 2,
        "image_results": [
            {
                "rank": 1,
                "score": 0.9,
                "metadata": {"id": "1", "caption": "a dog", "modality": "image"},
            }
        ],
        "audio_results": [
            {
                "rank": 1,
                "score": 0.8,
                "metadata": {"id": "audio_0", "caption": "barking", "modality": "audio"},
            }
        ],
    }
    return r


def test_health_and_search(fake_retriever: MagicMock) -> None:
    app = create_app(fake_retriever)
    with TestClient(app) as client:
        h = client.get("/health")
        assert h.status_code == 200
        body = h.json()
        assert body["status"] == "ok"
        assert body["image_index_size"] == 42
        assert body["audio_index_size"] == 7

        s = client.get("/search", params={"query": "dog", "top_k": 2})
        assert s.status_code == 200
        data = s.json()
        assert data["query"] == "dog"
        assert len(data["image_results"]) == 1
        assert data["image_results"][0]["score"] == 0.9
    fake_retriever.search.assert_called_once()


def test_search_rejects_empty_query(fake_retriever: MagicMock) -> None:
    # empty query should get 422 validation error
    app = create_app(fake_retriever)
    with TestClient(app) as client:
        r = client.get("/search", params={"query": ""})
        assert r.status_code == 422
