from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from cross_modal.retrieval import CrossModalRetriever, DEFAULT_EMBEDDINGS_DIR


class HealthResponse(BaseModel):
    status: str
    embeddings_dir: str
    image_index_size: int = Field(..., description="Number of image vectors in the index")
    audio_index_size: int = Field(..., description="Number of audio vectors in the index")


class SearchResponse(BaseModel):
    query: str
    modality: str = "text"
    top_k: int
    image_results: list[Dict[str, Any]]
    audio_results: list[Dict[str, Any]]


_retriever_singleton: Optional[CrossModalRetriever] = None


def get_retriever() -> CrossModalRetriever:
    global _retriever_singleton
    if _retriever_singleton is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    return _retriever_singleton


def create_app(retriever: Optional[CrossModalRetriever] = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        global _retriever_singleton
        if retriever is not None:
            _retriever_singleton = retriever
        else:
            path = os.environ.get("EMBEDDINGS_DIR", DEFAULT_EMBEDDINGS_DIR)
            r = CrossModalRetriever(embeddings_dir=path)
            r.load_all()
            _retriever_singleton = r
        yield
        _retriever_singleton = None

    app = FastAPI(title="Cross-modal retrieval", version="0.1.0", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        r = get_retriever()
        return HealthResponse(
            status="ok",
            embeddings_dir=str(r.embeddings_dir),
            image_index_size=r.image_index.size,
            audio_index_size=r.audio_index.size,
        )

    @app.get("/search", response_model=SearchResponse)
    def search(
        query: str = Query(..., min_length=1, description="Natural language query"),
        top_k: int = Query(10, ge=1, le=100),
        r: CrossModalRetriever = Depends(get_retriever),
    ) -> SearchResponse:
        payload = r.search(query, top_k=top_k)
        return SearchResponse(**payload)

    return app


app = create_app()
