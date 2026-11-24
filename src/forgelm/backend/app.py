"""FastAPI application for ForgeLM inference and text generation services."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from forgelm.backend.schemas import (
    ChatRequest,
    ChatResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfoResponse,
)
from forgelm.backend.service import model_service
from forgelm.utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan context manager for startup and shutdown hooks."""
    logger.info("Initializing ForgeLM backend service...")
    if not model_service.is_loaded:
        model_service.load()
    yield
    logger.info("ForgeLM backend service shutting down.")


app = FastAPI(
    title="ForgeLM Inference API",
    description="REST & SSE Streaming API for ForgeLM causal language model.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for web frontends (e.g. React / Vite)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Info"])
def root() -> dict[str, str]:
    """Root metadata endpoint."""
    return {
        "service": "ForgeLM Inference API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def get_health() -> HealthResponse:
    """Telemetry and GPU status check."""
    return model_service.get_health()


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Monitoring"])
def get_model_info() -> ModelInfoResponse:
    """Retrieve model architecture specifications."""
    try:
        return model_service.get_model_info()
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))


@app.post("/generate", response_model=GenerateResponse, tags=["Inference"])
def generate_text(req: GenerateRequest) -> GenerateResponse:
    """Generate text completion from a prompt."""
    try:
        return model_service.generate(req)
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/generate/stream", tags=["Inference"])
def stream_generate_text(req: GenerateRequest) -> StreamingResponse:
    """Stream generated tokens in Server-Sent Events (SSE) format."""

    def event_stream():
        try:
            for token in model_service.stream_generate(req):
                payload = json.dumps({"token": token, "done": False})
                yield f"data: {payload}\n\n"
            # Signal completion
            yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/chat", response_model=ChatResponse, tags=["Inference"])
def chat(req: ChatRequest) -> ChatResponse:
    """Generate assistant response from conversation messages."""
    try:
        return model_service.chat(req)
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/chat/stream", tags=["Inference"])
def stream_chat(req: ChatRequest) -> StreamingResponse:
    """Stream assistant response tokens in Server-Sent Events (SSE) format."""

    def event_stream():
        try:
            for token in model_service.stream_chat(req):
                payload = json.dumps({"token": token, "done": False})
                yield f"data: {payload}\n\n"
            # Signal completion
            yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
        except Exception as e:
            logger.error(f"Chat stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
